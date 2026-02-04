"""
Agent Graph Construction

Builds the LangGraph workflow for the Reflexive Financial Agent.

Architecture:
    - All LLM calls go through the vLLM Semantic Router (mandatory)
    - All tool calls go through the MCP server (mandatory)
    - Router handles model selection: Qwen3 for agent, Gemini for judge
"""

from typing import Literal, Optional, List, Callable

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from ..config import get_config
from ..tools import CORE_TOOLS
from .state import AgentState
from .nodes import (
    generator_node,
    tool_executor_node,
    reflector_node,
    revisor_node,
    create_generator_node,
    create_tool_executor_node,
    create_reflector_node,
    create_revisor_node,
)

config = get_config()


# =============================================================================
# CONDITIONAL ROUTING
# =============================================================================

def should_use_tools(state: AgentState) -> Literal["tools", "reflect"]:
    """
    Router after generator: check if tools need to be called.
    """
    messages = state["messages"]
    last_message = messages[-1]
    
    # If the LLM made tool calls, route to tool executor
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    
    # Otherwise, go to reflection
    return "reflect"


def after_tools(state: AgentState) -> Literal["generator", "reflect"]:
    """
    Router after tools: check where to go based on context.
    
    If we're in revision mode (iteration_count > 0), go directly to reflect.
    If we're in initial generation mode, go back to generator to process results.
    """
    # If we're in a revision cycle, go directly to reflect
    # (don't go back through generator which would make more tool calls)
    if state.get("iteration_count", 0) > 0:
        return "reflect"
    
    # Initial generation - go back to generator to process tool results
    return "generator"


def after_reflection(state: AgentState) -> Literal["revise", "end"]:
    """
    Router after reflection: decide whether to revise or finish.
    """
    # Check if passed reflection
    if state.get("critique_passed", False):
        return "end"
    
    # Check if we've hit max iterations
    max_iterations = config.agent.max_iterations
    if state.get("iteration_count", 0) >= max_iterations:
        return "end"
    
    # Need revision
    return "revise"


def after_revision(state: AgentState) -> Literal["tools", "reflect"]:
    """
    Router after revision: check if new tool calls are needed.
    """
    messages = state["messages"]
    last_message = messages[-1]
    
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    
    return "reflect"


# =============================================================================
# GRAPH CONSTRUCTION
# =============================================================================

def build_reflexive_agent(
    tools: Optional[List[Callable]] = None,
) -> StateGraph:
    """
    Construct the LangGraph workflow for the Reflexive Financial Agent.
    
    Graph Structure:
    
        START
          │
          ▼
      ┌───────────┐
      │ generator │◄────────────────┐
      └─────┬─────┘                 │
            │                       │
            ▼                       │
      [should_use_tools?]           │
            │                       │
       ┌────┴────┐                  │
       ▼         ▼                  │
    ┌──────┐  ┌─────────┐           │
    │tools │  │reflect  │           │
    └──┬───┘  └────┬────┘           │
       │           │                │
       │     [after_reflection?]    │
       │           │                │
       │      ┌────┴────┐           │
       │      ▼         ▼           │
       │   ┌──────┐   ┌───┐         │
       │   │revise│   │END│         │
       │   └──┬───┘   └───┘         │
       │      │                     │
       │      └─────────────────────┘
       │
       └──────────────► (back to generator for tool response handling)
    
    Architecture:
        - All LLM calls go through vLLM-SR router (mandatory)
        - Router routes to Qwen3 (agent) or Gemini (judge) based on model
        - All tool calls go through MCP server (mandatory)
    
    Tool Binding:
        Only the tools passed to this function are bound to the LLM in
        generator and revisor nodes. ToolRAG selects relevant tools based
        on the user query, and only those selected tools are passed here.
        This keeps the agent focused on relevant tools and reduces context size.
        
        Flow: Query → ToolRAG → Selected Tools → build_reflexive_agent(tools)
              → generator.bind_tools(tools) & revisor.bind_tools(tools)
    
    Args:
        tools: List of tools to bind to LLM nodes. These should be the
               tools selected by ToolRAG for the current query.
               Falls back to CORE_TOOLS if None.
        
    Returns:
        Compiled StateGraph with tools bound to generator and revisor nodes
    """
    tools = tools or CORE_TOOLS
    
    # Create node functions with configured tools
    gen_node = create_generator_node(tools=tools)
    tool_node = create_tool_executor_node(tools=tools)
    ref_node = create_reflector_node()
    rev_node = create_revisor_node(tools=tools)
    
    # Create the graph
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("generator", gen_node)
    workflow.add_node("tools", tool_node)
    workflow.add_node("reflect", ref_node)
    workflow.add_node("revise", rev_node)
    
    # Set entry point
    workflow.set_entry_point("generator")
    
    # Add conditional edges from generator
    workflow.add_conditional_edges(
        "generator",
        should_use_tools,
        {
            "tools": "tools",
            "reflect": "reflect",
        }
    )
    
    # Tools route based on context (initial vs revision)
    workflow.add_conditional_edges(
        "tools",
        after_tools,
        {
            "generator": "generator",  # Initial generation - process tool results
            "reflect": "reflect",       # Revision mode - go directly to reflection
        }
    )
    
    # Add conditional edges from reflector
    workflow.add_conditional_edges(
        "reflect",
        after_reflection,
        {
            "revise": "revise",
            "end": END,
        }
    )
    
    # Add conditional edges from revisor
    workflow.add_conditional_edges(
        "revise",
        after_revision,
        {
            "tools": "tools",
            "reflect": "reflect",
        }
    )
    
    return workflow


def create_agent(
    tools: Optional[List[Callable]] = None,
    checkpointer: Optional[MemorySaver] = None,
):
    """
    Create a compiled agent with memory.
    
    Required services (must be running):
    - llama.cpp server: Serves Qwen3 model
    - vLLM-SR router: Routes between agent and judge models
    - MCP server: Executes tool calls
    
    Args:
        tools: Optional list of tools to use (default: CORE_TOOLS via MCP)
        checkpointer: Optional memory checkpointer
        
    Returns:
        Compiled agent graph
    """
    workflow = build_reflexive_agent(tools=tools)
    checkpointer = checkpointer or MemorySaver()
    return workflow.compile(checkpointer=checkpointer)
