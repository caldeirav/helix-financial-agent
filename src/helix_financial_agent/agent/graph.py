"""
Agent Graph Construction

Builds the LangGraph workflow for the Reflexive Financial Agent.
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
    use_router: bool = False,
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
    
    Args:
        tools: Optional list of tools to use (default: CORE_TOOLS)
        use_router: Whether to use semantic router for model selection
        
    Returns:
        Compiled StateGraph
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
    
    # Tools always go back to generator (to process tool results)
    workflow.add_edge("tools", "generator")
    
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
    use_router: bool = False,
    checkpointer: Optional[MemorySaver] = None,
):
    """
    Create a compiled agent with memory.
    
    Args:
        tools: Optional list of tools to use
        use_router: Whether to use semantic router
        checkpointer: Optional memory checkpointer
        
    Returns:
        Compiled agent graph
    """
    workflow = build_reflexive_agent(tools=tools, use_router=use_router)
    checkpointer = checkpointer or MemorySaver()
    return workflow.compile(checkpointer=checkpointer)
