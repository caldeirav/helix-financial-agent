"""
Agent Node Implementations

Defines the nodes for the reflexive agent graph:
- generator_node: Drafts responses using tools
- tool_executor_node: Executes tool calls
- reflector_node: Critiques drafts
- revisor_node: Refines responses
"""

import json
import re
from typing import Dict, Any, List, Optional, Callable

from langchain_core.messages import AIMessage, SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import ToolNode

from ..config import get_config
from ..tools import CORE_TOOLS, ALL_TOOLS
from .state import AgentState
from .prompts import (
    GENERATOR_SYSTEM_PROMPT,
    REFLECTOR_SYSTEM_PROMPT,
    REVISOR_SYSTEM_PROMPT,
)

config = get_config()


def create_llm(
    use_router: bool = False,
    temperature: Optional[float] = None,
) -> ChatOpenAI:
    """
    Create an LLM instance configured for the agent.
    
    Args:
        use_router: Whether to use the semantic router endpoint
        temperature: Optional temperature override
        
    Returns:
        Configured ChatOpenAI instance
    """
    base_url = (
        config.router.router_endpoint 
        if use_router 
        else config.model.llama_cpp_base_url
    )
    
    return ChatOpenAI(
        base_url=base_url,
        api_key=config.model.llama_cpp_api_key,
        model=config.model.agent_model_name,
        temperature=temperature or config.model.agent_temperature,
        max_tokens=config.model.agent_max_tokens,
    )


def generator_node(
    state: AgentState,
    llm: Optional[ChatOpenAI] = None,
    tools: Optional[List[Callable]] = None,
) -> Dict[str, Any]:
    """
    Generator Node (initial_responder):
    - Takes user query and generates draft response using yfinance tools
    - This is the primary "thinking" node that gathers data and drafts answers
    
    Args:
        state: Current agent state
        llm: Optional LLM instance (creates default if not provided)
        tools: Optional list of tools to use
        
    Returns:
        State update with new message
    """
    llm = llm or create_llm()
    tools = tools or CORE_TOOLS
    
    messages = state["messages"]
    iteration = state["iteration_count"]
    
    # Bind tools to LLM
    llm_with_tools = llm.bind_tools(tools)
    
    # Build context with any previous critique
    system_message = SystemMessage(content=GENERATOR_SYSTEM_PROMPT)
    
    # If this is a revision iteration, include the critique context
    if iteration > 0 and state.get("critique"):
        revision_context = f"""
REVISION REQUIRED (Iteration {iteration}):
Previous critique: {state['critique']}

Please revise your response to address all issues raised.
"""
        messages = messages + [SystemMessage(content=revision_context)]
    
    # Invoke LLM with tools
    response = llm_with_tools.invoke([system_message] + messages)
    
    return {
        "messages": [response],
        "sender": "generator",
    }


def tool_executor_node(
    state: AgentState,
    tools: Optional[List[Callable]] = None,
) -> Dict[str, Any]:
    """
    Tool Executor Node:
    - Executes any tool calls made by the generator
    - Stores tool outputs for verification by reflector
    
    Args:
        state: Current agent state
        tools: Optional list of tools to use
        
    Returns:
        State update with tool results
    """
    tools = tools or CORE_TOOLS
    messages = state["messages"]
    last_message = messages[-1]
    
    tool_outputs = []
    
    # Check if there are tool calls to execute
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        tool_node = ToolNode(tools)
        result = tool_node.invoke({"messages": messages})
        
        # Extract tool outputs for verification
        for msg in result.get("messages", []):
            if hasattr(msg, "content"):
                tool_outputs.append({
                    "tool": msg.name if hasattr(msg, "name") else "unknown",
                    "output": msg.content
                })
        
        return {
            "messages": result.get("messages", []),
            "tool_outputs": state.get("tool_outputs", []) + tool_outputs,
            "sender": "tools",
        }
    
    return {"sender": "tools"}


def reflector_node(
    state: AgentState,
    llm: Optional[ChatOpenAI] = None,
) -> Dict[str, Any]:
    """
    Reflector Node:
    - Acts as "Senior Risk Analyst" critiquing the draft
    - Checks for hallucinations, advice violations, completeness
    - Returns structured critique with pass/fail decision
    
    Args:
        state: Current agent state
        llm: Optional LLM instance
        
    Returns:
        State update with critique
    """
    llm = llm or create_llm(temperature=0.3)  # Lower temp for evaluation
    
    messages = state["messages"]
    tool_outputs = state.get("tool_outputs", [])
    original_query = state.get("original_query", "")
    
    # Find the last AI response (the draft to evaluate)
    draft_response = None
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and msg.content:
            draft_response = msg.content
            break
    
    if not draft_response:
        return {
            "critique": "No draft response found to evaluate.",
            "critique_passed": False,
            "sender": "reflector",
        }
    
    # Build reflection prompt
    reflection_prompt = f"""
ORIGINAL USER QUERY:
{original_query}

TOOL OUTPUTS (Ground Truth):
{json.dumps(tool_outputs, indent=2)}

DRAFT RESPONSE TO EVALUATE:
{draft_response}

Please evaluate this draft response according to your criteria.
"""
    
    # Use the same LLM for reflection
    response = llm.invoke([
        SystemMessage(content=REFLECTOR_SYSTEM_PROMPT),
        HumanMessage(content=reflection_prompt)
    ])
    
    # Parse the critique
    critique_text = response.content
    critique_passed = False
    
    # Try to parse JSON from response
    try:
        json_match = re.search(r'\{[\s\S]*\}', critique_text)
        if json_match:
            critique_json = json.loads(json_match.group())
            critique_passed = critique_json.get("passed", False)
    except (json.JSONDecodeError, AttributeError):
        # If JSON parsing fails, look for explicit pass/fail indicators
        critique_passed = '"passed": true' in critique_text.lower() or '"passed":true' in critique_text.lower()
    
    return {
        "messages": [AIMessage(content=f"[REFLECTION]: {critique_text}")],
        "critique": critique_text,
        "critique_passed": critique_passed,
        "iteration_count": state["iteration_count"] + 1,
        "sender": "reflector",
    }


def revisor_node(
    state: AgentState,
    llm: Optional[ChatOpenAI] = None,
    tools: Optional[List[Callable]] = None,
) -> Dict[str, Any]:
    """
    Revisor Node:
    - Takes the critique and generates an improved response
    - Addresses all issues raised by the reflector
    
    Args:
        state: Current agent state
        llm: Optional LLM instance
        tools: Optional list of tools to use
        
    Returns:
        State update with revised response
    """
    llm = llm or create_llm()
    tools = tools or CORE_TOOLS
    
    messages = state["messages"]
    critique = state.get("critique", "")
    original_query = state.get("original_query", "")
    
    # Bind tools to LLM
    llm_with_tools = llm.bind_tools(tools)
    
    # Find the original draft
    draft_response = None
    for msg in messages:
        if isinstance(msg, AIMessage) and msg.content and not msg.content.startswith("[REFLECTION]"):
            draft_response = msg.content
    
    revision_prompt = f"""
ORIGINAL QUERY: {original_query}

PREVIOUS DRAFT:
{draft_response}

CRITIQUE FROM RISK ANALYST:
{critique}

Please generate an improved response that addresses all issues raised.
"""
    
    response = llm_with_tools.invoke([
        SystemMessage(content=REVISOR_SYSTEM_PROMPT),
        HumanMessage(content=revision_prompt)
    ])
    
    return {
        "messages": [response],
        "sender": "revisor",
    }


# =============================================================================
# NODE FACTORY FUNCTIONS
# =============================================================================

def create_generator_node(
    llm: Optional[ChatOpenAI] = None,
    tools: Optional[List[Callable]] = None,
) -> Callable[[AgentState], Dict[str, Any]]:
    """Create a generator node with specific LLM and tools."""
    def node(state: AgentState) -> Dict[str, Any]:
        return generator_node(state, llm=llm, tools=tools)
    return node


def create_tool_executor_node(
    tools: Optional[List[Callable]] = None,
) -> Callable[[AgentState], Dict[str, Any]]:
    """Create a tool executor node with specific tools."""
    def node(state: AgentState) -> Dict[str, Any]:
        return tool_executor_node(state, tools=tools)
    return node


def create_reflector_node(
    llm: Optional[ChatOpenAI] = None,
) -> Callable[[AgentState], Dict[str, Any]]:
    """Create a reflector node with specific LLM."""
    def node(state: AgentState) -> Dict[str, Any]:
        return reflector_node(state, llm=llm)
    return node


def create_revisor_node(
    llm: Optional[ChatOpenAI] = None,
    tools: Optional[List[Callable]] = None,
) -> Callable[[AgentState], Dict[str, Any]]:
    """Create a revisor node with specific LLM and tools."""
    def node(state: AgentState) -> Dict[str, Any]:
        return revisor_node(state, llm=llm, tools=tools)
    return node
