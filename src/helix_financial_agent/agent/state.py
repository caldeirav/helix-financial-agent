"""
Agent State Schema

Defines the typed state for the Reflexive Financial Agent using LangGraph.
"""

from typing import Dict, List, Any, Optional, Annotated, TypedDict

from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """
    State schema for the Reflexive Financial Agent.
    
    Attributes:
        messages: Conversation history with automatic message accumulation
        sender: Current node that sent the message ("generator", "reflector", "revisor")
        critique: Structured feedback from the reflector node
        critique_passed: Whether the current draft passed reflection
        iteration_count: Number of reflection cycles completed
        original_query: The user's original question (preserved for context)
        tool_outputs: Raw outputs from yfinance tools (for verification)
        selected_tools: Tools selected by ToolRAG for this query
    """
    messages: Annotated[List[BaseMessage], add_messages]
    sender: str
    critique: Optional[str]
    critique_passed: bool
    iteration_count: int
    original_query: str
    tool_outputs: List[Dict[str, Any]]
    selected_tools: List[str]


def create_initial_state(
    query: str,
    selected_tools: Optional[List[str]] = None,
) -> AgentState:
    """
    Create the initial state for a new agent invocation.
    
    Args:
        query: The user's financial question
        selected_tools: Optional list of tools selected by ToolRAG
        
    Returns:
        Initial agent state
    """
    return {
        "messages": [HumanMessage(content=query)],
        "sender": "user",
        "critique": None,
        "critique_passed": False,
        "iteration_count": 0,
        "original_query": query,
        "tool_outputs": [],
        "selected_tools": selected_tools or [],
    }
