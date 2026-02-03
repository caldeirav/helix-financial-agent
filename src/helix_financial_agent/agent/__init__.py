"""
Reflexive Financial Agent Module

Implements a self-correcting financial AI agent using LangGraph
with Generator → Reflector → Revisor architecture.
"""

from .state import AgentState, create_initial_state
from .prompts import (
    GENERATOR_SYSTEM_PROMPT,
    REFLECTOR_SYSTEM_PROMPT,
    REVISOR_SYSTEM_PROMPT,
)
from .nodes import (
    generator_node,
    tool_executor_node,
    reflector_node,
    revisor_node,
)
from .graph import build_reflexive_agent, create_agent
from .runner import run_agent, AgentRunner

__all__ = [
    "AgentState",
    "create_initial_state",
    "GENERATOR_SYSTEM_PROMPT",
    "REFLECTOR_SYSTEM_PROMPT",
    "REVISOR_SYSTEM_PROMPT",
    "generator_node",
    "tool_executor_node",
    "reflector_node",
    "revisor_node",
    "build_reflexive_agent",
    "create_agent",
    "run_agent",
    "AgentRunner",
]
