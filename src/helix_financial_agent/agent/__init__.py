"""
Reflexive Financial Agent Module

Implements a self-correcting financial AI agent using LangGraph
with Generator → Reflector → Revisor architecture.

Architecture:
    - All LLM calls go through vLLM Semantic Router (MANDATORY)
    - All tool calls go through MCP Server (MANDATORY)
    - Router handles model selection: Qwen3 for agent, Gemini for judge
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
    AGENT_MODEL,
    JUDGE_MODEL,
    AUTO_MODEL,
)
from .graph import build_reflexive_agent, create_agent
from .runner import (
    run_agent,
    run_random_benchmark_query,
    AgentRunner,
    ServiceError,
    verify_required_services,
    check_router_health,
    check_llama_server_health,
    load_benchmark_dataset,
    get_random_query,
)

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
    "run_random_benchmark_query",
    "AgentRunner",
    # Model routing constants
    "AGENT_MODEL",
    "JUDGE_MODEL",
    "AUTO_MODEL",
    # Service verification
    "ServiceError",
    "verify_required_services",
    "check_router_health",
    "check_llama_server_health",
    # Dataset utilities
    "load_benchmark_dataset",
    "get_random_query",
]
