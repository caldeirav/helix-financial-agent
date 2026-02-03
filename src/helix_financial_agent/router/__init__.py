"""
Semantic Router Module

Configuration and client for vLLM Semantic Router (vllm-sr).

Architecture:
- Router handles agent model routing (Qwen3-30B-A3B via llama.cpp)
- Gemini 2.5 Pro is called directly via langchain_google_genai SDK
  (not through router, as vLLM-SR requires IP addresses not domain names)

vLLM-SR Ports:
- 8801: HTTP entry point through Envoy (POST /v1/chat/completions)
- 8080: Classification API (GET /v1/models, GET /health)
- 9190: Prometheus metrics (GET /metrics)

See: https://vllm-semantic-router.com/docs/api/router/
"""

from .config import create_router_config, get_router_config, ROUTER_CONFIG_PATH
from .client import SemanticRouterClient, ModelTarget

__all__ = [
    "create_router_config",
    "get_router_config",
    "ROUTER_CONFIG_PATH",
    "SemanticRouterClient",
    "ModelTarget",
]
