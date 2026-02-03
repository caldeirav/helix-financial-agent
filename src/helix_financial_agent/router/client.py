"""
Semantic Router Client

Client for interacting with the vLLM Semantic Router.

Architecture:
- Router handles intelligent routing between models based on query semantics
- Agent model (Qwen3-30B-A3B via llama.cpp) - for financial analysis
- Judge model (Gemini 2.5 Pro via OpenAI-compatible API) - for evaluation/judging

Gemini OpenAI compatibility: https://ai.google.dev/gemini-api/docs/openai
- Base URL: https://generativelanguage.googleapis.com/v1beta/openai/
- Auth: Authorization: Bearer $GEMINI_API_KEY

vLLM-SR Ports (per https://vllm-semantic-router.com/docs/api/router/):
- 8801: HTTP entry point through Envoy (POST /v1/chat/completions)
- 8080: Classification API (GET /v1/models, GET /health)
- 9190: Prometheus metrics (GET /metrics)
"""

import httpx
from typing import Dict, Any, List, Optional, Union
from enum import Enum

from ..config import get_config

config = get_config()


class ModelTarget(Enum):
    """Target models for routing through vLLM-SR."""
    AGENT = "qwen3-30b-a3b"    # Local Qwen model via llama.cpp (financial analysis)
    JUDGE = "gemini-2.5-pro"   # Gemini via OpenAI-compatible API (evaluation/judging)
    GENERATOR = "gemini-2.5-pro"  # Gemini for data generation
    AUTO = "MoM"               # Model of Models - router auto-selects based on signals


class SemanticRouterClient:
    """
    Client for the vLLM Semantic Router.
    
    Routes requests to appropriate models based on query semantics:
    - Financial analysis → Qwen3-30B-A3B (local, fast)
    - Evaluation/judging → Gemini 2.5 Pro (high-quality reasoning)
    - Data generation → Gemini 2.5 Pro (diverse outputs)
    
    vLLM-SR exposes OpenAI-compatible API at port 8801 (through Envoy).
    
    Usage:
        client = SemanticRouterClient()
        
        # Auto-routing based on content semantics (MoM = Model of Models)
        response = client.chat_completion(messages, model=ModelTarget.AUTO)
        
        # Explicit routing to agent model (financial analysis)
        response = client.agent_completion(messages)
        
        # Explicit routing to judge model (evaluation)
        response = client.judge_completion(messages)
        
        # Health check
        if client.health_check():
            print("Router is healthy")
    """
    
    # Default ports per vLLM-SR documentation
    DEFAULT_HTTP_PORT = 8801      # Envoy HTTP entry point
    DEFAULT_CLASSIFY_PORT = 8080  # Classification API
    DEFAULT_METRICS_PORT = 9190   # Prometheus metrics
    
    def __init__(
        self,
        base_url: Optional[str] = None,
        host: Optional[str] = None,
        http_port: int = DEFAULT_HTTP_PORT,
    ):
        """
        Initialize the router client.
        
        Args:
            base_url: Full base URL (overrides host/port if provided)
            host: Router host (default: from config or localhost)
            http_port: HTTP port for chat completions (default: 8801)
        """
        if base_url:
            self.base_url = base_url.rstrip("/")
        else:
            router_host = host or config.router.router_host
            self.base_url = f"http://{router_host}:{http_port}"
        
        self.host = host or config.router.router_host
        self.client = httpx.Client(timeout=120.0)
    
    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: Union[ModelTarget, str] = ModelTarget.AUTO,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Send a chat completion request through the router.
        
        Endpoint: POST /v1/chat/completions (port 8801)
        
        The router will:
        1. Classify the query intent using signals (keywords, embeddings)
        2. Route to the appropriate model based on decisions
        3. Return the response with routing_metadata
        
        Args:
            messages: List of message dicts with role and content
            model: ModelTarget enum or string:
                   - ModelTarget.AUTO / "MoM": Router auto-selects based on signals
                   - ModelTarget.AGENT: Force route to Qwen (financial analysis)
                   - ModelTarget.JUDGE: Force route to Gemini (evaluation)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional parameters (tools, stream, etc.)
            
        Returns:
            Chat completion response with routing_metadata showing:
            - selected_model: Which model handled the request
            - confidence: Classification confidence score
            - processing_time_ms: Total processing time
        """
        # Convert ModelTarget enum to string value
        model_name = model.value if isinstance(model, ModelTarget) else model
        
        payload = {
            "model": model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            **kwargs,
        }
        
        response = self.client.post(
            f"{self.base_url}/v1/chat/completions",
            json=payload,
            headers={"Content-Type": "application/json"},
        )
        response.raise_for_status()
        return response.json()
    
    def agent_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Send a request to the agent model (Qwen3-30B-A3B).
        
        Use for financial analysis and user-facing agent interactions.
        Routes to local llama.cpp for fast inference.
        """
        return self.chat_completion(
            messages, model=ModelTarget.AGENT, temperature=temperature, **kwargs
        )
    
    def judge_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.0,  # Deterministic for evaluation
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Send a request to the judge model (Gemini 2.5 Pro).
        
        Use for response evaluation, scoring, and LLM-as-a-Judge tasks.
        Routes to Gemini via OpenAI-compatible API for high-quality reasoning.
        
        See: https://ai.google.dev/gemini-api/docs/openai
        """
        return self.chat_completion(
            messages, model=ModelTarget.JUDGE, temperature=temperature, **kwargs
        )
    
    def generate_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.8,  # Higher for diversity
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Send a request to the generator model (Gemini 2.5 Pro).
        
        Use for synthetic data generation and query creation.
        Routes to Gemini for diverse, high-quality outputs.
        """
        return self.chat_completion(
            messages, model=ModelTarget.GENERATOR, temperature=temperature, **kwargs
        )
    
    def list_models(self) -> Dict[str, Any]:
        """
        Get list of available models from the router.
        
        Endpoint: GET /v1/models (port 8080 - Classification API)
        
        Returns list including synthetic "MoM" (Model of Models) that
        uses router's intent classification to select the best model.
        """
        classify_url = f"http://{self.host}:{self.DEFAULT_CLASSIFY_PORT}/v1/models"
        response = self.client.get(classify_url)
        response.raise_for_status()
        return response.json()
    
    def health_check(self) -> bool:
        """
        Check if the router is healthy.
        
        Endpoint: GET /health (port 8080 - Classification API)
        """
        try:
            classify_url = f"http://{self.host}:{self.DEFAULT_CLASSIFY_PORT}/health"
            response = self.client.get(classify_url)
            return response.status_code == 200
        except Exception:
            return False
    
    def get_health_details(self) -> Dict[str, Any]:
        """
        Get detailed health status including model status and cache info.
        
        Endpoint: GET /health (port 8080)
        """
        try:
            classify_url = f"http://{self.host}:{self.DEFAULT_CLASSIFY_PORT}/health"
            response = self.client.get(classify_url)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get Prometheus metrics from the router.
        
        Endpoint: GET /metrics (port 9190)
        
        Metrics include:
        - semantic_router_requests_total
        - semantic_router_request_duration_seconds
        - semantic_router_classification_accuracy
        - semantic_router_cache_hit_ratio
        """
        try:
            metrics_url = f"http://{self.host}:{self.DEFAULT_METRICS_PORT}/metrics"
            response = self.client.get(metrics_url)
            response.raise_for_status()
            return {"status": "available", "content": response.text}
        except Exception as e:
            return {"status": "unavailable", "error": str(e)}
    
    def get_classifier_info(self) -> Dict[str, Any]:
        """
        Get classifier configuration details.
        
        Endpoint: GET /info/classifier (port 8080)
        """
        try:
            url = f"http://{self.host}:{self.DEFAULT_CLASSIFY_PORT}/info/classifier"
            response = self.client.get(url)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    def close(self):
        """Close the HTTP client."""
        self.client.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.close()
