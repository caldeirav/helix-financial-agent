"""
Semantic Router Configuration

Generates vLLM-SR configuration following the official documentation:
https://vllm-semantic-router.com/docs/installation/configuration/

Architecture:
- Agent model (Qwen3-30B-A3B via llama.cpp) - for financial analysis tasks
- Judge model (Gemini 2.5 Pro via OpenAI-compatible API) - for evaluation and data generation
  See: https://ai.google.dev/gemini-api/docs/openai

vLLM-SR Ports (per documentation):
- 8801: HTTP entry point through Envoy (OpenAI-compatible API)
- 8080: Classification API (health, /v1/models)
- 50051: gRPC ExtProc
- 9190: Prometheus metrics

Note: API keys are loaded from environment variables at runtime, NOT stored in config files.
"""

import os
import yaml
from pathlib import Path
from typing import Optional

from ..config import get_config

config = get_config()

# Default path for router config
ROUTER_CONFIG_PATH = Path(__file__).parent.parent.parent.parent / "config" / "router_config.yaml"

# Gemini OpenAI-compatible endpoint
# See: https://ai.google.dev/gemini-api/docs/openai
GEMINI_OPENAI_HOST = "generativelanguage.googleapis.com"
GEMINI_OPENAI_BASE_PATH = "/v1beta/openai"


def create_router_config(output_path: Optional[Path] = None) -> dict:
    """
    Create the vLLM-SR router configuration.
    
    The router is configured to route:
    - Financial analysis queries → Qwen3-30B-A3B via llama.cpp (local, fast)
    - Evaluation/judging queries → Gemini 2.5 Pro via OpenAI-compatible API
    - Data generation queries → Gemini 2.5 Pro via OpenAI-compatible API
    
    Gemini OpenAI compatibility: https://ai.google.dev/gemini-api/docs/openai
    - Base URL: https://generativelanguage.googleapis.com/v1beta/openai/
    - Auth: Authorization: Bearer $GEMINI_API_KEY
    
    Configuration follows vLLM-SR documentation:
    - https://vllm-semantic-router.com/docs/installation/configuration/
    - https://vllm-semantic-router.com/docs/tutorials/intelligent-route/keyword-routing/
    
    Args:
        output_path: Optional path to write the config file
        
    Returns:
        Configuration dictionary
    """
    # Parse llama.cpp endpoint from environment
    llama_cpp_url = os.getenv("LLAMA_CPP_BASE_URL", config.model.llama_cpp_base_url)
    llama_host = "127.0.0.1"  # vLLM-SR requires IP, not hostname
    llama_port = 8080
    if ":" in llama_cpp_url:
        try:
            port_part = llama_cpp_url.split(":")[-1].split("/")[0]
            llama_port = int(port_part)
        except (ValueError, IndexError):
            pass
    
    router_config = {
        # =================================================================
        # BERT model for semantic similarity (embedding-based signals)
        # =================================================================
        "bert_model": {
            "model_id": "sentence-transformers/all-MiniLM-L12-v2",
            "threshold": 0.6,
            "use_cpu": True,
        },
        
        # =================================================================
        # vLLM Endpoints - Backend model servers
        # =================================================================
        "vllm_endpoints": [
            # Local llama.cpp endpoint for Qwen3 agent model
            {
                "name": "llama_cpp_local",
                "address": llama_host,
                "port": llama_port,
                "models": ["qwen3-30b-a3b"],
                "weight": 1,
            },
            # Gemini via OpenAI-compatible API
            # See: https://ai.google.dev/gemini-api/docs/openai
            # API key loaded from GEMINI_API_KEY environment variable
            {
                "name": "gemini_openai",
                "address": GEMINI_OPENAI_HOST,
                "port": 443,
                "protocol": "https",
                "base_path": GEMINI_OPENAI_BASE_PATH,
                "models": ["gemini-2.5-pro"],
                "weight": 1,
                # API key reference - vLLM-SR reads from environment
                "api_key_env": "GEMINI_API_KEY",
            },
        ],
        
        # =================================================================
        # Model Configuration
        # =================================================================
        "model_config": {
            "qwen3-30b-a3b": {
                "preferred_endpoints": ["llama_cpp_local"],
                "pii_policy": {
                    "allow_by_default": True,
                    "pii_types_allowed": ["EMAIL_ADDRESS", "PERSON"],
                },
            },
            "gemini-2.5-pro": {
                "preferred_endpoints": ["gemini_openai"],
                "pii_policy": {
                    "allow_by_default": True,
                },
            },
        },
        
        # =================================================================
        # Categories - Define domain categories for classification
        # =================================================================
        "categories": [
            {"name": "financial_analysis"},
            {"name": "evaluation"},
            {"name": "data_generation"},
            {"name": "general"},
        ],
        
        # =================================================================
        # Signals - Signal extraction configuration
        # Following: https://vllm-semantic-router.com/docs/tutorials/intelligent-route/keyword-routing/
        # =================================================================
        "signals": {
            # Keyword-based signals (fast pattern matching)
            "keywords": [
                # Financial analysis keywords -> route to Qwen (local, fast)
                {
                    "name": "financial_keywords",
                    "operator": "OR",
                    "keywords": [
                        "stock", "price", "ticker", "PE ratio", "market cap",
                        "dividend", "revenue", "earnings", "balance sheet",
                        "income statement", "cash flow", "financial",
                        "portfolio", "investment", "trading", "shares",
                        "fundamentals", "analyst", "recommendation",
                    ],
                    "case_sensitive": False,
                },
                {
                    "name": "company_keywords",
                    "operator": "OR",
                    "keywords": [
                        "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA",
                        "META", "JPM", "V", "WMT", "Apple", "Microsoft",
                        "Google", "Amazon", "Tesla", "Nvidia",
                    ],
                    "case_sensitive": False,
                },
                # Evaluation keywords -> route to Gemini (high-quality judging)
                {
                    "name": "evaluation_keywords",
                    "operator": "OR",
                    "keywords": [
                        "evaluate", "judge", "critique", "review", "assess",
                        "score", "rate", "correctness", "accuracy", "quality",
                        "benchmark", "grading", "validation", "is correct",
                        "check response", "verify answer",
                    ],
                    "case_sensitive": False,
                },
                # Data generation keywords -> route to Gemini (diverse outputs)
                {
                    "name": "generation_keywords",
                    "operator": "OR",
                    "keywords": [
                        "generate", "create synthetic", "dataset", "test case",
                        "benchmark question", "sample query", "example question",
                        "diverse", "variation", "synthetic data",
                    ],
                    "case_sensitive": False,
                },
            ],
            
            # Embedding-based signals (semantic similarity)
            "embeddings": [
                {
                    "name": "stock_analysis_intent",
                    "threshold": 0.70,
                    "candidates": [
                        "analyze stock performance",
                        "get financial data for company",
                        "what is the PE ratio",
                        "show me the balance sheet",
                    ],
                    "aggregation_method": "max",
                },
                {
                    "name": "evaluation_intent",
                    "threshold": 0.75,
                    "candidates": [
                        "evaluate this response for correctness",
                        "judge the quality of this answer",
                        "score this financial analysis",
                        "assess the accuracy of the response",
                    ],
                    "aggregation_method": "max",
                },
                {
                    "name": "generation_intent",
                    "threshold": 0.75,
                    "candidates": [
                        "generate synthetic test questions",
                        "create benchmark dataset",
                        "produce diverse query examples",
                    ],
                    "aggregation_method": "max",
                },
            ],
        },
        
        # =================================================================
        # Decisions - Combine signals to make routing decisions
        # Following: https://vllm-semantic-router.com/docs/installation/configuration/
        # =================================================================
        "decisions": [
            # Route financial queries to local Qwen model (fast inference)
            {
                "name": "financial_analysis",
                "description": "Route financial analysis and market data queries to local Qwen model",
                "priority": 10,
                "rules": {
                    "operator": "OR",
                    "conditions": [
                        {"type": "keyword", "name": "financial_keywords"},
                        {"type": "keyword", "name": "company_keywords"},
                        {"type": "embedding", "name": "stock_analysis_intent"},
                    ],
                },
                "modelRefs": [
                    {
                        "model": "qwen3-30b-a3b",
                        "use_reasoning": True,
                    },
                ],
                "plugins": [
                    {
                        "type": "system_prompt",
                        "configuration": {
                            "enabled": True,
                            "prompt": "You are a financial analysis expert. Provide accurate, data-driven insights about stocks, markets, and financial metrics.",
                        },
                    },
                ],
            },
            # Route evaluation/judging to Gemini (high-quality reasoning)
            {
                "name": "evaluation",
                "description": "Route evaluation and LLM-as-a-Judge tasks to Gemini 2.5 Pro",
                "priority": 15,  # Higher priority than financial
                "rules": {
                    "operator": "OR",
                    "conditions": [
                        {"type": "keyword", "name": "evaluation_keywords"},
                        {"type": "embedding", "name": "evaluation_intent"},
                    ],
                },
                "modelRefs": [
                    {
                        "model": "gemini-2.5-pro",
                        "use_reasoning": True,
                    },
                ],
                "plugins": [
                    {
                        "type": "system_prompt",
                        "configuration": {
                            "enabled": True,
                            "prompt": "You are an expert evaluator. Assess responses for correctness, completeness, and quality. Provide structured JSON feedback.",
                        },
                    },
                ],
            },
            # Route data generation to Gemini (diverse, high-quality outputs)
            {
                "name": "data_generation",
                "description": "Route synthetic data generation to Gemini 2.5 Pro",
                "priority": 15,
                "rules": {
                    "operator": "OR",
                    "conditions": [
                        {"type": "keyword", "name": "generation_keywords"},
                        {"type": "embedding", "name": "generation_intent"},
                    ],
                },
                "modelRefs": [
                    {
                        "model": "gemini-2.5-pro",
                        "use_reasoning": False,
                    },
                ],
                "plugins": [
                    {
                        "type": "system_prompt",
                        "configuration": {
                            "enabled": True,
                            "prompt": "You are a dataset generator. Create diverse, realistic test queries for financial AI agents.",
                        },
                    },
                ],
            },
            # General fallback to local Qwen
            {
                "name": "general",
                "description": "Route general queries to local Qwen model",
                "priority": 5,
                "rules": {
                    "operator": "OR",
                    "conditions": [
                        {"type": "domain", "name": "general"},
                    ],
                },
                "modelRefs": [
                    {
                        "model": "qwen3-30b-a3b",
                        "use_reasoning": False,
                    },
                ],
            },
        ],
        
        # =================================================================
        # Default model fallback
        # =================================================================
        "default_model": "qwen3-30b-a3b",
        
        # =================================================================
        # Optional: Semantic Cache
        # =================================================================
        "semantic_cache": {
            "enabled": False,
            "backend_type": "memory",
            "similarity_threshold": 0.8,
        },
        
        # =================================================================
        # Optional: Prompt Guard (jailbreak protection)
        # =================================================================
        "prompt_guard": {
            "enabled": False,
            "threshold": 0.7,
            "use_cpu": True,
        },
    }
    
    # Write to file if path provided
    output_path = output_path or ROUTER_CONFIG_PATH
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        yaml.dump(router_config, f, default_flow_style=False, sort_keys=False)
    
    return router_config


def get_router_config() -> dict:
    """
    Get the router configuration.
    
    Returns existing config if available, otherwise creates default.
    """
    if ROUTER_CONFIG_PATH.exists():
        with open(ROUTER_CONFIG_PATH, 'r') as f:
            return yaml.safe_load(f)
    
    return create_router_config()
