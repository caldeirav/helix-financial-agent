"""
Configuration management for Helix Financial Agent.
Loads settings from environment variables and .env file.
"""

import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field

# Load .env file from project root
PROJECT_ROOT = Path(__file__).parent.parent.parent
load_dotenv(PROJECT_ROOT / ".env")


class ModelConfig(BaseModel):
    """Configuration for LLM models."""
    
    # llama.cpp served model (Agent)
    llama_cpp_base_url: str = Field(
        default_factory=lambda: os.getenv("LLAMA_CPP_BASE_URL", "http://localhost:8081/v1")
    )
    llama_cpp_api_key: str = Field(
        default_factory=lambda: os.getenv("LLAMA_CPP_API_KEY", "not-needed")
    )
    agent_model_name: str = Field(
        default_factory=lambda: os.getenv("AGENT_MODEL_NAME", "qwen3-30b-a3b-instruct-2507")
    )
    agent_temperature: float = Field(
        default_factory=lambda: float(os.getenv("AGENT_TEMPERATURE", "0.7"))
    )
    agent_top_p: float = Field(
        default_factory=lambda: float(os.getenv("AGENT_TOP_P", "0.8"))
    )
    agent_max_tokens: int = Field(
        default_factory=lambda: int(os.getenv("AGENT_MAX_TOKENS", "4096"))
    )
    
    # Gemini Judge
    gemini_api_key: Optional[str] = Field(
        default_factory=lambda: os.getenv("GEMINI_API_KEY")
    )
    gemini_model: str = Field(
        default_factory=lambda: os.getenv("GEMINI_MODEL", "gemini-2.5-pro")
    )


class RouterConfig(BaseModel):
    """
    Configuration for semantic router (vLLM-SR).
    
    vLLM-SR ports per documentation:
    - 8801: HTTP entry point through Envoy (OpenAI-compatible API)
    - 8889: Classification API (health, /v1/models)
    - 9190: Prometheus metrics
    
    See: https://vllm-semantic-router.com/docs/api/router/
    """
    
    router_host: str = Field(
        default_factory=lambda: os.getenv("ROUTER_HOST", "localhost")
    )
    router_http_port: int = Field(
        default_factory=lambda: int(os.getenv("ROUTER_HTTP_PORT", "8801"))
    )
    router_classify_port: int = Field(
        default_factory=lambda: int(os.getenv("ROUTER_CLASSIFY_PORT", "8889"))
    )
    router_metrics_port: int = Field(
        default_factory=lambda: int(os.getenv("ROUTER_METRICS_PORT", "9190"))
    )
    router_endpoint: str = Field(
        default_factory=lambda: os.getenv("ROUTER_ENDPOINT", "http://localhost:8801/v1")
    )


class MCPConfig(BaseModel):
    """Configuration for MCP server."""
    
    host: str = Field(
        default_factory=lambda: os.getenv("MCP_SERVER_HOST", "localhost")
    )
    port: int = Field(
        default_factory=lambda: int(os.getenv("MCP_SERVER_PORT", "8000"))
    )
    transport: str = Field(
        default_factory=lambda: os.getenv("MCP_TRANSPORT", "streamable-http")
    )


class ToolRAGConfig(BaseModel):
    """Configuration for ToolRAG system."""
    
    embedding_model: str = Field(
        default_factory=lambda: os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    )
    top_k: int = Field(
        default_factory=lambda: int(os.getenv("TOOL_RAG_TOP_K", "5"))
    )
    threshold: float = Field(
        default_factory=lambda: float(os.getenv("TOOL_RAG_THRESHOLD", "0.35"))
    )


class AgentConfig(BaseModel):
    """Configuration for agent behavior."""
    
    max_iterations: int = Field(
        default_factory=lambda: int(os.getenv("MAX_ITERATIONS", "3"))
    )


class TracingConfig(BaseModel):
    """Configuration for MLflow tracing."""
    
    tracking_uri: str = Field(
        default_factory=lambda: os.getenv("MLFLOW_TRACKING_URI", "./mlruns")
    )
    experiment_name: str = Field(
        default_factory=lambda: os.getenv("MLFLOW_EXPERIMENT_NAME", "helix-financial-agent")
    )


class PathConfig(BaseModel):
    """Configuration for file paths."""
    
    data_dir: Path = Field(
        default_factory=lambda: Path(os.getenv("DATA_DIR", "./data"))
    )
    log_dir: Path = Field(
        default_factory=lambda: Path(os.getenv("LOG_DIR", "./logs"))
    )
    log_level: str = Field(
        default_factory=lambda: os.getenv("LOG_LEVEL", "INFO")
    )


class Config(BaseModel):
    """Main configuration container."""
    
    model: ModelConfig = Field(default_factory=ModelConfig)
    router: RouterConfig = Field(default_factory=RouterConfig)
    mcp: MCPConfig = Field(default_factory=MCPConfig)
    tool_rag: ToolRAGConfig = Field(default_factory=ToolRAGConfig)
    agent: AgentConfig = Field(default_factory=AgentConfig)
    tracing: TracingConfig = Field(default_factory=TracingConfig)
    paths: PathConfig = Field(default_factory=PathConfig)


# Global configuration instance
config = Config()


def get_config() -> Config:
    """Get the global configuration instance."""
    return config
