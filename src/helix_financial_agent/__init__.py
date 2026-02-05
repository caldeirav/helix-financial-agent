"""
Helix Financial Agent - A Reflexive Financial AI Agent

Features:
- Reflexive metacognitive architecture with self-correction
- Semantic routing for model selection (vLLM-SR)
- ToolRAG for dynamic tool binding
- MCP server deployment via FastMCP
- Comprehensive evaluation with LLM-as-a-Judge
- MLflow tracing for end-to-end observability

MLflow Tracing:
    The agent automatically traces all LLM calls, tool executions, and routing
    decisions via MLflow. Per-trace assessments are logged:
    - tool_selection_successful: Y/N
    - model_selection_successful: Y/N
    - judge_score: 0-10
    
    View traces at http://localhost:5000 after running: mlflow ui --port 5000
"""

__version__ = "0.1.0"

# Expose tracing utilities
from .tracing import (
    setup_mlflow_tracing,
    log_run_assessments,
    TracingContext,
)
