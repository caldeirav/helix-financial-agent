"""
MLflow Tracing Integration for Helix Financial Agent.

Provides end-to-end tracing for LangGraph workflows including:
- Automatic tracing of LLM calls and tool executions
- Custom assessments for tool selection, model selection, and judge scores
- Benchmark run management with MLflow runs

Usage:
    from helix_financial_agent.tracing import setup_mlflow_tracing, log_run_assessments

    # Initialize tracing (call once at startup)
    setup_mlflow_tracing()

    # After agent execution, log assessments
    log_run_assessments(
        trace_id=trace_id,
        tool_selection_success=True,
        model_selection_success=True,
        judge_score=8.5,
        ...
    )
"""

import os
from typing import Any, Dict, List, Optional

import mlflow
from mlflow.entities import AssessmentSource, Expectation, Feedback

from .config import get_config

config = get_config()

# Module-level state
_tracing_initialized = False


def setup_mlflow_tracing(
    tracking_uri: Optional[str] = None,
    experiment_name: Optional[str] = None,
    enable_autolog: bool = True,
) -> bool:
    """
    Configure MLflow tracing for the financial agent.

    This should be called once at application startup. It configures:
    - MLflow tracking URI and experiment
    - LangChain/LangGraph autologging for automatic trace capture

    Args:
        tracking_uri: MLflow tracking URI (default from config)
        experiment_name: Experiment name (default from config)
        enable_autolog: Whether to enable LangChain autologging

    Returns:
        True if tracing was initialized successfully

    Example:
        >>> setup_mlflow_tracing()
        ✓ MLflow tracing configured
          Tracking URI: ./mlruns
          Experiment: helix-financial-agent
        True
    """
    global _tracing_initialized

    if _tracing_initialized:
        return True

    tracking_uri = tracking_uri or config.tracing.tracking_uri
    experiment_name = experiment_name or config.tracing.experiment_name

    # Configure MLflow
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    # Enable LangChain/LangGraph autologging
    # This captures all LLM calls, tool executions, and chain runs automatically
    if enable_autolog:
        try:
            mlflow.langchain.autolog()
            print("✓ LangChain autologging enabled")
        except Exception as e:
            print(f"⚠ LangChain autologging not available: {e}")
            print("  MLflow will still track runs and metrics, but without automatic LLM tracing.")

    print(f"\n✓ MLflow tracing configured")
    print(f"  Tracking URI: {tracking_uri}")
    print(f"  Experiment: {experiment_name}")

    _tracing_initialized = True
    return True


def is_tracing_enabled() -> bool:
    """Check if MLflow tracing has been initialized."""
    return _tracing_initialized


def get_current_trace_id() -> Optional[str]:
    """
    Get the trace ID of the most recently created trace.

    Returns:
        The trace ID string, or None if no trace is active
    """
    try:
        return mlflow.get_last_active_trace_id()
    except Exception:
        return None


def log_run_assessments(
    trace_id: str,
    tool_selection_success: Optional[bool] = None,
    model_selection_success: Optional[bool] = None,
    judge_score: Optional[float] = None,
    judge_reasoning: Optional[str] = None,
    judge_category: Optional[str] = None,
    selected_tools: Optional[List[str]] = None,
    expected_tools: Optional[List[str]] = None,
    tools_used: Optional[List[str]] = None,
    latency_seconds: Optional[float] = None,
    iteration_count: Optional[int] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, bool]:
    """
    Log custom assessments (Feedback) to an MLflow trace.

    This logs the key metrics for each agent run:
    - tool_selection_successful: Y/N - Did ToolRAG select the right tools?
    - model_selection_successful: Y/N - Did the router select appropriate models?
    - judge_score: 0-10 score from LLM-as-a-Judge

    Args:
        trace_id: The MLflow trace ID to attach assessments to
        tool_selection_success: Whether correct tools were selected
        model_selection_success: Whether correct models were routed to
        judge_score: Score from LLM-as-a-Judge (0-10)
        judge_reasoning: Explanation from the judge
        judge_category: "valid" or "hazard"
        selected_tools: List of tools selected by ToolRAG
        expected_tools: List of expected tools from benchmark
        tools_used: List of tools actually used during execution
        latency_seconds: Total execution time
        iteration_count: Number of revision iterations
        metadata: Additional metadata to attach

    Returns:
        Dict indicating which assessments were logged successfully

    Example:
        >>> log_run_assessments(
        ...     trace_id="tr-123",
        ...     tool_selection_success=True,
        ...     model_selection_success=True,
        ...     judge_score=8.5,
        ...     judge_reasoning="Accurate financial analysis",
        ... )
        {'tool_selection': True, 'model_selection': True, 'judge_score': True}
    """
    results = {}

    if not trace_id:
        print("⚠ No trace_id provided, skipping assessment logging")
        return results

    # 1. TOOL SELECTION SUCCESS (Y/N)
    if tool_selection_success is not None:
        try:
            rationale = f"Selected: {selected_tools or []}"
            if expected_tools:
                rationale += f", Expected: {expected_tools}"

            feedback = Feedback(
                name="tool_selection_successful",
                value=tool_selection_success,
                rationale=rationale[:500],
                source=AssessmentSource(source_type="CODE", source_id="tool_rag"),
                metadata={
                    "tools_selected": selected_tools or [],
                    "tools_expected": expected_tools or [],
                    "tools_used": tools_used or [],
                },
            )
            mlflow.log_assessment(trace_id=trace_id, assessment=feedback)
            results["tool_selection"] = True
        except Exception as e:
            print(f"⚠ Failed to log tool_selection assessment: {e}")
            results["tool_selection"] = False

    # 2. MODEL SELECTION SUCCESS (Y/N)
    if model_selection_success is not None:
        try:
            feedback = Feedback(
                name="model_selection_successful",
                value=model_selection_success,
                rationale="Checked if semantic router selected appropriate models per node",
                source=AssessmentSource(source_type="CODE", source_id="semantic_router"),
            )
            mlflow.log_assessment(trace_id=trace_id, assessment=feedback)
            results["model_selection"] = True
        except Exception as e:
            print(f"⚠ Failed to log model_selection assessment: {e}")
            results["model_selection"] = False

    # 3. JUDGE SCORE
    if judge_score is not None:
        try:
            feedback = Feedback(
                name="judge_score",
                value=judge_score,
                rationale=judge_reasoning[:500] if judge_reasoning else None,
                source=AssessmentSource(source_type="LLM_JUDGE", source_id="gemini-2.5-pro"),
                metadata={
                    "category": judge_category,
                    "max_score": 10,
                    **(metadata or {}),
                },
            )
            mlflow.log_assessment(trace_id=trace_id, assessment=feedback)
            results["judge_score"] = True
        except Exception as e:
            print(f"⚠ Failed to log judge_score assessment: {e}")
            results["judge_score"] = False

    # 4. LATENCY
    if latency_seconds is not None:
        try:
            feedback = Feedback(
                name="latency_seconds",
                value=round(latency_seconds, 3),
                source=AssessmentSource(source_type="CODE", source_id="agent_runner"),
            )
            mlflow.log_assessment(trace_id=trace_id, assessment=feedback)
            results["latency"] = True
        except Exception as e:
            print(f"⚠ Failed to log latency assessment: {e}")
            results["latency"] = False

    # 5. ITERATION COUNT
    if iteration_count is not None:
        try:
            feedback = Feedback(
                name="iteration_count",
                value=iteration_count,
                source=AssessmentSource(source_type="CODE", source_id="agent_runner"),
            )
            mlflow.log_assessment(trace_id=trace_id, assessment=feedback)
            results["iterations"] = True
        except Exception as e:
            print(f"⚠ Failed to log iteration_count assessment: {e}")
            results["iterations"] = False

    # 6. EXPECTED TOOLS (as Expectation)
    if expected_tools:
        try:
            expectation = Expectation(
                name="expected_tools",
                value=expected_tools,
                source=AssessmentSource(source_type="HUMAN", source_id="benchmark_dataset"),
            )
            mlflow.log_assessment(trace_id=trace_id, assessment=expectation)
            results["expected_tools"] = True
        except Exception as e:
            print(f"⚠ Failed to log expected_tools expectation: {e}")
            results["expected_tools"] = False

    return results


def evaluate_tool_selection(
    selected_tools: List[str],
    expected_tools: Optional[List[str]],
) -> bool:
    """
    Evaluate if tool selection was successful.

    Success criteria: All expected tools must be present in selected tools.

    Args:
        selected_tools: Tools selected by ToolRAG
        expected_tools: Expected tools from benchmark (can be None)

    Returns:
        True if all expected tools were selected, or if no expectations
    """
    if not expected_tools:
        return True  # No expectation = success by default

    expected_set = set(expected_tools) if isinstance(expected_tools, list) else {expected_tools}
    selected_set = set(selected_tools)

    return expected_set.issubset(selected_set)


def evaluate_model_selection(
    trace_log: List[Dict[str, Any]],
) -> bool:
    """
    Evaluate if model routing was correct based on node types.

    Expected routing:
    - generator, revisor nodes → Qwen3 (financial analysis)
    - reflect node → Gemini (evaluation/judging)

    Args:
        trace_log: List of trace entries from agent execution

    Returns:
        True if routing appears correct (default True if not verifiable)
    """
    # Parse trace_log to check routing decisions
    # This is a heuristic check - routing metadata may not always be available
    for entry in trace_log:
        event = entry.get("event", "")
        data = entry.get("data", {})

        # Check if routing metadata is available
        if "routing_metadata" in data:
            routing = data["routing_metadata"]
            model = routing.get("selected_model", "").lower()

            # Check if reflect node was routed to Gemini
            if "reflect" in event and "gemini" not in model:
                return False

            # Check if generator/revisor were routed to Qwen
            if ("generator" in event or "revisor" in event) and "qwen" not in model:
                return False

    # Default to True if we can't verify
    return True


def start_benchmark_run(
    run_name: str,
    params: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Start an MLflow run for a benchmark session.

    Args:
        run_name: Name for the run (e.g., "benchmark_20260205_143000")
        params: Parameters to log (sample_size, model, etc.)

    Returns:
        The run ID

    Example:
        >>> run_id = start_benchmark_run(
        ...     "benchmark_20260205",
        ...     {"sample_size": 10, "model": "qwen3-30b"}
        ... )
    """
    run = mlflow.start_run(run_name=run_name)

    if params:
        mlflow.log_params(params)

    return run.info.run_id


def end_benchmark_run(
    metrics: Optional[Dict[str, float]] = None,
    artifacts: Optional[List[str]] = None,
):
    """
    End the current MLflow run, logging final metrics and artifacts.

    Args:
        metrics: Final metrics to log (avg_score, success_rate, etc.)
        artifacts: Paths to artifact files to log
    """
    if metrics:
        mlflow.log_metrics(metrics)

    if artifacts:
        for artifact_path in artifacts:
            if os.path.exists(artifact_path):
                mlflow.log_artifact(artifact_path)

    mlflow.end_run()


class TracingContext:
    """
    Context manager for MLflow benchmark runs.

    Example:
        >>> with TracingContext("benchmark_run", {"sample_size": 10}) as ctx:
        ...     for query in queries:
        ...         result = runner.run(query)
        ...         ctx.add_result(result)
        ...     ctx.log_metrics({"avg_score": 8.5})
    """

    def __init__(self, run_name: str, params: Optional[Dict[str, Any]] = None):
        self.run_name = run_name
        self.params = params or {}
        self.run_id: Optional[str] = None
        self.results: List[Dict[str, Any]] = []

    def __enter__(self) -> "TracingContext":
        setup_mlflow_tracing()
        self.run_id = start_benchmark_run(self.run_name, self.params)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        mlflow.end_run()
        return False

    def add_result(self, result: Dict[str, Any]):
        """Add a result to track."""
        self.results.append(result)

    def log_metrics(self, metrics: Dict[str, float]):
        """Log metrics to the current run."""
        mlflow.log_metrics(metrics)

    def log_artifact(self, path: str):
        """Log an artifact to the current run."""
        if os.path.exists(path):
            mlflow.log_artifact(path)
