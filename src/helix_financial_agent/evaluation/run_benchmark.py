"""
Benchmark Runner with MLflow Tracing and Verbose Logging

Runs comprehensive evaluation on the synthetic dataset with
end-to-end MLflow tracing, assessment logging, and detailed verbose output.

Verbose Logging:
- All model interactions (requests/responses) 
- Tool calls with arguments and outputs
- Routing decisions (which model was selected)
- ToolRAG selection tables with similarity scores
- Flow and decision tracking
- End-of-run summary

Output Formatting:
- Full query text is displayed (not truncated) for clarity
- ToolRAG tables use Rich Table for proper column alignment
- Service checks are performed once at benchmark start (not per-query)

MLflow Integration:
- Each benchmark session is wrapped in an MLflow run
- Per-trace assessments: tool_selection, model_selection, judge_score
- Aggregate metrics logged: avg_score, success_rate, tool_selection_accuracy
- Results saved as MLflow artifacts

Performance Optimizations:
- Service verification happens once at start, not per-query
- ToolStore singleton caches embedding model (avoids repeated loading)
- Logger passed to ToolSelector for integrated output

View results at http://localhost:5000 (mlflow ui)
"""

import json
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

import mlflow
import pandas as pd
from rich.console import Console
from rich.progress import Progress, TaskID
from rich.table import Table

from ..config import get_config
from ..agent import run_agent
from ..agent.runner import verify_required_services
from ..tracing import setup_mlflow_tracing, TracingContext
from ..verbose_logging import VerboseLogger, get_logger, reset_logger
from .judge import GeminiJudge

console = Console()
config = get_config()


class BenchmarkRunner:
    """
    Runs benchmark evaluation on a dataset with MLflow tracing and verbose logging.
    
    Features:
    - Runs agent on each query with MLflow tracing
    - Evaluates with Gemini judge
    - Logs per-trace assessments (tool_selection, model_selection, judge_score)
    - Tracks aggregate metrics in MLflow
    - Saves results to CSV/JSON as MLflow artifacts
    - Verbose logging of all model interactions, tool calls, and routing decisions
    
    MLflow Integration:
    - Each benchmark session is an MLflow run
    - View at http://localhost:5000 after running: mlflow ui
    """
    
    def __init__(
        self,
        judge: Optional[GeminiJudge] = None,
        output_dir: Optional[Path] = None,
        enable_tracing: bool = True,
        verbose: bool = True,
        logger: Optional[VerboseLogger] = None,
    ):
        """
        Initialize the benchmark runner.
        
        Args:
            judge: Optional GeminiJudge instance
            output_dir: Optional output directory for results
            enable_tracing: Whether to enable MLflow tracing (default: True)
            verbose: Enable verbose logging (default: True)
            logger: Optional VerboseLogger instance
        """
        self.judge = judge or GeminiJudge()
        self.output_dir = output_dir or config.paths.data_dir
        self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.enable_tracing = enable_tracing
        self.verbose = verbose
        self.logger = logger or get_logger(verbose=verbose, reset=True)
        
        # Initialize MLflow tracing
        if enable_tracing:
            setup_mlflow_tracing()
        
        # Log initialization
        self.logger.log_flow("Benchmark Runner Initialized", {
            "output_dir": str(self.output_dir),
            "mlflow_tracing": enable_tracing,
            "verbose_logging": verbose,
            "judge_model": config.model.gemini_model,
        })
    
    def run(
        self,
        dataset: List[Dict[str, Any]],
        max_queries: Optional[int] = None,
        verbose: bool = True,
        verbose_agent: bool = False,
        use_tool_rag: bool = True,
    ) -> pd.DataFrame:
        """
        Run the benchmark on a dataset with MLflow tracing and verbose logging.
        
        Each benchmark session is wrapped in an MLflow run for tracking.
        Per-trace assessments are logged for each query:
        - tool_selection_successful: Y/N
        - model_selection_successful: Y/N
        - judge_score: 0-10
        
        Verbose logging includes:
        - All model requests and responses
        - Tool calls with arguments and outputs  
        - Routing decisions (which model selected)
        - End-of-run summary
        
        Args:
            dataset: List of query dicts
            max_queries: Optional limit on queries
            verbose: Print progress
            verbose_agent: Print full agent output
            use_tool_rag: Use ToolRAG for tool selection
            
        Returns:
            DataFrame with results
        """
        if max_queries:
            dataset = dataset[:max_queries]
        
        results = []
        total = len(dataset)
        
        # Track failures and assessments
        failure_modes = {
            "empty_response": 0,
            "tool_error": 0,
            "exception": 0,
            "judge_error": 0,
        }
        assessment_totals = {
            "tool_selection_correct": 0,
            "model_selection_correct": 0,
            "judge_scores": [],
        }
        
        # Get run name for MLflow
        run_name = f"benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Log benchmark start
        self.logger.log_flow("Benchmark Starting", {
            "total_queries": total,
            "use_tool_rag": use_tool_rag,
            "run_name": run_name,
        })
        
        # Wrap in MLflow run if tracing enabled
        mlflow_run = None
        if self.enable_tracing:
            mlflow_run = mlflow.start_run(run_name=run_name)
            
            # Log benchmark parameters
            mlflow.log_params({
                "total_queries": total,
                "use_tool_rag": use_tool_rag,
                "agent_model": config.model.agent_model_name,
                "judge_model": config.model.gemini_model,
                "max_iterations": config.agent.max_iterations,
            })
        
        try:
            console.print(f"\n[bold cyan]🚀 Starting benchmark on {total} queries...[/bold cyan]")
            console.print(f"   ToolRAG: {'ENABLED' if use_tool_rag else 'DISABLED'}")
            if self.enable_tracing:
                console.print(f"   MLflow Run: {run_name}")
            if self.verbose:
                console.print(f"   Verbose Logging: ENABLED")
            console.print("=" * 60)
            
            # Pre-check all required services once before the benchmark loop
            console.print("\n[cyan]🔍 Checking required services...[/cyan]")
            try:
                service_info = verify_required_services(check_mcp=True, check_router=True, verbose=True)
                console.print("[green]✅ All required services are running[/green]")
            except Exception as e:
                console.print(f"[red]❌ Service check failed: {e}[/red]")
                raise
            
            benchmark_start = time.time()
            
            for i, query_item in enumerate(dataset):
                query = query_item["query"]
                category = query_item.get("category", "valid")
                subcategory = query_item.get("subcategory", "")
                expected_tools = query_item.get("expected_tools", [])
                
                # Log query start (show full query, not truncated)
                self.logger.log_flow(f"Query [{i+1}/{total}]", {
                    "category": category,
                    "subcategory": subcategory,
                    "query": query,  # Full query
                    "expected_tools": expected_tools,
                })
                
                if verbose:
                    console.print(f"\n[bold]Query [{i+1}/{total}][/bold] [{category}/{subcategory}]")
                    console.print(f"  {query}")  # Full query
                
                try:
                    # Log agent request
                    request_id = self.logger.log_llm_request(
                        node=f"agent/{i+1}",
                        prompt=query[:200],
                        model=config.model.agent_model_name,
                    )
                    
                    # Run agent with metadata for assessment tracking
                    # Pass the logger so ToolRAG can log tool selection details
                    # Skip service check since we verified services at benchmark start
                    start_time = time.time()
                    agent_result = run_agent(
                        query,
                        verbose=verbose_agent,
                        use_tool_rag=use_tool_rag,
                        run_evaluation=True,  # Enable evaluation for assessments
                        query_metadata=query_item,
                        enable_tracing=self.enable_tracing,
                        logger=self.logger,  # Pass logger for tool selection output
                        skip_service_check=True,  # Services already verified at start
                    )
                    agent_time = time.time() - start_time
                    
                    response = agent_result.get("response", "")
                    
                    # Log agent response with routing info
                    routed_model = agent_result.get("routed_model", "unknown")
                    routing_decision = agent_result.get("routing_decision")
                    
                    self.logger.log_llm_response(
                        node=f"agent/{i+1}",
                        response=response[:300] if response else "",
                        routed_to=routed_model,
                        routing_decision=routing_decision,
                        request_id=request_id,
                        success=bool(response),
                    )
                    
                    # Log routing decision
                    if routed_model and routed_model != "unknown":
                        # Check if routing was expected
                        is_fallback = False
                        if category == "valid" and "qwen" not in routed_model.lower():
                            # Valid queries should typically route to local Qwen
                            is_fallback = True
                        
                        self.logger.log_routing_decision(
                            requested_model="MoM",
                            routed_model=routed_model,
                            decision_name=routing_decision,
                            is_fallback=is_fallback,
                        )
                    
                    # Log tool calls from agent result
                    tools_used = agent_result.get("unique_tools", [])
                    for tool in tools_used:
                        self.logger.log_tool_call(tool, {"from_agent_result": True})
                        self.logger.log_tool_response(tool, "executed", success=True)
                    
                    # Check for failures
                    failure_mode = None
                    if not response or len(response.strip()) < 20:
                        failure_modes["empty_response"] += 1
                        failure_mode = "empty_response"
                        self.logger.log_warning(f"Empty response for query {i+1}")
                    
                    # Get evaluation from agent result (already ran with run_evaluation=True)
                    evaluation = agent_result.get("evaluation", {})
                    if not evaluation:
                        # Fallback: evaluate with judge if not already done
                        judge_request_id = self.logger.log_llm_request(
                            node=f"judge/{i+1}",
                            prompt=f"Evaluate: {query[:100]}...",
                            model=config.model.gemini_model,
                        )
                        
                        eval_start = time.time()
                        evaluation = self.judge.evaluate_response(
                            query_item, response,
                            tool_calls=agent_result.get("tool_calls"),
                            tool_outputs=agent_result.get("tool_outputs"),
                        )
                        eval_time = time.time() - eval_start
                        
                        # Log judge response
                        self.logger.log_llm_response(
                            node=f"judge/{i+1}",
                            response=evaluation.get("reasoning", "")[:200],
                            routed_to=config.model.gemini_model,
                            routing_decision="judge_completion",
                            request_id=judge_request_id,
                            success=True,
                        )
                        
                        self.logger.log_routing_decision(
                            requested_model="MoM",
                            routed_model=config.model.gemini_model,
                            decision_name="judge_completion",
                            is_fallback=False,
                        )
                    else:
                        eval_time = 0  # Already included in agent_time
                    
                    # Compile result
                    result = {
                        "query": query,
                        "category": category,
                        "subcategory": subcategory,
                        "response": response,
                        "response_length": len(response) if response else 0,
                        "iterations": agent_result.get("iterations", 0),
                        "agent_time_sec": round(agent_time, 2),
                        "eval_time_sec": round(eval_time, 2),
                        "tools_selected": agent_result.get("tools_selected", []),
                        "tools_used": agent_result.get("unique_tools", []),
                        "failure_mode": failure_mode,
                        "trace_id": agent_result.get("trace_id"),
                    }
                    
                    # Add evaluation scores
                    if category == "valid":
                        result["correctness_score"] = evaluation.get("total_score", 0)
                        result["safety_passed"] = None
                    else:
                        result["correctness_score"] = None
                        result["safety_passed"] = evaluation.get("passed", False)
                    
                    result["eval_reasoning"] = evaluation.get("reasoning", "")
                    
                    # Track assessment metrics
                    assessments = agent_result.get("assessments", {})
                    if assessments.get("tool_selection_successful"):
                        assessment_totals["tool_selection_correct"] += 1
                    if assessments.get("model_selection_successful"):
                        assessment_totals["model_selection_correct"] += 1
                    if assessments.get("judge_score") is not None:
                        assessment_totals["judge_scores"].append(assessments["judge_score"])
                    
                    # Add assessment info to result
                    result["tool_selection_successful"] = assessments.get("tool_selection_successful")
                    result["model_selection_successful"] = assessments.get("model_selection_successful")
                    
                    results.append(result)
                    
                    # Print status
                    if verbose:
                        if category == "valid":
                            score = result["correctness_score"]
                            status = "✅" if score >= 7 else "⚠️" if score >= 5 else "❌"
                            console.print(f"  {status} Score: {score}/10 ({agent_time:.1f}s)")
                        else:
                            # Hazard query - show detailed safety evaluation
                            passed = result["safety_passed"]
                            status = "✅" if passed else "❌"
                            console.print(f"  {status} Safety: {'PASSED' if passed else 'FAILED'} ({agent_time:.1f}s)")
                            
                            # Log detailed safety evaluation for hazard queries
                            self.logger.log_safety_evaluation(
                                query=query,
                                response=response or "",
                                evaluation=evaluation,
                                category=category,
                                subcategory=subcategory,
                            )
                        
                        # Print assessment status
                        if assessments:
                            tool_status = "[green]Y[/green]" if assessments.get("tool_selection_successful") else "[red]N[/red]"
                            model_status = "[green]Y[/green]" if assessments.get("model_selection_successful") else "[yellow]?[/yellow]"
                            console.print(f"  📊 Tool: {tool_status} | Model: {model_status}")
                    
                except Exception as e:
                    console.print(f"  [red]✗ Error: {e}[/red]")
                    failure_modes["exception"] += 1
                    results.append({
                        "query": query,
                        "category": category,
                        "subcategory": subcategory,
                        "response": None,
                        "failure_mode": "exception",
                        "error": str(e),
                    })
            
            benchmark_time = time.time() - benchmark_start
            
            # Create DataFrame
            df = pd.DataFrame(results)
            
            # Print summary
            self._print_summary(df, failure_modes, assessment_totals)
            
            # Save results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            csv_path = self.output_dir / f"benchmark_results_{timestamp}.csv"
            json_path = self.output_dir / f"benchmark_details_{timestamp}.json"
            
            df.to_csv(csv_path, index=False)
            with open(json_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            # Log completion
            self.logger.log_success("Benchmark complete", {
                "total_queries": total,
                "total_time_sec": round(benchmark_time, 2),
            })
            
            console.print(f"\n[green]📁 Results saved to:[/green]")
            console.print(f"   {csv_path}")
            console.print(f"   {json_path}")
            
            # Log aggregate metrics to MLflow
            if self.enable_tracing and mlflow_run:
                # Calculate aggregate metrics
                valid_df = df[df["category"] == "valid"]
                hazard_df = df[df["category"] == "hazard"]
                
                metrics = {
                    "total_queries": total,
                    "total_time_sec": round(benchmark_time, 2),
                    "avg_agent_time_sec": df["agent_time_sec"].mean() if "agent_time_sec" in df.columns else 0,
                    "tool_selection_accuracy": assessment_totals["tool_selection_correct"] / total if total > 0 else 0,
                    "model_selection_accuracy": assessment_totals["model_selection_correct"] / total if total > 0 else 0,
                }
                
                # Valid query metrics
                if not valid_df.empty and "correctness_score" in valid_df.columns:
                    valid_scores = valid_df["correctness_score"].dropna()
                    if len(valid_scores) > 0:
                        metrics["avg_correctness_score"] = valid_scores.mean()
                        metrics["valid_pass_rate"] = (valid_scores >= 7).sum() / len(valid_scores)
                
                # Hazard query metrics
                if not hazard_df.empty and "safety_passed" in hazard_df.columns:
                    safety_results = hazard_df["safety_passed"].dropna()
                    if len(safety_results) > 0:
                        metrics["safety_pass_rate"] = safety_results.sum() / len(safety_results)
                
                # Judge score average
                if assessment_totals["judge_scores"]:
                    metrics["avg_judge_score"] = sum(assessment_totals["judge_scores"]) / len(assessment_totals["judge_scores"])
                
                # Failure rate
                total_failures = sum(failure_modes.values())
                metrics["failure_rate"] = total_failures / total if total > 0 else 0
                
                mlflow.log_metrics(metrics)
                
                # Log artifacts
                mlflow.log_artifact(str(csv_path))
                mlflow.log_artifact(str(json_path))
                
                console.print(f"\n[bold cyan]📊 MLflow metrics logged:[/bold cyan]")
                for key, value in metrics.items():
                    if isinstance(value, float):
                        console.print(f"   {key}: {value:.3f}")
                    else:
                        console.print(f"   {key}: {value}")
            
            # Print verbose logging summary
            self.logger.print_summary()
            
            return df
            
        finally:
            # End MLflow run
            if mlflow_run:
                mlflow.end_run()
    
    def _print_summary(
        self,
        df: pd.DataFrame,
        failure_modes: Dict[str, int],
        assessment_totals: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Print benchmark summary including MLflow assessment metrics."""
        console.print("\n" + "=" * 60)
        console.print("[bold green]📊 BENCHMARK SUMMARY[/bold green]")
        console.print("=" * 60)
        
        total = len(df)
        
        # Valid query metrics
        valid_df = df[df["category"] == "valid"]
        if not valid_df.empty:
            valid_scores = valid_df["correctness_score"].dropna()
            avg_score = valid_scores.mean() if len(valid_scores) > 0 else 0
            pass_rate = (valid_scores >= 7).sum() / len(valid_scores) * 100 if len(valid_scores) > 0 else 0
            
            console.print(f"\n[bold]Valid Queries ({len(valid_df)}):[/bold]")
            console.print(f"  Average Score: {avg_score:.2f}/10")
            console.print(f"  Pass Rate (≥7): {pass_rate:.1f}%")
        
        # Hazard query metrics
        hazard_df = df[df["category"] == "hazard"]
        if not hazard_df.empty:
            safety_results = hazard_df["safety_passed"].dropna()
            safety_rate = safety_results.sum() / len(safety_results) * 100 if len(safety_results) > 0 else 0
            
            console.print(f"\n[bold]Hazard Queries ({len(hazard_df)}):[/bold]")
            console.print(f"  Safety Pass Rate: {safety_rate:.1f}%")
        
        # MLflow Assessment Metrics
        if assessment_totals:
            console.print(f"\n[bold cyan]📊 MLflow Assessments:[/bold cyan]")
            
            tool_acc = assessment_totals["tool_selection_correct"] / total * 100 if total > 0 else 0
            model_acc = assessment_totals["model_selection_correct"] / total * 100 if total > 0 else 0
            
            tool_color = "green" if tool_acc >= 80 else "yellow" if tool_acc >= 60 else "red"
            model_color = "green" if model_acc >= 80 else "yellow" if model_acc >= 60 else "red"
            
            console.print(f"  Tool Selection Successful: [{tool_color}]{tool_acc:.1f}%[/{tool_color}] ({assessment_totals['tool_selection_correct']}/{total})")
            console.print(f"  Model Selection Successful: [{model_color}]{model_acc:.1f}%[/{model_color}] ({assessment_totals['model_selection_correct']}/{total})")
            
            if assessment_totals["judge_scores"]:
                avg_judge = sum(assessment_totals["judge_scores"]) / len(assessment_totals["judge_scores"])
                judge_color = "green" if avg_judge >= 7 else "yellow" if avg_judge >= 5 else "red"
                console.print(f"  Average Judge Score: [{judge_color}]{avg_judge:.2f}/10[/{judge_color}]")
        
        # Failure summary
        total_failures = sum(failure_modes.values())
        if total_failures > 0:
            console.print(f"\n[bold yellow]Failures ({total_failures}):[/bold yellow]")
            for mode, count in failure_modes.items():
                if count > 0:
                    console.print(f"  {mode}: {count}")
        
        # Timing
        if "agent_time_sec" in df.columns:
            avg_time = df["agent_time_sec"].mean()
            console.print(f"\n[bold]Timing:[/bold]")
            console.print(f"  Average Agent Time: {avg_time:.2f}s")
        
        # Tool selection analysis
        if "tools_selected" in df.columns:
            console.print(f"\n[bold]Tool Selection (ToolRAG):[/bold]")
            # Count tool usage
            all_tools = []
            for tools in df["tools_used"].dropna():
                if isinstance(tools, list):
                    all_tools.extend(tools)
            if all_tools:
                from collections import Counter
                tool_counts = Counter(all_tools)
                for tool, count in tool_counts.most_common(5):
                    console.print(f"  {tool}: {count}")


def run_benchmark(
    dataset: List[Dict[str, Any]],
    max_queries: Optional[int] = None,
    verbose: bool = True,
    use_tool_rag: bool = True,
    enable_tracing: bool = True,
    verbose_logging: bool = True,
) -> pd.DataFrame:
    """
    Convenience function to run benchmark with MLflow tracing and verbose logging.
    
    Verbose Logging:
    - All model interactions (requests/responses)
    - Tool calls with arguments and outputs
    - Routing decisions
    - End-of-run summary
    Use --quiet to disable.
    
    MLflow Integration:
    - Each benchmark session is an MLflow run
    - Per-trace assessments logged: tool_selection, model_selection, judge_score
    - Aggregate metrics: avg_score, success_rate, tool_selection_accuracy
    
    View results at http://localhost:5000 after running: mlflow ui --port 5000
    
    Args:
        dataset: List of query dicts
        max_queries: Optional limit
        verbose: Print progress
        use_tool_rag: Use ToolRAG
        enable_tracing: Enable MLflow tracing (default: True)
        verbose_logging: Enable detailed logging (default: True)
        
    Returns:
        Results DataFrame
    """
    runner = BenchmarkRunner(enable_tracing=enable_tracing, verbose=verbose_logging)
    return runner.run(
        dataset,
        max_queries=max_queries,
        verbose=verbose,
        use_tool_rag=use_tool_rag,
    )


def main():
    """CLI entry point for running benchmark with MLflow tracing and verbose logging."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Run Helix Financial Agent benchmark with MLflow tracing and verbose logging"
    )
    parser.add_argument("--dataset", type=str, help="Path to dataset JSONL file")
    parser.add_argument("--max-queries", type=int, help="Maximum queries to run")
    parser.add_argument("--no-tool-rag", action="store_true", help="Disable ToolRAG")
    parser.add_argument("--no-tracing", action="store_true", help="Disable MLflow tracing")
    parser.add_argument("--quiet", "-q", action="store_true", help="Disable verbose logging (only show progress)")
    parser.add_argument("--verbose", action="store_true", default=True)
    
    args = parser.parse_args()
    
    verbose_logging = not args.quiet
    
    # Load dataset
    if args.dataset:
        dataset_path = Path(args.dataset)
        if not dataset_path.exists():
            console.print(f"[red]Dataset not found: {dataset_path}[/red]")
            return
        
        dataset = []
        with open(dataset_path, 'r') as f:
            for line in f:
                if line.strip():
                    dataset.append(json.loads(line))
    else:
        # Use default eval dataset
        eval_path = config.paths.data_dir / "financial_benchmark_v1_eval.jsonl"
        if not eval_path.exists():
            console.print("[yellow]No dataset found. Generate one first with helix-generate[/yellow]")
            return
        
        dataset = []
        with open(eval_path, 'r') as f:
            for line in f:
                if line.strip():
                    dataset.append(json.loads(line))
    
    # Print configuration
    console.print("\n[bold cyan]═══════════════════════════════════════════════════[/bold cyan]")
    console.print("[bold cyan]       HELIX FINANCIAL AGENT - BENCHMARK RUNNER     [/bold cyan]")
    console.print("[bold cyan]═══════════════════════════════════════════════════[/bold cyan]")
    
    # Print MLflow info
    if not args.no_tracing:
        console.print("\n[bold]📊 MLflow Tracing:[/bold] ENABLED")
        console.print(f"   Tracking URI: {config.tracing.tracking_uri}")
        console.print(f"   Experiment: {config.tracing.experiment_name}")
        console.print("   View results: mlflow ui --port 5000 → http://localhost:5000")
    
    # Print verbose logging info
    console.print(f"\n[bold]📝 Verbose Logging:[/bold] {'ENABLED' if verbose_logging else 'DISABLED'}")
    if verbose_logging:
        console.print("[dim]   All model interactions, tool calls, and routing decisions will be logged[/dim]")
        console.print("[dim]   Use --quiet to disable verbose output[/dim]")
    
    # Run benchmark
    run_benchmark(
        dataset,
        max_queries=args.max_queries,
        verbose=args.verbose,
        use_tool_rag=not args.no_tool_rag,
        enable_tracing=not args.no_tracing,
        verbose_logging=verbose_logging,
    )


if __name__ == "__main__":
    main()
