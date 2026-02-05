"""
Benchmark Runner with MLflow Tracing

Runs comprehensive evaluation on the synthetic dataset with
end-to-end MLflow tracing and assessment logging.

MLflow Integration:
- Each benchmark session is wrapped in an MLflow run
- Per-trace assessments: tool_selection, model_selection, judge_score
- Aggregate metrics logged: avg_score, success_rate, tool_selection_accuracy
- Results saved as MLflow artifacts

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
from ..tracing import setup_mlflow_tracing, TracingContext
from .judge import GeminiJudge

console = Console()
config = get_config()


class BenchmarkRunner:
    """
    Runs benchmark evaluation on a dataset with MLflow tracing.
    
    Features:
    - Runs agent on each query with MLflow tracing
    - Evaluates with Gemini judge
    - Logs per-trace assessments (tool_selection, model_selection, judge_score)
    - Tracks aggregate metrics in MLflow
    - Saves results to CSV/JSON as MLflow artifacts
    
    MLflow Integration:
    - Each benchmark session is an MLflow run
    - View at http://localhost:5000 after running: mlflow ui
    """
    
    def __init__(
        self,
        judge: Optional[GeminiJudge] = None,
        output_dir: Optional[Path] = None,
        enable_tracing: bool = True,
    ):
        """
        Initialize the benchmark runner.
        
        Args:
            judge: Optional GeminiJudge instance
            output_dir: Optional output directory for results
            enable_tracing: Whether to enable MLflow tracing (default: True)
        """
        self.judge = judge or GeminiJudge()
        self.output_dir = output_dir or config.paths.data_dir
        self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.enable_tracing = enable_tracing
        
        # Initialize MLflow tracing
        if enable_tracing:
            setup_mlflow_tracing()
    
    def run(
        self,
        dataset: List[Dict[str, Any]],
        max_queries: Optional[int] = None,
        verbose: bool = True,
        verbose_agent: bool = False,
        use_tool_rag: bool = True,
    ) -> pd.DataFrame:
        """
        Run the benchmark on a dataset with MLflow tracing.
        
        Each benchmark session is wrapped in an MLflow run for tracking.
        Per-trace assessments are logged for each query:
        - tool_selection_successful: Y/N
        - model_selection_successful: Y/N
        - judge_score: 0-10
        
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
            console.print("=" * 60)
            
            benchmark_start = time.time()
            
            for i, query_item in enumerate(dataset):
                query = query_item["query"]
                category = query_item.get("category", "valid")
                subcategory = query_item.get("subcategory", "")
                expected_tools = query_item.get("expected_tools", [])
                
                if verbose:
                    console.print(f"\n[bold]Query [{i+1}/{total}][/bold] [{category}/{subcategory}]")
                    console.print(f"  {query[:80]}...")
                
                try:
                    # Run agent with metadata for assessment tracking
                    start_time = time.time()
                    agent_result = run_agent(
                        query,
                        verbose=verbose_agent,
                        use_tool_rag=use_tool_rag,
                        run_evaluation=True,  # Enable evaluation for assessments
                        query_metadata=query_item,
                        enable_tracing=self.enable_tracing,
                    )
                    agent_time = time.time() - start_time
                    
                    response = agent_result.get("response", "")
                    
                    # Check for failures
                    failure_mode = None
                    if not response or len(response.strip()) < 20:
                        failure_modes["empty_response"] += 1
                        failure_mode = "empty_response"
                    
                    # Get evaluation from agent result (already ran with run_evaluation=True)
                    evaluation = agent_result.get("evaluation", {})
                    if not evaluation:
                        # Fallback: evaluate with judge if not already done
                        eval_start = time.time()
                        evaluation = self.judge.evaluate_response(query_item, response)
                        eval_time = time.time() - eval_start
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
                            passed = result["safety_passed"]
                            status = "✅" if passed else "❌"
                            console.print(f"  {status} Safety: {'PASSED' if passed else 'FAILED'} ({agent_time:.1f}s)")
                        
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
) -> pd.DataFrame:
    """
    Convenience function to run benchmark with MLflow tracing.
    
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
        
    Returns:
        Results DataFrame
    """
    runner = BenchmarkRunner(enable_tracing=enable_tracing)
    return runner.run(
        dataset,
        max_queries=max_queries,
        verbose=verbose,
        use_tool_rag=use_tool_rag,
    )


def main():
    """CLI entry point for running benchmark with MLflow tracing."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Run Helix Financial Agent benchmark with MLflow tracing"
    )
    parser.add_argument("--dataset", type=str, help="Path to dataset JSONL file")
    parser.add_argument("--max-queries", type=int, help="Maximum queries to run")
    parser.add_argument("--no-tool-rag", action="store_true", help="Disable ToolRAG")
    parser.add_argument("--no-tracing", action="store_true", help="Disable MLflow tracing")
    parser.add_argument("--verbose", action="store_true", default=True)
    
    args = parser.parse_args()
    
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
    
    # Print MLflow info
    if not args.no_tracing:
        console.print("\n[bold cyan]📊 MLflow Tracing Enabled[/bold cyan]")
        console.print(f"   Tracking URI: {config.tracing.tracking_uri}")
        console.print(f"   Experiment: {config.tracing.experiment_name}")
        console.print("   View results: mlflow ui --port 5000 → http://localhost:5000")
    
    # Run benchmark
    run_benchmark(
        dataset,
        max_queries=args.max_queries,
        verbose=args.verbose,
        use_tool_rag=not args.no_tool_rag,
        enable_tracing=not args.no_tracing,
    )


if __name__ == "__main__":
    main()
