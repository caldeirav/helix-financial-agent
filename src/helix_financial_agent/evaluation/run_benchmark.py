"""
Benchmark Runner

Runs comprehensive evaluation on the synthetic dataset.
"""

import json
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

import pandas as pd
from rich.console import Console
from rich.progress import Progress, TaskID
from rich.table import Table

from ..config import get_config
from ..agent import run_agent
from .judge import GeminiJudge

console = Console()
config = get_config()


class BenchmarkRunner:
    """
    Runs benchmark evaluation on a dataset.
    
    Features:
    - Runs agent on each query
    - Evaluates with Gemini judge
    - Tracks metrics and failures
    - Saves results to CSV/JSON
    """
    
    def __init__(
        self,
        judge: Optional[GeminiJudge] = None,
        output_dir: Optional[Path] = None,
    ):
        """
        Initialize the benchmark runner.
        
        Args:
            judge: Optional GeminiJudge instance
            output_dir: Optional output directory for results
        """
        self.judge = judge or GeminiJudge()
        self.output_dir = output_dir or config.paths.data_dir
        self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def run(
        self,
        dataset: List[Dict[str, Any]],
        max_queries: Optional[int] = None,
        verbose: bool = True,
        verbose_agent: bool = False,
        use_tool_rag: bool = True,
    ) -> pd.DataFrame:
        """
        Run the benchmark on a dataset.
        
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
        
        # Track failures
        failure_modes = {
            "empty_response": 0,
            "tool_error": 0,
            "exception": 0,
            "judge_error": 0,
        }
        
        console.print(f"\n[bold cyan]🚀 Starting benchmark on {total} queries...[/bold cyan]")
        console.print(f"   ToolRAG: {'ENABLED' if use_tool_rag else 'DISABLED'}")
        console.print("=" * 60)
        
        for i, query_item in enumerate(dataset):
            query = query_item["query"]
            category = query_item.get("category", "valid")
            subcategory = query_item.get("subcategory", "")
            
            if verbose:
                console.print(f"\n[bold]Query [{i+1}/{total}][/bold] [{category}/{subcategory}]")
                console.print(f"  {query[:80]}...")
            
            try:
                # Run agent
                start_time = time.time()
                agent_result = run_agent(
                    query,
                    verbose=verbose_agent,
                    use_tool_rag=use_tool_rag,
                )
                agent_time = time.time() - start_time
                
                response = agent_result.get("response", "")
                
                # Check for failures
                failure_mode = None
                if not response or len(response.strip()) < 20:
                    failure_modes["empty_response"] += 1
                    failure_mode = "empty_response"
                
                # Evaluate with judge
                eval_start = time.time()
                evaluation = self.judge.evaluate_response(query_item, response)
                eval_time = time.time() - eval_start
                
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
                }
                
                # Add evaluation scores
                if category == "valid":
                    result["correctness_score"] = evaluation.get("total_score", 0)
                    result["safety_passed"] = None
                else:
                    result["correctness_score"] = None
                    result["safety_passed"] = evaluation.get("passed", False)
                
                result["eval_reasoning"] = evaluation.get("reasoning", "")
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
        
        # Create DataFrame
        df = pd.DataFrame(results)
        
        # Print summary
        self._print_summary(df, failure_modes)
        
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
        
        return df
    
    def _print_summary(self, df: pd.DataFrame, failure_modes: Dict[str, int]) -> None:
        """Print benchmark summary."""
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
) -> pd.DataFrame:
    """
    Convenience function to run benchmark.
    
    Args:
        dataset: List of query dicts
        max_queries: Optional limit
        verbose: Print progress
        use_tool_rag: Use ToolRAG
        
    Returns:
        Results DataFrame
    """
    runner = BenchmarkRunner()
    return runner.run(
        dataset,
        max_queries=max_queries,
        verbose=verbose,
        use_tool_rag=use_tool_rag,
    )


def main():
    """CLI entry point for running benchmark."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run Helix Financial Agent benchmark")
    parser.add_argument("--dataset", type=str, help="Path to dataset JSONL file")
    parser.add_argument("--max-queries", type=int, help="Maximum queries to run")
    parser.add_argument("--no-tool-rag", action="store_true", help="Disable ToolRAG")
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
    
    # Run benchmark
    run_benchmark(
        dataset,
        max_queries=args.max_queries,
        verbose=args.verbose,
        use_tool_rag=not args.no_tool_rag,
    )


if __name__ == "__main__":
    main()
