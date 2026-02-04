"""
Helix Financial Agent - Main Entry Point

A Reflexive Financial AI Agent with semantic routing, ToolRAG, and MCP deployment.

Architecture:
    - All LLM calls go through vLLM Semantic Router (MANDATORY)
    - Router routes to Qwen3 (agent) or Gemini (judge) based on task
    - All tool calls go through MCP server (MANDATORY)

Required Services:
    1. llama.cpp server: ./scripts/start_llama_server.sh
    2. vLLM-SR router: ./scripts/start_router.sh
    3. MCP server: ./scripts/start_mcp_server.sh
"""

import argparse
import sys
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .config import get_config
from .agent import run_agent, AgentRunner
from .agent.runner import ServiceError, run_random_benchmark_query, get_random_query, load_benchmark_dataset
from .tools import CORE_TOOLS, ALL_TOOLS

console = Console()


def print_banner():
    """Print the Helix banner."""
    banner = """
    ╦ ╦╔═╗╦  ╦═╗ ╦
    ╠═╣║╣ ║  ║╔╩╦╝
    ╩ ╩╚═╝╩═╝╩╩ ╚═
    FINANCIAL AGENT
    """
    console.print(Panel(
        banner,
        title="[bold cyan]Reflexive Financial AI[/bold cyan]",
        subtitle="Powered by Qwen3 + LangGraph + vLLM-SR + MCP",
        border_style="cyan"
    ))


def print_architecture():
    """Print the architecture diagram."""
    console.print("\n[bold]Architecture:[/bold]")
    
    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("Component", style="white")
    table.add_column("Purpose", style="dim")
    table.add_column("Endpoint", style="green")
    
    config = get_config()
    table.add_row(
        "llama.cpp",
        "Qwen3 model serving",
        config.model.llama_cpp_base_url
    )
    table.add_row(
        "vLLM-SR Router",
        "Semantic model routing",
        config.router.router_endpoint
    )
    table.add_row(
        "MCP Server",
        "Tool execution",
        f"http://{config.mcp.host}:{config.mcp.port}"
    )
    
    console.print(table)
    console.print("\n[dim]Agent → Router → {Qwen3 | Gemini}[/dim]")
    console.print("[dim]Agent → MCP → yfinance tools[/dim]")


def interactive_mode(run_evaluation: bool = False):
    """Run the agent in interactive mode."""
    config = get_config()
    
    print_banner()
    print_architecture()
    
    console.print("\n[bold]Model Configuration:[/bold]")
    console.print(f"  Agent Model: {config.model.agent_model_name}")
    console.print(f"  Judge Model: {config.model.gemini_model}")
    console.print(f"  Max Iterations: {config.agent.max_iterations}")
    console.print(f"  Evaluation: {'Enabled' if run_evaluation else 'Disabled'}")
    console.print("\nType 'quit' or 'exit' to exit, 'random' for random benchmark query.")
    console.print("-" * 50)
    
    try:
        runner = AgentRunner(verbose=True, use_tool_rag=True, run_evaluation=run_evaluation)
    except ServiceError as e:
        console.print(f"\n[red]Cannot start: {e}[/red]")
        sys.exit(1)
    
    while True:
        try:
            query = console.input("\n[bold cyan]You:[/bold cyan] ")
            
            if query.lower() in ['quit', 'exit', 'q']:
                console.print("\n[dim]Goodbye![/dim]")
                break
            
            if query.lower() == 'random':
                # Run random benchmark query
                try:
                    query_item = get_random_query()
                    query = query_item.get("query", "")
                    metadata = {
                        "id": query_item.get("id"),
                        "category": query_item.get("category"),
                        "subcategory": query_item.get("subcategory"),
                        "expected_tools": query_item.get("expected_tools"),
                    }
                    result = runner.run(query, query_metadata=metadata)
                except FileNotFoundError:
                    console.print("[yellow]⚠️ No benchmark dataset found. Generate with: helix-generate[/yellow]")
                continue
            
            if not query.strip():
                continue
            
            result = runner.run(query)
            
        except KeyboardInterrupt:
            console.print("\n[dim]Interrupted. Goodbye![/dim]")
            break
        except Exception as e:
            console.print(f"\n[red]Error: {e}[/red]")
            import traceback
            traceback.print_exc()


def single_query_mode(
    query: str,
    verbose: bool = True,
    use_tool_rag: bool = True,
    run_evaluation: bool = False,
    query_metadata: dict = None,
):
    """Run a single query."""
    print_banner()
    result = run_agent(
        query,
        verbose=verbose,
        use_tool_rag=use_tool_rag,
        run_evaluation=run_evaluation,
        query_metadata=query_metadata,
    )
    return result


def random_query_mode(
    dataset_path: str = None,
    verbose: bool = True,
    use_tool_rag: bool = True,
    run_evaluation: bool = True,
):
    """Run a random query from the benchmark dataset."""
    print_banner()
    
    path = Path(dataset_path) if dataset_path else None
    
    try:
        result = run_random_benchmark_query(
            dataset_path=path,
            verbose=verbose,
            use_tool_rag=use_tool_rag,
            run_evaluation=run_evaluation,
        )
        return result
    except FileNotFoundError as e:
        console.print(f"\n[red]Dataset not found: {e}[/red]")
        console.print("[yellow]Generate a dataset first with: helix-generate[/yellow]")
        sys.exit(1)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Helix Financial Agent - A Reflexive Financial AI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode
  helix-agent
  
  # Random query from benchmark dataset (with evaluation)
  helix-agent --random
  
  # Random query from specific dataset
  helix-agent --random --dataset data/my_benchmark.jsonl
  
  # Single query
  helix-agent --query "What is AAPL's PE ratio?"
  
  # Single query with evaluation
  helix-agent --query "What is NVDA's market cap?" --eval
  
  # Without ToolRAG
  helix-agent --query "Compare AAPL and MSFT" --no-tool-rag
        """
    )
    
    parser.add_argument(
        "--query", "-q",
        type=str,
        help="Run a single query instead of interactive mode"
    )
    parser.add_argument(
        "--random", "-r",
        action="store_true",
        help="Run a random query from the benchmark dataset"
    )
    parser.add_argument(
        "--dataset", "-d",
        type=str,
        default=None,
        help="Path to benchmark dataset JSONL file (default: data/financial_benchmark_v1_full.jsonl)"
    )
    parser.add_argument(
        "--eval", "-e",
        action="store_true",
        help="Run LLM-as-a-Judge evaluation after response"
    )
    parser.add_argument(
        "--no-tool-rag",
        action="store_true",
        help="Disable ToolRAG (use all core tools)"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce output verbosity"
    )
    
    args = parser.parse_args()
    
    if args.random:
        # Random query mode
        random_query_mode(
            dataset_path=args.dataset,
            verbose=not args.quiet,
            use_tool_rag=not args.no_tool_rag,
            run_evaluation=True,  # Always evaluate random queries
        )
    elif args.query:
        # Single query mode
        single_query_mode(
            args.query,
            verbose=not args.quiet,
            use_tool_rag=not args.no_tool_rag,
            run_evaluation=args.eval,
        )
    else:
        # Interactive mode
        interactive_mode(run_evaluation=args.eval)


if __name__ == "__main__":
    main()
