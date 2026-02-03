"""
Helix Financial Agent - Main Entry Point

A Reflexive Financial AI Agent with semantic routing, ToolRAG, and MCP deployment.
"""

import argparse
from rich.console import Console
from rich.panel import Panel

from .config import get_config
from .agent import run_agent, AgentRunner
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
        subtitle="Powered by Qwen3 + LangGraph + ToolRAG",
        border_style="cyan"
    ))


def interactive_mode():
    """Run the agent in interactive mode."""
    config = get_config()
    
    print_banner()
    console.print("\n[bold]Configuration:[/bold]")
    console.print(f"  Model: {config.model.agent_model_name}")
    console.print(f"  Endpoint: {config.model.llama_cpp_base_url}")
    console.print(f"  Max Iterations: {config.agent.max_iterations}")
    console.print("\nType 'quit' or 'exit' to exit.")
    console.print("-" * 50)
    
    runner = AgentRunner(verbose=True, use_tool_rag=True)
    
    while True:
        try:
            query = console.input("\n[bold cyan]You:[/bold cyan] ")
            
            if query.lower() in ['quit', 'exit', 'q']:
                console.print("\n[dim]Goodbye![/dim]")
                break
            
            if not query.strip():
                continue
            
            result = runner.run(query)
            
        except KeyboardInterrupt:
            console.print("\n[dim]Interrupted. Goodbye![/dim]")
            break
        except Exception as e:
            console.print(f"\n[red]Error: {e}[/red]")


def single_query_mode(query: str, verbose: bool = True, use_tool_rag: bool = True):
    """Run a single query."""
    print_banner()
    result = run_agent(query, verbose=verbose, use_tool_rag=use_tool_rag)
    return result


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Helix Financial Agent - A Reflexive Financial AI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode
  helix-agent
  
  # Single query
  helix-agent --query "What is AAPL's PE ratio?"
  
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
    
    if args.query:
        single_query_mode(
            args.query,
            verbose=not args.quiet,
            use_tool_rag=not args.no_tool_rag
        )
    else:
        interactive_mode()


if __name__ == "__main__":
    main()
