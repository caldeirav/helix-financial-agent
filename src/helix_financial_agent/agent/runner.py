"""
Agent Runner with Rich Logging

Provides a high-level interface for running the agent with
comprehensive logging, tracing, and beautiful console output.
"""

import json
import time
from typing import Dict, Any, List, Optional, Callable

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.syntax import Syntax
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.live import Live
from rich.text import Text

from langchain_core.messages import AIMessage, HumanMessage

from ..config import get_config
from ..tools import CORE_TOOLS
from ..tool_rag import ToolSelector
from .state import AgentState, create_initial_state
from .graph import create_agent

console = Console()
config = get_config()


class AgentRunner:
    """
    High-level runner for the Reflexive Financial Agent.
    
    Features:
    - Rich console output for demos
    - ToolRAG integration for dynamic tool selection
    - MLflow tracing integration
    - Comprehensive metrics collection
    """
    
    def __init__(
        self,
        tools: Optional[List[Callable]] = None,
        use_tool_rag: bool = True,
        use_router: bool = False,
        verbose: bool = True,
    ):
        """
        Initialize the agent runner.
        
        Args:
            tools: Optional list of tools (default: CORE_TOOLS)
            use_tool_rag: Whether to use ToolRAG for tool selection
            use_router: Whether to use semantic router
            verbose: Whether to print detailed output
        """
        self.all_tools = tools or CORE_TOOLS
        self.use_tool_rag = use_tool_rag
        self.use_router = use_router
        self.verbose = verbose
        
        # Initialize tool selector if using ToolRAG
        if use_tool_rag:
            self.tool_selector = ToolSelector()
            # Register tools
            for tool in self.all_tools:
                name = getattr(tool, "name", tool.__name__)
                self.tool_selector.register_tool(name, tool)
        else:
            self.tool_selector = None
        
        # Create agent (will be configured per-query if using ToolRAG)
        self._agent = None
    
    def _get_agent(self, tools: List[Callable]):
        """Get or create agent with specified tools."""
        return create_agent(tools=tools, use_router=self.use_router)
    
    def _select_tools(self, query: str) -> List[Callable]:
        """Select tools for a query using ToolRAG."""
        if not self.use_tool_rag or not self.tool_selector:
            return self.all_tools
        
        selected = self.tool_selector.get_tools_for_query(
            query, verbose=self.verbose
        )
        
        # If no tools selected, fall back to core tools
        if not selected:
            if self.verbose:
                console.print("[yellow]⚠️ No tools selected by ToolRAG, using core tools[/yellow]")
            return CORE_TOOLS
        
        return selected
    
    def run(
        self,
        query: str,
        thread_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Run the agent on a query.
        
        Args:
            query: The user's financial question
            thread_id: Optional thread ID for conversation continuity
            
        Returns:
            Dict with query, response, metrics, etc.
        """
        if thread_id is None:
            thread_id = str(time.time())
        
        start_time = time.time()
        
        # Print header
        if self.verbose:
            console.print()
            console.print(Panel(
                f"[bold cyan]{query}[/bold cyan]",
                title="[bold]💹 HELIX FINANCIAL AGENT[/bold]",
                border_style="cyan"
            ))
        
        # Select tools using ToolRAG
        selected_tools = self._select_tools(query)
        selected_tool_names = [getattr(t, "name", t.__name__) for t in selected_tools]
        
        # Create agent with selected tools
        agent = self._get_agent(selected_tools)
        
        # Initialize state
        initial_state = create_initial_state(query, selected_tool_names)
        agent_config = {"configurable": {"thread_id": thread_id}}
        
        # Track execution
        all_ai_responses = []
        tools_used = []
        iterations = 0
        total_steps = 0
        
        # Stream through the graph
        for step, state in enumerate(agent.stream(initial_state, agent_config)):
            node_name = list(state.keys())[0]
            node_state = state[node_name]
            total_steps += 1
            
            # Track iteration count
            if "iteration_count" in node_state:
                iterations = node_state["iteration_count"]
            
            if self.verbose:
                self._print_step(step + 1, node_name, node_state)
            
            # Process messages
            if "messages" in node_state:
                for msg in node_state["messages"]:
                    if hasattr(msg, 'tool_calls') and msg.tool_calls:
                        for tc in msg.tool_calls:
                            tools_used.append(tc['name'])
                    elif isinstance(msg, AIMessage) and msg.content:
                        if not msg.content.startswith("[REFLECTION]"):
                            all_ai_responses.append(msg.content)
        
        # Get final response
        final_response = all_ai_responses[-1] if all_ai_responses else None
        elapsed_time = time.time() - start_time
        
        # Compile result
        result = {
            "query": query,
            "response": final_response,
            "iterations": iterations,
            "elapsed_time": round(elapsed_time, 2),
            "thread_id": thread_id,
            "tools_selected": selected_tool_names,
            "tools_used": tools_used,
            "unique_tools": list(set(tools_used)),
            "total_steps": total_steps,
        }
        
        # Print summary
        if self.verbose:
            self._print_summary(result)
        
        return result
    
    def _print_step(self, step: int, node_name: str, node_state: Dict) -> None:
        """Print a step during execution."""
        # Node header
        node_icons = {
            "generator": "🤖",
            "tools": "🔧",
            "reflect": "🔍",
            "revise": "✏️",
        }
        icon = node_icons.get(node_name, "📍")
        
        console.print()
        console.print(f"{'─' * 60}")
        console.print(f"{icon} [bold]Step {step}: {node_name.upper()}[/bold]")
        console.print(f"{'─' * 60}")
        
        # Process messages
        if "messages" in node_state:
            for msg in node_state["messages"]:
                self._print_message(msg)
    
    def _print_message(self, msg) -> None:
        """Print a message with appropriate formatting."""
        if hasattr(msg, 'tool_calls') and msg.tool_calls:
            console.print("\n[bold magenta]🔧 TOOL CALLS:[/bold magenta]")
            for tc in msg.tool_calls:
                console.print(f"   Function: [cyan]{tc['name']}[/cyan]")
                console.print(f"   Args: {json.dumps(tc['args'], indent=2)}")
        
        elif hasattr(msg, 'name') and msg.name:
            # Tool output
            console.print(f"\n[bold green]📊 TOOL OUTPUT ({msg.name}):[/bold green]")
            try:
                data = json.loads(msg.content)
                console.print(Syntax(json.dumps(data, indent=2), "json", theme="monokai"))
            except:
                console.print(msg.content[:500] + "..." if len(msg.content) > 500 else msg.content)
        
        elif isinstance(msg, AIMessage) and msg.content:
            if msg.content.startswith("[REFLECTION]"):
                console.print("\n[bold yellow]🔍 REFLECTION:[/bold yellow]")
                content = msg.content.replace("[REFLECTION]: ", "")
                try:
                    data = json.loads(content)
                    passed = data.get("passed", False)
                    status = "[green]✅ PASSED[/green]" if passed else "[red]❌ NEEDS REVISION[/red]"
                    console.print(f"   Status: {status}")
                    if data.get("feedback"):
                        console.print(f"   Feedback: {data['feedback']}")
                except:
                    console.print(content[:300])
            else:
                console.print("\n[bold blue]💬 AGENT RESPONSE:[/bold blue]")
                console.print(Panel(msg.content, border_style="blue"))
    
    def _print_summary(self, result: Dict) -> None:
        """Print execution summary."""
        console.print()
        console.print("=" * 60)
        console.print("[bold green]📊 EXECUTION SUMMARY[/bold green]")
        console.print("=" * 60)
        
        table = Table(show_header=False, box=None)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="white")
        
        table.add_row("⏱️  Elapsed Time", f"{result['elapsed_time']:.2f}s")
        table.add_row("📈 Graph Steps", str(result['total_steps']))
        table.add_row("🔄 Iterations", str(result['iterations']))
        table.add_row("🔧 Tools Selected", ", ".join(result['tools_selected']))
        table.add_row("🛠️  Tools Used", ", ".join(result['unique_tools']) or "None")
        table.add_row("📝 Response Length", f"{len(result['response'] or ''):,} chars")
        
        console.print(table)
        console.print("=" * 60)
        console.print("[bold green]✅ Agent completed successfully[/bold green]")
        console.print("=" * 60)


def run_agent(
    query: str,
    thread_id: Optional[str] = None,
    verbose: bool = True,
    use_tool_rag: bool = True,
    use_router: bool = False,
) -> Dict[str, Any]:
    """
    Convenience function to run the agent on a query.
    
    Args:
        query: The user's financial question
        thread_id: Optional thread ID for conversation continuity
        verbose: Whether to print detailed output
        use_tool_rag: Whether to use ToolRAG for tool selection
        use_router: Whether to use semantic router
        
    Returns:
        Dict with query, response, metrics, etc.
    """
    runner = AgentRunner(
        verbose=verbose,
        use_tool_rag=use_tool_rag,
        use_router=use_router,
    )
    return runner.run(query, thread_id=thread_id)
