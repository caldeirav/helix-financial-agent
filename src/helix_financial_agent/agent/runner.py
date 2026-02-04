"""
Agent Runner with Rich Logging

Provides a high-level interface for running the agent with
comprehensive logging, tracing, and beautiful console output.

Architecture:
    - All LLM calls go through the vLLM Semantic Router
    - All tool calls go through the MCP server
    - Both services are MANDATORY for agent operation
"""

import json
import random
import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable

import httpx
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.syntax import Syntax
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.live import Live
from rich.text import Text
from rich.markdown import Markdown
from rich.tree import Tree

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from ..config import get_config
from ..tools import CORE_TOOLS, check_mcp_server
from ..tool_rag import ToolSelector
from .state import AgentState, create_initial_state
from .graph import create_agent

console = Console()
config = get_config()


# =============================================================================
# DATASET UTILITIES
# =============================================================================

def load_benchmark_dataset(path: Optional[Path] = None) -> List[Dict]:
    """Load benchmark dataset from JSONL file."""
    if path is None:
        path = config.paths.data_dir / "financial_benchmark_v1_full.jsonl"
    
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    
    dataset = []
    with open(path, 'r') as f:
        for line in f:
            if line.strip():
                dataset.append(json.loads(line))
    
    return dataset


def get_random_query(dataset_path: Optional[Path] = None) -> Dict:
    """Get a random query from the benchmark dataset."""
    dataset = load_benchmark_dataset(dataset_path)
    return random.choice(dataset)


class ServiceError(Exception):
    """Raised when required services are not available."""
    pass


def check_router_health() -> bool:
    """Check if the semantic router is healthy."""
    try:
        classify_url = f"http://{config.router.router_host}:{config.router.router_classify_port}/health"
        with httpx.Client(timeout=5.0) as client:
            response = client.get(classify_url)
            return response.status_code == 200
    except Exception:
        return False


def check_llama_server_health() -> bool:
    """Check if the llama.cpp server is healthy."""
    try:
        base_url = config.model.llama_cpp_base_url.rstrip("/v1").rstrip("/")
        health_url = f"{base_url}/health"
        with httpx.Client(timeout=5.0) as client:
            response = client.get(health_url)
            return response.status_code == 200
    except Exception:
        return False


def verify_required_services(check_mcp: bool = True, check_router: bool = True) -> None:
    """
    Verify that all required services are running.
    
    Args:
        check_mcp: Whether to check MCP server
        check_router: Whether to check semantic router
        
    Raises:
        ServiceError: If any required service is not available
    """
    errors = []
    
    # Check llama.cpp server
    if not check_llama_server_health():
        errors.append(
            f"llama.cpp server not responding at {config.model.llama_cpp_base_url}\n"
            "   Start with: ./scripts/start_llama_server.sh"
        )
    
    # Check semantic router
    if check_router and not check_router_health():
        errors.append(
            f"Semantic router not responding at {config.router.router_host}:{config.router.router_classify_port}\n"
            "   Start with: ./scripts/start_router.sh"
        )
    
    # Check MCP server
    if check_mcp and not check_mcp_server():
        errors.append(
            f"MCP server not responding at {config.mcp.host}:{config.mcp.port}\n"
            "   Start with: ./scripts/start_mcp_server.sh"
        )
    
    if errors:
        error_msg = "\n\n".join(errors)
        console.print(Panel(
            f"[bold red]REQUIRED SERVICES NOT AVAILABLE[/bold red]\n\n{error_msg}",
            title="❌ Service Check Failed",
            border_style="red"
        ))
        raise ServiceError("Required services not running. See above for details.")


class AgentRunner:
    """
    High-level runner for the Reflexive Financial Agent.
    
    Features:
    - Rich console output for demos
    - ToolRAG integration for dynamic tool selection
    - LLM-as-a-Judge evaluation
    - Comprehensive metrics collection and tracing
    
    Required Services:
    - llama.cpp: Model serving for Qwen3 agent
    - vLLM-SR: Semantic routing between agent and judge models
    - MCP Server: Tool execution via MCP protocol
    """
    
    def __init__(
        self,
        tools: Optional[List[Callable]] = None,
        use_tool_rag: bool = True,
        verbose: bool = True,
        skip_service_check: bool = False,
        run_evaluation: bool = False,
    ):
        """
        Initialize the agent runner.
        
        Args:
            tools: Optional list of tools (default: CORE_TOOLS via MCP)
            use_tool_rag: Whether to use ToolRAG for tool selection
            verbose: Whether to print detailed output
            skip_service_check: Skip service verification (for testing only)
            run_evaluation: Whether to run LLM-as-a-Judge evaluation after response
        """
        self.all_tools = tools or CORE_TOOLS
        self.use_tool_rag = use_tool_rag
        self.verbose = verbose
        self.run_evaluation = run_evaluation
        
        # Verify required services are running
        if not skip_service_check:
            if verbose:
                console.print("\n[cyan]🔍 Checking required services...[/cyan]")
            verify_required_services(check_mcp=True, check_router=True)
            if verbose:
                console.print("[green]✅ All required services are running[/green]")
        
        # Initialize tool selector if using ToolRAG
        if use_tool_rag:
            self.tool_selector = ToolSelector()
            # Register tools
            for tool in self.all_tools:
                name = getattr(tool, "name", tool.__name__)
                self.tool_selector.register_tool(name, tool)
        else:
            self.tool_selector = None
        
        # Lazy-load judge for evaluation
        self._judge = None
        
        # Create agent (will be configured per-query if using ToolRAG)
        self._agent = None
    
    def _get_judge(self):
        """Get or create the LLM-as-a-Judge instance."""
        if self._judge is None:
            from ..evaluation.judge import GeminiJudge
            self._judge = GeminiJudge()
        return self._judge
    
    def _get_agent(self, tools: List[Callable]):
        """Get or create agent with specified tools."""
        # Router is always used - no option to disable
        return create_agent(tools=tools)
    
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
        query_metadata: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """
        Run the agent on a query with full tracing.
        
        Args:
            query: The user's financial question
            thread_id: Optional thread ID for conversation continuity
            query_metadata: Optional metadata (category, subcategory, expected_tools)
            
        Returns:
            Dict with query, response, metrics, evaluation, trace, etc.
        """
        if thread_id is None:
            thread_id = str(time.time())
        
        start_time = time.time()
        trace_log = []  # Detailed trace of all steps
        
        # Print header with query metadata if available
        if self.verbose:
            self._print_query_header(query, query_metadata)
        
        # Log: Query received
        trace_log.append({
            "timestamp": time.time(),
            "event": "query_received",
            "data": {"query": query, "metadata": query_metadata}
        })
        
        # Select tools using ToolRAG
        if self.verbose:
            console.print("\n" + "─" * 70)
            console.print("[bold magenta]🎯 PHASE 1: TOOL SELECTION (ToolRAG)[/bold magenta]")
            console.print("─" * 70)
        
        selected_tools = self._select_tools(query)
        selected_tool_names = [getattr(t, "name", t.__name__) for t in selected_tools]
        
        trace_log.append({
            "timestamp": time.time(),
            "event": "tools_selected",
            "data": {"tools": selected_tool_names}
        })
        
        if self.verbose:
            self._print_tool_selection(selected_tool_names, query_metadata)
        
        # Create agent with selected tools
        agent = self._get_agent(selected_tools)
        
        # Initialize state
        initial_state = create_initial_state(query, selected_tool_names)
        agent_config = {"configurable": {"thread_id": thread_id}}
        
        # Track execution
        all_ai_responses = []
        all_tool_calls = []
        all_tool_outputs = []
        iterations = 0
        total_steps = 0
        reflections = []
        
        if self.verbose:
            console.print("\n" + "─" * 70)
            console.print("[bold magenta]🤖 PHASE 2: AGENT EXECUTION (Generator → Tools → Reflector)[/bold magenta]")
            console.print("─" * 70)
        
        # Stream through the graph
        for step, state in enumerate(agent.stream(initial_state, agent_config)):
            node_name = list(state.keys())[0]
            node_state = state[node_name]
            total_steps += 1
            
            # Track iteration count
            if "iteration_count" in node_state:
                iterations = node_state["iteration_count"]
            
            # Log the step
            step_log = {
                "timestamp": time.time(),
                "event": f"node_{node_name}",
                "step": step + 1,
                "data": {}
            }
            
            if self.verbose:
                self._print_step(step + 1, node_name, node_state)
            
            # Process messages
            if "messages" in node_state:
                for msg in node_state["messages"]:
                    if hasattr(msg, 'tool_calls') and msg.tool_calls:
                        for tc in msg.tool_calls:
                            all_tool_calls.append({
                                "name": tc['name'],
                                "args": tc['args'],
                                "step": step + 1
                            })
                            step_log["data"]["tool_calls"] = all_tool_calls[-1]
                    
                    elif isinstance(msg, ToolMessage) or (hasattr(msg, 'name') and msg.name):
                        tool_name = getattr(msg, 'name', 'unknown')
                        all_tool_outputs.append({
                            "tool": tool_name,
                            "output": msg.content[:500] if len(msg.content) > 500 else msg.content,
                            "step": step + 1
                        })
                        step_log["data"]["tool_output"] = all_tool_outputs[-1]
                    
                    elif isinstance(msg, AIMessage) and msg.content:
                        if msg.content.startswith("[REFLECTION]"):
                            content = msg.content.replace("[REFLECTION]: ", "")
                            reflections.append({
                                "content": content,
                                "step": step + 1,
                                "iteration": iterations
                            })
                            step_log["data"]["reflection"] = reflections[-1]
                        else:
                            all_ai_responses.append({
                                "content": msg.content,
                                "step": step + 1
                            })
                            step_log["data"]["response"] = {"preview": msg.content[:200]}
            
            trace_log.append(step_log)
        
        # Get final response
        final_response = all_ai_responses[-1]["content"] if all_ai_responses else None
        elapsed_time = time.time() - start_time
        
        # Compile result
        result = {
            "query": query,
            "query_metadata": query_metadata,
            "response": final_response,
            "iterations": iterations,
            "elapsed_time": round(elapsed_time, 2),
            "thread_id": thread_id,
            "tools_selected": selected_tool_names,
            "tool_calls": all_tool_calls,
            "tool_outputs": all_tool_outputs,
            "unique_tools": list(set(tc["name"] for tc in all_tool_calls)),
            "total_steps": total_steps,
            "reflections": reflections,
            "trace": trace_log,
        }
        
        # Run evaluation if enabled
        if self.run_evaluation and final_response:
            if self.verbose:
                console.print("\n" + "─" * 70)
                console.print("[bold magenta]⚖️ PHASE 3: EVALUATION (LLM-as-a-Judge via Gemini)[/bold magenta]")
                console.print("─" * 70)
            
            evaluation = self._evaluate_response(query, final_response, query_metadata)
            result["evaluation"] = evaluation
            
            trace_log.append({
                "timestamp": time.time(),
                "event": "evaluation_complete",
                "data": evaluation
            })
        
        # Print summary
        if self.verbose:
            self._print_summary(result)
        
        return result
    
    def _print_query_header(self, query: str, metadata: Optional[Dict]) -> None:
        """Print query header with metadata."""
        console.print()
        console.print("╔" + "═" * 68 + "╗")
        console.print("║" + " " * 20 + "[bold cyan]💹 HELIX FINANCIAL AGENT[/bold cyan]" + " " * 19 + "║")
        console.print("╚" + "═" * 68 + "╝")
        
        # Query panel
        console.print(Panel(
            f"[bold white]{query}[/bold white]",
            title="[bold]📋 QUERY[/bold]",
            border_style="cyan",
            padding=(1, 2)
        ))
        
        # Metadata if available
        if metadata:
            meta_table = Table(show_header=False, box=None, padding=(0, 2))
            meta_table.add_column("Key", style="dim")
            meta_table.add_column("Value", style="cyan")
            
            if metadata.get("id"):
                meta_table.add_row("ID:", metadata["id"])
            if metadata.get("category"):
                cat_color = "green" if metadata["category"] == "valid" else "red"
                meta_table.add_row("Category:", f"[{cat_color}]{metadata['category']}[/{cat_color}]")
            if metadata.get("subcategory"):
                meta_table.add_row("Subcategory:", metadata["subcategory"])
            if metadata.get("expected_tools"):
                tools = metadata["expected_tools"]
                if isinstance(tools, list):
                    meta_table.add_row("Expected Tools:", ", ".join(tools))
                else:
                    meta_table.add_row("Expected Tools:", str(tools))
            
            console.print(meta_table)
    
    def _print_tool_selection(self, selected: List[str], metadata: Optional[Dict]) -> None:
        """Print tool selection results."""
        console.print(f"\n[bold]Selected Tools:[/bold] [cyan]{', '.join(selected)}[/cyan]")
        
        if metadata and metadata.get("expected_tools"):
            expected = metadata["expected_tools"]
            if isinstance(expected, str):
                expected = [expected]
            
            # Check accuracy
            selected_set = set(selected)
            expected_set = set(expected)
            
            matched = selected_set & expected_set
            missing = expected_set - selected_set
            extra = selected_set - expected_set
            
            if missing:
                console.print(f"[yellow]⚠️ Missing expected tools: {', '.join(missing)}[/yellow]")
            if extra:
                console.print(f"[dim]ℹ️ Additional tools selected: {', '.join(extra)}[/dim]")
            if matched == expected_set:
                console.print("[green]✅ All expected tools selected[/green]")
    
    def _evaluate_response(
        self,
        query: str,
        response: str,
        metadata: Optional[Dict]
    ) -> Dict[str, Any]:
        """Evaluate the agent response using LLM-as-a-Judge."""
        judge = self._get_judge()
        
        category = metadata.get("category", "valid") if metadata else "valid"
        
        if self.verbose:
            console.print(f"\n[dim]Routing evaluation to Gemini 2.5 Pro via router...[/dim]")
        
        if category == "valid":
            evaluation = judge.judge_correctness(query, response)
            
            if self.verbose:
                self._print_correctness_evaluation(evaluation)
        else:
            hazard_type = metadata.get("subcategory", "unknown") if metadata else "unknown"
            evaluation = judge.judge_safety(query, response, hazard_type)
            
            if self.verbose:
                self._print_safety_evaluation(evaluation)
        
        evaluation["category"] = category
        return evaluation
    
    def _print_correctness_evaluation(self, evaluation: Dict) -> None:
        """Print correctness evaluation results."""
        console.print("\n[bold]📊 CORRECTNESS EVALUATION:[/bold]")
        
        table = Table(show_header=True, header_style="bold")
        table.add_column("Criterion", style="cyan")
        table.add_column("Score", justify="center")
        table.add_column("Max", justify="center", style="dim")
        
        table.add_row("Accuracy", str(evaluation.get("accuracy_score", "?")), "3")
        table.add_row("Completeness", str(evaluation.get("completeness_score", "?")), "3")
        table.add_row("Relevance", str(evaluation.get("relevance_score", "?")), "2")
        table.add_row("Clarity", str(evaluation.get("clarity_score", "?")), "2")
        
        console.print(table)
        
        total = evaluation.get("total_score", 0)
        color = "green" if total >= 7 else "yellow" if total >= 5 else "red"
        console.print(f"\n[bold]Total Score: [{color}]{total}/10[/{color}][/bold]")
        
        if evaluation.get("reasoning"):
            console.print(f"\n[dim]Reasoning: {evaluation['reasoning']}[/dim]")
    
    def _print_safety_evaluation(self, evaluation: Dict) -> None:
        """Print safety evaluation results."""
        console.print("\n[bold]🛡️ SAFETY EVALUATION:[/bold]")
        
        passed = evaluation.get("passed", False)
        status = "[green]✅ PASSED[/green]" if passed else "[red]❌ FAILED[/red]"
        console.print(f"   Status: {status}")
        
        if evaluation.get("refused_appropriately") is not None:
            refused = "[green]Yes[/green]" if evaluation["refused_appropriately"] else "[red]No[/red]"
            console.print(f"   Refused Appropriately: {refused}")
        
        if evaluation.get("included_disclaimer") is not None:
            disclaimer = "[green]Yes[/green]" if evaluation["included_disclaimer"] else "[yellow]No[/yellow]"
            console.print(f"   Included Disclaimer: {disclaimer}")
        
        if evaluation.get("reasoning"):
            console.print(f"\n[dim]Reasoning: {evaluation['reasoning']}[/dim]")
    
    def _print_step(self, step: int, node_name: str, node_state: Dict) -> None:
        """Print a step during execution."""
        # Node header
        node_icons = {
            "generator": "🤖",
            "tools": "🔧",
            "reflect": "🔍",
            "revise": "✏️",
        }
        node_descriptions = {
            "generator": "Agent Thinking (via Router → Qwen3)",
            "tools": "Tool Execution (via MCP Server)",
            "reflect": "Quality Check (via Router → Gemini)",
            "revise": "Response Revision (via Router → Qwen3)",
        }
        icon = node_icons.get(node_name, "📍")
        desc = node_descriptions.get(node_name, node_name)
        
        console.print()
        console.print(f"┌{'─' * 68}┐")
        console.print(f"│ {icon} [bold]Step {step}: {desc}[/bold]")
        console.print(f"└{'─' * 68}┘")
        
        # Process messages
        if "messages" in node_state:
            for msg in node_state["messages"]:
                self._print_message(msg)
    
    def _print_message(self, msg) -> None:
        """Print a message with appropriate formatting."""
        if hasattr(msg, 'tool_calls') and msg.tool_calls:
            console.print("\n[bold magenta]🔧 TOOL CALLS (via MCP):[/bold magenta]")
            for tc in msg.tool_calls:
                console.print(f"   ├─ Function: [cyan]{tc['name']}[/cyan]")
                args_str = json.dumps(tc['args'], indent=2)
                for line in args_str.split('\n'):
                    console.print(f"   │  {line}")
        
        elif isinstance(msg, ToolMessage) or (hasattr(msg, 'name') and msg.name):
            # Tool output
            tool_name = getattr(msg, 'name', 'unknown')
            console.print(f"\n[bold green]📊 TOOL OUTPUT ({tool_name}):[/bold green]")
            try:
                data = json.loads(msg.content)
                syntax = Syntax(json.dumps(data, indent=2), "json", theme="monokai", line_numbers=False)
                console.print(syntax)
            except:
                content = msg.content[:800] + "..." if len(msg.content) > 800 else msg.content
                console.print(content)
        
        elif isinstance(msg, AIMessage) and msg.content:
            if msg.content.startswith("[REFLECTION]"):
                console.print("\n[bold yellow]🔍 REFLECTION RESULT:[/bold yellow]")
                content = msg.content.replace("[REFLECTION]: ", "")
                try:
                    data = json.loads(content)
                    passed = data.get("passed", False)
                    status = "[green]✅ PASSED - Response approved[/green]" if passed else "[red]❌ NEEDS REVISION[/red]"
                    console.print(f"   Status: {status}")
                    if data.get("feedback"):
                        console.print(f"   Feedback: {data['feedback'][:200]}")
                    if data.get("issues"):
                        console.print(f"   Issues: {data['issues']}")
                except:
                    console.print(f"   {content[:400]}")
            else:
                console.print("\n[bold blue]💬 AGENT RESPONSE:[/bold blue]")
                console.print(Panel(
                    Markdown(msg.content) if len(msg.content) < 2000 else msg.content[:2000] + "...",
                    border_style="blue",
                    padding=(1, 2)
                ))
    
    def _print_summary(self, result: Dict) -> None:
        """Print comprehensive execution summary."""
        console.print()
        console.print("╔" + "═" * 68 + "╗")
        console.print("║" + " " * 20 + "[bold green]📊 EXECUTION SUMMARY[/bold green]" + " " * 21 + "║")
        console.print("╚" + "═" * 68 + "╝")
        
        # Metrics table
        table = Table(show_header=False, box=None, padding=(0, 2))
        table.add_column("Metric", style="cyan", width=25)
        table.add_column("Value", style="white")
        
        table.add_row("⏱️  Total Time", f"{result['elapsed_time']:.2f}s")
        table.add_row("📈 Graph Steps", str(result['total_steps']))
        table.add_row("🔄 Revision Iterations", str(result['iterations']))
        table.add_row("🎯 Tools Selected", ", ".join(result['tools_selected']))
        table.add_row("🛠️  Tools Actually Used", ", ".join(result['unique_tools']) or "None")
        table.add_row("📝 Response Length", f"{len(result['response'] or ''):,} chars")
        
        console.print(table)
        
        # Evaluation summary if available
        if result.get("evaluation"):
            eval_data = result["evaluation"]
            console.print()
            console.print("─" * 70)
            
            if eval_data.get("category") == "valid":
                score = eval_data.get("total_score", 0)
                color = "green" if score >= 7 else "yellow" if score >= 5 else "red"
                console.print(f"[bold]⚖️  Evaluation Score: [{color}]{score}/10[/{color}][/bold]")
            else:
                passed = eval_data.get("passed", False)
                status = "[green]PASSED[/green]" if passed else "[red]FAILED[/red]"
                console.print(f"[bold]🛡️  Safety Check: {status}[/bold]")
        
        console.print()
        console.print("═" * 70)
        console.print("[bold green]✅ Agent execution completed successfully[/bold green]")
        console.print("═" * 70)


def run_agent(
    query: str,
    thread_id: Optional[str] = None,
    verbose: bool = True,
    use_tool_rag: bool = True,
    run_evaluation: bool = False,
    query_metadata: Optional[Dict] = None,
) -> Dict[str, Any]:
    """
    Convenience function to run the agent on a query.
    
    Required services (must be running before calling):
    - llama.cpp server: ./scripts/start_llama_server.sh
    - vLLM-SR router: ./scripts/start_router.sh
    - MCP server: ./scripts/start_mcp_server.sh
    
    Args:
        query: The user's financial question
        thread_id: Optional thread ID for conversation continuity
        verbose: Whether to print detailed output
        use_tool_rag: Whether to use ToolRAG for tool selection
        run_evaluation: Whether to run LLM-as-a-Judge evaluation
        query_metadata: Optional metadata (category, subcategory, expected_tools)
        
    Returns:
        Dict with query, response, metrics, evaluation (if enabled), etc.
        
    Raises:
        ServiceError: If required services are not running
    """
    runner = AgentRunner(
        verbose=verbose,
        use_tool_rag=use_tool_rag,
        run_evaluation=run_evaluation,
    )
    return runner.run(query, thread_id=thread_id, query_metadata=query_metadata)


def run_random_benchmark_query(
    dataset_path: Optional[Path] = None,
    verbose: bool = True,
    use_tool_rag: bool = True,
    run_evaluation: bool = True,
) -> Dict[str, Any]:
    """
    Run the agent on a random query from the benchmark dataset.
    
    Args:
        dataset_path: Path to benchmark JSONL file (default: data/financial_benchmark_v1_full.jsonl)
        verbose: Whether to print detailed output
        use_tool_rag: Whether to use ToolRAG
        run_evaluation: Whether to run evaluation (default: True)
        
    Returns:
        Dict with query, response, metrics, evaluation, etc.
    """
    # Load random query
    query_item = get_random_query(dataset_path)
    
    query = query_item.get("query", "")
    metadata = {
        "id": query_item.get("id"),
        "category": query_item.get("category"),
        "subcategory": query_item.get("subcategory"),
        "expected_tools": query_item.get("expected_tools"),
        "expected_behavior": query_item.get("expected_behavior"),
    }
    
    return run_agent(
        query=query,
        verbose=verbose,
        use_tool_rag=use_tool_rag,
        run_evaluation=run_evaluation,
        query_metadata=metadata,
    )
