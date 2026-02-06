"""
Agent Runner with Rich Logging and MLflow Tracing

Provides a high-level interface for running the agent with
comprehensive logging, tracing, and beautiful console output.

Architecture:
    - All LLM calls go through the vLLM Semantic Router
    - All tool calls go through the MCP server
    - Both services are MANDATORY for agent operation
    - MLflow tracing captures end-to-end execution

MLflow Integration:
    - Automatic tracing of LLM calls and tool executions via autolog
    - Custom assessments logged per trace:
        - tool_selection_successful: Y/N
        - model_selection_successful: Y/N
        - judge_score: 0-10
    - Benchmark runs wrapped in MLflow runs for aggregation
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
from ..tools import CORE_TOOLS, ALL_TOOLS, check_mcp_server, get_mcp_client
from ..tool_rag import ToolSelector
from ..tracing import (
    setup_mlflow_tracing,
    get_current_trace_id,
    log_run_assessments,
    evaluate_tool_selection,
    evaluate_model_selection,
    is_tracing_enabled,
)
from ..verbose_logging import VerboseLogger, get_logger
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
        # Use removesuffix instead of rstrip to avoid stripping port digits
        base_url = config.model.llama_cpp_base_url
        if base_url.endswith("/v1"):
            base_url = base_url[:-3]  # Remove "/v1" suffix
        base_url = base_url.rstrip("/")
        health_url = f"{base_url}/health"
        with httpx.Client(timeout=5.0) as client:
            response = client.get(health_url)
            return response.status_code == 200
    except Exception:
        return False


def verify_required_services(check_mcp: bool = True, check_router: bool = True, verbose: bool = True) -> Dict[str, Any]:
    """
    Verify that all required services are running.
    
    Args:
        check_mcp: Whether to check MCP server
        check_router: Whether to check semantic router
        verbose: Whether to print status messages
        
    Returns:
        Dict with service status information including MCP tools list
        
    Raises:
        ServiceError: If any required service is not available
    """
    errors = []
    service_info = {
        "llama_cpp": False,
        "router": False,
        "mcp": False,
        "mcp_tools": [],
    }
    
    # Check llama.cpp server
    if check_llama_server_health():
        service_info["llama_cpp"] = True
    else:
        errors.append(
            f"llama.cpp server not responding at {config.model.llama_cpp_base_url}\n"
            "   Start with: ./scripts/start_llama_server.sh"
        )
    
    # Check semantic router
    if check_router:
        if check_router_health():
            service_info["router"] = True
        else:
            errors.append(
                f"Semantic router not responding at {config.router.router_host}:{config.router.router_classify_port}\n"
                "   Start with: ./scripts/start_router.sh"
            )
    else:
        service_info["router"] = None  # Not checked
    
    # Check MCP server with detailed health check
    if check_mcp:
        mcp_client = get_mcp_client()
        mcp_health = mcp_client.health_check_detailed()
        
        if mcp_health["healthy"]:
            service_info["mcp"] = True
            service_info["mcp_tools"] = mcp_health["tools"]
            service_info["mcp_tool_count"] = mcp_health["tool_count"]
            
            if verbose:
                console.print(f"[green]✅ MCP server healthy with {mcp_health['tool_count']} tools available:[/green]")
                # Display tools in a compact format
                tools_str = ", ".join(mcp_health["tools"])
                console.print(f"   [cyan]{tools_str}[/cyan]")
        else:
            error_msg = mcp_health.get("error", "No tools available")
            errors.append(
                f"MCP server not functional at {config.mcp.host}:{config.mcp.port}\n"
                f"   Error: {error_msg}\n"
                "   Start with: ./scripts/start_mcp_server.sh"
            )
    else:
        service_info["mcp"] = None  # Not checked
    
    if errors:
        error_msg = "\n\n".join(errors)
        console.print(Panel(
            f"[bold red]REQUIRED SERVICES NOT AVAILABLE[/bold red]\n\n{error_msg}",
            title="❌ Service Check Failed",
            border_style="red"
        ))
        raise ServiceError("Required services not running. See above for details.")
    
    return service_info


class AgentRunner:
    """
    High-level runner for the Reflexive Financial Agent.
    
    Features:
    - Rich console output for demos
    - ToolRAG integration for dynamic tool selection
    - LLM-as-a-Judge evaluation
    - Comprehensive metrics collection and tracing
    - MLflow tracing for end-to-end observability
    
    Required Services:
    - llama.cpp: Model serving for Qwen3 agent
    - vLLM-SR: Semantic routing between agent and judge models
    - MCP Server: Tool execution via MCP protocol
    
    MLflow Tracing:
    - Automatic LLM/tool tracing via mlflow.langchain.autolog()
    - Custom assessments per trace: tool_selection, model_selection, judge_score
    - View traces at http://localhost:5000 (mlflow ui)
    """
    
    def __init__(
        self,
        tools: Optional[List[Callable]] = None,
        use_tool_rag: bool = True,
        verbose: bool = True,
        skip_service_check: bool = False,
        run_evaluation: bool = False,
        enable_tracing: bool = True,
        logger: Optional[VerboseLogger] = None,
    ):
        """
        Initialize the agent runner.
        
        Args:
            tools: Optional list of tools (default: CORE_TOOLS via MCP)
            use_tool_rag: Whether to use ToolRAG for tool selection
            verbose: Whether to print detailed output
            skip_service_check: Skip service verification (for testing only)
            run_evaluation: Whether to run LLM-as-a-Judge evaluation after response
            enable_tracing: Whether to enable MLflow tracing (default: True)
            logger: Optional VerboseLogger for benchmark output
        """
        self.all_tools = tools or CORE_TOOLS
        self.use_tool_rag = use_tool_rag
        self.verbose = verbose
        self.run_evaluation = run_evaluation
        self.enable_tracing = enable_tracing
        # Use a VerboseLogger when verbose and none passed (e.g. interactive/random) so ToolRAG table is shown
        self.logger = logger
        if logger is None and verbose and use_tool_rag:
            self.logger = get_logger(verbose=True, reset=True)
        
        # Initialize MLflow tracing
        if enable_tracing:
            setup_mlflow_tracing()
        
        # Verify required services are running
        if not skip_service_check:
            if verbose:
                console.print("\n[cyan]🔍 Checking required services...[/cyan]")
            verify_required_services(check_mcp=True, check_router=True, verbose=verbose)
            if verbose:
                console.print("[green]✅ All required services are running[/green]")
        
        # Initialize tool selector if using ToolRAG
        if use_tool_rag:
            self.tool_selector = ToolSelector(logger=self.logger)
            # Register all tools that can be selected (tool store has core + distraction)
            # so that any ToolRAG-selected name resolves to a callable
            tools_to_register = tools if tools is not None else ALL_TOOLS
            for tool in tools_to_register:
                # Handle both regular functions and LangChain StructuredTool objects
                name = getattr(tool, "name", None) or getattr(tool, "__name__", str(tool))
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
    
    def _select_tools(self, query: str, expected_tools: Optional[List[str]] = None) -> List[Callable]:
        """
        Select tools for a query using ToolRAG semantic search.
        
        Only tools that meet the similarity threshold are selected and
        subsequently bound to the LLM. This ensures the agent operates
        with a focused set of relevant tools rather than all available tools.
        
        Args:
            query: The user's financial query
            expected_tools: Optional list of expected tool names (for logging accuracy)
            
        Returns:
            List of selected tool callables to bind to the LLM.
            Falls back to CORE_TOOLS if no tools meet threshold.
            
        Flow:
            1. Query embedded via sentence-transformers
            2. Compared against tool embeddings in ChromaDB
            3. Tools with similarity >= threshold returned
            4. These tools are then bound to generator/revisor nodes
        """
        if not self.use_tool_rag or not self.tool_selector:
            return self.all_tools
        
        selected = self.tool_selector.get_tools_for_query(
            query, 
            verbose=self.verbose,
            expected_tools=expected_tools,
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
        
        # Extract expected tools from metadata for accuracy logging
        expected_tools = query_metadata.get("expected_tools", []) if query_metadata else []
        
        selected_tools = self._select_tools(query, expected_tools=expected_tools)
        selected_tool_names = [getattr(t, "name", None) or getattr(t, "__name__", str(t)) for t in selected_tools]
        
        # Capture tool selection details for UI (without logging)
        tool_selection_details = None
        if self.use_tool_rag and self.tool_selector:
            try:
                tool_selection_details = self.tool_selector.get_selection_details(
                    query, expected_tools=expected_tools
                )
            except Exception:
                pass  # Non-fatal; UI can still show tools_selected names
        
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
        max_steps = config.agent.max_agent_steps
        agent_config = {
            "configurable": {"thread_id": thread_id},
            "recursion_limit": max_steps,
        }

        # Track execution
        all_ai_responses = []
        all_tool_calls = []
        all_tool_outputs = []
        iterations = 0
        total_steps = 0
        reflections = []
        routed_models = []  # Track which models were used
        stopped_at_limit = False

        if self.verbose:
            console.print("\n" + "─" * 70)
            console.print("[bold magenta]🤖 PHASE 2: AGENT EXECUTION (Generator → Tools → Reflector)[/bold magenta]")
            console.print("─" * 70)

        try:
            # Stream through the graph (recursion_limit in agent_config caps graph steps)
            for step, state in enumerate(agent.stream(initial_state, agent_config)):
                node_name = list(state.keys())[0]
                node_state = state[node_name]
                total_steps += 1

                # Failsafe: stop consuming if we hit max steps (in case graph didn't raise)
                if total_steps >= max_steps:
                    stopped_at_limit = True
                    if self.verbose:
                        console.print(f"\n[yellow]⚠ Max agent steps ({max_steps}) reached; stopping.[/yellow]")
                    break

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
                            # Keep full output for judge evaluation (cap 8k per output to avoid huge prompts)
                            out_content = msg.content
                            if len(out_content) > 8000:
                                out_content = out_content[:8000] + "\n... (truncated)"
                            all_tool_outputs.append({
                                "tool": tool_name,
                                "output": out_content,
                                "step": step + 1
                            })
                            step_log["data"]["tool_output"] = all_tool_outputs[-1]

                        elif isinstance(msg, AIMessage) and msg.content:
                            # Extract routing metadata from response
                            response_metadata = getattr(msg, 'response_metadata', {}) or {}
                            model_name = (
                                response_metadata.get('model_name') or
                                response_metadata.get('model') or
                                response_metadata.get('routing_metadata', {}).get('selected_model')
                            )
                            if model_name:
                                routed_models.append(model_name)

                            if msg.content.startswith("[REFLECTION]"):
                                content = msg.content.replace("[REFLECTION]: ", "")
                                reflections.append({
                                    "content": content,
                                    "step": step + 1,
                                    "iteration": iterations,
                                    "model": model_name,
                                })
                                step_log["data"]["reflection"] = reflections[-1]

                                # Log metacognitive step if logger available
                                if self.logger:
                                    try:
                                        import json as _json
                                        reflection_data = _json.loads(content)
                                        passed = reflection_data.get("passed", False)
                                        feedback = reflection_data.get("feedback", "")
                                        issues = reflection_data.get("issues", [])
                                        if isinstance(issues, str):
                                            issues = [issues]
                                        self.logger.log_metacognitive_step(
                                            step_type="reflection",
                                            iteration=iterations,
                                            passed=passed,
                                            feedback=feedback,
                                            issues=issues,
                                        )
                                    except Exception:
                                        pass  # Skip if not valid JSON
                            else:
                                all_ai_responses.append({
                                    "content": msg.content,
                                    "step": step + 1,
                                    "model": model_name,
                                })
                                step_log["data"]["response"] = {"preview": msg.content[:200], "model": model_name}

                trace_log.append(step_log)

        except Exception as e:
            # LangGraph raises when recursion_limit is hit; treat as stopped-at-limit
            if "recursion" in type(e).__name__.lower() or "recursion" in str(e).lower():
                stopped_at_limit = True
                if self.verbose:
                    console.print(f"\n[yellow]⚠ Agent step limit reached ({max_steps}); stopping.[/yellow]")
            else:
                raise

        # Get final response (or fallback if we stopped at limit without a final answer)
        final_response = all_ai_responses[-1]["content"] if all_ai_responses else None
        if stopped_at_limit and final_response is None and all_tool_outputs:
            last_out = all_tool_outputs[-1].get("output", "")
            preview = (last_out[:300] + "...") if len(last_out) > 300 else last_out
            final_response = (
                "Agent stopped: maximum steps reached. The model repeatedly called tools without producing a final answer. "
                "Last tool result (preview): " + preview
            )
        elif stopped_at_limit and final_response is None:
            final_response = (
                "Agent stopped: maximum steps reached. Consider rephrasing the query or simplifying the request."
            )
        elapsed_time = time.time() - start_time
        
        # Determine the primary routed model (most common or last used)
        routed_model = None
        if routed_models:
            # Use the most frequently used model, or the first one
            from collections import Counter
            model_counts = Counter(routed_models)
            routed_model = model_counts.most_common(1)[0][0] if model_counts else routed_models[0]
        
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
            "routed_model": routed_model,
            "routed_models": routed_models,  # All models used during execution
            "tool_selection_details": tool_selection_details,
            "stopped_at_limit": stopped_at_limit,
        }
        
        # Run evaluation if enabled
        if self.run_evaluation and final_response:
            if self.verbose:
                console.print("\n" + "─" * 70)
                console.print("[bold magenta]⚖️ PHASE 3: EVALUATION (LLM-as-a-Judge via Gemini)[/bold magenta]")
                console.print("─" * 70)
            
            evaluation = self._evaluate_response(
                query, final_response, query_metadata,
                tool_calls=result.get("tool_calls"),
                tool_outputs=result.get("tool_outputs"),
            )
            result["evaluation"] = evaluation
            
            trace_log.append({
                "timestamp": time.time(),
                "event": "evaluation_complete",
                "data": evaluation
            })
        
        # =================================================================
        # MLFLOW TRACING: Log custom assessments to the trace
        # =================================================================
        if self.enable_tracing:
            trace_id = get_current_trace_id()
            result["trace_id"] = trace_id
            
            if trace_id:
                if self.verbose:
                    console.print("\n" + "─" * 70)
                    console.print("[bold magenta]📊 PHASE 4: MLFLOW TRACING[/bold magenta]")
                    console.print("─" * 70)
                
                # Evaluate tool selection success
                expected_tools = query_metadata.get("expected_tools", []) if query_metadata else []
                tool_selection_success = evaluate_tool_selection(
                    selected_tool_names, expected_tools
                )
                
                # Evaluate model selection success (based on trace log)
                model_selection_success = evaluate_model_selection(trace_log)
                
                # Extract judge score from evaluation
                judge_score = None
                judge_reasoning = None
                judge_category = None
                if result.get("evaluation"):
                    eval_data = result["evaluation"]
                    judge_category = eval_data.get("category", "valid")
                    if judge_category == "valid":
                        judge_score = eval_data.get("total_score", 0)
                    else:
                        # Safety evaluation: passed=True → 10, passed=False → 0
                        judge_score = 10 if eval_data.get("passed", False) else 0
                    judge_reasoning = eval_data.get("reasoning", "")
                
                # Log all assessments to the trace
                assessment_results = log_run_assessments(
                    trace_id=trace_id,
                    tool_selection_success=tool_selection_success,
                    model_selection_success=model_selection_success,
                    judge_score=judge_score,
                    judge_reasoning=judge_reasoning,
                    judge_category=judge_category,
                    selected_tools=selected_tool_names,
                    expected_tools=expected_tools if expected_tools else None,
                    tools_used=result.get("unique_tools", []),
                    latency_seconds=elapsed_time,
                    iteration_count=iterations,
                )
                
                # Store assessment results
                result["assessments"] = {
                    "tool_selection_successful": tool_selection_success,
                    "model_selection_successful": model_selection_success,
                    "judge_score": judge_score,
                    "trace_id": trace_id,
                }
                
                if self.verbose:
                    console.print(f"  📝 Trace ID: [cyan]{trace_id[:30]}...[/cyan]")
                    console.print(f"  🎯 Tool Selection: [{'green' if tool_selection_success else 'red'}]{'✅ SUCCESS' if tool_selection_success else '❌ FAILED'}[/{'green' if tool_selection_success else 'red'}]")
                    console.print(f"  🔀 Model Selection: [{'green' if model_selection_success else 'yellow'}]{'✅ SUCCESS' if model_selection_success else '⚠️ UNVERIFIED'}[/{'green' if model_selection_success else 'yellow'}]")
                    if judge_score is not None:
                        score_color = "green" if judge_score >= 7 else "yellow" if judge_score >= 5 else "red"
                        console.print(f"  ⚖️  Judge Score: [{score_color}]{judge_score}/10[/{score_color}]")
                    console.print(f"  ✓ Logged {sum(assessment_results.values())} assessments to MLflow")
            else:
                if self.verbose:
                    console.print("\n[yellow]⚠️ No MLflow trace captured for this run[/yellow]")
        
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
        metadata: Optional[Dict],
        tool_calls: Optional[list] = None,
        tool_outputs: Optional[list] = None,
    ) -> Dict[str, Any]:
        """Evaluate the agent response using LLM-as-a-Judge (flow-based: query → tools → outputs → response)."""
        judge = self._get_judge()
        
        category = metadata.get("category", "valid") if metadata else "valid"
        
        if self.verbose:
            console.print(f"\n[dim]Routing evaluation to Gemini 2.5 Pro via router...[/dim]")
        
        if category == "valid":
            evaluation = judge.judge_correctness(
                query, response,
                tool_calls=tool_calls,
                tool_outputs=tool_outputs,
            )
            
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
            "generator": "Agent Thinking",
            "tools": "Tool Execution (via MCP Server)",
            "reflect": "Quality Check",
            "revise": "Response Revision",
        }
        icon = node_icons.get(node_name, "📍")
        desc = node_descriptions.get(node_name, node_name)
        
        console.print()
        console.print(f"┌{'─' * 68}┐")
        console.print(f"│ {icon} [bold]Step {step}: {desc}[/bold]")
        console.print(f"└{'─' * 68}┘")
        
        # Process messages and extract routing info
        if "messages" in node_state:
            for msg in node_state["messages"]:
                self._print_message(msg, node_name=node_name)
    
    def _print_message(self, msg, node_name: str = None) -> None:
        """Print a message with appropriate formatting including routing info."""
        # Extract and display routing metadata from AIMessage
        if isinstance(msg, AIMessage):
            self._print_routing_info(msg, node_name)
        
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
    
    def _print_routing_info(self, msg: AIMessage, node_name: str = None) -> None:
        """Print routing/model selection information from AIMessage metadata."""
        # Get response metadata from LangChain AIMessage
        response_metadata = getattr(msg, 'response_metadata', {}) or {}
        
        # Extract model info from response metadata
        model_name = response_metadata.get('model_name') or response_metadata.get('model', '')
        
        # Check for routing metadata (if router includes it)
        routing_meta = response_metadata.get('routing_metadata', {})
        
        # Determine the expected model based on node type
        expected_models = {
            "generator": ("qwen3", "Qwen3-30B-A3B", "Financial Analysis"),
            "reflect": ("gemini", "Gemini 2.5 Pro", "Quality Evaluation"),
            "revise": ("qwen3", "Qwen3-30B-A3B", "Response Revision"),
        }
        
        if node_name in expected_models:
            expected_prefix, expected_display, purpose = expected_models[node_name]
            
            console.print(f"\n[bold cyan]🔀 ROUTING DECISION:[/bold cyan]")
            console.print(f"   ├─ Purpose: [white]{purpose}[/white]")
            console.print(f"   ├─ Request Model: [yellow]MoM[/yellow] (Model of Models - auto-select)")
            
            # Show actual model used if available
            if model_name:
                # Determine if routing was correct
                model_lower = model_name.lower()
                if expected_prefix in model_lower or expected_display.lower() in model_lower:
                    console.print(f"   ├─ Routed To: [green]{model_name}[/green] ✓")
                    console.print(f"   └─ Rationale: [dim]Semantic router detected {purpose.lower()} intent[/dim]")
                else:
                    console.print(f"   ├─ Routed To: [yellow]{model_name}[/yellow]")
                    console.print(f"   └─ Expected: [dim]{expected_display}[/dim]")
            else:
                console.print(f"   ├─ Expected Model: [cyan]{expected_display}[/cyan]")
                console.print(f"   └─ Rationale: [dim]Router auto-selects based on prompt content[/dim]")
            
            # Show confidence if available
            if routing_meta.get('confidence'):
                console.print(f"   │  Confidence: {routing_meta['confidence']:.2%}")
    
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
        
        # MLflow Assessments summary if available
        if result.get("assessments"):
            assessments = result["assessments"]
            console.print()
            console.print("─" * 70)
            console.print("[bold]📊 MLflow Assessments:[/bold]")
            
            # Tool Selection
            tool_success = assessments.get("tool_selection_successful")
            if tool_success is not None:
                status = "[green]Y[/green]" if tool_success else "[red]N[/red]"
                console.print(f"   Tool Selection Successful: {status}")
            
            # Model Selection
            model_success = assessments.get("model_selection_successful")
            if model_success is not None:
                status = "[green]Y[/green]" if model_success else "[yellow]?[/yellow]"
                console.print(f"   Model Selection Successful: {status}")
            
            # Judge Score
            judge_score = assessments.get("judge_score")
            if judge_score is not None:
                color = "green" if judge_score >= 7 else "yellow" if judge_score >= 5 else "red"
                console.print(f"   Judge Score: [{color}]{judge_score}/10[/{color}]")
            
            # Trace ID
            trace_id = assessments.get("trace_id")
            if trace_id:
                console.print(f"   Trace ID: [dim]{trace_id[:40]}...[/dim]")
        
        console.print()
        console.print("═" * 70)
        console.print("[bold green]✅ Agent execution completed successfully[/bold green]")
        if result.get("trace_id"):
            console.print("[dim]View trace: mlflow ui --port 5000 → http://localhost:5000[/dim]")
        console.print("═" * 70)


def run_agent(
    query: str,
    thread_id: Optional[str] = None,
    verbose: bool = True,
    use_tool_rag: bool = True,
    run_evaluation: bool = False,
    query_metadata: Optional[Dict] = None,
    enable_tracing: bool = True,
    logger: Optional[VerboseLogger] = None,
    skip_service_check: bool = False,
) -> Dict[str, Any]:
    """
    Convenience function to run the agent on a query.
    
    Required services (must be running before calling):
    - llama.cpp server: ./scripts/start_llama_server.sh
    - vLLM-SR router: ./scripts/start_router.sh
    - MCP server: ./scripts/start_mcp_server.sh
    
    MLflow Tracing:
    When enable_tracing=True (default), the result includes:
    - trace_id: MLflow trace identifier
    - assessments: Dict with tool_selection_successful, model_selection_successful, judge_score
    
    View traces at http://localhost:5000 after running: mlflow ui --port 5000
    
    Args:
        query: The user's financial question
        thread_id: Optional thread ID for conversation continuity
        verbose: Whether to print detailed output
        use_tool_rag: Whether to use ToolRAG for tool selection
        run_evaluation: Whether to run LLM-as-a-Judge evaluation
        query_metadata: Optional metadata (category, subcategory, expected_tools)
        enable_tracing: Whether to enable MLflow tracing (default: True)
        logger: Optional VerboseLogger for benchmark output
        skip_service_check: Skip service verification (use when already checked)
        
    Returns:
        Dict with query, response, metrics, evaluation (if enabled), assessments, trace_id, etc.
        
    Raises:
        ServiceError: If required services are not running
    """
    runner = AgentRunner(
        verbose=verbose,
        use_tool_rag=use_tool_rag,
        run_evaluation=run_evaluation,
        enable_tracing=enable_tracing,
        logger=logger,
        skip_service_check=skip_service_check,
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
