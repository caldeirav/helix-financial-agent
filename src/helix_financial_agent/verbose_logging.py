"""
Verbose Logging Module for Helix Financial Agent

Provides detailed logging of all model interactions, tool calls, routing decisions,
and flow throughout the agent execution.

Features:
- Logs all LLM requests and responses
- Tracks routing decisions (which model was selected)  
- Shows tool calls with arguments and outputs
- Captures timing information
- ToolRAG selection logging with detailed similarity tables
- Provides end-of-run summary

ToolRAG Logging:
    The log_tool_selection() method displays a formatted table showing:
    - All tools ranked by semantic similarity to the query
    - Selection status for each tool:
        ✓ SEL: Selected (above threshold, within max_tools)
        ~ CAP: Capped (above threshold but excluded by max_tools limit)
        ✗ REJ: Rejected (below similarity threshold)
    - Accuracy comparison vs expected tools (for benchmark evaluation)

    This helps debug tool selection issues and understand why certain
    tools were or weren't selected for a given query.

Output Formatting:
    - Query and expected_tools fields are NOT truncated (shown in full)
    - Other long fields are truncated to 100 characters for readability
    - Tables use Rich Table for proper column alignment

Usage:
    from helix_financial_agent.verbose_logging import VerboseLogger, get_logger

    # Initialize logger
    logger = get_logger(verbose=True)

    # Log model interaction
    logger.log_llm_request("generator", prompt, model="MoM")
    logger.log_llm_response("generator", response, routed_to="qwen3-30b-a3b")

    # Log tool selection (from ToolRAG)
    logger.log_tool_selection(
        query="What is AAPL's PE ratio?",
        all_matches=matches,
        selected=selected,
        threshold=0.3,
        expected_tools=["get_stock_fundamentals"],
        max_tools=4,
        above_threshold_count=8,
    )

    # Print summary
    logger.print_summary()
"""

import json
import textwrap
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from enum import Enum

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.syntax import Syntax
from rich.text import Text
from rich.tree import Tree

console = Console()


class LogLevel(Enum):
    """Log levels for filtering output."""
    DEBUG = 0
    INFO = 1
    WARNING = 2
    ERROR = 3


@dataclass
class LogEntry:
    """A single log entry."""
    timestamp: float
    category: str  # llm, tool, routing, flow, error
    event: str
    details: Dict[str, Any] = field(default_factory=dict)
    duration_ms: Optional[float] = None


@dataclass
class LLMInteraction:
    """Record of a single LLM interaction."""
    node: str
    request_model: str
    routed_model: Optional[str]
    prompt_preview: str
    response_preview: str
    tokens_in: Optional[int]
    tokens_out: Optional[int]
    duration_ms: float
    routing_decision: Optional[str]
    success: bool


@dataclass
class ToolInteraction:
    """Record of a single tool interaction."""
    tool_name: str
    arguments: Dict[str, Any]
    output_preview: str
    duration_ms: float
    success: bool
    error: Optional[str] = None


class VerboseLogger:
    """
    Comprehensive logger for agent execution.
    
    Captures all model interactions, tool calls, and routing decisions
    with detailed output suitable for debugging and monitoring.
    """
    
    def __init__(
        self,
        verbose: bool = True,
        log_level: LogLevel = LogLevel.INFO,
        show_full_prompts: bool = False,
        show_full_responses: bool = False,
        max_preview_length: int = 200,
    ):
        """
        Initialize the verbose logger.
        
        Args:
            verbose: If True, print logs in real-time
            log_level: Minimum log level to display
            show_full_prompts: Show complete prompts (vs preview)
            show_full_responses: Show complete responses (vs preview)
            max_preview_length: Max characters for previews
        """
        self.verbose = verbose
        self.log_level = log_level
        self.show_full_prompts = show_full_prompts
        self.show_full_responses = show_full_responses
        self.max_preview_length = max_preview_length
        
        # Storage
        self.entries: List[LogEntry] = []
        self.llm_interactions: List[LLMInteraction] = []
        self.tool_interactions: List[ToolInteraction] = []
        self.routing_decisions: List[Dict[str, Any]] = []
        
        # Timing
        self.start_time = time.time()
        self._pending_requests: Dict[str, float] = {}
        
        # Counters
        self.counters = {
            "llm_requests": 0,
            "llm_successes": 0,
            "llm_failures": 0,
            "tool_calls": 0,
            "tool_successes": 0,
            "tool_failures": 0,
            "routing_to_qwen": 0,
            "routing_to_gemini": 0,
            "routing_fallback": 0,
            "routing_unknown": 0,
        }
    
    def _truncate(self, text: str, max_length: Optional[int] = None) -> str:
        """Truncate text to max length."""
        max_len = max_length or self.max_preview_length
        if len(text) > max_len:
            return text[:max_len] + "..."
        return text
    
    def _log(self, level: LogLevel, category: str, event: str, details: Dict[str, Any] = None):
        """Internal logging method."""
        entry = LogEntry(
            timestamp=time.time(),
            category=category,
            event=event,
            details=details or {},
        )
        self.entries.append(entry)
        
        if self.verbose and level.value >= self.log_level.value:
            self._print_log(entry, level)
    
    def _print_log(self, entry: LogEntry, level: LogLevel):
        """Print a log entry with formatting."""
        # Color mapping
        colors = {
            LogLevel.DEBUG: "dim",
            LogLevel.INFO: "cyan",
            LogLevel.WARNING: "yellow",
            LogLevel.ERROR: "red",
        }
        color = colors.get(level, "white")
        
        # Icons
        category_icons = {
            "llm": "🤖",
            "tool": "🔧",
            "routing": "🔀",
            "flow": "📍",
            "error": "❌",
            "success": "✅",
        }
        icon = category_icons.get(entry.category, "•")
        
        # Format timestamp
        elapsed = entry.timestamp - self.start_time
        time_str = f"[{elapsed:6.2f}s]"
        
        console.print(f"  {time_str} {icon} [{color}]{entry.event}[/{color}]")
        
        # Print details if any
        if entry.details:
            for key, value in entry.details.items():
                # Don't truncate these important fields - show them in full
                no_truncate_fields = ("query", "expected_tools", "prompt_preview", "prompt")
                if isinstance(value, str) and len(value) > 100 and key not in no_truncate_fields:
                    value = self._truncate(value, 100)
                console.print(f"           └─ {key}: {value}")
    
    # =========================================================================
    # LLM LOGGING
    # =========================================================================
    
    def log_llm_request(
        self,
        node: str,
        prompt: Union[str, List[Dict]],
        model: str = "MoM",
        request_id: Optional[str] = None,
    ):
        """Log an LLM request."""
        self.counters["llm_requests"] += 1
        
        # Convert messages to string - don't truncate, show full prompt
        if isinstance(prompt, list):
            prompt_str = " | ".join([
                f"{m.get('role', 'user')}: {m.get('content', '')}"
                for m in prompt[:3]
            ])
        else:
            prompt_str = str(prompt)
        
        # Don't truncate prompt_preview - important for understanding queries
        preview = prompt_str
        
        # Track timing
        req_id = request_id or f"{node}_{time.time()}"
        self._pending_requests[req_id] = time.time()
        
        self._log(LogLevel.INFO, "llm", f"LLM Request", {
            "model_requested": model,
            "prompt_preview": preview,
        })
        
        return req_id
    
    def log_llm_response(
        self,
        node: str,
        response: Union[str, Dict, Any],
        routed_to: Optional[str] = None,
        routing_decision: Optional[str] = None,
        tokens_in: Optional[int] = None,
        tokens_out: Optional[int] = None,
        request_id: Optional[str] = None,
        success: bool = True,
        error: Optional[str] = None,
    ):
        """Log an LLM response."""
        # Calculate duration
        req_id = request_id or f"{node}_{time.time()}"
        start = self._pending_requests.pop(req_id, time.time())
        duration_ms = (time.time() - start) * 1000
        
        # Extract response text
        if hasattr(response, 'content'):
            response_str = response.content
        elif isinstance(response, dict):
            response_str = response.get('content', str(response))
        else:
            response_str = str(response)
        
        preview = self._truncate(response_str) if not self.show_full_responses else response_str
        
        # Update counters
        if success:
            self.counters["llm_successes"] += 1
        else:
            self.counters["llm_failures"] += 1
        
        # Track routing
        if routed_to:
            if "qwen" in routed_to.lower():
                self.counters["routing_to_qwen"] += 1
            elif "gemini" in routed_to.lower():
                self.counters["routing_to_gemini"] += 1
            else:
                self.counters["routing_unknown"] += 1
        
        # Store interaction
        interaction = LLMInteraction(
            node=node,
            request_model="MoM",
            routed_model=routed_to,
            prompt_preview=preview[:100],
            response_preview=preview[:200],
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            duration_ms=duration_ms,
            routing_decision=routing_decision,
            success=success,
        )
        self.llm_interactions.append(interaction)
        
        # Log
        level = LogLevel.INFO if success else LogLevel.ERROR
        event = f"LLM Response [{node}]" if success else f"LLM Error [{node}]"
        
        details = {
            "routed_to": routed_to or "unknown",
            "duration": f"{duration_ms:.0f}ms",
            "response_preview": preview,
        }
        if routing_decision:
            details["routing_decision"] = routing_decision
        if tokens_out:
            details["tokens"] = f"in={tokens_in}, out={tokens_out}"
        if error:
            details["error"] = error
        
        self._log(level, "llm", event, details)
    
    def log_routing_decision(
        self,
        requested_model: str,
        routed_model: str,
        decision_name: Optional[str] = None,
        confidence: Optional[float] = None,
        is_fallback: bool = False,
    ):
        """Log a routing decision."""
        self.routing_decisions.append({
            "requested": requested_model,
            "routed_to": routed_model,
            "decision": decision_name,
            "confidence": confidence,
            "is_fallback": is_fallback,
            "timestamp": time.time(),
        })
        
        if is_fallback:
            self.counters["routing_fallback"] += 1
        
        level = LogLevel.WARNING if is_fallback else LogLevel.INFO
        event = "Routing Decision (FALLBACK)" if is_fallback else "Routing Decision"
        
        details = {
            "requested": requested_model,
            "routed_to": routed_model,
        }
        if decision_name:
            details["decision"] = decision_name
        if confidence:
            details["confidence"] = f"{confidence:.2%}"
        
        self._log(level, "routing", event, details)
    
    # =========================================================================
    # TOOL SELECTION LOGGING (ToolRAG)
    # =========================================================================
    
    def log_tool_selection(
        self,
        query: str,
        all_matches: List[Dict[str, Any]],
        selected: List[Dict[str, Any]],
        threshold: float,
        expected_tools: Optional[List[str]] = None,
        max_tools: Optional[int] = None,
        above_threshold_count: Optional[int] = None,
    ):
        """
        Log ToolRAG tool selection results.
        
        Args:
            query: The user's query
            all_matches: All tools with their similarity scores
            selected: Tools that passed the threshold
            threshold: The similarity threshold used
            expected_tools: Optional list of expected tool names for accuracy check
            max_tools: Maximum tools limit that was applied
            above_threshold_count: Number of tools above threshold before max_tools limit
        """
        selected_names = [s["name"] for s in selected]
        
        # Calculate selection accuracy if expected tools provided
        accuracy_info = {}
        if expected_tools:
            expected_set = set(expected_tools)
            selected_set = set(selected_names)
            true_positives = selected_set & expected_set
            false_negatives = expected_set - selected_set
            
            accuracy_info = {
                "expected": expected_tools,
                "matched": list(true_positives),
                "missing": list(false_negatives),
                "accuracy": f"{len(true_positives)}/{len(expected_set)}" if expected_set else "N/A",
            }
        
        # Build log details
        log_details = {
            "threshold": threshold,
            "total_tools": len(all_matches),
            "selected_count": len(selected),
            "selected_tools": ", ".join(selected_names) if selected_names else "None",
        }
        
        # Add max_tools info if capping was applied
        if max_tools and above_threshold_count and above_threshold_count > len(selected):
            log_details["max_tools"] = max_tools
            log_details["above_threshold"] = above_threshold_count
        
        # Log summary
        self._log(LogLevel.INFO, "tool", "ToolRAG Selection", log_details)
        
        # Log each tool's similarity in verbose mode
        if self.verbose:
            self._print_tool_selection_table(
                all_matches, selected, threshold, accuracy_info,
                max_tools=max_tools, above_threshold_count=above_threshold_count
            )
    
    def _print_tool_selection_table(
        self,
        all_matches: List[Dict[str, Any]],
        selected: List[Dict[str, Any]],
        threshold: float,
        accuracy_info: Dict[str, Any],
        max_tools: Optional[int] = None,
        above_threshold_count: Optional[int] = None,
    ):
        """Print a formatted table of tool selection results."""
        selected_names = set(s["name"] for s in selected)
        
        # Column widths (adjusted to fit tool names properly)
        col_rank = 3
        col_name = 35
        col_sim = 8
        col_status = 7
        col_cat = 10
        total_width = col_rank + col_name + col_sim + col_status + col_cat + 14  # +14 for borders and padding
        
        console.print()
        console.print("┌" + "─" * (total_width - 2) + "┐")
        title = "🎯 TOOLRAG SELECTION"
        padding = (total_width - 2 - len(title) - 8) // 2  # -8 for markup characters
        console.print("│" + " " * padding + f"[bold cyan]{title}[/bold cyan]" + " " * (total_width - 2 - padding - len(title) - 8) + "│")
        console.print("└" + "─" * (total_width - 2) + "┘")
        
        # Show summary info
        summary_parts = [
            f"[bold]Threshold:[/bold] {threshold:.2f}",
            f"[bold]Total:[/bold] {len(all_matches)}",
            f"[bold]Selected:[/bold] {len(selected)}",
        ]
        if max_tools and above_threshold_count and above_threshold_count > len(selected):
            summary_parts.append(f"[yellow](capped from {above_threshold_count} by max_tools={max_tools})[/yellow]")
        
        console.print(f"\n   {' │ '.join(summary_parts)}")
        console.print()
        
        # Table header - use rich Table for proper formatting
        table = Table(show_header=True, header_style="bold", box=None, padding=(0, 1))
        table.add_column("#", justify="right", width=col_rank)
        table.add_column("Tool Name", width=col_name)
        table.add_column("Sim", justify="right", width=col_sim)
        table.add_column("Status", width=col_status)
        table.add_column("Category", width=col_cat)
        
        for i, match in enumerate(all_matches, 1):
            name = match["name"]
            if len(name) > col_name:
                name = name[:col_name - 3] + "..."
            
            sim = match["similarity"]
            is_selected = match["name"] in selected_names
            category = match.get("category", "?")
            if len(category) > col_cat:
                category = category[:col_cat - 1] + "…"
            
            # Format similarity with color
            if is_selected:
                sim_str = f"[green]{sim:.4f}[/green]"
                status = "[green]✓ SEL[/green]"
            elif sim >= threshold:
                # Above threshold but not selected (capped by max_tools)
                sim_str = f"[yellow]{sim:.4f}[/yellow]"
                status = "[yellow]~ CAP[/yellow]"
            else:
                sim_str = f"[dim]{sim:.4f}[/dim]"
                status = "[dim]✗ REJ[/dim]"
            
            table.add_row(str(i), name, sim_str, status, category)
        
        console.print(table)
        
        # Show accuracy if expected tools provided
        if accuracy_info:
            console.print()
            expected = accuracy_info.get("expected", [])
            matched = accuracy_info.get("matched", [])
            missing = accuracy_info.get("missing", [])
            
            if expected:
                console.print(f"   [bold]Expected:[/bold] {', '.join(expected)}")
                if matched:
                    console.print(f"   [green]✓ Matched:[/green] {', '.join(matched)}")
                if missing:
                    console.print(f"   [red]✗ Missing:[/red] {', '.join(missing)}")
                console.print(f"   [bold]Accuracy:[/bold] {accuracy_info.get('accuracy', 'N/A')}")
        
        console.print()
    
    # =========================================================================
    # SAFETY EVALUATION LOGGING (Hazard Queries)
    # =========================================================================
    
    def log_safety_evaluation(
        self,
        query: str,
        response: str,
        evaluation: Dict[str, Any],
        category: str = "hazard",
        subcategory: str = "",
    ):
        """
        Log detailed safety evaluation for hazard queries.
        
        Shows how the metacognitive loop and guardrails handled
        potentially dangerous queries (regulated advice, unrealistic promises, etc.)
        
        Args:
            query: The hazard query
            response: The agent's response
            evaluation: Safety evaluation results from judge
            category: Query category (hazard)
            subcategory: Hazard type (regulated_advice, unrealistic_promises, etc.)
        """
        passed = evaluation.get("passed", False)
        refused = evaluation.get("refused_appropriately", False)
        disclaimer = evaluation.get("included_disclaimer", False)
        reasoning = evaluation.get("reasoning", "")
        
        if not self.verbose:
            return
        
        console.print()
        console.print("┌" + "─" * 76 + "┐")
        title = "🛡️ SAFETY GUARDRAILS EVALUATION"
        padding = (76 - len(title) - 8) // 2
        console.print("│" + " " * padding + f"[bold yellow]{title}[/bold yellow]" + " " * (76 - padding - len(title) - 8) + "│")
        console.print("└" + "─" * 76 + "┘")
        
        # Hazard type
        hazard_descriptions = {
            "regulated_advice": "Request for specific investment advice (requires licensure)",
            "unrealistic_promises": "Request for guaranteed/risk-free returns",
            "market_manipulation": "Request that could facilitate market manipulation",
            "illegal_activity": "Request for potentially illegal financial activity",
        }
        hazard_desc = hazard_descriptions.get(subcategory, subcategory)
        
        console.print(f"\n   [bold]Hazard Type:[/bold] [yellow]{subcategory}[/yellow]")
        console.print(f"   [dim]{hazard_desc}[/dim]")
        
        # Overall result
        if passed:
            console.print(f"\n   [bold green]✅ SAFETY CHECK: PASSED[/bold green]")
            console.print(f"   [dim]The agent handled this hazard query appropriately[/dim]")
        else:
            console.print(f"\n   [bold red]❌ SAFETY CHECK: FAILED[/bold red]")
            console.print(f"   [dim]The agent may have provided inappropriate advice[/dim]")
        
        # Guardrail indicators
        console.print(f"\n   [bold]Guardrail Indicators:[/bold]")
        
        if refused is not None:
            refused_icon = "[green]✓[/green]" if refused else "[red]✗[/red]"
            console.print(f"   {refused_icon} Refused to provide specific advice: {'Yes' if refused else 'No'}")
        
        if disclaimer is not None:
            disclaimer_icon = "[green]✓[/green]" if disclaimer else "[yellow]○[/yellow]"
            console.print(f"   {disclaimer_icon} Included appropriate disclaimers: {'Yes' if disclaimer else 'No'}")
        
        # Response preview (show how agent actually responded)
        console.print(f"\n   [bold]Agent Response Preview:[/bold]")
        response_preview = response[:400] if response else "(empty)"
        if len(response) > 400:
            response_preview += "..."
        # Wrap long lines for readability
        wrapped = textwrap.fill(response_preview, width=70, initial_indent="   ", subsequent_indent="   ")
        console.print(f"[dim]{wrapped}[/dim]")
        
        # Judge reasoning
        if reasoning:
            console.print(f"\n   [bold]Judge Reasoning:[/bold]")
            reasoning_preview = reasoning[:300] if len(reasoning) > 300 else reasoning
            wrapped_reasoning = textwrap.fill(reasoning_preview, width=70, initial_indent="   ", subsequent_indent="   ")
            console.print(f"[cyan]{wrapped_reasoning}[/cyan]")
        
        console.print()
    
    def log_metacognitive_step(
        self,
        step_type: str,
        iteration: int,
        passed: bool,
        feedback: Optional[str] = None,
        issues: Optional[List[str]] = None,
    ):
        """
        Log a metacognitive loop step (reflection/revision).
        
        Shows how the agent's self-reflection mechanism evaluated
        and potentially revised its response.
        
        Args:
            step_type: "reflection" or "revision"
            iteration: Current iteration number
            passed: Whether the reflection passed
            feedback: Feedback from reflection
            issues: List of issues identified
        """
        if not self.verbose:
            return
        
        if step_type == "reflection":
            if passed:
                console.print(f"\n   [bold green]🔍 REFLECTION (Iteration {iteration}): PASSED[/bold green]")
                console.print(f"   [dim]Response quality meets standards - no revision needed[/dim]")
            else:
                console.print(f"\n   [bold yellow]🔍 REFLECTION (Iteration {iteration}): NEEDS REVISION[/bold yellow]")
                if issues:
                    console.print(f"   [dim]Issues identified: {', '.join(issues)}[/dim]")
                if feedback:
                    console.print(f"   [dim]Feedback: {feedback[:200]}[/dim]")
        
        elif step_type == "revision":
            console.print(f"\n   [bold cyan]✏️ REVISION (Iteration {iteration})[/bold cyan]")
            console.print(f"   [dim]Agent revising response based on reflection feedback[/dim]")
    
    # =========================================================================
    # TOOL CALL LOGGING
    # =========================================================================
    
    def log_tool_call(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        request_id: Optional[str] = None,
    ):
        """Log a tool call."""
        self.counters["tool_calls"] += 1
        
        req_id = request_id or f"{tool_name}_{time.time()}"
        self._pending_requests[req_id] = time.time()
        
        args_str = json.dumps(arguments, default=str)[:100]
        
        self._log(LogLevel.INFO, "tool", f"Tool Call [{tool_name}]", {
            "arguments": args_str,
        })
        
        return req_id
    
    def log_tool_response(
        self,
        tool_name: str,
        output: Any,
        request_id: Optional[str] = None,
        success: bool = True,
        error: Optional[str] = None,
    ):
        """Log a tool response."""
        req_id = request_id or f"{tool_name}_{time.time()}"
        start = self._pending_requests.pop(req_id, time.time())
        duration_ms = (time.time() - start) * 1000
        
        if success:
            self.counters["tool_successes"] += 1
        else:
            self.counters["tool_failures"] += 1
        
        output_str = str(output)[:200] if output else "None"
        
        interaction = ToolInteraction(
            tool_name=tool_name,
            arguments={},
            output_preview=output_str,
            duration_ms=duration_ms,
            success=success,
            error=error,
        )
        self.tool_interactions.append(interaction)
        
        level = LogLevel.INFO if success else LogLevel.ERROR
        event = f"Tool Response [{tool_name}]" if success else f"Tool Error [{tool_name}]"
        
        details = {
            "duration": f"{duration_ms:.0f}ms",
            "output_preview": self._truncate(output_str, 100),
        }
        if error:
            details["error"] = error
        
        self._log(level, "tool", event, details)
    
    # =========================================================================
    # FLOW LOGGING
    # =========================================================================
    
    def log_flow(self, event: str, details: Dict[str, Any] = None):
        """Log a flow event (phase, step, etc.)."""
        self._log(LogLevel.INFO, "flow", event, details)
    
    def log_error(self, event: str, error: str, details: Dict[str, Any] = None):
        """Log an error."""
        full_details = details or {}
        full_details["error"] = error
        self._log(LogLevel.ERROR, "error", event, full_details)
    
    def log_warning(self, event: str, details: Dict[str, Any] = None):
        """Log a warning."""
        self._log(LogLevel.WARNING, "flow", event, details)
    
    def log_success(self, event: str, details: Dict[str, Any] = None):
        """Log a success event."""
        self._log(LogLevel.INFO, "success", event, details)
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    
    def print_summary(self):
        """Print a comprehensive summary of all logged activity."""
        if not self.verbose:
            return
        
        total_time = time.time() - self.start_time
        
        console.print()
        console.print("╔" + "═" * 68 + "╗")
        console.print("║" + " " * 18 + "[bold cyan]📊 EXECUTION SUMMARY[/bold cyan]" + " " * 21 + "║")
        console.print("╚" + "═" * 68 + "╝")
        
        # Overview table
        table = Table(show_header=False, box=None, padding=(0, 2))
        table.add_column("Metric", style="cyan", width=30)
        table.add_column("Value", style="white")
        
        table.add_row("⏱️  Total Time", f"{total_time:.2f}s")
        table.add_row("📝 Log Entries", str(len(self.entries)))
        
        console.print(table)
        
        # LLM Interactions
        if self.llm_interactions:
            console.print()
            console.print("─" * 70)
            console.print("[bold]🤖 LLM Interactions:[/bold]")
            
            llm_table = Table(show_header=True, header_style="bold")
            llm_table.add_column("Node", style="cyan", width=15)
            llm_table.add_column("Routed To", width=18)
            llm_table.add_column("Duration", justify="right", width=10)
            llm_table.add_column("Status", width=10)
            
            for interaction in self.llm_interactions:
                status = "[green]✓[/green]" if interaction.success else "[red]✗[/red]"
                routed = interaction.routed_model or "unknown"
                if "qwen" in routed.lower():
                    routed = f"[yellow]{routed}[/yellow]"
                elif "gemini" in routed.lower():
                    routed = f"[magenta]{routed}[/magenta]"
                
                llm_table.add_row(
                    interaction.node,
                    routed,
                    f"{interaction.duration_ms:.0f}ms",
                    status,
                )
            
            console.print(llm_table)
            
            # LLM stats
            total_llm_time = sum(i.duration_ms for i in self.llm_interactions)
            console.print(f"   Total LLM time: {total_llm_time:.0f}ms ({total_llm_time/1000:.2f}s)")
            console.print(f"   Requests: {self.counters['llm_requests']} (✓{self.counters['llm_successes']} / ✗{self.counters['llm_failures']})")
        
        # Tool Interactions
        if self.tool_interactions:
            console.print()
            console.print("─" * 70)
            console.print("[bold]🔧 Tool Interactions:[/bold]")
            
            tool_table = Table(show_header=True, header_style="bold")
            tool_table.add_column("Tool", style="cyan", width=30)
            tool_table.add_column("Duration", justify="right", width=10)
            tool_table.add_column("Status", width=10)
            
            for interaction in self.tool_interactions:
                status = "[green]✓[/green]" if interaction.success else "[red]✗[/red]"
                tool_table.add_row(
                    interaction.tool_name,
                    f"{interaction.duration_ms:.0f}ms",
                    status,
                )
            
            console.print(tool_table)
            console.print(f"   Tool calls: {self.counters['tool_calls']} (✓{self.counters['tool_successes']} / ✗{self.counters['tool_failures']})")
        
        # Routing Summary
        console.print()
        console.print("─" * 70)
        console.print("[bold]🔀 Routing Summary:[/bold]")
        console.print(f"   → Qwen3 (local):  {self.counters['routing_to_qwen']}")
        console.print(f"   → Gemini (API):   {self.counters['routing_to_gemini']}")
        if self.counters['routing_fallback'] > 0:
            console.print(f"   ⚠️  Fallbacks:     [yellow]{self.counters['routing_fallback']}[/yellow]")
        if self.counters['routing_unknown'] > 0:
            console.print(f"   ❓ Unknown:        {self.counters['routing_unknown']}")
        
        # Errors
        errors = [e for e in self.entries if e.category == "error"]
        if errors:
            console.print()
            console.print("─" * 70)
            console.print(f"[bold red]❌ Errors ({len(errors)}):[/bold red]")
            for err in errors[:5]:  # Show first 5
                console.print(f"   • {err.event}: {err.details.get('error', 'Unknown')}")
            if len(errors) > 5:
                console.print(f"   ... and {len(errors) - 5} more")
        
        console.print()
        console.print("═" * 70)


# Global logger instance
_global_logger: Optional[VerboseLogger] = None


def get_logger(
    verbose: bool = True,
    reset: bool = False,
    **kwargs,
) -> VerboseLogger:
    """
    Get the global verbose logger instance.
    
    Args:
        verbose: Whether to enable verbose logging
        reset: If True, create a new logger instance
        **kwargs: Additional arguments for VerboseLogger
        
    Returns:
        The global VerboseLogger instance
    """
    global _global_logger
    
    if _global_logger is None or reset:
        _global_logger = VerboseLogger(verbose=verbose, **kwargs)
    elif verbose != _global_logger.verbose:
        _global_logger.verbose = verbose
    
    return _global_logger


def reset_logger():
    """Reset the global logger."""
    global _global_logger
    _global_logger = None
