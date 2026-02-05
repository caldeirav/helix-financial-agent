"""
Verbose Logging Module for Helix Financial Agent

Provides detailed logging of all model interactions, tool calls, routing decisions,
and flow throughout the agent execution.

Features:
- Logs all LLM requests and responses
- Tracks routing decisions (which model was selected)
- Shows tool calls with arguments and outputs
- Captures timing information
- Provides end-of-run summary

Usage:
    from helix_financial_agent.verbose_logging import VerboseLogger, get_logger

    # Initialize logger
    logger = get_logger(verbose=True)

    # Log model interaction
    logger.log_llm_request("generator", prompt, model="MoM")
    logger.log_llm_response("generator", response, routed_to="qwen3-30b-a3b")

    # Log tool call
    logger.log_tool_call("get_stock_fundamentals", {"ticker": "AAPL"})
    logger.log_tool_response("get_stock_fundamentals", result)

    # Print summary
    logger.print_summary()
"""

import json
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
                if isinstance(value, str) and len(value) > 100:
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
        
        # Convert messages to string preview
        if isinstance(prompt, list):
            prompt_str = " | ".join([
                f"{m.get('role', 'user')}: {m.get('content', '')[:50]}"
                for m in prompt[:3]
            ])
        else:
            prompt_str = str(prompt)
        
        preview = self._truncate(prompt_str) if not self.show_full_prompts else prompt_str
        
        # Track timing
        req_id = request_id or f"{node}_{time.time()}"
        self._pending_requests[req_id] = time.time()
        
        self._log(LogLevel.INFO, "llm", f"LLM Request [{node}]", {
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
    # TOOL LOGGING
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
