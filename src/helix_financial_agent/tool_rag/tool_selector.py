"""
Tool Selector - Dynamic tool selection using ToolRAG.

Analyzes user queries and selects the most relevant tools
based on semantic similarity, then binds them to the agent.

Key Design Principle:
    Only selected tools are bound to the LLM. This keeps the agent
    focused on relevant tools and prevents context bloat from
    irrelevant tool schemas.

Selection Flow:
    1. User query is embedded via sentence-transformers
    2. Query embedding compared against tool embeddings (ChromaDB)
    3. Tools with similarity >= threshold are selected
    4. Tools are capped to max_tools to prevent context overflow
    5. Selected tools (or fallback core tools) are bound to LLM

Context Overflow Prevention:
    The max_tools parameter limits how many tools are passed to the LLM,
    even if more tools exceed the similarity threshold. This prevents
    errors like:
        "request (16454 tokens) exceeds context size (16384 tokens)"
    
    Each tool adds ~1000-2000 tokens to the prompt. With max_tools=4,
    approximately 6000 tokens are used for tools, leaving room for the
    query, system prompt, and response in a 16K context model.
    
    In the output table, tools above threshold but excluded due to
    max_tools are marked with "~ CAP" status.

Fallback Behavior:
    If no tools meet the similarity threshold, the selector falls
    back to CORE_TOOLS to ensure the agent can still function.

Logging:
    Integrates with VerboseLogger for benchmark output.
    Shows all tools ranked by similarity with selection status:
    - ✓ SEL: Selected (above threshold, within max_tools limit)
    - ~ CAP: Capped (above threshold but excluded by max_tools)
    - ✗ REJ: Rejected (below threshold)
"""

from typing import List, Dict, Any, Optional, Callable, TYPE_CHECKING
import json

from rich.console import Console
from rich.table import Table

from ..config import get_config
from .tool_store import ToolStore, create_default_tool_store

if TYPE_CHECKING:
    from ..verbose_logging import VerboseLogger

console = Console()


class ToolSelector:
    """
    Selects relevant tools for a query using semantic search.
    
    The selector uses ToolRAG to find tools most relevant to the
    user's query, filtering out irrelevant "distraction" tools.
    
    Integrates with VerboseLogger for benchmark output showing:
    - All tools ranked by similarity score
    - Selection status (selected vs rejected)
    - Accuracy vs expected tools (when provided)
    """
    
    def __init__(
        self,
        tool_store: Optional[ToolStore] = None,
        tool_registry: Optional[Dict[str, Callable]] = None,
        logger: Optional["VerboseLogger"] = None,
    ):
        """
        Initialize the tool selector.
        
        Args:
            tool_store: Optional pre-initialized tool store
            tool_registry: Optional mapping of tool names to callables
            logger: Optional VerboseLogger for benchmark output
        """
        self.config = get_config()
        self.tool_store = tool_store or create_default_tool_store()
        self.tool_registry = tool_registry or {}
        self.logger = logger
        
        # Selection parameters
        self.top_k = self.config.tool_rag.top_k
        self.threshold = self.config.tool_rag.threshold
        self.max_tools = self.config.tool_rag.max_tools
    
    def register_tool(self, name: str, func: Callable) -> None:
        """Register a callable tool function."""
        self.tool_registry[name] = func
    
    def register_tools(self, tools: Dict[str, Callable]) -> None:
        """Register multiple callable tool functions."""
        self.tool_registry.update(tools)
    
    def set_logger(self, logger: "VerboseLogger") -> None:
        """Set the VerboseLogger for benchmark output."""
        self.logger = logger
    
    def select_tools(
        self,
        query: str,
        top_k: Optional[int] = None,
        threshold: Optional[float] = None,
        max_tools: Optional[int] = None,
        verbose: bool = False,
        show_all_tools: bool = True,
        expected_tools: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Select relevant tools for a query.
        
        Args:
            query: The user's query
            top_k: Number of tools to retrieve (default from config, or all if show_all_tools=True)
            threshold: Minimum similarity threshold (default from config)
            max_tools: Maximum number of tools to select (prevents context overflow)
            verbose: Whether to print selection details (rich console output)
            show_all_tools: If True, search and display all tools (not just top_k)
            expected_tools: Optional list of expected tool names for accuracy logging
            
        Returns:
            List of selected tool info dicts
        """
        threshold = threshold or self.threshold
        max_tools = max_tools or self.max_tools
        
        # Search for ALL tools if show_all_tools is True, otherwise use top_k
        total_tools = len(self.tool_store._tools)
        search_k = total_tools if show_all_tools else (top_k or self.top_k)
        
        # Search for relevant tools
        matches = self.tool_store.search(query, top_k=search_k)
        
        # Filter by threshold
        above_threshold = [m for m in matches if m["similarity"] >= threshold]
        
        # Limit to max_tools (take highest similarity)
        selected = above_threshold[:max_tools] if max_tools else above_threshold
        
        # Log to VerboseLogger if available (for benchmark output)
        if self.logger:
            self.logger.log_tool_selection(
                query=query,
                all_matches=matches,
                selected=selected,
                threshold=threshold,
                expected_tools=expected_tools,
                max_tools=max_tools,
                above_threshold_count=len(above_threshold),
            )
        
        # Also print detailed selection if verbose (for interactive use)
        if verbose:
            self._print_selection(query, matches, selected, threshold)
        
        return selected

    def get_selection_details(
        self,
        query: str,
        top_k: Optional[int] = None,
        threshold: Optional[float] = None,
        max_tools: Optional[int] = None,
        show_all_tools: bool = True,
        expected_tools: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Return tool selection details (all_matches, selected) without logging.
        Used by UIs (e.g. Streamlit) to display the same table as VerboseLogger.
        """
        threshold = threshold or self.threshold
        max_tools = max_tools or self.max_tools
        total_tools = len(self.tool_store._tools)
        search_k = total_tools if show_all_tools else (top_k or self.top_k)
        matches = self.tool_store.search(query, top_k=search_k)
        above_threshold = [m for m in matches if m["similarity"] >= threshold]
        selected = above_threshold[:max_tools] if max_tools else above_threshold
        return {
            "all_matches": matches,
            "selected": selected,
            "threshold": threshold,
            "max_tools": max_tools,
            "above_threshold_count": len(above_threshold),
            "expected_tools": expected_tools,
        }

    def get_tools_for_query(
        self,
        query: str,
        top_k: Optional[int] = None,
        threshold: Optional[float] = None,
        verbose: bool = False,
        expected_tools: Optional[List[str]] = None,
    ) -> List[Callable]:
        """
        Get callable tool functions for a query.
        
        Args:
            query: The user's query
            top_k: Number of tools to retrieve
            threshold: Minimum similarity threshold
            verbose: Whether to print selection details
            expected_tools: Optional list of expected tool names for accuracy logging
            
        Returns:
            List of callable tool functions
        """
        selected = self.select_tools(
            query, top_k, threshold, verbose, 
            expected_tools=expected_tools
        )
        
        tools = []
        for tool_info in selected:
            name = tool_info["name"]
            if name in self.tool_registry:
                tools.append(self.tool_registry[name])
        
        return tools
    
    def get_tool_names(
        self,
        query: str,
        top_k: Optional[int] = None,
        threshold: Optional[float] = None,
    ) -> List[str]:
        """Get names of selected tools for a query."""
        selected = self.select_tools(query, top_k, threshold)
        return [t["name"] for t in selected]
    
    def analyze_tool_selection(
        self,
        query: str,
        expected_tools: List[str],
        top_k: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Analyze tool selection accuracy against expected tools.
        
        Useful for evaluating ToolRAG performance.
        
        Args:
            query: The user's query
            expected_tools: List of tool names that should be selected
            top_k: Number of tools to retrieve
            
        Returns:
            Analysis results including precision, recall, F1
        """
        selected = self.select_tools(query, top_k=top_k or len(expected_tools) + 2)
        selected_names = set(t["name"] for t in selected)
        expected_set = set(expected_tools)
        
        # Calculate metrics
        true_positives = len(selected_names & expected_set)
        false_positives = len(selected_names - expected_set)
        false_negatives = len(expected_set - selected_names)
        
        precision = true_positives / len(selected_names) if selected_names else 0
        recall = true_positives / len(expected_set) if expected_set else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            "query": query,
            "expected": list(expected_set),
            "selected": list(selected_names),
            "true_positives": list(selected_names & expected_set),
            "false_positives": list(selected_names - expected_set),
            "false_negatives": list(expected_set - selected_names),
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "f1_score": round(f1, 3),
        }
    
    def _print_selection(
        self,
        query: str,
        all_matches: List[Dict],
        selected: List[Dict],
        threshold: float,
    ) -> None:
        """Print detailed tool selection information using rich."""
        console.print()
        console.print("┏" + "━" * 78 + "┓")
        console.print("┃" + " " * 25 + "[bold cyan]🎯 TOOLRAG SELECTION DETAILS[/bold cyan]" + " " * 24 + "┃")
        console.print("┗" + "━" * 78 + "┛")
        
        total_tools = len(self.tool_store._tools)
        
        # Configuration section
        console.print("\n[bold yellow]📋 CONFIGURATION[/bold yellow]")
        console.print(f"   Embedding Model: [cyan]{self.config.tool_rag.embedding_model}[/cyan]")
        console.print(f"   Similarity Threshold: [cyan]{threshold}[/cyan]")
        console.print(f"   Total Tools in Store: [cyan]{total_tools}[/cyan]")
        console.print(f"   Searching: [cyan]ALL {total_tools} tools[/cyan] (ranked by similarity)")
        
        # Query section
        console.print("\n[bold yellow]🔍 QUERY ANALYSIS[/bold yellow]")
        console.print(f"   Input Query: [white]\"{query}\"[/white]")
        console.print(f"   Query Length: [dim]{len(query)} characters[/dim]")
        
        # Search results section
        console.print("\n[bold yellow]📊 SEMANTIC SEARCH RESULTS (ALL TOOLS RANKED)[/bold yellow]")
        console.print(f"   Comparing query against all {len(all_matches)} tools in vector DB\n")
        
        # Detailed results for each tool
        for i, match in enumerate(all_matches, 1):
            is_selected = match in selected
            status_icon = "✅" if is_selected else "❌"
            status_reason = f"[green]SELECTED (similarity {match['similarity']:.3f} >= {threshold})[/green]" if is_selected else f"[red]REJECTED (similarity {match['similarity']:.3f} < {threshold})[/red]"
            
            # Tool header
            console.print(f"   ┌─ {status_icon} [bold cyan]#{i} {match['name']}[/bold cyan] ({match['category']})")
            
            # Similarity score with visual bar
            sim_pct = int(match['similarity'] * 100)
            bar_filled = int(sim_pct / 5)  # 20 chars max
            bar_empty = 20 - bar_filled
            sim_color = "green" if is_selected else "red"
            bar = f"[{sim_color}]{'█' * bar_filled}[/{sim_color}][dim]{'░' * bar_empty}[/dim]"
            console.print(f"   │  Similarity: {bar} [{sim_color}]{match['similarity']:.4f}[/{sim_color}] ({sim_pct}%)")
            
            # Description (truncated)
            desc = match.get('description', 'N/A')
            if len(desc) > 100:
                desc = desc[:100] + "..."
            console.print(f"   │  Description: [dim]{desc}[/dim]")
            
            # Keywords
            keywords = match.get('keywords', [])
            if keywords:
                kw_str = ", ".join(keywords[:6])
                if len(keywords) > 6:
                    kw_str += f" (+{len(keywords)-6} more)"
                console.print(f"   │  Keywords: [dim]{kw_str}[/dim]")
            
            # Use cases (show first 2)
            use_cases = match.get('use_cases', [])
            if use_cases:
                console.print(f"   │  Example queries: [dim]{use_cases[0]}[/dim]")
                if len(use_cases) > 1:
                    console.print(f"   │                   [dim]{use_cases[1]}[/dim]")
            
            # Selection decision
            console.print(f"   └─ Decision: {status_reason}")
            console.print()
        
        # Summary section
        selected_names = [s['name'] for s in selected]
        rejected_names = [m['name'] for m in all_matches if m not in selected]
        
        console.print("[bold yellow]📝 SELECTION SUMMARY[/bold yellow]")
        console.print(f"   ✅ Selected ({len(selected)}): [green]{', '.join(selected_names) if selected_names else 'None'}[/green]")
        if rejected_names:
            console.print(f"   ❌ Rejected ({len(rejected_names)}): [dim]{', '.join(rejected_names)}[/dim]")
        
        # Decision explanation
        console.print("\n[bold yellow]💡 SELECTION RATIONALE[/bold yellow]")
        if selected:
            top_tool = selected[0]
            console.print(f"   Top match '[cyan]{top_tool['name']}[/cyan]' selected because:")
            console.print(f"   → Highest semantic similarity ({top_tool['similarity']:.4f}) to query")
            console.print(f"   → Query keywords likely matched: {', '.join(top_tool.get('keywords', [])[:3])}")
        else:
            console.print(f"   [yellow]⚠️ No tools met the threshold ({threshold})[/yellow]")
            console.print(f"   [dim]Consider lowering the threshold or rephrasing the query[/dim]")
        
        console.print("\n" + "─" * 80)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def create_tool_selector_with_langchain_tools(tools: List[Callable]) -> ToolSelector:
    """
    Create a ToolSelector pre-registered with LangChain tools.
    
    Args:
        tools: List of LangChain tool functions
        
    Returns:
        Configured ToolSelector
    """
    selector = ToolSelector()
    
    for tool in tools:
        # Handle both regular functions and LangChain StructuredTool objects
        name = getattr(tool, "name", None) or getattr(tool, "__name__", str(tool))
        selector.register_tool(name, tool)
    
    return selector
