"""
Tool Selector - Dynamic tool selection using ToolRAG.

Analyzes user queries and selects the most relevant tools
based on semantic similarity, then binds them to the agent.
"""

from typing import List, Dict, Any, Optional, Callable
import json

from rich.console import Console
from rich.table import Table

from ..config import get_config
from .tool_store import ToolStore, create_default_tool_store

console = Console()


class ToolSelector:
    """
    Selects relevant tools for a query using semantic search.
    
    The selector uses ToolRAG to find tools most relevant to the
    user's query, filtering out irrelevant "distraction" tools.
    """
    
    def __init__(
        self,
        tool_store: Optional[ToolStore] = None,
        tool_registry: Optional[Dict[str, Callable]] = None,
    ):
        """
        Initialize the tool selector.
        
        Args:
            tool_store: Optional pre-initialized tool store
            tool_registry: Optional mapping of tool names to callables
        """
        self.config = get_config()
        self.tool_store = tool_store or create_default_tool_store()
        self.tool_registry = tool_registry or {}
        
        # Selection parameters
        self.top_k = self.config.tool_rag.top_k
        self.threshold = self.config.tool_rag.threshold
    
    def register_tool(self, name: str, func: Callable) -> None:
        """Register a callable tool function."""
        self.tool_registry[name] = func
    
    def register_tools(self, tools: Dict[str, Callable]) -> None:
        """Register multiple callable tool functions."""
        self.tool_registry.update(tools)
    
    def select_tools(
        self,
        query: str,
        top_k: Optional[int] = None,
        threshold: Optional[float] = None,
        verbose: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Select relevant tools for a query.
        
        Args:
            query: The user's query
            top_k: Number of tools to retrieve (default from config)
            threshold: Minimum similarity threshold (default from config)
            verbose: Whether to print selection details
            
        Returns:
            List of selected tool info dicts
        """
        top_k = top_k or self.top_k
        threshold = threshold or self.threshold
        
        # Search for relevant tools
        matches = self.tool_store.search(query, top_k=top_k)
        
        # Filter by threshold
        selected = [m for m in matches if m["similarity"] >= threshold]
        
        if verbose:
            self._print_selection(query, matches, selected, threshold)
        
        return selected
    
    def get_tools_for_query(
        self,
        query: str,
        top_k: Optional[int] = None,
        threshold: Optional[float] = None,
        verbose: bool = False,
    ) -> List[Callable]:
        """
        Get callable tool functions for a query.
        
        Args:
            query: The user's query
            top_k: Number of tools to retrieve
            threshold: Minimum similarity threshold
            verbose: Whether to print selection details
            
        Returns:
            List of callable tool functions
        """
        selected = self.select_tools(query, top_k, threshold, verbose)
        
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
        """Print tool selection details using rich."""
        console.print("\n" + "=" * 60)
        console.print("[bold cyan]🔧 TOOL SELECTION (ToolRAG)[/bold cyan]")
        console.print("=" * 60)
        console.print(f"[bold]Query:[/bold] {query}")
        console.print(f"[bold]Threshold:[/bold] {threshold}")
        console.print()
        
        # Create results table
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Tool", style="cyan")
        table.add_column("Category", style="dim")
        table.add_column("Similarity", justify="right")
        table.add_column("Selected", justify="center")
        
        for match in all_matches:
            is_selected = match in selected
            status = "✅" if is_selected else "❌"
            sim_style = "green" if is_selected else "red"
            
            table.add_row(
                match["name"],
                match["category"],
                f"[{sim_style}]{match['similarity']:.3f}[/{sim_style}]",
                status,
            )
        
        console.print(table)
        console.print(f"\n[bold]Selected {len(selected)} of {len(all_matches)} tools[/bold]")
        console.print("=" * 60 + "\n")


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
        name = getattr(tool, "name", tool.__name__)
        selector.register_tool(name, tool)
    
    return selector
