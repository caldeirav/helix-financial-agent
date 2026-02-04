"""
MCP Client for Financial Tools

Provides a client to interact with the MCP server for tool invocation.
The agent uses this client instead of calling tools directly.
"""

import json
import httpx
from typing import Dict, Any, List, Optional, Callable
from functools import wraps

from langchain_core.tools import tool
from rich.console import Console

from ..config import get_config

console = Console()
config = get_config()


class MCPClient:
    """
    Client for interacting with the Helix Financial Tools MCP Server.
    
    Provides methods to invoke tools via the MCP protocol.
    """
    
    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        timeout: float = 60.0,
    ):
        """
        Initialize the MCP client.
        
        Args:
            host: MCP server host (default: from config)
            port: MCP server port (default: from config)
            timeout: Request timeout in seconds
        """
        self.host = host or config.mcp.host
        self.port = port or config.mcp.port
        self.base_url = f"http://{self.host}:{self.port}"
        self.client = httpx.Client(timeout=timeout)
        self._tools_cache: Optional[Dict[str, Any]] = None
    
    def invoke_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Invoke a tool on the MCP server.
        
        Args:
            tool_name: Name of the tool to invoke
            arguments: Tool arguments
            
        Returns:
            Tool result as dictionary
        """
        try:
            # FastMCP uses JSON-RPC style endpoints
            payload = {
                "jsonrpc": "2.0",
                "method": "tools/call",
                "params": {
                    "name": tool_name,
                    "arguments": arguments,
                },
                "id": 1,
            }
            
            response = self.client.post(
                f"{self.base_url}/mcp/v1/messages",
                json=payload,
                headers={"Content-Type": "application/json"},
            )
            response.raise_for_status()
            
            result = response.json()
            if "error" in result:
                return {"error": result["error"]["message"]}
            
            return result.get("result", {})
            
        except httpx.ConnectError:
            return {
                "error": f"Cannot connect to MCP server at {self.base_url}. "
                         "Ensure the MCP server is running: ./scripts/start_mcp_server.sh"
            }
        except Exception as e:
            return {"error": f"MCP call failed: {str(e)}"}
    
    def list_tools(self) -> List[str]:
        """Get list of available tools from the MCP server."""
        try:
            payload = {
                "jsonrpc": "2.0",
                "method": "tools/list",
                "params": {},
                "id": 1,
            }
            
            response = self.client.post(
                f"{self.base_url}/mcp/v1/messages",
                json=payload,
                headers={"Content-Type": "application/json"},
            )
            response.raise_for_status()
            
            result = response.json()
            tools = result.get("result", {}).get("tools", [])
            return [t.get("name") for t in tools]
            
        except Exception as e:
            console.print(f"[yellow]Warning: Could not list MCP tools: {e}[/yellow]")
            return []
    
    def health_check(self) -> bool:
        """Check if the MCP server is healthy."""
        try:
            response = self.client.get(f"{self.base_url}/health")
            return response.status_code == 200
        except Exception:
            return False
    
    def close(self):
        """Close the HTTP client."""
        self.client.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.close()


# =============================================================================
# MCP-BACKED LANGCHAIN TOOLS
# =============================================================================
# These tools are LangChain-compatible wrappers that call the MCP server

# Global MCP client instance
_mcp_client: Optional[MCPClient] = None


def get_mcp_client() -> MCPClient:
    """Get or create the global MCP client."""
    global _mcp_client
    if _mcp_client is None:
        _mcp_client = MCPClient()
    return _mcp_client


def mcp_tool(tool_name: str, description: str):
    """
    Decorator to create a LangChain tool that calls the MCP server.
    
    Args:
        tool_name: Name of the tool on the MCP server
        description: Tool description for the LLM
    """
    def decorator(func: Callable) -> Callable:
        @tool
        @wraps(func)
        def wrapper(*args, **kwargs) -> str:
            # Get arguments from function signature
            client = get_mcp_client()
            result = client.invoke_tool(tool_name, kwargs)
            return json.dumps(result, indent=2)
        
        # Update the docstring for LangChain
        wrapper.__doc__ = description
        wrapper.__name__ = tool_name
        return wrapper
    return decorator


# =============================================================================
# MCP-BACKED CORE TOOLS
# =============================================================================

@tool
def get_stock_fundamentals(ticker: str) -> str:
    """
    Retrieves fundamental data for a given stock ticker via MCP server.
    
    Use this for questions about PE ratios, market cap, dividend yield, 
    sector, business summary, or company address.
    
    Args:
        ticker: The stock symbol (e.g., "AAPL", "NVDA", "MSFT")
        
    Returns:
        JSON string containing key fundamental metrics
    """
    client = get_mcp_client()
    result = client.invoke_tool("get_stock_fundamentals", {"ticker": ticker})
    return json.dumps(result, indent=2)


@tool
def get_historical_prices(ticker: str, period: str = "1mo") -> str:
    """
    Fetches historical price data via MCP server.
    
    Use this for technical analysis, moving averages, or performance over time.
    
    Args:
        ticker: The stock symbol (e.g., "AAPL", "TSLA")
        period: Time period - "1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "ytd", "max"
        
    Returns:
        JSON string with price history and summary statistics
    """
    client = get_mcp_client()
    result = client.invoke_tool("get_historical_prices", {"ticker": ticker, "period": period})
    return json.dumps(result, indent=2)


@tool
def get_financial_statements(ticker: str) -> str:
    """
    Retrieves the latest balance sheet, income statement, and cash flow via MCP server.
    
    Use this for deep-dive questions about debt, revenue growth, assets, or liabilities.
    
    Args:
        ticker: The stock symbol (e.g., "AAPL", "GOOGL")
        
    Returns:
        JSON string with key financial line items
    """
    client = get_mcp_client()
    result = client.invoke_tool("get_financial_statements", {"ticker": ticker})
    return json.dumps(result, indent=2)


@tool
def get_company_news(ticker: str) -> str:
    """
    Fetches the latest news headlines via MCP server.
    
    Use this for questions about recent events, sentiment, or why a stock is moving.
    
    Args:
        ticker: The stock symbol (e.g., "AAPL", "NVDA")
        
    Returns:
        JSON string with news items
    """
    client = get_mcp_client()
    result = client.invoke_tool("get_company_news", {"ticker": ticker})
    return json.dumps(result, indent=2)


# =============================================================================
# MCP-BACKED DISTRACTION TOOLS
# =============================================================================

@tool
def get_options_chain(ticker: str, expiration_date: str = None) -> str:
    """
    Retrieves options chain data via MCP server.
    
    Args:
        ticker: The stock symbol
        expiration_date: Optional expiration date (YYYY-MM-DD)
    """
    client = get_mcp_client()
    args = {"ticker": ticker}
    if expiration_date:
        args["expiration_date"] = expiration_date
    result = client.invoke_tool("get_options_chain", args)
    return json.dumps(result, indent=2)


@tool
def get_institutional_holders(ticker: str) -> str:
    """Retrieves institutional ownership data via MCP server."""
    client = get_mcp_client()
    result = client.invoke_tool("get_institutional_holders", {"ticker": ticker})
    return json.dumps(result, indent=2)


@tool
def get_insider_transactions(ticker: str) -> str:
    """Retrieves recent insider trading activity via MCP server."""
    client = get_mcp_client()
    result = client.invoke_tool("get_insider_transactions", {"ticker": ticker})
    return json.dumps(result, indent=2)


@tool
def get_analyst_recommendations(ticker: str) -> str:
    """Retrieves analyst ratings and recommendations via MCP server."""
    client = get_mcp_client()
    result = client.invoke_tool("get_analyst_recommendations", {"ticker": ticker})
    return json.dumps(result, indent=2)


@tool
def get_earnings_calendar(ticker: str) -> str:
    """Retrieves earnings calendar data via MCP server."""
    client = get_mcp_client()
    result = client.invoke_tool("get_earnings_calendar", {"ticker": ticker})
    return json.dumps(result, indent=2)


@tool
def get_sustainability_scores(ticker: str) -> str:
    """Retrieves ESG sustainability scores via MCP server."""
    client = get_mcp_client()
    result = client.invoke_tool("get_sustainability_scores", {"ticker": ticker})
    return json.dumps(result, indent=2)


@tool
def get_dividend_history(ticker: str, years: int = 5) -> str:
    """Retrieves dividend payment history via MCP server."""
    client = get_mcp_client()
    result = client.invoke_tool("get_dividend_history", {"ticker": ticker, "years": years})
    return json.dumps(result, indent=2)


@tool
def calculate_technical_indicators(ticker: str, indicators: str = "RSI,MACD,BB") -> str:
    """Calculates technical indicators via MCP server (stub)."""
    client = get_mcp_client()
    result = client.invoke_tool(
        "calculate_technical_indicators", 
        {"ticker": ticker, "indicators": indicators.split(",")}
    )
    return json.dumps(result, indent=2)


@tool
def compare_sector_performance(sector: str, period: str = "1y") -> str:
    """Compares sector performance via MCP server (stub)."""
    client = get_mcp_client()
    result = client.invoke_tool(
        "compare_sector_performance",
        {"sector": sector, "period": period}
    )
    return json.dumps(result, indent=2)


# =============================================================================
# TOOL COLLECTIONS (MCP-backed)
# =============================================================================

MCP_CORE_TOOLS = [
    get_stock_fundamentals,
    get_historical_prices,
    get_financial_statements,
    get_company_news,
]

MCP_DISTRACTION_TOOLS = [
    get_options_chain,
    get_institutional_holders,
    get_insider_transactions,
    get_analyst_recommendations,
    get_earnings_calendar,
    get_sustainability_scores,
    get_dividend_history,
    calculate_technical_indicators,
    compare_sector_performance,
]

MCP_ALL_TOOLS = MCP_CORE_TOOLS + MCP_DISTRACTION_TOOLS


def check_mcp_server() -> bool:
    """
    Check if the MCP server is running and accessible.
    
    Returns:
        True if server is healthy, False otherwise
    """
    client = get_mcp_client()
    return client.health_check()
