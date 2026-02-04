"""
Financial Tools Module

Contains all yfinance-based tools for the financial agent:
- Core tools: Required for main use cases
- Distraction tools: Additional tools for ToolRAG testing
- MCP tools: Tools that call the MCP server (RECOMMENDED)

Architecture:
    The agent should use MCP_CORE_TOOLS and MCP_ALL_TOOLS which route
    tool calls through the MCP server. This enables centralized tool
    management and monitoring.
    
    Local tools (CORE_TOOLS, ALL_TOOLS) are available for testing
    but not recommended for production use.
"""

# Local tools (direct yfinance calls) - for testing only
from .core_tools import (
    get_stock_fundamentals as local_get_stock_fundamentals,
    get_historical_prices as local_get_historical_prices,
    get_financial_statements as local_get_financial_statements,
    get_company_news as local_get_company_news,
    CORE_TOOLS as LOCAL_CORE_TOOLS,
)

from .distraction_tools import (
    get_options_chain as local_get_options_chain,
    get_institutional_holders as local_get_institutional_holders,
    get_insider_transactions as local_get_insider_transactions,
    get_analyst_recommendations as local_get_analyst_recommendations,
    get_earnings_calendar as local_get_earnings_calendar,
    get_sustainability_scores as local_get_sustainability_scores,
    get_major_holders as local_get_major_holders,
    calculate_technical_indicators as local_calculate_technical_indicators,
    compare_sector_performance as local_compare_sector_performance,
    get_dividend_history as local_get_dividend_history,
    DISTRACTION_TOOLS as LOCAL_DISTRACTION_TOOLS,
)

# MCP-backed tools (calls go through MCP server) - RECOMMENDED
from .mcp_client import (
    MCPClient,
    get_mcp_client,
    check_mcp_server,
    get_stock_fundamentals,
    get_historical_prices,
    get_financial_statements,
    get_company_news,
    get_options_chain,
    get_institutional_holders,
    get_insider_transactions,
    get_analyst_recommendations,
    get_earnings_calendar,
    get_sustainability_scores,
    get_dividend_history,
    calculate_technical_indicators,
    compare_sector_performance,
    MCP_CORE_TOOLS,
    MCP_DISTRACTION_TOOLS,
    MCP_ALL_TOOLS,
)

# Default to MCP-backed tools
CORE_TOOLS = MCP_CORE_TOOLS
DISTRACTION_TOOLS = MCP_DISTRACTION_TOOLS
ALL_TOOLS = MCP_ALL_TOOLS

__all__ = [
    # MCP Client
    "MCPClient",
    "get_mcp_client",
    "check_mcp_server",
    # MCP-backed tools (default)
    "get_stock_fundamentals",
    "get_historical_prices",
    "get_financial_statements",
    "get_company_news",
    "get_options_chain",
    "get_institutional_holders",
    "get_insider_transactions",
    "get_analyst_recommendations",
    "get_earnings_calendar",
    "get_sustainability_scores",
    "get_dividend_history",
    "calculate_technical_indicators",
    "compare_sector_performance",
    # Tool collections (MCP-backed by default)
    "CORE_TOOLS",
    "DISTRACTION_TOOLS",
    "ALL_TOOLS",
    # MCP-specific collections
    "MCP_CORE_TOOLS",
    "MCP_DISTRACTION_TOOLS",
    "MCP_ALL_TOOLS",
    # Local tools (for testing only)
    "LOCAL_CORE_TOOLS",
    "LOCAL_DISTRACTION_TOOLS",
]
