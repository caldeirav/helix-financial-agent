"""
Financial Tools Module

Contains all yfinance-based tools for the financial agent:
- Core tools: Required for main use cases
- Distraction tools: Additional tools for ToolRAG testing
"""

from .core_tools import (
    get_stock_fundamentals,
    get_historical_prices,
    get_financial_statements,
    get_company_news,
    CORE_TOOLS,
)

from .distraction_tools import (
    get_options_chain,
    get_institutional_holders,
    get_insider_transactions,
    get_analyst_recommendations,
    get_earnings_calendar,
    get_sustainability_scores,
    get_major_holders,
    calculate_technical_indicators,
    compare_sector_performance,
    get_dividend_history,
    DISTRACTION_TOOLS,
)

# All tools combined
ALL_TOOLS = CORE_TOOLS + DISTRACTION_TOOLS

__all__ = [
    # Core tools
    "get_stock_fundamentals",
    "get_historical_prices",
    "get_financial_statements",
    "get_company_news",
    "CORE_TOOLS",
    # Distraction tools
    "get_options_chain",
    "get_institutional_holders",
    "get_insider_transactions",
    "get_analyst_recommendations",
    "get_earnings_calendar",
    "get_sustainability_scores",
    "get_major_holders",
    "calculate_technical_indicators",
    "compare_sector_performance",
    "get_dividend_history",
    "DISTRACTION_TOOLS",
    # All tools
    "ALL_TOOLS",
]
