"""
MCP Server for Financial Tools

Deploys all financial tools via FastMCP as an MCP server.
Supports SSE transport for real-time tool invocation.
"""

import json
from typing import Optional, List

import yfinance as yf
from fastmcp import FastMCP

from ..config import get_config

config = get_config()

# Create FastMCP server instance
mcp = FastMCP(
    name="Helix Financial Tools",
    version="0.1.0",
    description="Financial data tools powered by yfinance for stock analysis, fundamentals, and market data"
)


# =============================================================================
# CORE FINANCIAL TOOLS (Required for main use cases)
# =============================================================================

@mcp.tool
def get_stock_fundamentals(ticker: str) -> dict:
    """
    Retrieves fundamental data for a given stock ticker.
    
    Use this for questions about PE ratios, market cap, dividend yield, 
    sector, business summary, or company address.
    
    Args:
        ticker: The stock symbol (e.g., "AAPL", "NVDA", "MSFT")
        
    Returns:
        Dictionary containing key fundamental metrics
    """
    try:
        stock = yf.Ticker(ticker.upper())
        info = stock.info
        
        return {
            "ticker": ticker.upper(),
            "name": info.get("longName", "N/A"),
            "sector": info.get("sector", "N/A"),
            "industry": info.get("industry", "N/A"),
            "market_cap": info.get("marketCap", "N/A"),
            "pe_ratio": info.get("trailingPE", "N/A"),
            "forward_pe": info.get("forwardPE", "N/A"),
            "peg_ratio": info.get("pegRatio", "N/A"),
            "price_to_book": info.get("priceToBook", "N/A"),
            "dividend_yield": info.get("dividendYield", "N/A"),
            "dividend_rate": info.get("dividendRate", "N/A"),
            "beta": info.get("beta", "N/A"),
            "52_week_high": info.get("fiftyTwoWeekHigh", "N/A"),
            "52_week_low": info.get("fiftyTwoWeekLow", "N/A"),
            "current_price": info.get("currentPrice", info.get("regularMarketPrice", "N/A")),
            "target_mean_price": info.get("targetMeanPrice", "N/A"),
            "recommendation": info.get("recommendationKey", "N/A"),
            "business_summary": (info.get("longBusinessSummary", "N/A")[:500] + "...") 
                if info.get("longBusinessSummary") else "N/A",
        }
    except Exception as e:
        return {"error": str(e), "ticker": ticker}


@mcp.tool
def get_historical_prices(ticker: str, period: str = "1mo") -> dict:
    """
    Fetches historical price data (Open, High, Low, Close, Volume) for a specified period.
    
    Use this for technical analysis, moving averages, or performance over time.
    
    Args:
        ticker: The stock symbol (e.g., "AAPL", "TSLA")
        period: Time period - "1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "ytd", "max"
        
    Returns:
        Dictionary with price history and summary statistics
    """
    try:
        stock = yf.Ticker(ticker.upper())
        hist = stock.history(period=period)
        
        if hist.empty:
            return {"error": f"No data found for {ticker}", "ticker": ticker}
        
        summary = {
            "ticker": ticker.upper(),
            "period": period,
            "start_date": str(hist.index[0].date()),
            "end_date": str(hist.index[-1].date()),
            "start_price": round(hist['Close'].iloc[0], 2),
            "end_price": round(hist['Close'].iloc[-1], 2),
            "period_return_pct": round(((hist['Close'].iloc[-1] / hist['Close'].iloc[0]) - 1) * 100, 2),
            "highest_price": round(hist['High'].max(), 2),
            "lowest_price": round(hist['Low'].min(), 2),
            "avg_volume": int(hist['Volume'].mean()),
            "volatility_std": round(hist['Close'].std(), 2),
        }
        
        if len(hist) >= 20:
            summary["sma_20"] = round(hist['Close'].tail(20).mean(), 2)
        if len(hist) >= 50:
            summary["sma_50"] = round(hist['Close'].tail(50).mean(), 2)
        
        return summary
    except Exception as e:
        return {"error": str(e), "ticker": ticker}


@mcp.tool
def get_financial_statements(ticker: str) -> dict:
    """
    Retrieves the latest balance sheet, income statement, and cash flow statement.
    
    Use this for deep-dive questions about debt, revenue growth, assets, or liabilities.
    
    Args:
        ticker: The stock symbol (e.g., "AAPL", "GOOGL")
        
    Returns:
        Dictionary with key financial line items
    """
    try:
        stock = yf.Ticker(ticker.upper())
        
        income_stmt = stock.financials
        balance_sheet = stock.balance_sheet
        cash_flow = stock.cashflow
        
        statements = {
            "ticker": ticker.upper(),
            "income_statement": {},
            "balance_sheet": {},
            "cash_flow": {}
        }
        
        if not income_stmt.empty:
            latest = income_stmt.iloc[:, 0]
            statements["income_statement"] = {
                "period": str(income_stmt.columns[0].date()) if hasattr(income_stmt.columns[0], 'date') else str(income_stmt.columns[0]),
                "total_revenue": float(latest.get("Total Revenue", 0)),
                "gross_profit": float(latest.get("Gross Profit", 0)),
                "operating_income": float(latest.get("Operating Income", 0)),
                "net_income": float(latest.get("Net Income", 0)),
                "ebitda": float(latest.get("EBITDA", 0)),
            }
        
        if not balance_sheet.empty:
            latest = balance_sheet.iloc[:, 0]
            statements["balance_sheet"] = {
                "period": str(balance_sheet.columns[0].date()) if hasattr(balance_sheet.columns[0], 'date') else str(balance_sheet.columns[0]),
                "total_assets": float(latest.get("Total Assets", 0)),
                "total_liabilities": float(latest.get("Total Liabilities Net Minority Interest", 0)),
                "total_equity": float(latest.get("Stockholders Equity", latest.get("Total Equity Gross Minority Interest", 0))),
                "total_debt": float(latest.get("Total Debt", 0)),
                "cash_and_equivalents": float(latest.get("Cash And Cash Equivalents", 0)),
            }
        
        if not cash_flow.empty:
            latest = cash_flow.iloc[:, 0]
            statements["cash_flow"] = {
                "period": str(cash_flow.columns[0].date()) if hasattr(cash_flow.columns[0], 'date') else str(cash_flow.columns[0]),
                "operating_cash_flow": float(latest.get("Operating Cash Flow", 0)),
                "investing_cash_flow": float(latest.get("Investing Cash Flow", 0)),
                "financing_cash_flow": float(latest.get("Financing Cash Flow", 0)),
                "free_cash_flow": float(latest.get("Free Cash Flow", 0)),
                "capital_expenditure": float(latest.get("Capital Expenditure", 0)),
            }
        
        return statements
    except Exception as e:
        return {"error": str(e), "ticker": ticker}


@mcp.tool
def get_company_news(ticker: str) -> dict:
    """
    Fetches the latest news headlines and links for a specific company.
    
    Use this for questions about recent events, sentiment, or why a stock is moving today.
    
    Args:
        ticker: The stock symbol (e.g., "AAPL", "NVDA")
        
    Returns:
        Dictionary with news items
    """
    try:
        stock = yf.Ticker(ticker.upper())
        news = stock.news
        
        if not news:
            return {"ticker": ticker.upper(), "news": [], "message": "No recent news found"}
        
        news_items = []
        for item in news[:10]:
            news_items.append({
                "title": item.get("title", "N/A"),
                "publisher": item.get("publisher", "N/A"),
                "link": item.get("link", "N/A"),
                "published": item.get("providerPublishTime", "N/A"),
                "type": item.get("type", "N/A"),
            })
        
        return {
            "ticker": ticker.upper(),
            "news_count": len(news_items),
            "news": news_items
        }
    except Exception as e:
        return {"error": str(e), "ticker": ticker}


# =============================================================================
# DISTRACTION TOOLS (For ToolRAG testing)
# =============================================================================

@mcp.tool
def get_options_chain(ticker: str, expiration_date: Optional[str] = None) -> dict:
    """
    Retrieves options chain data including calls and puts for a stock.
    
    Use this for options trading analysis, finding strike prices, 
    implied volatility, and open interest data.
    
    Args:
        ticker: The stock symbol (e.g., "AAPL", "SPY")
        expiration_date: Optional specific expiration date (YYYY-MM-DD format)
        
    Returns:
        Dictionary with options chain data
    """
    try:
        stock = yf.Ticker(ticker.upper())
        expirations = stock.options
        
        if not expirations:
            return {"ticker": ticker.upper(), "error": "No options available"}
        
        exp_to_use = expiration_date if expiration_date in expirations else expirations[0]
        opt_chain = stock.option_chain(exp_to_use)
        
        return {
            "ticker": ticker.upper(),
            "available_expirations": list(expirations[:10]),
            "selected_expiration": exp_to_use,
            "calls_count": len(opt_chain.calls) if not opt_chain.calls.empty else 0,
            "puts_count": len(opt_chain.puts) if not opt_chain.puts.empty else 0,
        }
    except Exception as e:
        return {"error": str(e), "ticker": ticker}


@mcp.tool
def get_institutional_holders(ticker: str) -> dict:
    """
    Retrieves institutional ownership data including top institutional holders.
    
    Use this for analyzing who owns a stock, major institutional positions,
    and ownership concentration.
    
    Args:
        ticker: The stock symbol (e.g., "AAPL", "MSFT")
        
    Returns:
        Dictionary with institutional holder information
    """
    try:
        stock = yf.Ticker(ticker.upper())
        result = {"ticker": ticker.upper()}
        
        institutional = stock.institutional_holders
        if institutional is not None and not institutional.empty:
            result["top_holders"] = institutional.head(5).to_dict(orient='records')
            result["total_count"] = len(institutional)
        
        return result
    except Exception as e:
        return {"error": str(e), "ticker": ticker}


@mcp.tool
def get_insider_transactions(ticker: str) -> dict:
    """
    Retrieves recent insider trading activity including buys and sells.
    
    Use this for tracking executive and insider trading patterns.
    
    Args:
        ticker: The stock symbol
        
    Returns:
        Dictionary with insider transaction data
    """
    try:
        stock = yf.Ticker(ticker.upper())
        result = {"ticker": ticker.upper()}
        
        transactions = stock.insider_transactions
        if transactions is not None and not transactions.empty:
            result["recent_transactions"] = transactions.head(10).to_dict(orient='records')
        
        return result
    except Exception as e:
        return {"error": str(e), "ticker": ticker}


@mcp.tool
def get_analyst_recommendations(ticker: str) -> dict:
    """
    Retrieves analyst ratings and recommendations history.
    
    Use this for understanding Wall Street sentiment and consensus ratings.
    
    Args:
        ticker: The stock symbol
        
    Returns:
        Dictionary with analyst recommendations
    """
    try:
        stock = yf.Ticker(ticker.upper())
        result = {"ticker": ticker.upper()}
        
        rec = stock.recommendations
        if rec is not None and not rec.empty:
            result["recent_recommendations"] = rec.tail(5).reset_index().to_dict(orient='records')
        
        return result
    except Exception as e:
        return {"error": str(e), "ticker": ticker}


@mcp.tool
def get_earnings_calendar(ticker: str) -> dict:
    """
    Retrieves earnings calendar and historical earnings data.
    
    Use this for tracking upcoming earnings dates and historical EPS.
    
    Args:
        ticker: The stock symbol
        
    Returns:
        Dictionary with earnings information
    """
    try:
        stock = yf.Ticker(ticker.upper())
        result = {"ticker": ticker.upper()}
        
        earnings = stock.earnings_dates
        if earnings is not None and not earnings.empty:
            result["upcoming"] = earnings.head(3).reset_index().to_dict(orient='records')
        
        return result
    except Exception as e:
        return {"error": str(e), "ticker": ticker}


@mcp.tool
def get_sustainability_scores(ticker: str) -> dict:
    """
    Retrieves ESG (Environmental, Social, Governance) sustainability scores.
    
    Use this for ESG analysis and sustainable investing research.
    
    Args:
        ticker: The stock symbol
        
    Returns:
        Dictionary with sustainability scores
    """
    try:
        stock = yf.Ticker(ticker.upper())
        result = {"ticker": ticker.upper()}
        
        sustainability = stock.sustainability
        if sustainability is not None and not sustainability.empty:
            scores = {}
            for idx in sustainability.index:
                val = sustainability.loc[idx].values[0] if len(sustainability.loc[idx].values) > 0 else None
                scores[str(idx)] = val
            result["scores"] = scores
        else:
            result["message"] = "No sustainability data available"
        
        return result
    except Exception as e:
        return {"error": str(e), "ticker": ticker}


@mcp.tool
def get_dividend_history(ticker: str, years: int = 5) -> dict:
    """
    Retrieves dividend payment history and dividend growth analysis.
    
    Use this for dividend investing analysis and yield calculations.
    
    Args:
        ticker: The stock symbol
        years: Number of years of history (default: 5)
        
    Returns:
        Dictionary with dividend history
    """
    try:
        stock = yf.Ticker(ticker.upper())
        result = {"ticker": ticker.upper()}
        
        dividends = stock.dividends
        if dividends is not None and not dividends.empty:
            from datetime import datetime, timedelta
            cutoff = datetime.now() - timedelta(days=years*365)
            recent = dividends[dividends.index >= cutoff]
            
            if not recent.empty:
                result["payments_count"] = len(recent)
                result["total_paid"] = round(recent.sum(), 2)
                result["latest_payment"] = round(recent.iloc[-1], 4)
        
        return result
    except Exception as e:
        return {"error": str(e), "ticker": ticker}


@mcp.tool
def calculate_technical_indicators(ticker: str, indicators: Optional[List[str]] = None) -> dict:
    """
    Calculates advanced technical indicators like RSI, MACD, Bollinger Bands.
    
    NOTE: This is a stub implementation for ToolRAG testing.
    
    Args:
        ticker: The stock symbol
        indicators: List of indicators to calculate
        
    Returns:
        Dictionary with status (stub implementation)
    """
    return {
        "ticker": ticker.upper(),
        "status": "stub_implementation",
        "message": "Technical indicator calculation not fully implemented",
        "requested_indicators": indicators or ["RSI", "MACD", "BB"],
    }


@mcp.tool
def compare_sector_performance(sector: str, period: str = "1y") -> dict:
    """
    Compares performance of stocks within a specific sector.
    
    NOTE: This is a stub implementation for ToolRAG testing.
    
    Args:
        sector: The sector name
        period: Time period for comparison
        
    Returns:
        Dictionary with status (stub implementation)
    """
    return {
        "sector": sector,
        "period": period,
        "status": "stub_implementation",
        "message": "Sector comparison not fully implemented",
    }


# =============================================================================
# SERVER CONFIGURATION
# =============================================================================

@mcp.resource("resource://tools/list")
def list_available_tools() -> dict:
    """List all available financial tools."""
    return {
        "core_tools": [
            "get_stock_fundamentals",
            "get_historical_prices",
            "get_financial_statements",
            "get_company_news",
        ],
        "distraction_tools": [
            "get_options_chain",
            "get_institutional_holders",
            "get_insider_transactions",
            "get_analyst_recommendations",
            "get_earnings_calendar",
            "get_sustainability_scores",
            "get_dividend_history",
            "calculate_technical_indicators",
            "compare_sector_performance",
        ]
    }


def main():
    """Run the MCP server."""
    print("🚀 Starting Helix Financial Tools MCP Server")
    print(f"   Host: {config.mcp.host}")
    print(f"   Port: {config.mcp.port}")
    print(f"   Transport: {config.mcp.transport}")
    mcp.run()


if __name__ == "__main__":
    main()
