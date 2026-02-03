"""
Core Financial Tools

These are the essential tools required for the financial agent's main use cases:
- Stock fundamentals (PE ratio, market cap, etc.)
- Historical prices and technical data
- Financial statements (balance sheet, income, cash flow)
- Company news and sentiment
"""

import json
from typing import Optional

import yfinance as yf
from langchain_core.tools import tool


@tool
def get_stock_fundamentals(ticker: str) -> str:
    """
    Retrieves fundamental data for a given stock ticker.
    
    Use this for questions about PE ratios, market cap, dividend yield, 
    sector, business summary, or company address.
    
    Args:
        ticker: The stock symbol (e.g., "AAPL", "NVDA", "MSFT")
        
    Returns:
        JSON string containing key fundamental metrics
    """
    try:
        stock = yf.Ticker(ticker.upper())
        info = stock.info
        
        # Extract key fundamental metrics
        fundamentals = {
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
            "business_summary": info.get("longBusinessSummary", "N/A")[:500] + "..." 
                if info.get("longBusinessSummary") else "N/A",
        }
        
        return json.dumps(fundamentals, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e), "ticker": ticker})


@tool
def get_historical_prices(ticker: str, period: str = "1mo") -> str:
    """
    Fetches historical price data (Open, High, Low, Close, Volume) for a specified period.
    
    Use this for technical analysis, moving averages, or performance over time.
    
    Args:
        ticker: The stock symbol (e.g., "AAPL", "TSLA")
        period: Time period - "1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "ytd", "max"
        
    Returns:
        JSON with price history and summary statistics
    """
    try:
        stock = yf.Ticker(ticker.upper())
        hist = stock.history(period=period)
        
        if hist.empty:
            return json.dumps({"error": f"No data found for {ticker}", "ticker": ticker})
        
        # Calculate summary statistics
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
        
        # Calculate moving averages if enough data
        if len(hist) >= 20:
            summary["sma_20"] = round(hist['Close'].tail(20).mean(), 2)
        if len(hist) >= 50:
            summary["sma_50"] = round(hist['Close'].tail(50).mean(), 2)
        
        # Recent price data (last 5 days)
        recent = hist.tail(5)[['Open', 'High', 'Low', 'Close', 'Volume']].round(2)
        summary["recent_prices"] = recent.reset_index().to_dict(orient='records')
        
        return json.dumps(summary, indent=2, default=str)
    except Exception as e:
        return json.dumps({"error": str(e), "ticker": ticker})


@tool
def get_financial_statements(ticker: str) -> str:
    """
    Retrieves the latest balance sheet, income statement, and cash flow statement.
    
    Use this for deep-dive questions about debt, revenue growth, assets, or liabilities.
    
    Args:
        ticker: The stock symbol (e.g., "AAPL", "GOOGL")
        
    Returns:
        JSON summary of key financial line items
    """
    try:
        stock = yf.Ticker(ticker.upper())
        
        # Get financial statements
        income_stmt = stock.financials
        balance_sheet = stock.balance_sheet
        cash_flow = stock.cashflow
        
        statements = {
            "ticker": ticker.upper(),
            "income_statement": {},
            "balance_sheet": {},
            "cash_flow": {}
        }
        
        # Extract key income statement items
        if not income_stmt.empty:
            latest = income_stmt.iloc[:, 0]  # Most recent period
            statements["income_statement"] = {
                "period": str(income_stmt.columns[0].date()) if hasattr(income_stmt.columns[0], 'date') else str(income_stmt.columns[0]),
                "total_revenue": float(latest.get("Total Revenue", 0)),
                "gross_profit": float(latest.get("Gross Profit", 0)),
                "operating_income": float(latest.get("Operating Income", 0)),
                "net_income": float(latest.get("Net Income", 0)),
                "ebitda": float(latest.get("EBITDA", 0)),
            }
        
        # Extract key balance sheet items
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
        
        # Extract key cash flow items
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
        
        return json.dumps(statements, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e), "ticker": ticker})


@tool
def get_company_news(ticker: str) -> str:
    """
    Fetches the latest news headlines and links for a specific company.
    
    Use this for questions about recent events, sentiment, or why a stock is moving today.
    
    Args:
        ticker: The stock symbol (e.g., "AAPL", "NVDA")
        
    Returns:
        List of news items with title, publisher, and link
    """
    try:
        stock = yf.Ticker(ticker.upper())
        news = stock.news
        
        if not news:
            return json.dumps({"ticker": ticker.upper(), "news": [], "message": "No recent news found"})
        
        # Extract relevant news fields (limit to 10 most recent)
        news_items = []
        for item in news[:10]:
            news_items.append({
                "title": item.get("title", "N/A"),
                "publisher": item.get("publisher", "N/A"),
                "link": item.get("link", "N/A"),
                "published": item.get("providerPublishTime", "N/A"),
                "type": item.get("type", "N/A"),
            })
        
        return json.dumps({
            "ticker": ticker.upper(),
            "news_count": len(news_items),
            "news": news_items
        }, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e), "ticker": ticker})


# List of core tools
CORE_TOOLS = [
    get_stock_fundamentals,
    get_historical_prices,
    get_financial_statements,
    get_company_news,
]
