"""
Distraction Tools for ToolRAG Testing

These additional yfinance tools are NOT required for the core financial agent use cases.
They serve as "distraction" tools to test the ToolRAG system's ability to select
only the relevant tools for a given query.

Categories:
- Options data tools
- Holder/ownership tools  
- Analyst and earnings tools
- Sustainability tools
- Technical analysis tools (stub implementations)
"""

import json
from typing import Optional, List

import yfinance as yf
from langchain_core.tools import tool


# =============================================================================
# OPTIONS DATA TOOLS
# =============================================================================

@tool
def get_options_chain(ticker: str, expiration_date: Optional[str] = None) -> str:
    """
    Retrieves options chain data including calls and puts for a stock.
    
    Use this for options trading analysis, finding strike prices, 
    implied volatility, and open interest data.
    
    Args:
        ticker: The stock symbol (e.g., "AAPL", "SPY")
        expiration_date: Optional specific expiration date (YYYY-MM-DD format)
        
    Returns:
        JSON with available expirations and options data
    """
    try:
        stock = yf.Ticker(ticker.upper())
        
        # Get available expiration dates
        expirations = stock.options
        
        if not expirations:
            return json.dumps({
                "ticker": ticker.upper(),
                "error": "No options available for this ticker"
            })
        
        result = {
            "ticker": ticker.upper(),
            "available_expirations": list(expirations[:10]),  # First 10
            "total_expirations": len(expirations),
        }
        
        # Get specific expiration if provided, else use nearest
        exp_to_use = expiration_date if expiration_date in expirations else expirations[0]
        opt_chain = stock.option_chain(exp_to_use)
        
        # Summarize calls
        calls = opt_chain.calls
        if not calls.empty:
            result["calls_summary"] = {
                "expiration": exp_to_use,
                "total_contracts": len(calls),
                "strike_range": [float(calls['strike'].min()), float(calls['strike'].max())],
                "top_volume": calls.nlargest(3, 'volume')[['strike', 'lastPrice', 'volume', 'openInterest']].to_dict(orient='records')
            }
        
        # Summarize puts
        puts = opt_chain.puts
        if not puts.empty:
            result["puts_summary"] = {
                "expiration": exp_to_use,
                "total_contracts": len(puts),
                "strike_range": [float(puts['strike'].min()), float(puts['strike'].max())],
                "top_volume": puts.nlargest(3, 'volume')[['strike', 'lastPrice', 'volume', 'openInterest']].to_dict(orient='records')
            }
        
        return json.dumps(result, indent=2, default=str)
    except Exception as e:
        return json.dumps({"error": str(e), "ticker": ticker})


# =============================================================================
# HOLDER/OWNERSHIP TOOLS
# =============================================================================

@tool
def get_institutional_holders(ticker: str) -> str:
    """
    Retrieves institutional ownership data including top institutional holders.
    
    Use this for analyzing who owns a stock, major institutional positions,
    and ownership concentration.
    
    Args:
        ticker: The stock symbol (e.g., "AAPL", "MSFT")
        
    Returns:
        JSON with institutional holder information
    """
    try:
        stock = yf.Ticker(ticker.upper())
        
        result = {"ticker": ticker.upper()}
        
        # Major holders breakdown
        major = stock.major_holders
        if major is not None and not major.empty:
            result["major_holders"] = major.to_dict()[0]
        
        # Institutional holders
        institutional = stock.institutional_holders
        if institutional is not None and not institutional.empty:
            result["top_institutional_holders"] = institutional.head(10).to_dict(orient='records')
            result["total_institutional_holders"] = len(institutional)
        
        return json.dumps(result, indent=2, default=str)
    except Exception as e:
        return json.dumps({"error": str(e), "ticker": ticker})


@tool
def get_insider_transactions(ticker: str) -> str:
    """
    Retrieves recent insider trading activity including buys and sells.
    
    Use this for tracking executive and insider trading patterns,
    insider sentiment, and compliance monitoring.
    
    Args:
        ticker: The stock symbol (e.g., "AAPL", "TSLA")
        
    Returns:
        JSON with insider transaction data
    """
    try:
        stock = yf.Ticker(ticker.upper())
        
        result = {"ticker": ticker.upper()}
        
        # Insider transactions
        transactions = stock.insider_transactions
        if transactions is not None and not transactions.empty:
            result["recent_transactions"] = transactions.head(15).to_dict(orient='records')
            result["total_transactions"] = len(transactions)
        
        # Insider purchases summary
        purchases = stock.insider_purchases
        if purchases is not None and not purchases.empty:
            result["purchases_summary"] = purchases.to_dict(orient='records')
        
        # Insider roster
        roster = stock.insider_roster_holders
        if roster is not None and not roster.empty:
            result["insider_roster"] = roster.head(10).to_dict(orient='records')
        
        return json.dumps(result, indent=2, default=str)
    except Exception as e:
        return json.dumps({"error": str(e), "ticker": ticker})


@tool
def get_major_holders(ticker: str) -> str:
    """
    Retrieves major shareholder breakdown including insider and institutional percentages.
    
    Use this for ownership structure analysis and control distribution.
    
    Args:
        ticker: The stock symbol
        
    Returns:
        JSON with major holder breakdown
    """
    try:
        stock = yf.Ticker(ticker.upper())
        
        result = {"ticker": ticker.upper()}
        
        major = stock.major_holders
        if major is not None and not major.empty:
            # Convert to readable format
            holder_data = {}
            for idx, val in major[0].items():
                holder_data[str(major[1][idx])] = val
            result["ownership_breakdown"] = holder_data
        
        return json.dumps(result, indent=2, default=str)
    except Exception as e:
        return json.dumps({"error": str(e), "ticker": ticker})


# =============================================================================
# ANALYST & EARNINGS TOOLS
# =============================================================================

@tool
def get_analyst_recommendations(ticker: str) -> str:
    """
    Retrieves analyst ratings and recommendations history.
    
    Use this for understanding Wall Street sentiment, consensus ratings,
    and price target ranges.
    
    Args:
        ticker: The stock symbol (e.g., "AAPL", "NVDA")
        
    Returns:
        JSON with analyst recommendations
    """
    try:
        stock = yf.Ticker(ticker.upper())
        
        result = {"ticker": ticker.upper()}
        
        # Recommendations
        rec = stock.recommendations
        if rec is not None and not rec.empty:
            result["recent_recommendations"] = rec.tail(10).reset_index().to_dict(orient='records')
            result["total_recommendations"] = len(rec)
        
        # Recommendations summary
        rec_summary = stock.recommendations_summary
        if rec_summary is not None and not rec_summary.empty:
            result["recommendation_summary"] = rec_summary.to_dict(orient='records')
        
        # Upgrades/Downgrades
        upgrades = stock.upgrades_downgrades
        if upgrades is not None and not upgrades.empty:
            result["recent_changes"] = upgrades.tail(5).reset_index().to_dict(orient='records')
        
        return json.dumps(result, indent=2, default=str)
    except Exception as e:
        return json.dumps({"error": str(e), "ticker": ticker})


@tool
def get_earnings_calendar(ticker: str) -> str:
    """
    Retrieves earnings calendar and historical earnings data.
    
    Use this for tracking upcoming earnings dates, historical EPS,
    and earnings surprises.
    
    Args:
        ticker: The stock symbol (e.g., "AAPL", "MSFT")
        
    Returns:
        JSON with earnings information
    """
    try:
        stock = yf.Ticker(ticker.upper())
        
        result = {"ticker": ticker.upper()}
        
        # Earnings dates
        earnings = stock.earnings_dates
        if earnings is not None and not earnings.empty:
            result["upcoming_earnings"] = earnings.head(5).reset_index().to_dict(orient='records')
        
        # Historical quarterly earnings
        quarterly = stock.quarterly_earnings
        if quarterly is not None and not quarterly.empty:
            result["quarterly_earnings"] = quarterly.reset_index().to_dict(orient='records')
        
        # Earnings history
        earnings_hist = stock.earnings_history
        if earnings_hist is not None and not earnings_hist.empty:
            result["earnings_history"] = earnings_hist.to_dict(orient='records')
        
        return json.dumps(result, indent=2, default=str)
    except Exception as e:
        return json.dumps({"error": str(e), "ticker": ticker})


# =============================================================================
# SUSTAINABILITY & ESG TOOLS
# =============================================================================

@tool
def get_sustainability_scores(ticker: str) -> str:
    """
    Retrieves ESG (Environmental, Social, Governance) sustainability scores.
    
    Use this for ESG analysis, sustainable investing research,
    and corporate responsibility assessment.
    
    Args:
        ticker: The stock symbol (e.g., "AAPL", "XOM")
        
    Returns:
        JSON with sustainability scores
    """
    try:
        stock = yf.Ticker(ticker.upper())
        
        result = {"ticker": ticker.upper()}
        
        sustainability = stock.sustainability
        if sustainability is not None and not sustainability.empty:
            # Convert to readable format
            scores = {}
            for idx in sustainability.index:
                val = sustainability.loc[idx].values[0] if len(sustainability.loc[idx].values) > 0 else None
                scores[str(idx)] = val
            result["sustainability_scores"] = scores
        else:
            result["message"] = "No sustainability data available"
        
        return json.dumps(result, indent=2, default=str)
    except Exception as e:
        return json.dumps({"error": str(e), "ticker": ticker})


# =============================================================================
# DIVIDEND TOOLS
# =============================================================================

@tool
def get_dividend_history(ticker: str, years: int = 5) -> str:
    """
    Retrieves dividend payment history and dividend growth analysis.
    
    Use this for dividend investing analysis, yield calculations,
    and dividend growth research.
    
    Args:
        ticker: The stock symbol (e.g., "KO", "JNJ")
        years: Number of years of history (default: 5)
        
    Returns:
        JSON with dividend history and statistics
    """
    try:
        stock = yf.Ticker(ticker.upper())
        
        result = {"ticker": ticker.upper()}
        
        # Get dividend data
        dividends = stock.dividends
        if dividends is not None and not dividends.empty:
            # Filter to requested years
            from datetime import datetime, timedelta
            cutoff = datetime.now() - timedelta(days=years*365)
            recent_divs = dividends[dividends.index >= cutoff]
            
            if not recent_divs.empty:
                result["dividend_history"] = [
                    {"date": str(d.date()), "amount": round(v, 4)}
                    for d, v in recent_divs.items()
                ][-20:]  # Last 20 payments
                
                result["statistics"] = {
                    "total_payments": len(recent_divs),
                    "total_paid": round(recent_divs.sum(), 2),
                    "average_payment": round(recent_divs.mean(), 4),
                    "latest_payment": round(recent_divs.iloc[-1], 4),
                }
        else:
            result["message"] = "No dividend history available"
        
        return json.dumps(result, indent=2, default=str)
    except Exception as e:
        return json.dumps({"error": str(e), "ticker": ticker})


# =============================================================================
# STUB/PLACEHOLDER TOOLS (For ToolRAG distraction testing)
# =============================================================================

@tool
def calculate_technical_indicators(ticker: str, indicators: List[str] = None) -> str:
    """
    Calculates advanced technical indicators like RSI, MACD, Bollinger Bands.
    
    Use this for quantitative technical analysis, trading signals,
    and momentum indicators.
    
    NOTE: This is a stub implementation for ToolRAG testing.
    
    Args:
        ticker: The stock symbol
        indicators: List of indicators to calculate (RSI, MACD, BB, etc.)
        
    Returns:
        JSON with calculated technical indicators
    """
    # Stub implementation
    return json.dumps({
        "ticker": ticker.upper(),
        "status": "stub_implementation",
        "message": "Technical indicator calculation is not fully implemented",
        "requested_indicators": indicators or ["RSI", "MACD", "BB"],
        "note": "This tool exists for ToolRAG testing purposes"
    }, indent=2)


@tool
def compare_sector_performance(sector: str, period: str = "1y") -> str:
    """
    Compares performance of stocks within a specific sector.
    
    Use this for sector-level analysis, finding sector leaders,
    and relative performance comparison.
    
    NOTE: This is a stub implementation for ToolRAG testing.
    
    Args:
        sector: The sector name (e.g., "Technology", "Healthcare")
        period: Time period for comparison
        
    Returns:
        JSON with sector performance comparison
    """
    # Stub implementation
    return json.dumps({
        "sector": sector,
        "period": period,
        "status": "stub_implementation",
        "message": "Sector comparison is not fully implemented",
        "note": "This tool exists for ToolRAG testing purposes"
    }, indent=2)


# List of distraction tools
DISTRACTION_TOOLS = [
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
]
