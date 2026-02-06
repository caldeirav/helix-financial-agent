"""
Tool Store - Vector database for tool definitions.

Stores tool metadata and descriptions in a ChromaDB collection
for semantic retrieval. Tools are embedded using a query-centric format
that prioritizes natural language use cases for better matching with
user queries.

Embedding Strategy:
    Tools are embedded as: "[use cases] [keywords] [description]"
    This format puts example queries first, ensuring high similarity
    when user queries match the tool's intended use cases.

Example:
    User query: "What is AAPL's PE ratio?"
    Tool embedding: "What is AAPL's PE ratio? Get the market cap... PE ratio..."
    Result: High cosine similarity → tool selected
"""

import json
import os
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from pathlib import Path

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

from ..config import get_config


def _ensure_hf_token():
    """
    Ensure HuggingFace token is properly configured before loading models.
    
    This function addresses the warning:
        "Warning: You are sending unauthenticated requests to the HF Hub.
         Please set a HF_TOKEN to enable higher rate limits and faster downloads."
    
    The issue occurs because SentenceTransformer may not automatically pick up
    the HF_TOKEN from .env. This function:
    
    1. Loads HF_TOKEN from .env if not already in environment
    2. Sets multiple HF token environment variables for compatibility
    3. Optionally logs in via huggingface_hub for full authentication
    
    Called automatically by ToolStore.__init__ before loading the embedding model.
    """
    # Load from .env if not already set
    if not os.environ.get("HF_TOKEN"):
        from dotenv import load_dotenv
        load_dotenv()
    
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        # Set all known HuggingFace token environment variables
        os.environ["HF_TOKEN"] = hf_token
        os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token
        os.environ["HUGGINGFACE_HUB_TOKEN"] = hf_token
        
        # Try to login via huggingface_hub for full authentication
        try:
            from huggingface_hub import login
            login(token=hf_token, add_to_git_credential=False)
        except Exception:
            pass  # Login failed, but env vars are set which may be enough


@dataclass
class ToolDefinition:
    """Definition of a tool for storage and retrieval."""
    
    name: str
    description: str
    parameters: Dict[str, Any]
    category: str  # "core" or "distraction"
    keywords: List[str] = field(default_factory=list)
    use_cases: List[str] = field(default_factory=list)
    
    def to_embedding_text(self) -> str:
        """
        Create text representation for embedding.
        
        Uses a query-centric format that embeds well against user questions.
        The use cases are natural language questions, so we put them first
        to maximize similarity with user queries.
        """
        # Build a query-centric embedding that matches how users ask questions
        # Start with use cases (natural questions) which match user query format
        parts = []
        
        # Use cases first - these are most similar to actual user queries
        if self.use_cases:
            parts.append(" ".join(self.use_cases))
        
        # Keywords as phrases users might mention
        if self.keywords:
            parts.append(" ".join(self.keywords))
        
        # Description provides context
        parts.append(self.description)
        
        return " ".join(parts)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
            "category": self.category,
            "keywords": self.keywords,
            "use_cases": self.use_cases,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ToolDefinition":
        """Create from dictionary."""
        return cls(
            name=data["name"],
            description=data["description"],
            parameters=data.get("parameters", {}),
            category=data.get("category", "unknown"),
            keywords=data.get("keywords", []),
            use_cases=data.get("use_cases", []),
        )


class ToolStore:
    """
    Vector store for tool definitions using ChromaDB.
    
    Provides semantic search over tool descriptions to find
    relevant tools for a given query.
    """
    
    def __init__(
        self,
        collection_name: str = "helix_tools",
        persist_directory: Optional[str] = None,
    ):
        """
        Initialize the tool store.
        
        Args:
            collection_name: Name of the ChromaDB collection
            persist_directory: Optional path to persist the database
        """
        config = get_config()
        
        # Ensure HF token is set before loading model
        _ensure_hf_token()
        
        # Initialize embedding model (suppress progress bar noise)
        self.embedding_model = SentenceTransformer(
            config.tool_rag.embedding_model,
            token=os.environ.get("HF_TOKEN"),
        )
        
        # Initialize ChromaDB
        if persist_directory:
            self.client = chromadb.PersistentClient(
                path=persist_directory,
                settings=Settings(anonymized_telemetry=False)
            )
        else:
            self.client = chromadb.Client(
                settings=Settings(anonymized_telemetry=False)
            )
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        
        # Local cache of tool definitions
        self._tools: Dict[str, ToolDefinition] = {}
    
    def add_tool(self, tool_def: ToolDefinition) -> None:
        """
        Add a tool definition to the store.
        
        Args:
            tool_def: Tool definition to add
        """
        # Generate embedding
        embedding_text = tool_def.to_embedding_text()
        embedding = self.embedding_model.encode(embedding_text).tolist()
        
        # Add to ChromaDB
        self.collection.upsert(
            ids=[tool_def.name],
            embeddings=[embedding],
            metadatas=[{
                "name": tool_def.name,
                "category": tool_def.category,
                "keywords": json.dumps(tool_def.keywords),
                "use_cases": json.dumps(tool_def.use_cases),
            }],
            documents=[tool_def.description],
        )
        
        # Cache locally
        self._tools[tool_def.name] = tool_def
    
    def add_tools(self, tool_defs: List[ToolDefinition]) -> None:
        """Add multiple tool definitions."""
        for tool_def in tool_defs:
            self.add_tool(tool_def)
    
    def search(
        self,
        query: str,
        top_k: int = 5,
        category_filter: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search for relevant tools based on query.
        
        Args:
            query: The user's query
            top_k: Number of results to return
            category_filter: Optional filter by category ("core" or "distraction")
            
        Returns:
            List of tool matches with scores
        """
        # Generate query embedding
        query_embedding = self.embedding_model.encode(query).tolist()
        
        # Build where clause for filtering
        where = None
        if category_filter:
            where = {"category": category_filter}
        
        # Search ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where,
            include=["metadatas", "documents", "distances"]
        )
        
        # Format results
        matches = []
        if results["ids"] and results["ids"][0]:
            for i, tool_id in enumerate(results["ids"][0]):
                metadata = results["metadatas"][0][i] if results["metadatas"] else {}
                distance = results["distances"][0][i] if results["distances"] else 1.0
                
                matches.append({
                    "name": tool_id,
                    "description": results["documents"][0][i] if results["documents"] else "",
                    "category": metadata.get("category", "unknown"),
                    "keywords": json.loads(metadata.get("keywords", "[]")),
                    "use_cases": json.loads(metadata.get("use_cases", "[]")),
                    "similarity": 1.0 - distance,  # Convert distance to similarity
                })
        
        return matches
    
    def get_tool(self, name: str) -> Optional[ToolDefinition]:
        """Get a specific tool by name."""
        return self._tools.get(name)
    
    def get_all_tools(self) -> List[ToolDefinition]:
        """Get all stored tools."""
        return list(self._tools.values())
    
    def clear(self) -> None:
        """Clear all tools from the store."""
        self.client.delete_collection(self.collection.name)
        self.collection = self.client.create_collection(
            name=self.collection.name,
            metadata={"hnsw:space": "cosine"}
        )
        self._tools.clear()


# =============================================================================
# DEFAULT TOOL DEFINITIONS
# =============================================================================

DEFAULT_TOOL_DEFINITIONS = [
    # Core Tools
    ToolDefinition(
        name="get_stock_fundamentals",
        description="Retrieves fundamental data for a given stock ticker: PE ratios (trailing and forward), PEG ratio, Price-to-Sales (P/S), Price-to-Book (P/B), market cap, dividend yield, EPS, sector, business summary, and company info. Also provides upcoming dividend dates: ex-dividend date, payment date, and next dividend. Use for valuation ratios, per-share metrics, and next dividend or ex-dividend date; not for earnings report dates.",
        parameters={"ticker": {"type": "string", "description": "Stock symbol like AAPL, NVDA, MSFT"}},
        category="core",
        keywords=[
            "PE ratio", "P/E", "price-to-earnings", "forward P/E", "forward PE", "trailing PE", "trailing P/E",
            "PEG ratio", "Price-to-Sales", "P/S ratio", "Price-to-Book", "P/B ratio",
            "market cap", "fundamentals", "valuation", "valuation ratio", "dividend yield", "sector", "beta",
            "52-week high", "52-week low", "current price", "stock price", "EPS", "earnings per share",
            "enterprise value", "book value",
            "ex-dividend date", "ex dividend date", "payment date", "next dividend", "dividend date", "upcoming dividend",
        ],
        use_cases=[
            "What is AAPL's PE ratio?",
            "Fetch the forward price-to-earnings ratio for Apple Inc.",
            "What is the forward P/E for a ticker?",
            "Get market cap and valuation ratios for NVDA",
            "What sector is MSFT in?",
            "Tell me about GOOGL's dividend yield",
            "What is the current price of TSLA?",
            "What is the PEG ratio for WMT?",
            "Show me the Price-to-Sales ratio for AMZN",
            "What is the P/B ratio?",
            "List the PE ratio and dividend yield",
            "Get key financial ratios for analysis",
            "What are the fundamental metrics?",
            "What is the ex-dividend date and payment date for the next dividend?",
            "When is the next dividend payment date for a stock?",
        ],
    ),
    ToolDefinition(
        name="get_historical_prices",
        description="Fetches historical price data: Open, High, Low, Close, Volume (OHLCV) over a time period. Use for price history, returns, moving averages, and volatility from past prices. Not for earnings dates or financial statements.",
        parameters={
            "ticker": {"type": "string", "description": "Stock symbol"},
            "period": {"type": "string", "description": "Time period: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, ytd, max"},
        },
        category="core",
        keywords=[
            "price history", "historical prices", "OHLCV", "returns", "moving average",
            "SMA", "EMA", "50-day", "200-day", "volatility", "performance", "price data",
            "daily prices", "closing price", "open price", "high low", "volume",
            "30-day", "90-day", "standard deviation", "price change", "YTD return"
        ],
        use_cases=[
            "What's AAPL's YTD return?",
            "Show price history for NVDA",
            "What was TSLA's highest price this month?",
            "Calculate the 20-day moving average for MSFT",
            "How far apart are the 50-day and 200-day moving averages?",
            "What is the 200-day moving average for AAPL?",
            "Calculate moving average crossover",
            "Calculate the 90-day volatility for a stock",
            "What is the volatility of this stock?",
            "Show historical volatility calculation",
            "What is the standard deviation of returns?",
            "Show me the historical closing prices",
            "Get daily price data for the last month",
            "What were the prices over the past year?",
        ],
    ),
    ToolDefinition(
        name="get_financial_statements",
        description="Retrieves balance sheet, income statement, and cash flow statement from filings (10-K, 10-Q): total debt, assets, liabilities, revenue, net income, cash flows. Use for reported financials and statement line items, not for valuation ratios or earnings report dates.",
        parameters={"ticker": {"type": "string", "description": "Stock symbol"}},
        category="core",
        keywords=[
            "revenue", "income statement", "balance sheet", "cash flow statement", "total debt",
            "total assets", "total liabilities", "EBITDA", "net income", "operating income",
            "10-K", "10-Q", "quarterly report", "annual report", "financial report", "filing",
            "free cash flow", "operating cash flow", "long-term debt", "short-term debt",
            "shareholders equity", "retained earnings", "gross profit"
        ],
        use_cases=[
            "What is AAPL's total revenue?",
            "Get the income statement data",
            "Show quarterly revenue figures",
            "How much debt does MSFT have?",
            "Find the total debt for a company",
            "What is the company's long-term debt?",
            "Get GOOGL's cash flow statement",
            "Show the cash flow statement from the most recent 10-K",
            "What is free cash flow?",
            "What are NVDA's total assets?",
            "Show me the balance sheet for the latest quarter",
            "What is the company's total liabilities?",
            "Get financial statement data",
            "Show me the latest 10-K financials",
        ],
    ),
    ToolDefinition(
        name="get_company_news",
        description="Fetches latest news headlines and links for a company: recent events, announcements, press, and sentiment. Use for news and headlines only, not for financial ratios or price data.",
        parameters={"ticker": {"type": "string", "description": "Stock symbol"}},
        category="core",
        keywords=["news", "headlines", "events", "sentiment", "announcements", "press", "media", "articles", "recent news"],
        use_cases=[
            "What's the latest news on AAPL?",
            "Any recent announcements from TSLA?",
            "Why is NVDA stock moving today?",
            "Get recent news articles for a company",
            "Show me headlines about the stock",
        ],
    ),
    
    # Distraction Tools
    ToolDefinition(
        name="get_options_chain",
        description="Retrieves options chain data: calls and puts, strike prices, expiration dates for options trading. Use only for options contracts and derivatives, not for stock price or fundamentals.",
        parameters={
            "ticker": {"type": "string", "description": "Stock symbol"},
            "expiration_date": {"type": "string", "description": "Optional expiration date"},
        },
        category="distraction",
        keywords=["options", "calls", "puts", "strike price", "expiration", "implied volatility", "open interest", "options chain", "derivatives"],
        use_cases=[
            "Show options chain for SPY",
            "What are the available call options for AAPL?",
            "Get put options for TSLA",
        ],
    ),
    ToolDefinition(
        name="get_institutional_holders",
        description="Retrieves who holds the stock: institutional ownership and top institutional holders (funds, institutions). Use for ownership and holder names, not for valuation ratios or price.",
        parameters={"ticker": {"type": "string", "description": "Stock symbol"}},
        category="distraction",
        keywords=["institutional", "ownership", "holders", "funds", "institutions", "who holds", "institutional ownership"],
        use_cases=[
            "Who are the major institutional holders of AAPL?",
            "What percentage is held by institutions?",
        ],
    ),
    ToolDefinition(
        name="get_insider_transactions",
        description="Retrieves recent insider trading activity: buys and sells by executives and insiders. Use for insider transaction history only, not for analyst views or valuation.",
        parameters={"ticker": {"type": "string", "description": "Stock symbol"}},
        category="distraction",
        keywords=["insider", "trading", "executives", "buys", "sells", "transactions", "insider buying", "insider selling"],
        use_cases=[
            "Any recent insider trading in TSLA?",
            "Have executives been buying or selling AAPL?",
        ],
    ),
    ToolDefinition(
        name="get_analyst_recommendations",
        description="Retrieves analyst opinions: Wall Street ratings, buy/sell recommendations, upgrades, downgrades, and price targets. Use for analyst views only, not for raw P/E or fundamentals from the company.",
        parameters={"ticker": {"type": "string", "description": "Stock symbol"}},
        category="distraction",
        keywords=["analyst", "ratings", "recommendations", "price target", "upgrade", "downgrade", "analyst rating", "Wall Street"],
        use_cases=[
            "What do analysts think about NVDA?",
            "Any recent upgrades for AAPL?",
        ],
    ),
    ToolDefinition(
        name="get_earnings_calendar",
        description="Retrieves earnings report dates and calendar: when a company reports earnings (announcement date, schedule). Use only for when earnings are reported, not for P/E ratios, EPS values, or valuation metrics.",
        parameters={"ticker": {"type": "string", "description": "Stock symbol"}},
        category="distraction",
        keywords=["earnings date", "earnings report date", "when is earnings", "earnings schedule", "announcement date", "calendar", "quarterly report date"],
        use_cases=[
            "When is AAPL's next earnings report?",
            "When does MSFT report earnings?",
            "Earnings announcement schedule for a ticker",
        ],
    ),
    ToolDefinition(
        name="get_sustainability_scores",
        description="Retrieves ESG sustainability scores: environmental, social, and governance (ESG) metrics. Use for ESG and sustainability only, not for financial ratios or price.",
        parameters={"ticker": {"type": "string", "description": "Stock symbol"}},
        category="distraction",
        keywords=["ESG", "sustainability", "environmental", "social", "governance", "green", "ESG score"],
        use_cases=[
            "What's AAPL's ESG score?",
            "Is XOM sustainable?",
        ],
    ),
    ToolDefinition(
        name="get_dividend_history",
        description="Retrieves past dividend payment history over time: historical dividends paid and growth. Use only for history of past payments (how much paid in prior years). Not for next dividend date, ex-dividend date, or payment date (use fundamentals for upcoming dividend dates); not for current dividend yield (that is a fundamental metric).",
        parameters={
            "ticker": {"type": "string", "description": "Stock symbol"},
            "years": {"type": "integer", "description": "Years of history"},
        },
        category="distraction",
        keywords=["dividend history", "past dividend payments", "historical dividends", "dividend growth", "payment history", "how much paid in dividends"],
        use_cases=[
            "Show dividend history for KO",
            "How much has JNJ paid in dividends over the years?",
        ],
    ),
    ToolDefinition(
        name="calculate_technical_indicators",
        description="Calculates technical indicators from price data: RSI, MACD, Bollinger Bands, ATR, stochastic, and other momentum/volatility indicators. Use for derived indicators only, not for raw fundamentals or valuation ratios.",
        parameters={
            "ticker": {"type": "string", "description": "Stock symbol"},
            "indicators": {"type": "array", "description": "List of indicators"},
        },
        category="distraction",
        keywords=[
            "RSI", "MACD", "Bollinger Bands", "technical indicators", "momentum",
            "ATR", "average true range", "stochastic", "Williams %R", "CCI",
            "relative strength index", "signal line", "histogram"
        ],
        use_cases=[
            "Calculate RSI for AAPL",
            "What's the MACD signal for TSLA?",
            "Show Bollinger Bands for NVDA",
            "What is the ATR indicator?",
        ],
    ),
    ToolDefinition(
        name="compare_sector_performance",
        description="Compares performance of multiple stocks within a sector or industry over a period. Use for sector-level or multi-stock comparison only, not for a single stock's fundamentals or price.",
        parameters={
            "sector": {"type": "string", "description": "Sector name"},
            "period": {"type": "string", "description": "Time period"},
        },
        category="distraction",
        keywords=["sector", "comparison", "performance", "industry", "sector performance", "compare stocks"],
        use_cases=[
            "How is the tech sector performing?",
            "Compare healthcare stocks",
        ],
    ),
]


# =============================================================================
# SINGLETON TOOL STORE - Caches embedding model and tool embeddings
# =============================================================================
#
# Problem Solved:
#   During benchmark evaluation, the embedding model was being reloaded for
#   every query, causing:
#   - Repeated "Loading weights: 100%|..." messages in output
#   - Slow benchmark execution (~3s per query just for model loading)
#   - Unnecessary GPU memory churn
#
# Solution:
#   Singleton pattern caches the ToolStore (and thus the SentenceTransformer
#   model) after first initialization. Subsequent calls return the cached
#   instance, eliminating redundant model loads.
#
# =============================================================================

_cached_tool_store: Optional[ToolStore] = None
_cached_persist_directory: Optional[str] = None


def create_default_tool_store(persist_directory: Optional[str] = None, force_reload: bool = False) -> ToolStore:
    """
    Create or return cached tool store with default tool definitions.
    
    Uses singleton pattern to avoid reloading the embedding model and 
    re-computing tool embeddings on every query. This significantly speeds
    up benchmark runs (from ~3s to ~0.01s per query for tool selection).
    
    The first call loads the SentenceTransformer model and computes embeddings
    for all tools. Subsequent calls return the cached instance immediately.
    
    Args:
        persist_directory: Optional path to persist the database
        force_reload: If True, forces recreation of the tool store
        
    Returns:
        Initialized ToolStore with all tools (cached after first call)
        
    Example:
        # First call - loads model and computes embeddings (~3s)
        store = create_default_tool_store()
        
        # Second call - returns cached instance (~0.01s)
        store = create_default_tool_store()
    """
    global _cached_tool_store, _cached_persist_directory
    
    # Return cached store if available and directory matches
    if (
        _cached_tool_store is not None 
        and not force_reload 
        and _cached_persist_directory == persist_directory
    ):
        return _cached_tool_store
    
    # Create new store
    store = ToolStore(persist_directory=persist_directory)
    store.add_tools(DEFAULT_TOOL_DEFINITIONS)
    
    # Cache it
    _cached_tool_store = store
    _cached_persist_directory = persist_directory
    
    return store


def reset_tool_store_cache():
    """Reset the cached tool store (useful for testing)."""
    global _cached_tool_store, _cached_persist_directory
    _cached_tool_store = None
    _cached_persist_directory = None
