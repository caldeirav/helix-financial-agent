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
    """Ensure HF_TOKEN is set from config before loading models."""
    # Load from .env if not already set
    if not os.environ.get("HF_TOKEN"):
        from dotenv import load_dotenv
        load_dotenv()
    
    # Also set HUGGING_FACE_HUB_TOKEN for sentence-transformers
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token
        os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "0"


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
        description="Retrieves fundamental data for a given stock ticker including PE ratios, PEG ratio, Price-to-Sales (P/S), Price-to-Book (P/B), market cap, dividend yield, EPS, sector, business summary, and company info.",
        parameters={"ticker": {"type": "string", "description": "Stock symbol like AAPL, NVDA, MSFT"}},
        category="core",
        keywords=[
            "PE ratio", "P/E", "PEG ratio", "Price-to-Sales", "P/S ratio", "Price-to-Book", "P/B ratio",
            "market cap", "fundamentals", "valuation", "dividend yield", "sector", "beta", 
            "52-week high", "52-week low", "current price", "stock price", "EPS", "earnings per share",
            "forward PE", "trailing PE", "enterprise value", "book value"
        ],
        use_cases=[
            # Basic fundamentals
            "What is AAPL's PE ratio?",
            "Get the market cap of NVDA",
            "What sector is MSFT in?",
            "Tell me about GOOGL's dividend yield",
            "What is the current price of TSLA?",
            # Valuation ratios
            "What is the PEG ratio for WMT?",
            "Show me the Price-to-Sales ratio for AMZN",
            "Provide a side-by-side of the PEG ratio and Price-to-Sales ratio",
            "What is the P/B ratio?",
            "List the PE ratio and dividend yield",
            # Comparative queries
            "Compare valuation metrics for a stock",
            "Get key financial ratios for analysis",
            "What are the fundamental metrics?",
        ],
    ),
    ToolDefinition(
        name="get_historical_prices",
        description="Fetches historical price data including Open, High, Low, Close, Volume for technical analysis, moving averages, volatility calculations, and performance tracking.",
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
            # Returns and performance
            "What's AAPL's YTD return?",
            "Show price history for NVDA",
            "What was TSLA's highest price this month?",
            # Moving averages - many variations
            "Calculate the 20-day moving average for MSFT",
            "How far apart are the 50-day and 200-day moving averages?",
            "What is the 200-day moving average for AAPL?",
            "Show the 50-day and 200-day moving average difference",
            "Calculate moving average crossover",
            # Volatility - many variations
            "Calculate the 90-day volatility for pharmaceutical stocks",
            "What is the volatility of this stock?",
            "Calculate and list the volatility for BMY TMO AMGN",
            "Show historical volatility calculation",
            "What is the standard deviation of returns?",
            # General historical data
            "Show me the historical closing prices",
            "Get daily price data for the last month",
            "What were the prices over the past year?",
        ],
    ),
    ToolDefinition(
        name="get_financial_statements",
        description="Retrieves balance sheet, income statement, and cash flow statement data for financial analysis including total debt, assets, liabilities, revenue, and cash flows.",
        parameters={"ticker": {"type": "string", "description": "Stock symbol"}},
        category="core",
        keywords=[
            "revenue", "income statement", "balance sheet", "cash flow statement", "total debt",
            "total assets", "total liabilities", "EBITDA", "net income", "operating income",
            "10-K", "10-Q", "quarterly report", "annual report", "financial report",
            "free cash flow", "operating cash flow", "long-term debt", "short-term debt",
            "shareholders equity", "retained earnings", "gross profit"
        ],
        use_cases=[
            # Revenue and income
            "What is AAPL's total revenue?",
            "Get the income statement data",
            "Show quarterly revenue figures",
            # Debt queries - many variations
            "How much debt does MSFT have?",
            "Find the total debt for Morgan Stanley",
            "Can you find the total debt as of their latest financial report?",
            "What is the company's long-term debt?",
            # Cash flow
            "Get GOOGL's cash flow statement",
            "I need to see the statement of cash flows from their 10-K filing",
            "Show the cash flow statement from the most recent 10-K",
            "What is free cash flow?",
            # Balance sheet
            "What are NVDA's total assets?",
            "Show me the balance sheet for the latest quarter",
            "What is the company's total liabilities?",
            # General financial statements
            "Get financial statement data",
            "Show me the latest 10-K financials",
        ],
    ),
    ToolDefinition(
        name="get_company_news",
        description="Fetches latest news headlines and links for a company to understand recent events and sentiment.",
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
        description="Retrieves options chain data including calls and puts for options trading analysis.",
        parameters={
            "ticker": {"type": "string", "description": "Stock symbol"},
            "expiration_date": {"type": "string", "description": "Optional expiration date"},
        },
        category="distraction",
        keywords=["options", "calls", "puts", "strike price", "expiration", "implied volatility", "open interest"],
        use_cases=[
            "Show options chain for SPY",
            "What are the available call options for AAPL?",
            "Get put options for TSLA",
        ],
    ),
    ToolDefinition(
        name="get_institutional_holders",
        description="Retrieves institutional ownership data and top institutional holders.",
        parameters={"ticker": {"type": "string", "description": "Stock symbol"}},
        category="distraction",
        keywords=["institutional", "ownership", "holders", "funds", "institutions"],
        use_cases=[
            "Who are the major institutional holders of AAPL?",
            "What percentage is held by institutions?",
        ],
    ),
    ToolDefinition(
        name="get_insider_transactions",
        description="Retrieves recent insider trading activity including buys and sells by executives.",
        parameters={"ticker": {"type": "string", "description": "Stock symbol"}},
        category="distraction",
        keywords=["insider", "trading", "executives", "buys", "sells", "transactions"],
        use_cases=[
            "Any recent insider trading in TSLA?",
            "Have executives been buying or selling AAPL?",
        ],
    ),
    ToolDefinition(
        name="get_analyst_recommendations",
        description="Retrieves analyst ratings, recommendations, and price targets from Wall Street.",
        parameters={"ticker": {"type": "string", "description": "Stock symbol"}},
        category="distraction",
        keywords=["analyst", "ratings", "recommendations", "price target", "upgrade", "downgrade"],
        use_cases=[
            "What do analysts think about NVDA?",
            "Any recent upgrades for AAPL?",
        ],
    ),
    ToolDefinition(
        name="get_earnings_calendar",
        description="Retrieves earnings calendar and historical earnings data.",
        parameters={"ticker": {"type": "string", "description": "Stock symbol"}},
        category="distraction",
        keywords=["earnings", "EPS", "calendar", "quarterly", "report"],
        use_cases=[
            "When is AAPL's next earnings?",
            "What was MSFT's last quarter EPS?",
        ],
    ),
    ToolDefinition(
        name="get_sustainability_scores",
        description="Retrieves ESG sustainability scores for environmental, social, and governance analysis.",
        parameters={"ticker": {"type": "string", "description": "Stock symbol"}},
        category="distraction",
        keywords=["ESG", "sustainability", "environmental", "social", "governance", "green"],
        use_cases=[
            "What's AAPL's ESG score?",
            "Is XOM sustainable?",
        ],
    ),
    ToolDefinition(
        name="get_dividend_history",
        description="Retrieves dividend payment history and dividend growth analysis.",
        parameters={
            "ticker": {"type": "string", "description": "Stock symbol"},
            "years": {"type": "integer", "description": "Years of history"},
        },
        category="distraction",
        keywords=["dividend", "history", "payments", "yield", "growth"],
        use_cases=[
            "Show dividend history for KO",
            "How much has JNJ paid in dividends?",
        ],
    ),
    ToolDefinition(
        name="calculate_technical_indicators",
        description="Calculates advanced technical indicators like RSI, MACD, Bollinger Bands, ATR, and other momentum/volatility indicators.",
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
        description="Compares performance of stocks within a specific sector.",
        parameters={
            "sector": {"type": "string", "description": "Sector name"},
            "period": {"type": "string", "description": "Time period"},
        },
        category="distraction",
        keywords=["sector", "comparison", "performance", "industry"],
        use_cases=[
            "How is the tech sector performing?",
            "Compare healthcare stocks",
        ],
    ),
]


# =============================================================================
# SINGLETON TOOL STORE - Caches embedding model and tool embeddings
# =============================================================================

_cached_tool_store: Optional[ToolStore] = None
_cached_persist_directory: Optional[str] = None


def create_default_tool_store(persist_directory: Optional[str] = None, force_reload: bool = False) -> ToolStore:
    """
    Create or return cached tool store with default tool definitions.
    
    Uses singleton pattern to avoid reloading the embedding model and 
    re-computing tool embeddings on every query. This significantly speeds
    up benchmark runs.
    
    Args:
        persist_directory: Optional path to persist the database
        force_reload: If True, forces recreation of the tool store
        
    Returns:
        Initialized ToolStore with all tools (cached after first call)
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
