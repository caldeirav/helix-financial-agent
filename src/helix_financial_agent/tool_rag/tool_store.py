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
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from pathlib import Path

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

from ..config import get_config


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
        
        # Initialize embedding model
        self.embedding_model = SentenceTransformer(config.tool_rag.embedding_model)
        
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
        description="Retrieves fundamental data for a given stock ticker including PE ratios, market cap, dividend yield, sector, business summary, and company info.",
        parameters={"ticker": {"type": "string", "description": "Stock symbol like AAPL, NVDA, MSFT"}},
        category="core",
        keywords=["PE ratio", "market cap", "fundamentals", "valuation", "dividend", "sector", "beta", "52-week", "price"],
        use_cases=[
            "What is AAPL's PE ratio?",
            "Get the market cap of NVDA",
            "What sector is MSFT in?",
            "Tell me about GOOGL's dividend yield",
            "What is the current price of TSLA?",
        ],
    ),
    ToolDefinition(
        name="get_historical_prices",
        description="Fetches historical price data including Open, High, Low, Close, Volume for technical analysis and performance tracking.",
        parameters={
            "ticker": {"type": "string", "description": "Stock symbol"},
            "period": {"type": "string", "description": "Time period: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, ytd, max"},
        },
        category="core",
        keywords=["price history", "historical", "OHLCV", "returns", "moving average", "SMA", "volatility", "performance"],
        use_cases=[
            "What's AAPL's YTD return?",
            "Show price history for NVDA",
            "Calculate the 20-day moving average for MSFT",
            "What was TSLA's highest price this month?",
        ],
    ),
    ToolDefinition(
        name="get_financial_statements",
        description="Retrieves balance sheet, income statement, and cash flow statement data for financial analysis.",
        parameters={"ticker": {"type": "string", "description": "Stock symbol"}},
        category="core",
        keywords=["revenue", "income", "balance sheet", "cash flow", "debt", "assets", "liabilities", "EBITDA", "net income"],
        use_cases=[
            "What is AAPL's total revenue?",
            "How much debt does MSFT have?",
            "Get GOOGL's cash flow statement",
            "What are NVDA's total assets?",
        ],
    ),
    ToolDefinition(
        name="get_company_news",
        description="Fetches latest news headlines and links for a company to understand recent events and sentiment.",
        parameters={"ticker": {"type": "string", "description": "Stock symbol"}},
        category="core",
        keywords=["news", "headlines", "events", "sentiment", "announcements", "press", "media"],
        use_cases=[
            "What's the latest news on AAPL?",
            "Any recent announcements from TSLA?",
            "Why is NVDA stock moving today?",
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
        description="Calculates advanced technical indicators like RSI, MACD, Bollinger Bands.",
        parameters={
            "ticker": {"type": "string", "description": "Stock symbol"},
            "indicators": {"type": "array", "description": "List of indicators"},
        },
        category="distraction",
        keywords=["RSI", "MACD", "Bollinger", "technical", "indicators", "momentum"],
        use_cases=[
            "Calculate RSI for AAPL",
            "What's the MACD signal for TSLA?",
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


def create_default_tool_store(persist_directory: Optional[str] = None) -> ToolStore:
    """
    Create a tool store with default tool definitions.
    
    Args:
        persist_directory: Optional path to persist the database
        
    Returns:
        Initialized ToolStore with all tools
    """
    store = ToolStore(persist_directory=persist_directory)
    store.add_tools(DEFAULT_TOOL_DEFINITIONS)
    return store
