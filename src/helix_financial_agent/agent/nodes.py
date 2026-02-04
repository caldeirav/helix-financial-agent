"""
Agent Node Implementations

Defines the nodes for the reflexive agent graph:
- generator_node: Drafts responses using tools (routes to Qwen3)
- tool_executor_node: Executes tool calls via MCP server
- reflector_node: Critiques drafts (routes to Gemini)
- revisor_node: Refines responses (routes to Qwen3)

================================================================================
SEMANTIC ROUTING ARCHITECTURE
================================================================================

All LLM calls use `model="MoM"` (Model of Models), which tells the vLLM Semantic
Router to automatically select the best model based on the request content.

Request Flow:
    Node → ChatOpenAI(model="MoM") → Router → Signal Analysis → Backend Selection
                                                                  ├─ Qwen3 (llama.cpp)
                                                                  └─ Gemini API

How Routing Works:
    1. Node calls create_llm() which returns ChatOpenAI pointing to router endpoint
    2. LLM call includes model="MoM" (semantic routing mode)
    3. Router extracts signals from the prompt:
       - Keyword signals: Pattern matching for domain terms
       - Embedding signals: Semantic similarity to intent candidates
    4. Router matches signals against priority-ordered decisions
    5. Router forwards request to the selected backend model

================================================================================
ROUTING DECISIONS (from router_config.yaml)
================================================================================

| Decision            | Priority | Trigger Keywords                    | Model     |
|---------------------|----------|-------------------------------------|-----------|
| evaluation          | 15       | evaluate, judge, assess, score      | Gemini    |
| data_generation     | 15       | generate, synthetic, dataset        | Gemini    |
| financial_analysis  | 10       | stock, price, PE ratio, dividend    | Qwen3     |
| general             | 5        | (fallback)                          | Qwen3     |

Higher priority decisions are checked first. The first matching decision wins.

================================================================================
INTENT MARKERS
================================================================================

Each node includes intent markers in prompts to help the router classify:

    Generator Node:
        "[FINANCIAL_ANALYSIS] Analyze the stock..." → Routes to Qwen3
        
    Reflector Node:
        "[EVALUATE] Assess and judge this response..." → Routes to Gemini
        
    Revisor Node:
        "[FINANCIAL_ANALYSIS] REVISION TASK: Improve..." → Routes to Qwen3

These markers are combined with natural keywords for robust classification.
The router uses both keyword matching AND semantic embeddings, so markers
help but aren't strictly required.

================================================================================
CODE INTEGRATION
================================================================================

LLM Creation (create_llm function):
    - Points ChatOpenAI to router endpoint (http://localhost:8801/v1)
    - Sets model="MoM" for automatic model selection
    - Router handles authentication per backend (GEMINI_API_KEY, etc.)

Tool Binding:
    - Only ToolRAG-selected tools are bound to LLM in generator/revisor
    - This keeps prompts focused and reduces context size

Response Metadata:
    - Router adds routing_metadata to responses
    - Contains: selected_model, confidence, processing_time_ms
    - Displayed in verbose output for observability

================================================================================
SEE ALSO
================================================================================

    - router/config.py: Router configuration generation
    - router/client.py: Direct router client for health/metrics
    - config/router_config.yaml: Active routing rules
    - https://vllm-semantic-router.com/docs/tutorials/intelligent-route/

================================================================================
"""

import json
import re
from typing import Dict, Any, List, Optional, Callable

from langchain_core.messages import AIMessage, SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import ToolNode

from ..config import get_config
from ..tools import CORE_TOOLS, ALL_TOOLS
from .state import AgentState
from .prompts import (
    GENERATOR_SYSTEM_PROMPT,
    REFLECTOR_SYSTEM_PROMPT,
    REVISOR_SYSTEM_PROMPT,
)

config = get_config()


# =============================================================================
# SEMANTIC ROUTING CONFIGURATION
# =============================================================================
#
# MODEL SELECTION STRATEGY
# ------------------------
# We use "MoM" (Model of Models) for all LLM requests. This tells the vLLM
# Semantic Router to automatically select the best backend model based on
# the content of each request.
#
# WHY SEMANTIC ROUTING?
# ---------------------
# Instead of hardcoding model selection in Python:
#   ❌ llm = ChatOpenAI(model="gpt-4")  # Hardcoded
#   ✅ llm = ChatOpenAI(model="MoM")    # Router decides
#
# Benefits:
#   - Change models without code changes (edit router_config.yaml)
#   - Centralized routing logic (observable, testable)
#   - A/B testing between models
#   - Metrics and observability (port 9190)
#
# HOW IT WORKS
# ------------
# 1. All LLM calls go through ChatOpenAI pointed at router (port 8801)
# 2. Router extracts signals from prompt content:
#    - Keyword signals: "stock", "evaluate", "generate" → triggers decisions
#    - Embedding signals: semantic similarity to intent candidates
# 3. Router matches against priority-ordered decisions
# 4. Request forwarded to selected backend with routing_metadata
#
# ROUTING DECISIONS (see config/router_config.yaml)
# -------------------------------------------------
# Priority | Decision           | Triggers                    | Routes To
# ---------|--------------------|-----------------------------|----------
# 15       | evaluation         | evaluate, judge, assess     | Gemini
# 15       | data_generation    | generate, synthetic         | Gemini
# 10       | financial_analysis | stock, price, PE ratio      | Qwen3
# 5        | general            | (fallback)                  | Qwen3
#
# =============================================================================

# Semantic routing model - router auto-selects based on content
# When ChatOpenAI receives model="MoM", it sends this to the router
# The router then analyzes the prompt and selects the appropriate backend
SEMANTIC_ROUTER_MODEL = "MoM"

# Explicit model names for direct routing (bypasses semantic analysis)
# Use these for debugging or when you need guaranteed model selection
AGENT_MODEL = "qwen3-30b-a3b"      # Direct route to llama.cpp (Qwen3)
JUDGE_MODEL = "gemini-2.5-pro"     # Direct route to Gemini API
AUTO_MODEL = "MoM"                  # Alias for semantic routing


def create_llm(
    temperature: Optional[float] = None,
    use_semantic_routing: bool = True,
) -> ChatOpenAI:
    """
    Create an LLM instance connected to the vLLM Semantic Router.
    
    This is the primary interface between the agent and the LLM backends.
    All LLM calls flow through the semantic router, which automatically
    selects the best model based on the content of each request.
    
    Connection Setup:
        - base_url: Points to router's HTTP endpoint (default: localhost:8801/v1)
        - model: "MoM" enables automatic model selection
        - api_key: Set to "not-needed" (router handles auth per backend)
    
    Routing Flow:
        1. ChatOpenAI sends request to router with model="MoM"
        2. Router extracts signals from prompt (keywords + embeddings)
        3. Router matches against decisions (evaluation > financial > general)
        4. Router forwards to selected backend (Qwen3 or Gemini)
        5. Response includes routing_metadata (model, confidence, time)
    
    Backend Selection:
        The router selects based on prompt content:
        - Financial keywords ("stock", "PE ratio") → Qwen3 (llama.cpp, fast)
        - Evaluation keywords ("evaluate", "judge") → Gemini (high-quality)
        - Generation keywords ("generate", "synthetic") → Gemini (diverse)
        - Other requests → Qwen3 (fallback)
    
    Args:
        temperature: Sampling temperature (default from config.model.agent_temperature)
        use_semantic_routing: If True (default), uses model="MoM" for auto-routing.
                             If False, uses explicit agent model (bypasses router logic).
        
    Returns:
        ChatOpenAI instance configured to use the semantic router.
        All calls through this instance go through the router.
        
    Example:
        # Create LLM with semantic routing (recommended)
        llm = create_llm()
        
        # These prompts route to different models automatically:
        llm.invoke("What is AAPL's PE ratio?")      # → Qwen3 (financial)
        llm.invoke("Evaluate this response...")     # → Gemini (evaluation)
        
        # Bypass routing for debugging (direct to Qwen3)
        llm_direct = create_llm(use_semantic_routing=False)
        
    See Also:
        - config/router_config.yaml: Routing rules and signal configuration
        - router/config.py: Router configuration generator
    """
    model = SEMANTIC_ROUTER_MODEL if use_semantic_routing else AGENT_MODEL
    
    return ChatOpenAI(
        base_url=config.router.router_endpoint,
        api_key="not-needed",  # Router handles auth per backend
        model=model,
        temperature=temperature or config.model.agent_temperature,
        max_tokens=config.model.agent_max_tokens,
    )


def generator_node(
    state: AgentState,
    llm: Optional[ChatOpenAI] = None,
    tools: Optional[List[Callable]] = None,
) -> Dict[str, Any]:
    """
    Generator Node (initial_responder):
    - Takes user query and generates draft response using yfinance tools
    - This is the primary "thinking" node that gathers data and drafts answers
    
    Tool Binding:
        Only the tools passed to this function are bound to the LLM.
        ToolRAG pre-selects relevant tools for the query, and only those
        tools are passed here. This ensures the agent operates with a
        focused toolset rather than all 13+ available tools.
    
    Semantic Routing:
        Uses model="MoM" with financial analysis intent markers.
        The router detects keywords like "stock", "price", "PE ratio" and
        routes to Qwen3 (llama.cpp) for financial analysis tasks.
    
    Args:
        state: Current agent state
        llm: Optional LLM instance (creates default if not provided)
        tools: Tools selected by ToolRAG to bind to LLM (falls back to CORE_TOOLS)
        
    Returns:
        State update with new message
    """
    # Create LLM with semantic routing - router will detect financial analysis intent
    llm = llm or create_llm()
    tools = tools or CORE_TOOLS
    
    messages = state["messages"]
    iteration = state["iteration_count"]
    
    # Bind tools to LLM
    llm_with_tools = llm.bind_tools(tools)
    
    # Build system prompt with intent markers for semantic routing
    # The "[FINANCIAL_ANALYSIS]" marker helps the router classify this request
    system_message = SystemMessage(content=GENERATOR_SYSTEM_PROMPT)
    
    # If this is a revision iteration, include the critique context
    if iteration > 0 and state.get("critique"):
        revision_context = f"""
[FINANCIAL_ANALYSIS] REVISION REQUIRED (Iteration {iteration}):
Previous critique: {state['critique']}

Please revise your response to address all issues raised.
Use the financial tools to gather accurate data.
"""
        messages = messages + [SystemMessage(content=revision_context)]
    
    # Invoke LLM with tools
    # Router analyzes content → detects financial keywords → routes to Qwen3
    response = llm_with_tools.invoke([system_message] + messages)
    
    return {
        "messages": [response],
        "sender": "generator",
    }


def tool_executor_node(
    state: AgentState,
    tools: Optional[List[Callable]] = None,
) -> Dict[str, Any]:
    """
    Tool Executor Node:
    - Executes any tool calls made by the generator
    - Stores tool outputs for verification by reflector
    
    Args:
        state: Current agent state
        tools: Optional list of tools to use
        
    Returns:
        State update with tool results
    """
    tools = tools or CORE_TOOLS
    messages = state["messages"]
    last_message = messages[-1]
    
    tool_outputs = []
    
    # Check if there are tool calls to execute
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        tool_node = ToolNode(tools)
        result = tool_node.invoke({"messages": messages})
        
        # Extract tool outputs for verification
        for msg in result.get("messages", []):
            if hasattr(msg, "content"):
                tool_outputs.append({
                    "tool": msg.name if hasattr(msg, "name") else "unknown",
                    "output": msg.content
                })
        
        return {
            "messages": result.get("messages", []),
            "tool_outputs": state.get("tool_outputs", []) + tool_outputs,
            "sender": "tools",
        }
    
    return {"sender": "tools"}


def reflector_node(
    state: AgentState,
    llm: Optional[ChatOpenAI] = None,
) -> Dict[str, Any]:
    """
    Reflector Node (LLM-as-a-Judge):
    - Acts as "Senior Risk Analyst" critiquing the draft
    - Checks for hallucinations, advice violations, completeness
    - Scores response (0-10) and determines if revision needed
    
    This node implements the "reflection" step of the reflexive architecture.
    It evaluates the generator's draft against the tool outputs (ground truth)
    to catch errors before they reach the user.
    
    Semantic Routing to Gemini:
        This node's prompts are designed to trigger the "evaluation" decision
        in the router, which routes to Gemini 2.5 Pro for high-quality judging.
        
        Intent Markers Used:
            "[EVALUATE] TASK: Assess and judge the quality..."
            "evaluate this draft response for correctness, accuracy..."
            
        Why Gemini for Evaluation:
            - Gemini 2.5 Pro excels at nuanced quality assessment
            - Better at catching subtle errors than local models
            - Provides structured, detailed feedback
            
        Router Decision Matching:
            Keywords: "evaluate", "judge", "assess", "score", "accuracy"
            Decision: evaluation (priority 15)
            Routes to: gemini-2.5-pro
    
    Score Interpretation:
        - Score >= 8.0: Response passes, no revision needed
        - Score < 8.0: Response needs revision, goes to revisor_node
    
    Args:
        state: Current agent state containing draft response and tool outputs
        llm: Optional LLM instance (creates default with semantic routing)
        
    Returns:
        State update with critique text, critique_passed boolean, and
        incremented iteration_count
    """
    # Create LLM with semantic routing - router will detect evaluation intent
    # and route to Gemini 2.5 Pro for quality assessment
    llm = llm or create_llm(temperature=0.0)
    
    messages = state["messages"]
    tool_outputs = state.get("tool_outputs", [])
    original_query = state.get("original_query", "")
    
    # Find the last AI response (the draft to evaluate)
    draft_response = None
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and msg.content:
            draft_response = msg.content
            break
    
    if not draft_response:
        return {
            "critique": "No draft response found to evaluate.",
            "critique_passed": False,
            "sender": "reflector",
        }
    
    # Build reflection prompt with EVALUATION intent markers
    # These keywords trigger the router to route to Gemini 2.5 Pro:
    # "evaluate", "assess", "critique", "judge", "correctness", "accuracy"
    reflection_prompt = f"""[EVALUATE] TASK: Assess and judge the quality of this financial response.

ORIGINAL USER QUERY:
{original_query}

TOOL OUTPUTS (Ground Truth to validate against):
{json.dumps(tool_outputs, indent=2)}

DRAFT RESPONSE TO EVALUATE AND CRITIQUE:
{draft_response}

Please evaluate this draft response for correctness, accuracy, and quality.
Score the response and provide detailed feedback on any issues found.
"""
    
    # Invoke with evaluation-focused system prompt
    # Router analyzes content → detects evaluation keywords → routes to Gemini
    response = llm.invoke([
        SystemMessage(content=REFLECTOR_SYSTEM_PROMPT),
        HumanMessage(content=reflection_prompt)
    ])
    
    # Parse the critique
    critique_text = response.content
    critique_passed = False
    
    # Passing threshold - responses scoring >= this value are considered acceptable
    PASSING_SCORE_THRESHOLD = 8.0
    
    # Try multiple methods to determine if critique passed
    
    # Method 1: Try to parse JSON from response
    try:
        json_match = re.search(r'\{[\s\S]*\}', critique_text)
        if json_match:
            critique_json = json.loads(json_match.group())
            if "passed" in critique_json:
                critique_passed = critique_json.get("passed", False)
            elif "score" in critique_json:
                # JSON with numeric score
                critique_passed = float(critique_json.get("score", 0)) >= PASSING_SCORE_THRESHOLD
    except (json.JSONDecodeError, AttributeError, ValueError):
        pass
    
    # Method 2: If JSON parsing didn't work, look for explicit pass/fail indicators
    if not critique_passed:
        if '"passed": true' in critique_text.lower() or '"passed":true' in critique_text.lower():
            critique_passed = True
    
    # Method 3: Extract numeric score from markdown format (e.g., "Score: 8.5 / 10" or "8.5/10")
    if not critique_passed:
        # Match patterns like "Score: 8.5 / 10", "8.5/10", "Score: 9", "**8.5 / 10**"
        score_patterns = [
            r'[Ss]core[:\s]+\*?\*?(\d+(?:\.\d+)?)\s*/\s*10',  # "Score: 8.5 / 10"
            r'[Ss]core[:\s]+\*?\*?(\d+(?:\.\d+)?)',           # "Score: 8.5"
            r'(\d+(?:\.\d+)?)\s*/\s*10',                       # "8.5 / 10"
            r'[Ee]valuation[:\s]+\*?\*?(\d+(?:\.\d+)?)',      # "Evaluation: 8.5"
        ]
        for pattern in score_patterns:
            match = re.search(pattern, critique_text)
            if match:
                try:
                    score = float(match.group(1))
                    critique_passed = score >= PASSING_SCORE_THRESHOLD
                    break
                except ValueError:
                    continue
    
    return {
        "messages": [AIMessage(content=f"[REFLECTION]: {critique_text}")],
        "critique": critique_text,
        "critique_passed": critique_passed,
        "iteration_count": state["iteration_count"] + 1,
        "sender": "reflector",
    }


def revisor_node(
    state: AgentState,
    llm: Optional[ChatOpenAI] = None,
    tools: Optional[List[Callable]] = None,
) -> Dict[str, Any]:
    """
    Revisor Node:
    - Takes the critique and generates an improved response
    - Addresses all issues raised by the reflector
    
    Tool Binding:
        Only the tools passed to this function are bound to the LLM.
        These are the same tools selected by ToolRAG for the original query,
        ensuring consistency between initial generation and revision.
    
    Semantic Routing:
        Uses model="MoM" with financial analysis intent markers.
        The router detects this is a revision task for financial content
        and routes to Qwen3 (llama.cpp) for response improvement.
    
    Args:
        state: Current agent state
        llm: Optional LLM instance
        tools: Tools selected by ToolRAG to bind to LLM (falls back to CORE_TOOLS)
        
    Returns:
        State update with revised response
    """
    # Create LLM with semantic routing - router will detect financial revision intent
    llm = llm or create_llm()
    tools = tools or CORE_TOOLS
    
    messages = state["messages"]
    critique = state.get("critique", "")
    original_query = state.get("original_query", "")
    
    # Bind tools to LLM
    llm_with_tools = llm.bind_tools(tools)
    
    # Find the original draft
    draft_response = None
    for msg in messages:
        if isinstance(msg, AIMessage) and msg.content and not msg.content.startswith("[REFLECTION]"):
            draft_response = msg.content
    
    # Build revision prompt with FINANCIAL_ANALYSIS intent markers
    # Keywords like "stock", "financial", "portfolio" trigger routing to Qwen3
    revision_prompt = f"""[FINANCIAL_ANALYSIS] REVISION TASK: Improve this stock market analysis.

ORIGINAL FINANCIAL QUERY: {original_query}

PREVIOUS DRAFT RESPONSE:
{draft_response}

CRITIQUE FROM RISK ANALYST (issues to address):
{critique}

Please generate an improved financial analysis response that:
1. Addresses all issues raised in the critique
2. Uses accurate data from the financial tools
3. Provides proper stock market context
4. Includes appropriate investment disclaimers
"""
    
    # Router analyzes content → detects financial keywords → routes to Qwen3
    response = llm_with_tools.invoke([
        SystemMessage(content=REVISOR_SYSTEM_PROMPT),
        HumanMessage(content=revision_prompt)
    ])
    
    return {
        "messages": [response],
        "sender": "revisor",
    }


# =============================================================================
# NODE FACTORY FUNCTIONS
# =============================================================================

def create_generator_node(
    llm: Optional[ChatOpenAI] = None,
    tools: Optional[List[Callable]] = None,
) -> Callable[[AgentState], Dict[str, Any]]:
    """Create a generator node with specific LLM and tools."""
    def node(state: AgentState) -> Dict[str, Any]:
        return generator_node(state, llm=llm, tools=tools)
    return node


def create_tool_executor_node(
    tools: Optional[List[Callable]] = None,
) -> Callable[[AgentState], Dict[str, Any]]:
    """Create a tool executor node with specific tools."""
    def node(state: AgentState) -> Dict[str, Any]:
        return tool_executor_node(state, tools=tools)
    return node


def create_reflector_node(
    llm: Optional[ChatOpenAI] = None,
) -> Callable[[AgentState], Dict[str, Any]]:
    """Create a reflector node with specific LLM."""
    def node(state: AgentState) -> Dict[str, Any]:
        return reflector_node(state, llm=llm)
    return node


def create_revisor_node(
    llm: Optional[ChatOpenAI] = None,
    tools: Optional[List[Callable]] = None,
) -> Callable[[AgentState], Dict[str, Any]]:
    """Create a revisor node with specific LLM and tools."""
    def node(state: AgentState) -> Dict[str, Any]:
        return revisor_node(state, llm=llm, tools=tools)
    return node
