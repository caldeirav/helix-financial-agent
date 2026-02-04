"""
Agent Node Implementations

Defines the nodes for the reflexive agent graph:
- generator_node: Drafts responses using tools
- tool_executor_node: Executes tool calls  
- reflector_node: Critiques drafts
- revisor_node: Refines responses

================================================================================
SEMANTIC ROUTING ARCHITECTURE
================================================================================

All LLM calls use `model="MoM"` (Model of Models), which tells the vLLM Semantic
Router to automatically select the best model based on the request content.

How it works:
    1. Application sends request with model="MoM"
    2. Router analyzes the prompt content using:
       - Keyword signals: Detects domain-specific terms
       - Embedding signals: Semantic similarity to intent candidates
    3. Router matches against configured "decisions" (routing rules)
    4. Router forwards to the appropriate backend:
       - Financial analysis queries → Qwen3 (llama.cpp)
       - Evaluation/judging tasks → Gemini 2.5 Pro

Intent Markers:
    Prompts include subtle intent markers that help the router classify requests:
    - "[FINANCIAL_ANALYSIS]" → Routes to Qwen3 for stock/market queries
    - "[EVALUATE]" → Routes to Gemini for quality assessment
    - "[REVISE]" → Routes to Qwen3 for response improvement

Benefits:
    - No hardcoded model selection in application code
    - Routing logic centralized in router_config.yaml
    - Easy to add new models or change routing without code changes
    - Observable via router metrics (port 9190)

See: config/router_config.yaml for routing rules
See: https://vllm-semantic-router.com/docs/tutorials/intelligent-route/

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
# We use "MoM" (Model of Models) for all requests, letting the semantic router
# decide which model is most appropriate based on the content.
#
# The router detects intent from:
#   1. Keywords in the prompt (e.g., "stock", "evaluate", "generate")
#   2. Semantic embeddings compared against intent candidates
#   3. Priority-based decision matching
#
# Routing decisions (see router_config.yaml):
#   - financial_analysis (priority 10) → qwen3-30b-a3b (llama.cpp)
#   - evaluation (priority 15) → gemini-2.5-pro (Gemini API)
#   - data_generation (priority 15) → gemini-2.5-pro (Gemini API)
#   - general (priority 5) → qwen3-30b-a3b (fallback)
#
# =============================================================================

# Semantic routing model - router auto-selects based on content
SEMANTIC_ROUTER_MODEL = "MoM"

# Legacy explicit model names (kept for reference/debugging)
AGENT_MODEL = "qwen3-30b-a3b"      # Direct route to llama.cpp
JUDGE_MODEL = "gemini-2.5-pro"     # Direct route to Gemini
AUTO_MODEL = "MoM"                  # Alias for semantic routing


def create_llm(
    temperature: Optional[float] = None,
    use_semantic_routing: bool = True,
) -> ChatOpenAI:
    """
    Create an LLM instance using the vLLM Semantic Router.
    
    By default, uses "MoM" (Model of Models) which enables automatic model
    selection based on the content of each request. The router analyzes
    keywords and semantic embeddings to route to the most appropriate model.
    
    Routing Flow:
        Application → Router (model="MoM") → Analysis → Decision → Backend
                                                                    ├─ Qwen3 (financial)
                                                                    └─ Gemini (evaluation)
    
    Args:
        temperature: Temperature for response generation (default from config)
        use_semantic_routing: If True, uses MoM for auto-routing (recommended).
                             If False, uses default agent model directly.
        
    Returns:
        Configured ChatOpenAI instance pointing to the semantic router
        
    Example:
        # Semantic routing (recommended) - router decides model
        llm = create_llm()
        
        # The router will analyze the prompt and route accordingly:
        # - "What is AAPL's PE ratio?" → Qwen3 (financial_analysis)
        # - "Evaluate this response..." → Gemini (evaluation)
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
    
    Semantic Routing:
        Uses model="MoM" with financial analysis intent markers.
        The router detects keywords like "stock", "price", "PE ratio" and
        routes to Qwen3 (llama.cpp) for financial analysis tasks.
    
    Args:
        state: Current agent state
        llm: Optional LLM instance (creates default if not provided)
        tools: Optional list of tools to use
        
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
    Reflector Node:
    - Acts as "Senior Risk Analyst" critiquing the draft
    - Checks for hallucinations, advice violations, completeness
    - Returns structured critique with pass/fail decision
    
    Semantic Routing:
        Uses model="MoM" with evaluation intent markers.
        The router detects keywords like "evaluate", "assess", "critique", "judge"
        and semantic similarity to evaluation intent, then routes to Gemini 2.5 Pro
        for high-quality response assessment.
    
    Args:
        state: Current agent state
        llm: Optional LLM instance
        
    Returns:
        State update with critique
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
    
    Semantic Routing:
        Uses model="MoM" with financial analysis intent markers.
        The router detects this is a revision task for financial content
        and routes to Qwen3 (llama.cpp) for response improvement.
    
    Args:
        state: Current agent state
        llm: Optional LLM instance
        tools: Optional list of tools to use
        
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
