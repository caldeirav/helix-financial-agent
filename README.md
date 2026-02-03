# Helix Financial Agent

A **Reflexive Financial AI Agent** with semantic routing, dynamic tool selection (ToolRAG), and MCP server deployment.

## Features

- **Reflexive Architecture**: Self-correcting agent with Generator, Reflector, Revisor loop
- **Semantic Routing**: Route requests via vLLM Semantic Router (vllm-sr)
- **ToolRAG**: Dynamic tool selection based on query semantics
- **MCP Deployment**: Tools deployed via FastMCP
- **LLM-as-a-Judge**: Evaluation using Gemini 2.5 Pro

## Quick Start

### 1. Installation

```bash
chmod +x scripts/setup.sh
./scripts/setup.sh

# Or manually:
uv venv --python 3.11
source .venv/bin/activate
uv pip install -e .
```

### 2. Start llama.cpp Server

```bash
./scripts/start_llama_server.sh
```

### 3. Run the Agent

```bash
source .venv/bin/activate
helix-agent --query "What is AAPL's PE ratio?"
```

## Architecture

```
User Query -> ToolRAG (Select Tools) -> Generator (Qwen3) -> Tools (yfinance)
                                              |
                                              v
                                        Reflector (Critique)
                                              |
                                    [Pass?] --+-- [Fail]
                                       |            |
                                    Output       Revisor -> Generator
```

## Technology Stack

| Component | Technology |
|-----------|------------|
| Hardware | DGX Spark (GB10) |
| Model Serving | llama.cpp + CUDA |
| Agent Model | Qwen3-30B-A3B |
| Orchestration | LangGraph |
| Market Data | yfinance |
| Judge Model | Gemini 2.5 Pro |
| Tool Serving | FastMCP |
| Semantic Router | vLLM-SR |
| Tool Selection | ToolRAG (ChromaDB) |

## Financial Tools

**Core Tools:**
- `get_stock_fundamentals` - PE, market cap, dividends
- `get_historical_prices` - Price history, moving averages
- `get_financial_statements` - Balance sheet, income
- `get_company_news` - Headlines, sentiment

**Distraction Tools (for ToolRAG testing):**
- Options, institutional holders, insider transactions
- Analyst recommendations, earnings, sustainability
- Dividend history, technical indicators

## Model Setup

### Build llama.cpp

```bash
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp && mkdir build && cd build
cmake .. -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="90"
make -j$(nproc)
```

### Download Model

```bash
hf download bartowski/Qwen_Qwen3-30B-A3B-Instruct-2507-GGUF \
    Qwen_Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf \
    --local-dir ./models
```

## Configuration (.env)

```bash
LLAMA_CPP_BASE_URL=http://localhost:8080/v1
GEMINI_API_KEY=your-api-key
AGENT_TEMPERATURE=0.7
MAX_ITERATIONS=3
TOOL_RAG_TOP_K=5
```

## Evaluation

```bash
# Generate dataset
helix-generate

# Run benchmark
helix-eval --dataset ./data/financial_benchmark_v1_eval.jsonl
```

## License

MIT
