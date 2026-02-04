# Helix Financial Agent

A **Reflexive Financial AI Agent** with semantic routing, dynamic tool selection (ToolRAG), and MCP server deployment.

## Features

- **Reflexive Architecture**: Self-correcting agent with Generator → Reflector → Revisor loop
- **Semantic Routing**: Intelligent model selection via vLLM Semantic Router (MoM architecture)
- **ToolRAG**: Dynamic tool selection - only relevant tools are bound to the LLM
- **MCP Deployment**: All tools executed via FastMCP (Model Context Protocol)
- **LLM-as-a-Judge**: Response evaluation using Gemini 2.5 Pro (score >= 8.0 passes)

## Architecture

```
                    ┌─────────────────────────────────────────────────────────┐
                    │              vLLM SEMANTIC ROUTER (model="MoM")         │
                    │    Financial queries → Qwen3-30B-A3B (llama.cpp)       │
                    │    Evaluation/Judge → Gemini 2.5 Pro (Google API)      │
                    └─────────────────────────────────────────────────────────┘
                                              │
                                              ▼
User Query ──► ToolRAG ──► Generator ──► MCP Server ──► yfinance
               (select)    (via Router)    (tools)
                  │             │
                  │             ▼
                  │       Reflector (via Router → Gemini)
                  │             │
                  │    [Score >= 8?] ──┬── [Score < 8]
                  │         │               │
                  │      Output          Revisor ──► retry (max 3)
                  │
                  └─► Only selected tools bound to LLM
```

### Request Flow

| Step | Flow | Description |
|------|------|-------------|
| 1 | Query → ToolRAG | Select relevant tools via semantic search |
| 2 | Agent → Router → Qwen3 | Generate response using selected tools |
| 3 | Agent → Router → Gemini | Evaluate response quality (score 0-10) |
| 4 | Score >= 8 → Output | Pass threshold, return response |
| 4 | Score < 8 → Revisor | Revise and re-evaluate (max 3 iterations) |

## Technology Stack

| Component | Technology | Description |
|-----------|------------|-------------|
| Model Serving | llama.cpp + CUDA | GGUF quantized model serving |
| Agent Model | Qwen3-30B-A3B-Instruct | 30B MoE model (3B active params) |
| Judge Model | Gemini 2.5 Pro | LLM-as-a-Judge evaluation |
| Orchestration | LangGraph | Stateful graph-based workflow |
| Market Data | yfinance | Stock fundamentals, prices, news |
| Tool Serving | FastMCP | Model Context Protocol server |
| Semantic Router | vLLM-SR | Intelligent request routing |
| Tool Selection | ChromaDB | Vector-based tool retrieval |

---

## Quick Start

### Prerequisites

- Python 3.11+
- CUDA-capable GPU (DGX Spark / ZGX Nano recommended)
- ~20GB disk space for model
- HuggingFace token (model download)
- Google AI Studio API key (Gemini evaluation)

### Step 1: Installation

```bash
cd /home/vincent/Code/helix-financial-agent

# Make scripts executable
chmod +x scripts/*.sh

# Run setup script
./scripts/setup.sh

# Configure environment
cp .env.example .env
nano .env  # Add your API keys
```

**Required in `.env`:**
```bash
HF_TOKEN=hf_your_token_here          # https://huggingface.co/settings/tokens
GEMINI_API_KEY=your_gemini_key_here  # https://aistudio.google.com/app/apikey
```

### Step 2: Build llama.cpp (if needed)

```bash
git clone https://github.com/ggerganov/llama.cpp.git ~/llama.cpp
cd ~/llama.cpp
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="90"
cmake --build build -j$(nproc)
```

### Step 3: Start Services

Open **3 terminal windows**. All services are required.

**Terminal 1 - Model Server:**
```bash
./scripts/start_llama_server.sh
# Wait for: "llama server listening at http://0.0.0.0:8081"
```

**Terminal 2 - Semantic Router:**
```bash
./scripts/start_router.sh
# Provides MoM routing between Qwen3 and Gemini
```

**Terminal 3 - MCP Server:**
```bash
./scripts/start_mcp_server.sh
# Serves 13 financial tools via FastMCP
```

### Step 4: Run the Agent

```bash
source .venv/bin/activate

# Random benchmark query (recommended for demo)
helix-agent --random

# Single query
helix-agent --query "What is AAPL's PE ratio?"

# Interactive mode
helix-agent
```

---

## Service Ports

| Service | Port | Description |
|---------|------|-------------|
| llama.cpp | 8081 | Model inference (OpenAI-compatible) |
| MCP Server | 8000 | FastMCP tool server (streamable-http) |
| vLLM-SR HTTP | 8801 | Semantic routing entry point |
| vLLM-SR Classify | 8889 | Health checks, model listing |
| vLLM-SR Metrics | 9190 | Prometheus metrics |
| **vLLM-SR Hub UI** | **8080** | **Router dashboard (forward for remote access)** |
| **MLflow UI** | **5000** | **Experiment tracking (forward for remote access)** |

### Port Forwarding for Web UIs

If accessing web interfaces from another machine, forward the UI ports:

```bash
# Forward web UI ports only
ssh -L 8080:localhost:8080 \
    -L 5000:localhost:5000 \
    user@zgx-nano

# Or use the provided script
./scripts/ssh_port_forward.sh user@zgx-nano
```

**After port forwarding, access:**
- Semantic Router Hub UI: http://localhost:8080
- MLflow UI: http://localhost:5000 (start with `mlflow ui`)

---

## ToolRAG: Dynamic Tool Selection

ToolRAG selects only relevant tools for each query, keeping the LLM focused and efficient.

### How It Works

1. **Embed Query**: User query embedded via sentence-transformers
2. **Search Tools**: Compare against tool embeddings in ChromaDB
3. **Filter**: Select tools with similarity >= 0.35
4. **Bind**: Only selected tools bound to LLM (generator + revisor)
5. **Fallback**: Use core tools if none selected

### Configuration

| Parameter | Default | Environment Variable |
|-----------|---------|---------------------|
| Threshold | 0.35 | `TOOL_RAG_THRESHOLD` |
| Embedding Model | all-MiniLM-L6-v2 | `EMBEDDING_MODEL` |

### Why It Matters

| Aspect | All 13 Tools | Selected Tools |
|--------|--------------|----------------|
| LLM Context | Bloated | Focused |
| Tool Selection | May pick wrong tool | Precise |
| Latency | Slower | Faster |

---

## Semantic Routing (MoM Architecture)

The router automatically selects the best model based on request content using `model="MoM"` (Model of Models).

### Routing Decisions

| Decision | Priority | Triggers | Routes To |
|----------|----------|----------|-----------|
| `evaluation` | 15 | evaluate, judge, assess, score | Gemini 2.5 Pro |
| `data_generation` | 15 | generate, synthetic, dataset | Gemini 2.5 Pro |
| `financial_analysis` | 10 | stock, price, PE ratio, dividend | Qwen3 (llama.cpp) |
| `general` | 5 | (fallback) | Qwen3 (llama.cpp) |

### Intent Markers

Prompts include markers to help classification:

```python
"[FINANCIAL_ANALYSIS] What is AAPL's PE ratio?"  # → Qwen3
"[EVALUATE] Assess this response for accuracy..."  # → Gemini
```

### Code Integration

```python
# All LLM calls use MoM routing
llm = ChatOpenAI(
    base_url="http://localhost:8801/v1",  # Router endpoint
    model="MoM",                           # Auto-select model
)
```

| Node | Purpose | Routes To |
|------|---------|-----------|
| Generator | Draft response with tools | Qwen3 |
| Reflector | Evaluate quality (0-10) | Gemini |
| Revisor | Improve based on critique | Qwen3 |

### Benefits

- **No hardcoded models** - Router decides based on content
- **Centralized config** - Edit `config/router_config.yaml`
- **Observable** - Metrics at port 9190
- **Easy model swaps** - No code changes needed

---

## Financial Tools

### Core Tools

| Tool | Description |
|------|-------------|
| `get_stock_fundamentals` | PE ratio, market cap, dividends, beta |
| `get_historical_prices` | OHLCV, returns, moving averages |
| `get_financial_statements` | Balance sheet, income, cash flow |
| `get_company_news` | Recent headlines |

### Distraction Tools (ToolRAG Testing)

| Tool | Description |
|------|-------------|
| `get_options_chain` | Options data |
| `get_institutional_holders` | Institutional ownership |
| `get_insider_transactions` | Insider trading |
| `get_analyst_recommendations` | Analyst ratings |
| `get_earnings_calendar` | Earnings dates |
| `get_sustainability_scores` | ESG scores |
| `get_dividend_history` | Historical dividends |
| `calculate_technical_indicators` | Technical analysis |
| `compare_sector_performance` | Sector comparison |

---

## CLI Reference

### Commands

| Command | Description |
|---------|-------------|
| `helix-agent` | Interactive mode |
| `helix-agent --random` | Random benchmark query |
| `helix-agent -q "query"` | Single query |
| `helix-generate` | Generate synthetic dataset |
| `helix-eval` | Run evaluation benchmark |
| `helix-mcp` | Start MCP server |

### Agent Options

| Option | Description |
|--------|-------------|
| `--random, -r` | Random query from benchmark |
| `--query, -q` | Specific query |
| `--dataset, -d` | Custom dataset path |
| `--eval, -e` | Enable evaluation |
| `--no-tool-rag` | Use all tools |
| `--quiet` | Less verbose |

---

## Debugging

### VS Code / Cursor

1. Open Debug Panel: `Ctrl+Shift+D`
2. Select configuration:
   - `Helix Agent - Single Query`
   - `Helix Agent - Interactive`
   - `MCP Server - Debug`
3. Set breakpoints in:
   - `agent/nodes.py` - Generator, Reflector, Revisor
   - `agent/runner.py` - Main runner
   - `tool_rag/tool_selector.py` - Tool selection
4. Press `F5`

**Note:** All 3 services must be running before debugging.

### Python Interpreter

If debugging fails with path errors:
1. `Ctrl+Shift+P` → "Python: Select Interpreter"
2. Select `.venv/bin/python`
3. Restart VS Code

---

## Project Structure

```
helix-financial-agent/
├── src/helix_financial_agent/
│   ├── agent/           # LangGraph nodes, graph, runner
│   ├── tools/           # Financial tools + MCP server
│   ├── tool_rag/        # ChromaDB tool selection
│   ├── router/          # vLLM-SR client & config
│   ├── evaluation/      # LLM-as-a-Judge
│   └── data_generation/ # Synthetic data
├── scripts/
│   ├── start_llama_server.sh
│   ├── start_router.sh
│   ├── start_mcp_server.sh
│   └── ssh_port_forward.sh
├── config/
│   └── router_config.yaml
└── .vscode/
    └── launch.json
```

---

## Troubleshooting

### Services won't start

```bash
# Check llama.cpp
curl http://localhost:8081/health

# Check router
curl http://localhost:8889/health

# Check MCP
curl http://localhost:8000/mcp
```

### Model download fails

```bash
# Verify HF_TOKEN
grep HF_TOKEN .env

# Manual download
huggingface-cli download bartowski/Qwen_Qwen3-30B-A3B-Instruct-2507-GGUF \
    Qwen_Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf \
    --local-dir ~/llama.cpp/models
```

### ToolRAG selects no tools

Lower the threshold in `.env`:
```bash
TOOL_RAG_THRESHOLD=0.25
```

### Reflection always fails

Check if score parsing is working. Scores >= 8.0 pass.
The reflector uses Gemini which returns markdown scores like "Score: 8.5 / 10".

---

## License

Apache 2.0
