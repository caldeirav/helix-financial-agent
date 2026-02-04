# Helix Financial Agent

A **Reflexive Financial AI Agent** with semantic routing, dynamic tool selection (ToolRAG), and MCP server deployment.

## Features

- **Reflexive Architecture**: Self-correcting agent with Generator → Reflector → Revisor loop
- **Semantic Routing** (MANDATORY): Intelligent request routing via vLLM Semantic Router (vllm-sr)
- **ToolRAG**: Dynamic tool selection based on query semantics using ChromaDB
- **MCP Deployment** (MANDATORY): All tools executed via FastMCP (Model Context Protocol)
- **LLM-as-a-Judge**: Response evaluation using Gemini 2.5 Pro (via router)
- **Synthetic Benchmark**: Dataset generation for comprehensive testing

## Architecture

**All LLM calls go through the Semantic Router. All tool calls go through MCP Server.**

```
                    ┌─────────────────────────────────────────────────────────┐
                    │              vLLM SEMANTIC ROUTER (MANDATORY)           │
                    │    Agent Queries (model=qwen3) → Qwen3 (llama.cpp)     │
                    │    Judge/Eval (model=gemini-2.5-pro) → Gemini API      │
                    └─────────────────────────────────────────────────────────┘
                                              │
                                              ▼
User Query ──► ToolRAG (Select Tools) ──► Generator (via Router) ──► MCP Server ──► yfinance
                                              │
                                              ▼
                                        Reflector (via Router → Gemini)
                                              │
                                    [Pass?] ──┼── [Fail]
                                       │            │
                                    Output       Revisor (via Router) ──► retry
```

### Request Flow

1. **Agent Processing**: `Agent → Router (model=qwen3-30b-a3b) → llama.cpp`
2. **Evaluation/Judge**: `Agent → Router (model=gemini-2.5-pro) → Gemini API`
3. **Tool Execution**: `Agent → MCP Client → MCP Server → yfinance`

## Technology Stack

| Component | Technology | Description |
|-----------|------------|-------------|
| Hardware | DGX Spark / ZGX Nano | ARM64 + NVIDIA GB10 (128GB unified memory) |
| Model Serving | llama.cpp + CUDA | GGUF quantized model serving |
| Agent Model | Qwen3-30B-A3B-Instruct | 30B parameter MoE model (3B active) |
| Judge Model | Gemini 2.5 Pro | LLM-as-a-Judge for evaluation |
| Orchestration | LangGraph | Stateful graph-based agent workflow |
| Market Data | yfinance | Real-time stock fundamentals, prices, news |
| Tool Serving | FastMCP | Model Context Protocol server |
| Semantic Router | vLLM-SR | Intelligent request routing |
| Tool Selection | ToolRAG (ChromaDB) | Vector-based tool retrieval |

---

## Quick Start

### Prerequisites

- Python 3.11+
- CUDA-capable GPU (DGX Spark / ZGX Nano recommended)
- ~20GB disk space for model
- HuggingFace account (for model download)
- Google AI Studio API key (for Gemini evaluation)

---

## Step 1: Installation

### 1.1 Clone and Setup

```bash
cd /home/vincent/Code/helix-financial-agent

# Make scripts executable
chmod +x scripts/*.sh

# Run setup script
./scripts/setup.sh
```

The setup script will:
- Create Python virtual environment with uv
- Install all dependencies (including vllm-sr beta)
- Verify installation
- Generate router configuration

### 1.2 Configure Environment

```bash
# Copy template to .env
cp .env.example .env

# Edit .env with your API keys
nano .env
```

**Required environment variables:**

```bash
# HuggingFace (for model download)
HF_TOKEN=hf_your_token_here          # Get from: https://huggingface.co/settings/tokens

# Gemini (for evaluation/judging)
GEMINI_API_KEY=your_gemini_key_here  # Get from: https://aistudio.google.com/app/apikey
```

### 1.3 Build llama.cpp (if not already installed)

```bash
# Clone llama.cpp
git clone https://github.com/ggerganov/llama.cpp.git ~/llama.cpp
cd ~/llama.cpp

# Build with CUDA support
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="90"
cmake --build build -j$(nproc)
```

---

## Step 2: Start Services (ALL REQUIRED)

Open **3 separate terminal windows** for each service. **All services are mandatory.**

### Terminal 1: Model Server (llama.cpp) - REQUIRED

```bash
cd /home/vincent/Code/helix-financial-agent
./scripts/start_llama_server.sh
```

**Features:**
- Auto-downloads model if missing (uses `HF_TOKEN` from `.env`)
- Serves Qwen3-30B-A3B on port 8080
- Wait for: `llama server listening at http://0.0.0.0:8081`

### Terminal 2: Semantic Router - REQUIRED

```bash
cd /home/vincent/Code/helix-financial-agent
./scripts/start_router.sh
```

**Mandatory for demo.** Routes all LLM requests based on model name:
- `model=qwen3-30b-a3b` → llama.cpp (agent queries)
- `model=gemini-2.5-pro` → Gemini API (evaluation/judging)

### Terminal 3: MCP Server - REQUIRED

```bash
cd /home/vincent/Code/helix-financial-agent
./scripts/start_mcp_server.sh
```

**Mandatory for demo.** All tool execution goes through MCP:
- Serves financial tools via Model Context Protocol
- Enables centralized tool monitoring and logging

---

## Step 3: Run the Agent

### Random Benchmark Query (Recommended for Demo)

Runs a random query from the benchmark dataset with full tracing and evaluation:

```bash
source .venv/bin/activate
helix-agent --random
```

This will:
1. Select a random query from `data/financial_benchmark_v1_full.jsonl`
2. Show query metadata (category, expected tools)
3. Display ToolRAG selection results
4. Trace all agent steps (Generator → Tools → Reflector)
5. Run LLM-as-a-Judge evaluation via Gemini
6. Display comprehensive execution summary

### Interactive Mode

```bash
source .venv/bin/activate
helix-agent
```

In interactive mode, type `random` to run a random benchmark query.

### Single Query

```bash
source .venv/bin/activate
helix-agent --query "What is AAPL's PE ratio?"

# With evaluation
helix-agent --query "What is NVDA's market cap?" --eval
```

### All CLI Options

```bash
# Random query from benchmark dataset
helix-agent --random

# Random query from custom dataset
helix-agent --random --dataset data/my_benchmark.jsonl

# Single query with evaluation
helix-agent --query "AAPL fundamentals" --eval

# Disable ToolRAG (use all tools)
helix-agent --query "Compare AAPL and MSFT" --no-tool-rag

# Quiet mode (less verbose output)
helix-agent --query "NVDA dividend yield?" --quiet
```

---

## Step 4: Generate Synthetic Data

Generate benchmark dataset using Gemini 2.5 Pro:

```bash
source .venv/bin/activate

# Generate 100 queries (default)
helix-generate

# Generate custom count
helix-generate --count 50 --output data/my_dataset.jsonl
```

---

## Step 5: Run Evaluation Benchmark

Evaluate agent responses using LLM-as-a-Judge:

```bash
source .venv/bin/activate

# Run on generated dataset
helix-eval --dataset data/financial_benchmark_v1_eval.jsonl

# Limit queries for testing
helix-eval --dataset data/test_dataset.jsonl --limit 10
```

---

## Debugging in Cursor/VS Code

The project includes pre-configured debug configurations in `.vscode/launch.json`.

### How to Debug

1. **Open the Debug Panel**: Press `Ctrl+Shift+D` (or click the bug icon in sidebar)

2. **Select a Configuration** from the dropdown:
   - `Helix Agent - Interactive` - Chat mode
   - `Helix Agent - Single Query` - Run one query
   - `Data Generation - Debug` - Generate synthetic data
   - `Evaluation Benchmark - Debug` - Run evaluation
   - `MCP Server - Debug` - Debug tool server

3. **Set Breakpoints**: Click left of line numbers in:
   - `src/helix_financial_agent/agent/nodes.py` - Generator, Reflector, Revisor logic
   - `src/helix_financial_agent/agent/runner.py` - Main agent runner
   - `src/helix_financial_agent/tool_rag/tool_selector.py` - ToolRAG selection
   - `src/helix_financial_agent/data_generation/generate.py` - Data generation
   - `src/helix_financial_agent/evaluation/judge.py` - LLM-as-a-Judge

4. **Start Debugging**: Press `F5` or click the green play button

### Debug Data Generation

1. Select **"Data Generation - Debug"** from dropdown
2. Set breakpoints in `src/helix_financial_agent/data_generation/generate.py`
3. Press `F5`
4. Step through code with:
   - `F10` - Step Over
   - `F11` - Step Into
   - `Shift+F11` - Step Out
   - `F5` - Continue

### Debug the Agent

**All 3 services must be running before debugging the agent:**

1. **Start all services** in separate terminals:
   ```bash
   # Terminal 1
   ./scripts/start_llama_server.sh
   
   # Terminal 2
   ./scripts/start_router.sh
   
   # Terminal 3
   ./scripts/start_mcp_server.sh
   ```

2. Select **"Helix Agent - Single Query"** from dropdown
3. Set breakpoints in agent files
4. Press `F5`
5. Watch the agent flow: Generator → Tools (MCP) → Reflector (Gemini) → Revisor

### Debug Tips

- **View Variables**: Hover over variables or check the Variables panel
- **Watch Expressions**: Add expressions to the Watch panel
- **Debug Console**: Execute Python in the debug context
- **Call Stack**: See the execution path in the Call Stack panel

---

## Service Ports

| Service | Port | Description |
|---------|------|-------------|
| llama.cpp | 8081 | Model inference (OpenAI-compatible) |
| vLLM-SR (Envoy) | 8801 | Semantic routing HTTP entry |
| vLLM-SR (Classify) | 8889 | Health & models API |
| vLLM-SR (Metrics) | 9190 | Prometheus metrics |
| MCP Server | 8000 | FastMCP tool server |

---

## Financial Tools

### Core Tools (Required for Agent)

| Tool | Description | Data Source |
|------|-------------|-------------|
| `get_stock_fundamentals` | PE ratio, market cap, EPS, dividends, beta | yfinance |
| `get_historical_prices` | Price history, returns, moving averages | yfinance |
| `get_financial_statements` | Balance sheet, income statement, cash flow | yfinance |
| `get_company_news` | Recent headlines and news | yfinance |

### Distraction Tools (For ToolRAG Testing)

| Tool | Description |
|------|-------------|
| `get_options_chain` | Options data (calls/puts) |
| `get_institutional_holders` | Institutional ownership |
| `get_insider_transactions` | Insider buying/selling |
| `get_analyst_recommendations` | Analyst ratings |
| `get_earnings_calendar` | Earnings dates |
| `get_sustainability_scores` | ESG scores |
| `get_major_holders` | Major shareholders |
| `get_dividend_history` | Historical dividends |
| `calculate_technical_indicators` | Technical analysis (stub) |
| `compare_sector_performance` | Sector comparison (stub) |

---

## Semantic Routing (MoM Architecture)

The agent uses **MoM (Model of Models)** semantic routing - the router automatically selects the best model based on the content of each request, not hardcoded model names.

### How It Works

1. **Application sends request with `model="MoM"`**
2. **Router analyzes prompt content** using:
   - Keyword signals (e.g., "stock", "evaluate", "generate")
   - Semantic embeddings (sentence-transformers)
3. **Router matches against decisions** (priority-ordered rules)
4. **Router forwards to the appropriate backend**

### Routing Decisions

| Decision | Priority | Trigger Keywords | Routes To |
|----------|----------|------------------|-----------|
| `evaluation` | 15 | evaluate, judge, assess, score, accuracy | Gemini 2.5 Pro |
| `data_generation` | 15 | generate, create, synthetic, dataset | Gemini 2.5 Pro |
| `financial_analysis` | 10 | stock, price, PE ratio, dividend, market | Qwen3 (llama.cpp) |
| `general` | 5 | (fallback) | Qwen3 (llama.cpp) |

### Intent Markers in Code

Prompts include intent markers to help the router classify requests:

```python
# Financial analysis (routes to Qwen3)
"[FINANCIAL_ANALYSIS] Analyze the stock price of AAPL..."

# Evaluation (routes to Gemini)
"[EVALUATE] Assess and judge this response for correctness..."

# Data generation (routes to Gemini)
"[GENERATE] Create synthetic benchmark queries..."
```

### Benefits of MoM Routing

| Benefit | Description |
|---------|-------------|
| No hardcoded models | Application code uses `model="MoM"` everywhere |
| Centralized config | All routing logic in `router_config.yaml` |
| Easy model swaps | Change models without code changes |
| Observable | Monitor routing via metrics (port 9190) |
| A/B testing | Test different models by updating config |

### Example Request Flow

```
User: "What is NVDA's PE ratio?"
  ↓
Agent sends to Router (model="MoM")
  ↓
Router detects: "PE ratio", "NVDA" → financial_keywords
  ↓
Router matches: financial_analysis decision (priority 10)
  ↓
Router forwards to: Qwen3 (llama.cpp)
```

Configuration: `config/router_config.yaml`

---

## Project Structure

```
helix-financial-agent/
├── src/helix_financial_agent/
│   ├── agent/           # LangGraph agent (nodes, graph, runner)
│   ├── tools/           # Financial tools + MCP server
│   ├── tool_rag/        # ToolRAG (ChromaDB + embeddings)
│   ├── router/          # vLLM-SR configuration & client
│   ├── evaluation/      # LLM-as-a-Judge benchmark
│   ├── data_generation/ # Synthetic dataset generator
│   ├── config.py        # Configuration management
│   └── main.py          # CLI entry point
├── scripts/
│   ├── setup.sh         # Installation script
│   ├── start_llama_server.sh  # Model server (auto-downloads model)
│   ├── start_router.sh  # Semantic router
│   └── start_mcp_server.sh    # MCP tool server
├── config/
│   └── router_config.yaml     # vLLM-SR routing rules
├── .vscode/
│   └── launch.json      # Debug configurations
├── .env.example         # Environment template
└── pyproject.toml       # Dependencies
```

---

## CLI Commands

| Command | Description |
|---------|-------------|
| `helix-agent` | Run the financial agent (interactive mode) |
| `helix-agent --random` | Run random benchmark query with full trace + evaluation |
| `helix-agent -q "query"` | Run a specific query |
| `helix-agent -q "query" --eval` | Run query with LLM-as-a-Judge evaluation |
| `helix-generate` | Generate synthetic benchmark dataset |
| `helix-eval` | Run evaluation benchmark on dataset |
| `helix-mcp` | Start MCP tool server |

### Agent CLI Options

| Option | Description |
|--------|-------------|
| `--random, -r` | Run random query from benchmark dataset |
| `--query, -q` | Run specific query |
| `--dataset, -d` | Path to benchmark JSONL file |
| `--eval, -e` | Enable LLM-as-a-Judge evaluation |
| `--no-tool-rag` | Disable dynamic tool selection |
| `--quiet` | Reduce output verbosity |

---

## Troubleshooting

### Model server won't start

```bash
# Check if llama-server exists
ls ~/llama.cpp/build/bin/llama-server

# Rebuild if needed
cd ~/llama.cpp && cmake --build build -j
```

### Model download fails

```bash
# Verify HF_TOKEN is set
grep HF_TOKEN .env

# Manual download
hf download bartowski/Qwen_Qwen3-30B-A3B-Instruct-2507-GGUF \
    Qwen_Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf \
    --local-dir ~/llama.cpp/models
```

### Agent can't connect to model

```bash
# Check if server is running
curl http://localhost:8080/health

# Test inference
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "qwen3", "messages": [{"role": "user", "content": "Hello"}]}'
```

### Debug mode not working

1. Ensure `.venv` is activated in VS Code
2. Check Python interpreter: `Ctrl+Shift+P` → "Python: Select Interpreter"
3. Select: `.venv/bin/python`

---

## License

Apache 2.0

---

## Semantic Routing Deep Dive (MoM Architecture)

The agent uses **MoM (Model of Models)** semantic routing - the router automatically selects the best model based on the content of each request.

### How MoM Routing Works

1. Application sends request with `model="MoM"`
2. Router analyzes prompt content using keywords and embeddings
3. Router matches against priority-ordered decision rules
4. Router forwards to the appropriate backend

### Routing Decision Table

| Decision | Priority | Trigger Keywords | Routes To |
|----------|----------|------------------|-----------|
| evaluation | 15 | evaluate, judge, assess, score, accuracy | Gemini 2.5 Pro |
| data_generation | 15 | generate, create, synthetic, dataset | Gemini 2.5 Pro |
| financial_analysis | 10 | stock, price, PE ratio, dividend, market | Qwen3 (llama.cpp) |
| general | 5 | (fallback) | Qwen3 (llama.cpp) |

### Intent Markers in Prompts

Prompts include intent markers to help the router classify requests reliably:

- `[FINANCIAL_ANALYSIS]` - Triggers routing to Qwen3 for stock/market queries
- `[EVALUATE]` - Triggers routing to Gemini for quality assessment
- `[GENERATE]` - Triggers routing to Gemini for data generation

### Benefits of MoM vs Explicit Model Names

| Aspect | Explicit Model | MoM Routing |
|--------|----------------|-------------|
| Model selection | Hardcoded in Python | Router decides from content |
| Changing models | Requires code changes | Update config/router_config.yaml |
| Observability | None | Metrics at port 9190 |
| A/B testing | Requires code changes | Just update config |

### Example Request Flow

User: "What is NVDA's PE ratio?"
1. Agent sends to Router with model="MoM"
2. Router detects: "PE ratio", "NVDA" → financial_keywords signal
3. Router matches: financial_analysis decision (priority 10)
4. Router forwards to: Qwen3 via llama.cpp

Configuration file: `config/router_config.yaml`
