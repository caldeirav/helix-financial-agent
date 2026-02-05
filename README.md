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
                            ┌───────────────────────────────────────┐
                            │       vLLM SEMANTIC ROUTER            │
                            │          (model="MoM")                │
                            │                                       │
                            │    Qwen3 (llama.cpp)     Gemini 2.5   │
                            └───────┬───────────────────────┬───────┘
                                    │                       │
                         ┌──────────┴──────────┐            │
                         ▼                     ▼            ▼
User Query ──► ToolRAG ──► Generator ──► Reflector ◄───── Revisor
                 │             │             │                 ▲
                 │             │             │                 │
                 │             ▼             ▼                 │
                 │        MCP Server    [Score >= 8?]          │
                 │             │             │                 │
                 │             ▼        ┌────┴────┐            │
                 │         yfinance     ▼         ▼            │
                 │                   Output    Revise ─────────┘
                 │                             (max 3)
                 │
                 └─► Only selected tools bound to Generator/Revisor
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

Run these commands on your **local machine** (laptop/desktop) to access web UIs on the remote server:

```bash
# Forward web UI ports (replace <user>@<host> with your server)
# Note: Uses local port 8180 to avoid conflicts with Cursor IDE (which uses 8080)
ssh -L 8180:localhost:8080 \
    -L 5000:localhost:5000 \
    <user>@<host>

# Or use the provided script
./scripts/ssh_port_forward.sh <user>@<host>
```

**After port forwarding, open in your local browser:**
- Semantic Router Hub UI: http://localhost:8180
- MLflow UI: http://localhost:5000

**Note:** Start MLflow UI on the server first: `mlflow ui --host 0.0.0.0`

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

## MLflow Tracing

End-to-end observability for the agent with automatic tracing and custom assessments.

### What Gets Traced

| Component | Traced As | Details |
|-----------|-----------|---------|
| Generator → ChatOpenAI | CHAT_MODEL span | Prompts, outputs, token usage |
| Tool Executor → ToolNode | TOOL spans | Tool name, arguments, outputs |
| Reflector → ChatOpenAI | CHAT_MODEL span | Evaluation prompts, responses |
| Revisor → ChatOpenAI | CHAT_MODEL span | Revision prompts, responses |
| Full graph execution | CHAIN span | End-to-end timeline |

### Custom Assessments

Per-trace assessments logged for each agent run:

| Assessment | Type | Description |
|------------|------|-------------|
| `tool_selection_successful` | Y/N | Did ToolRAG select the correct tools? |
| `model_selection_successful` | Y/N | Did the router select appropriate models? |
| `judge_score` | 0-10 | Score from LLM-as-a-Judge evaluation |
| `latency_seconds` | float | Total execution time |
| `iteration_count` | int | Number of revision iterations |

### Usage

Tracing is enabled by default. View traces at http://localhost:5000 after starting the MLflow UI:

```bash
# Start MLflow UI
mlflow ui --port 5000

# Run agent (tracing enabled by default)
helix-agent -q "What is AAPL's PE ratio?"

# Run benchmark with tracing
helix-eval --max-queries 10

# Disable tracing
helix-agent -q "query" --no-tracing
```

### Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `MLFLOW_TRACKING_URI` | `./mlruns` | MLflow tracking URI |
| `MLFLOW_EXPERIMENT_NAME` | `helix-financial-agent` | Experiment name |

### Benchmark Metrics

When running benchmarks, aggregate metrics are logged to MLflow:

| Metric | Description |
|--------|-------------|
| `avg_correctness_score` | Average judge score for valid queries |
| `valid_pass_rate` | % of valid queries scoring >= 7 |
| `safety_pass_rate` | % of hazard queries correctly refused |
| `tool_selection_accuracy` | % of queries with correct tool selection |
| `avg_agent_time_sec` | Average execution time per query |

### Remote Tracking Server

For production deployments, use a remote MLflow server:

```bash
# Set remote tracking URI
export MLFLOW_TRACKING_URI=http://mlflow-server:5000

# Or in .env file
MLFLOW_TRACKING_URI=http://mlflow-server:5000
```

---

## Verbose Logging

Detailed, real-time logging of all agent interactions for debugging and monitoring.

### What Gets Logged

| Category | Details |
|----------|---------|
| LLM Requests | Model requested, prompt preview, timing |
| LLM Responses | Routed model, response preview, duration |
| Routing Decisions | Requested vs routed model, fallback warnings |
| Tool Calls | Tool name, arguments, outputs |
| Flow Events | Phase transitions, decisions |
| Errors | Full error details with context |

### Usage

Verbose logging is **enabled by default**. Disable with `--quiet`:

```bash
# Data generation with verbose logging (default)
helix-generate --count 20

# Data generation without verbose logging
helix-generate --count 20 --quiet

# Benchmark with verbose logging (default)
helix-eval --max-queries 10

# Benchmark without verbose logging
helix-eval --max-queries 10 --quiet
```

### Output Format

Real-time log entries show:
```
  [  0.05s] 📍 Generator Initialized
           └─ model: MoM
           └─ router_endpoint: http://localhost:8801/v1
  [  0.12s] 🤖 LLM Request [generator/fundamental_basic]
           └─ model_requested: MoM
           └─ prompt_preview: [GENERATE SYNTHETIC DATA]...
  [  1.85s] 🤖 LLM Response [generator/fundamental_basic]
           └─ routed_to: gemini-2.5-pro
           └─ duration: 1730ms
  [  1.86s] 🔀 Routing Decision
           └─ requested: MoM
           └─ routed_to: gemini-2.5-pro
```

### End-of-Run Summary

After completion, a summary table is printed:

```
╔══════════════════════════════════════════════════════════════════════╗
║                  📊 EXECUTION SUMMARY                                 ║
╚══════════════════════════════════════════════════════════════════════╝
⏱️  Total Time                   45.23s
📝 Log Entries                   127

──────────────────────────────────────────────────────────────────────
🤖 LLM Interactions:
┌─────────────────┬──────────────────┬──────────┬────────┐
│ Node            │ Routed To        │ Duration │ Status │
├─────────────────┼──────────────────┼──────────┼────────┤
│ generator/basic │ gemini-2.5-pro   │   1730ms │   ✓    │
│ generator/adv   │ gemini-2.5-pro   │   2150ms │   ✓    │
└─────────────────┴──────────────────┴──────────┴────────┘
   Total LLM time: 12500ms (12.50s)
   Requests: 10 (✓10 / ✗0)

──────────────────────────────────────────────────────────────────────
🔀 Routing Summary:
   → Qwen3 (local):  0
   → Gemini (API):   10
```

### Routing Fallback Detection

When the router selects an unexpected model (e.g., routing to local Qwen when Gemini was expected for data generation), verbose logging highlights this:

```
  [  1.85s] 🔀 Routing Decision (FALLBACK)
           └─ requested: MoM
           └─ routed_to: qwen3-30b-a3b
           └─ decision: fallback_to_local
  [  1.85s] 📍 Routing fallback for fundamental_basic
           └─ expected: gemini-2.5-pro
           └─ got: qwen3-30b-a3b
           └─ hint: Generation keywords may not be triggering data_generation decision
```

This helps identify when the semantic router's rules need adjustment without requiring code changes.

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
| `--no-tracing` | Disable MLflow tracing |
| `--quiet, -q` | Disable verbose logging |

### Data Generation Options

| Option | Description |
|--------|-------------|
| `--count, -n` | Total queries to generate (default: 100) |
| `--output-dir, -o` | Output directory (default: ./data) |
| `--eval-ratio` | Ratio for evaluation split (default: 0.10) |
| `--valid-ratio` | Ratio of valid vs hazard queries (default: 0.80) |
| `--quiet, -q` | Disable verbose logging |

### Benchmark Options

| Option | Description |
|--------|-------------|
| `--dataset` | Path to JSONL dataset |
| `--max-queries` | Maximum queries to run |
| `--no-tool-rag` | Disable ToolRAG |
| `--no-tracing` | Disable MLflow tracing |
| `--quiet, -q` | Disable verbose logging |

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
