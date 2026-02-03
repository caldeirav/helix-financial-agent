#!/bin/bash
# vLLM Semantic Router Startup
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"
[ -f ".venv/bin/activate" ] && source .venv/bin/activate

CONFIG_PATH="$PROJECT_ROOT/config/router_config.yaml"
[ ! -f "$CONFIG_PATH" ] && python -c "from helix_financial_agent.router.config import create_router_config; create_router_config()"

echo "Starting vLLM Semantic Router on port 8888..."
exec vllm-sr serve --config "$CONFIG_PATH"
