#!/bin/bash
# FastMCP Server for Financial Tools
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"
[ -f ".venv/bin/activate" ] && source .venv/bin/activate

export MCP_SERVER_HOST="${MCP_SERVER_HOST:-localhost}"
export MCP_SERVER_PORT="${MCP_SERVER_PORT:-8000}"

echo "Starting MCP Server on $MCP_SERVER_HOST:$MCP_SERVER_PORT..."
exec python -m helix_financial_agent.tools.mcp_server
