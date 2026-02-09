#!/bin/bash
# =============================================================================
# SSH Port Forwarding for Web UIs
# =============================================================================
# Run this script on your LOCAL machine (laptop/desktop) to access web
# interfaces running on the remote server (ZGX Nano / DGX Spark).
#
# Usage:
#   ./scripts/ssh_port_forward.sh <user>@<host>
#
# Local ports are read from .env (see LOCAL_STREAMLIT_PORT, LOCAL_MLFLOW_PORT,
# LOCAL_ROUTER_HUB_PORT). Defaults apply if .env is missing or variables unset.
#
# Forwards (remote → local):
#   - Streamlit Eval & Run UI:  remote STREAMLIT_PORT (8501) → local LOCAL_STREAMLIT_PORT
#   - Semantic Router Hub UI:   remote 8700 (vLLM-SR dashboard) → local LOCAL_ROUTER_HUB_PORT
#   - MLflow UI:                remote MLFLOW_PORT (5000) → local LOCAL_MLFLOW_PORT
# =============================================================================

if [ -z "$1" ]; then
    echo "Usage: $0 <user>@<host>"
    echo ""
    echo "Local ports are read from .env (LOCAL_STREAMLIT_PORT, LOCAL_MLFLOW_PORT, LOCAL_ROUTER_HUB_PORT)."
    echo ""
    echo "Example: $0 vincent@dgx-spark.local"
    echo ""
    echo "Run this on your LOCAL machine to forward web UI ports."
    exit 1
fi

# Project root: directory containing this script's parent (scripts/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
ENV_FILE="$PROJECT_ROOT/.env"

if [ -f "$ENV_FILE" ]; then
    set -a
    # shellcheck source=/dev/null
    source "$ENV_FILE"
    set +a
fi

# Server ports (where services listen on remote); must match .env on server
# Router UI (Hub/dashboard) is on 8700 per vLLM-SR (DASHBOARD_PORT), separate from Classify (8889)
STREAMLIT_PORT="${STREAMLIT_PORT:-8501}"
MLFLOW_PORT="${MLFLOW_PORT:-5000}"
ROUTER_HUB_PORT="${ROUTER_HUB_PORT:-8700}"

# Local ports (bind on this machine when forwarding); defaults if not in .env
LOCAL_STREAMLIT_PORT="${LOCAL_STREAMLIT_PORT:-8501}"
LOCAL_MLFLOW_PORT="${LOCAL_MLFLOW_PORT:-5000}"
LOCAL_ROUTER_HUB_PORT="${LOCAL_ROUTER_HUB_PORT:-8700}"

SSH_TARGET="$1"

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║     SSH Port Forwarding for Web UIs                            ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""
echo "Connecting to: $SSH_TARGET"
echo "Ports from: ${ENV_FILE} (defaults if not set)"
echo ""
echo "Once connected, access these web interfaces locally:"
echo ""
echo "  Streamlit Eval & Run UI:  http://localhost:$LOCAL_STREAMLIT_PORT"
echo "  Semantic Router Hub UI:   http://localhost:$LOCAL_ROUTER_HUB_PORT"
echo "  MLflow UI:                http://localhost:$LOCAL_MLFLOW_PORT"
echo ""
echo "On the server, start services with (ports from .env):"
echo "  ./scripts/run_streamlit.sh     # Streamlit on port $STREAMLIT_PORT"
echo "  ./scripts/run_mlflow_ui.sh     # MLflow UI on port $MLFLOW_PORT"
echo ""
echo "Press Ctrl+C to close the tunnel."
echo ""

# Forward: local port -> remote port (both from .env for consistency)
# Router UI (dashboard) is on server at 8700
ssh -L "${LOCAL_STREAMLIT_PORT}:localhost:${STREAMLIT_PORT}" \
    -L "${LOCAL_ROUTER_HUB_PORT}:localhost:${ROUTER_HUB_PORT}" \
    -L "${LOCAL_MLFLOW_PORT}:localhost:${MLFLOW_PORT}" \
    -N "$SSH_TARGET"
