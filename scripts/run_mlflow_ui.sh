#!/bin/bash
# =============================================================================
# Run MLflow UI
# =============================================================================
# Starts MLflow UI using MLFLOW_PORT from .env. Use when viewing traces
# (e.g. on a remote server with port forwarding).
# Usage: ./scripts/run_mlflow_ui.sh
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

if [ -f ".env" ]; then
    set -a
    # shellcheck source=/dev/null
    source .env
    set +a
fi

MLFLOW_PORT="${MLFLOW_PORT:-5000}"

[ -f ".venv/bin/activate" ] && source .venv/bin/activate

exec mlflow ui --port "$MLFLOW_PORT" --host 0.0.0.0
