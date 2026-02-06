#!/bin/bash
# =============================================================================
# Run Streamlit Eval & Run UI
# =============================================================================
# Starts the Streamlit app using STREAMLIT_PORT from .env.
# Usage: ./scripts/run_streamlit.sh
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

STREAMLIT_PORT="${STREAMLIT_PORT:-8501}"

[ -f ".venv/bin/activate" ] && source .venv/bin/activate

exec uv run streamlit run app.py --server.port "$STREAMLIT_PORT"
