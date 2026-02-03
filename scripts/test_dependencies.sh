#!/bin/bash
# Test dependencies
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"
[ -f ".venv/bin/activate" ] && source .venv/bin/activate
echo "Testing Python dependencies..."
python -c "
from helix_financial_agent import __version__
print(f'helix_financial_agent: v{__version__}')
import langchain, langgraph, yfinance, mlflow, rich
print('All core dependencies OK')
"
