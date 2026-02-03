#!/bin/bash
# =============================================================================
# Helix Financial Agent - Setup Script
# =============================================================================
# This script sets up the Python environment and installs all dependencies.
# Run from the project root: ./scripts/setup.sh
# =============================================================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

echo -e "${CYAN}"
echo "╔════════════════════════════════════════════════════════════╗"
echo "║         HELIX FINANCIAL AGENT - SETUP                      ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# Check for uv
if ! command -v uv &> /dev/null; then
    echo -e "${RED}Error: 'uv' is not installed.${NC}"
    echo "Install with: curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

echo -e "${GREEN}✓ uv found${NC}"

# Navigate to project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

echo -e "\n${CYAN}Project root: $PROJECT_ROOT${NC}"

# =============================================================================
# Step 1: Create/sync virtual environment with uv
# =============================================================================
echo -e "\n${YELLOW}Step 1: Setting up Python environment...${NC}"

if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    uv venv --python 3.11
fi

echo -e "${GREEN}✓ Virtual environment ready${NC}"

# =============================================================================
# Step 2: Install project dependencies (including vllm-sr beta)
# =============================================================================
echo -e "\n${YELLOW}Step 2: Installing dependencies...${NC}"

# Install the project in editable mode with pre-release support for vllm-sr
# The --prerelease=allow flag is required because vllm-sr only has beta versions
uv pip install --prerelease=allow -e .

echo -e "${GREEN}✓ All dependencies installed (including vllm-sr)${NC}"

# =============================================================================
# Step 3: Verify vllm-sr installation
# =============================================================================
echo -e "\n${YELLOW}Step 3: Verifying vLLM Semantic Router...${NC}"

VLLM_SR_VERSION=$(vllm-sr --version 2>/dev/null)
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ $VLLM_SR_VERSION${NC}"
else
    echo -e "${RED}✗ vllm-sr CLI not found${NC}"
    echo "   Try: uv pip install --prerelease=allow vllm-sr"
    exit 1
fi

# =============================================================================
# Step 4: Create necessary directories
# =============================================================================
echo -e "\n${YELLOW}Step 4: Creating directories...${NC}"

mkdir -p data logs config models

echo -e "${GREEN}✓ Directories created${NC}"

# =============================================================================
# Step 5: Generate router configuration
# =============================================================================
echo -e "\n${YELLOW}Step 5: Generating router configuration...${NC}"

# Activate venv and generate config
source .venv/bin/activate
python -c "
from helix_financial_agent.router.config import create_router_config
create_router_config()
print('Router config generated at config/router_config.yaml')
" 2>/dev/null || echo -e "${YELLOW}⚠ Could not generate router config (will be created on first run)${NC}"

# =============================================================================
# Step 6: Verify installation
# =============================================================================
echo -e "\n${YELLOW}Step 6: Verifying installation...${NC}"

python -c "
import sys
print(f'Python: {sys.version}')

# Check critical imports
checks = []

try:
    import langchain
    checks.append(('langchain', langchain.__version__))
except ImportError as e:
    checks.append(('langchain', f'FAILED: {e}'))

try:
    import langgraph
    checks.append(('langgraph', 'OK'))
except ImportError as e:
    checks.append(('langgraph', f'FAILED: {e}'))

try:
    import yfinance
    checks.append(('yfinance', yfinance.__version__))
except ImportError as e:
    checks.append(('yfinance', f'FAILED: {e}'))

try:
    import fastmcp
    checks.append(('fastmcp', 'OK'))
except ImportError as e:
    checks.append(('fastmcp', f'FAILED: {e}'))

try:
    import chromadb
    checks.append(('chromadb', chromadb.__version__))
except ImportError as e:
    checks.append(('chromadb', f'FAILED: {e}'))

try:
    import sentence_transformers
    checks.append(('sentence-transformers', 'OK'))
except ImportError as e:
    checks.append(('sentence-transformers', f'FAILED: {e}'))

try:
    import mlflow
    checks.append(('mlflow', mlflow.__version__))
except ImportError as e:
    checks.append(('mlflow', f'FAILED: {e}'))

try:
    import rich
    checks.append(('rich', 'OK'))
except ImportError as e:
    checks.append(('rich', f'FAILED: {e}'))

print()
for pkg, status in checks:
    symbol = '✓' if 'FAILED' not in str(status) else '✗'
    print(f'  {symbol} {pkg}: {status}')
"

# =============================================================================
# Step 7: Check .env file
# =============================================================================
echo -e "\n${YELLOW}Step 7: Checking configuration...${NC}"

if [ -f ".env" ]; then
    echo -e "${GREEN}✓ .env file exists${NC}"
    
    # Check for required variables
    source .env 2>/dev/null
    
    if [ -n "$GEMINI_API_KEY" ]; then
        echo -e "${GREEN}✓ GEMINI_API_KEY is set${NC}"
    else
        echo -e "${YELLOW}⚠ GEMINI_API_KEY not set (required for evaluation)${NC}"
    fi
    
    if [ -n "$LLAMA_CPP_BASE_URL" ]; then
        echo -e "${GREEN}✓ LLAMA_CPP_BASE_URL: $LLAMA_CPP_BASE_URL${NC}"
    fi
else
    echo -e "${YELLOW}⚠ .env file not found. Copy from template:${NC}"
    echo "   cp .env.example .env"
fi

# =============================================================================
# Done!
# =============================================================================
echo -e "\n${GREEN}"
echo "╔════════════════════════════════════════════════════════════╗"
echo "║                    SETUP COMPLETE!                          ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

echo -e "Next steps:"
echo -e "  1. ${CYAN}Start llama.cpp server:${NC} ./scripts/start_llama_server.sh"
echo -e "  2. ${CYAN}(Optional) Start router:${NC} ./scripts/start_router.sh"
echo -e "  3. ${CYAN}(Optional) Start MCP:${NC} ./scripts/start_mcp_server.sh"
echo -e "  4. ${CYAN}Run the agent:${NC} source .venv/bin/activate && helix-agent"
echo ""
echo -e "Or run a quick test:"
echo -e "  source .venv/bin/activate"
echo -e "  helix-agent --query \"What is AAPL's PE ratio?\""
echo ""
