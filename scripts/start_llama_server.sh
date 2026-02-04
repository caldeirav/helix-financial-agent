#!/bin/bash
# =============================================================================
# llama.cpp Server Startup for DGX Spark / ZGX Nano
# =============================================================================
# Automatically downloads the model if missing using HuggingFace credentials
# from .env file.
# =============================================================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

# =============================================================================
# Configuration
# =============================================================================

# Get script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Load environment variables from .env if it exists
if [ -f "$PROJECT_ROOT/.env" ]; then
    set -a
    source "$PROJECT_ROOT/.env"
    set +a
fi

# Model configuration
LLAMA_CPP_DIR="${LLAMA_CPP_DIR:-$HOME/llama.cpp}"
MODEL_DIR="${MODEL_DIR:-$LLAMA_CPP_DIR/models}"
MODEL_NAME="${MODEL_NAME:-Qwen_Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf}"
MODEL_PATH="${MODEL_PATH:-$MODEL_DIR/$MODEL_NAME}"

# HuggingFace model info
HF_REPO="${HF_REPO:-bartowski/Qwen_Qwen3-30B-A3B-Instruct-2507-GGUF}"
HF_FILE="${HF_FILE:-Qwen_Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf}"

# Server configuration
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8081}"
CTX_SIZE="${CTX_SIZE:-32768}"
N_GPU_LAYERS="${N_GPU_LAYERS:--1}"
PARALLEL="${PARALLEL:-2}"

# =============================================================================
# Functions
# =============================================================================

download_model() {
    echo -e "${CYAN}Model not found. Downloading from HuggingFace...${NC}"
    echo -e "  Repository: ${HF_REPO}"
    echo -e "  File: ${HF_FILE}"
    echo -e "  Destination: ${MODEL_DIR}"
    
    # Create model directory if it doesn't exist
    mkdir -p "$MODEL_DIR"
    
    # Check if huggingface-cli is available
    if ! command -v huggingface-cli &> /dev/null; then
        echo -e "${YELLOW}huggingface-cli not found. Trying 'hf' command...${NC}"
        HF_CMD="hf"
    else
        HF_CMD="huggingface-cli"
    fi
    
    # Check for HF_TOKEN in environment (from .env)
    if [ -n "$HF_TOKEN" ]; then
        echo -e "${GREEN}✓ Using HF_TOKEN from .env for authentication${NC}"
        export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
    else
        echo -e "${YELLOW}⚠ HF_TOKEN not set in .env. Attempting anonymous download...${NC}"
        echo -e "  If download fails, add HF_TOKEN to your .env file"
        echo -e "  Get token at: https://huggingface.co/settings/tokens"
    fi
    
    # Download the model
    echo -e "\n${CYAN}Starting download (this may take 15-30 minutes)...${NC}"
    
    if [ "$HF_CMD" = "hf" ]; then
        $HF_CMD download "$HF_REPO" "$HF_FILE" --local-dir "$MODEL_DIR"
    else
        $HF_CMD download "$HF_REPO" "$HF_FILE" --local-dir "$MODEL_DIR"
    fi
    
    # Verify download
    if [ -f "$MODEL_PATH" ]; then
        echo -e "${GREEN}✓ Model downloaded successfully!${NC}"
    else
        echo -e "${RED}✗ Download failed. Please download manually:${NC}"
        echo -e "  hf download $HF_REPO $HF_FILE --local-dir $MODEL_DIR"
        exit 1
    fi
}

check_llama_server() {
    # Find llama-server executable
    LLAMA_SERVER=""
    
    # Check common locations
    for path in \
        "$LLAMA_CPP_DIR/build/bin/llama-server" \
        "$LLAMA_CPP_DIR/llama-server" \
        "/usr/local/bin/llama-server" \
        "$(which llama-server 2>/dev/null)"; do
        if [ -f "$path" ] && [ -x "$path" ]; then
            LLAMA_SERVER="$path"
            break
        fi
    done
    
    if [ -z "$LLAMA_SERVER" ]; then
        echo -e "${RED}Error: llama-server not found!${NC}"
        echo -e ""
        echo -e "Please build llama.cpp first:"
        echo -e "  ${CYAN}git clone https://github.com/ggerganov/llama.cpp.git ~/llama.cpp${NC}"
        echo -e "  ${CYAN}cd ~/llama.cpp && cmake -B build -DGGML_CUDA=ON && cmake --build build -j${NC}"
        echo -e ""
        echo -e "Or set LLAMA_CPP_DIR to your llama.cpp installation:"
        echo -e "  ${CYAN}export LLAMA_CPP_DIR=/path/to/llama.cpp${NC}"
        exit 1
    fi
    
    echo -e "${GREEN}✓ Found llama-server: $LLAMA_SERVER${NC}"
}

# =============================================================================
# Main
# =============================================================================

echo -e "${CYAN}"
echo "╔════════════════════════════════════════════════════════════╗"
echo "║         LLAMA.CPP MODEL SERVER                             ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# Check for llama-server
check_llama_server

# Check if model exists, download if missing
if [ ! -f "$MODEL_PATH" ]; then
    download_model
fi

echo -e "\n${CYAN}Starting server...${NC}"
echo -e "  Model: $MODEL_PATH"
echo -e "  Host: $HOST"
echo -e "  Port: $PORT"
echo -e "  Context: $CTX_SIZE tokens"
echo -e "  GPU Layers: $N_GPU_LAYERS"
echo -e "  Parallel: $PARALLEL"

# Set CUDA environment
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

echo -e "\n${GREEN}Server starting on http://$HOST:$PORT${NC}"
echo -e "${YELLOW}Press Ctrl+C to stop${NC}\n"

# Start the server
exec "$LLAMA_SERVER" \
    --model "$MODEL_PATH" \
    --host "$HOST" \
    --port "$PORT" \
    --ctx-size "$CTX_SIZE" \
    --n-gpu-layers "$N_GPU_LAYERS" \
    --chat-template chatml \
    --parallel "$PARALLEL" \
    --metrics
