#!/bin/bash
# llama.cpp Server Startup for DGX Spark
set -e

LLAMA_CPP_DIR="${LLAMA_CPP_DIR:-$HOME/llama.cpp}"
MODEL_DIR="${MODEL_DIR:-$LLAMA_CPP_DIR/models}"
MODEL_NAME="${MODEL_NAME:-Qwen_Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf}"
MODEL_PATH="${MODEL_PATH:-$MODEL_DIR/$MODEL_NAME}"

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8080}"
CTX_SIZE="${CTX_SIZE:-32768}"
N_GPU_LAYERS="${N_GPU_LAYERS:--1}"
PARALLEL="${PARALLEL:-2}"

echo "Starting llama.cpp server..."
echo "  Model: $MODEL_PATH"
echo "  Port: $PORT"

LLAMA_SERVER="$LLAMA_CPP_DIR/build/bin/llama-server"
[ ! -f "$LLAMA_SERVER" ] && LLAMA_SERVER="$LLAMA_CPP_DIR/llama-server"
[ ! -f "$LLAMA_SERVER" ] && { echo "Error: llama-server not found"; exit 1; }
[ ! -f "$MODEL_PATH" ] && { echo "Error: Model not found"; exit 1; }

export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

exec "$LLAMA_SERVER" \
    --model "$MODEL_PATH" \
    --host "$HOST" --port "$PORT" \
    --ctx-size "$CTX_SIZE" \
    --n-gpu-layers "$N_GPU_LAYERS" \
    --chat-template chatml \
    --parallel "$PARALLEL" --metrics
