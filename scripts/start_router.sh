#!/bin/bash
# =============================================================================
# vLLM Semantic Router Startup Script
# =============================================================================
# Starts the vLLM-SR container with clean progress monitoring.
#
# USAGE:
#   ./scripts/start_router.sh           # Start with progress monitoring (default)
#   ./scripts/start_router.sh --detach  # Start in background immediately
#   ./scripts/start_router.sh --raw     # Show raw unfiltered logs
# =============================================================================

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
DIM='\033[2m'
NC='\033[0m'
BOLD='\033[1m'

# Parse arguments
DETACH=false
RAW_LOGS=false
for arg in "$@"; do
    case $arg in
        --detach) DETACH=true ;;
        --raw) RAW_LOGS=true ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --detach    Start in background without waiting"
            echo "  --raw       Show raw unfiltered container logs"
            echo "  -h, --help  Show this help"
            exit 0
            ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

[ -f ".venv/bin/activate" ] && source .venv/bin/activate
if [ -f "$PROJECT_ROOT/.env" ]; then
    set -a; source "$PROJECT_ROOT/.env"; set +a
fi

# Configuration
CONFIG_PATH="$PROJECT_ROOT/config/router_config.yaml"
VLLM_SR_DIR="$PROJECT_ROOT/config/.vllm-sr"
CONTAINER_NAME="vllm-sr-container"
NETWORK_NAME="vllm-sr-network"
IMAGE="ghcr.io/vllm-project/semantic-router/vllm-sr:latest"
HTTP_PORT="${ROUTER_HTTP_PORT:-8801}"
CLASSIFY_PORT="${ROUTER_CLASSIFY_PORT:-8889}"
METRICS_PORT="${ROUTER_METRICS_PORT:-9190}"

echo ""
echo -e "${BOLD}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${BOLD}           vLLM SEMANTIC ROUTER${NC}"
echo -e "${BOLD}═══════════════════════════════════════════════════════════════${NC}"
echo ""

# =============================================================================
# Stage 1: Preprocess Config
# =============================================================================
echo -e "${CYAN}[1/4]${NC} Preprocessing configuration..."

[ ! -f "$CONFIG_PATH" ] && python -c "from helix_financial_agent.router.config import create_router_config; create_router_config()"
mkdir -p "$VLLM_SR_DIR"

PROCESSED_CONFIG="$VLLM_SR_DIR/processed_config.yaml"
RESOLVED=$(python3 << 'PYEOF'
import os, sys, yaml
config_path = os.environ.get('CONFIG_PATH', 'config/router_config.yaml')
output_path = os.environ.get('PROCESSED_CONFIG', 'config/.vllm-sr/processed_config.yaml')
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)
count = 0
for model in config.get('providers', {}).get('models', []):
    if 'access_key_env' in model:
        env_var = model.pop('access_key_env')
        value = os.environ.get(env_var)
        if value:
            model['access_key'] = value
            count += 1
with open(output_path, 'w') as f:
    yaml.dump(config, f, default_flow_style=False, sort_keys=False)
print(count)
PYEOF
)
[ -f "$PROCESSED_CONFIG" ] && CONFIG_PATH="$PROCESSED_CONFIG"
echo -e "      ${GREEN}✓${NC} Resolved ${RESOLVED} API key(s)"

# =============================================================================
# Stage 2: Docker Setup
# =============================================================================
echo -e "${CYAN}[2/4]${NC} Setting up Docker..."

# Cleanup
for c in "$CONTAINER_NAME" vllm-sr-jaeger vllm-sr-prometheus vllm-sr-grafana; do
    docker stop "$c" 2>/dev/null && docker rm "$c" 2>/dev/null || true
done

docker network ls --format '{{.Name}}' | grep -q "^${NETWORK_NAME}$" || docker network create "$NETWORK_NAME" >/dev/null
echo -e "      ${GREEN}✓${NC} Network ready"

# Start observability
docker run -d --name vllm-sr-jaeger --network "$NETWORK_NAME" -p 16686:16686 -p 4318:4317 jaegertracing/all-in-one:latest >/dev/null 2>&1 || true
docker run -d --name vllm-sr-prometheus --network "$NETWORK_NAME" -p 9090:9090 prom/prometheus:latest >/dev/null 2>&1 || true
docker run -d --name vllm-sr-grafana --network "$NETWORK_NAME" -p 3000:3000 grafana/grafana:latest >/dev/null 2>&1 || true
echo -e "      ${GREEN}✓${NC} Observability stack started"

# =============================================================================
# Stage 3: Start Container
# =============================================================================
echo -e "${CYAN}[3/4]${NC} Starting router container..."

ENV_VARS=""
[ -n "$HF_TOKEN" ] && ENV_VARS="$ENV_VARS -e HF_TOKEN=$HF_TOKEN"
[ -n "$HF_ENDPOINT" ] && ENV_VARS="$ENV_VARS -e HF_ENDPOINT=$HF_ENDPOINT"
[ -n "$GEMINI_API_KEY" ] && ENV_VARS="$ENV_VARS -e GEMINI_API_KEY=$GEMINI_API_KEY"

ENVOY_TEMPLATE="$VLLM_SR_DIR/envoy.template.yaml"

# Create persistent cache directories for ML models (avoids re-downloading on restart)
CACHE_DIR="$HOME/.cache/vllm-sr"
MODELS_DIR="$HOME/.local/share/vllm-sr/models"
mkdir -p "$CACHE_DIR" "$MODELS_DIR"

docker run -d \
    --name "$CONTAINER_NAME" \
    --network "$NETWORK_NAME" \
    --add-host=host.docker.internal:host-gateway \
    -p ${HTTP_PORT}:8801 \
    -p ${CLASSIFY_PORT}:8080 \
    -p ${METRICS_PORT}:9190 \
    -p 8700:8700 \
    -p 50051:50051 \
    -v "$CONFIG_PATH":/app/config.yaml:ro \
    -v "$VLLM_SR_DIR":/app/.vllm-sr \
    -v "$ENVOY_TEMPLATE":/app/cli/templates/envoy.template.yaml:ro \
    -v "$CACHE_DIR":/root/.cache/huggingface \
    -v "$MODELS_DIR":/app/models \
    $ENV_VARS \
    "$IMAGE" >/dev/null

echo -e "      ${GREEN}✓${NC} Container started"

if [ "$DETACH" = true ]; then
    echo ""
    echo -e "${GREEN}Router starting in background.${NC}"
    echo "  Monitor: ./scripts/start_router.sh  (run again to see status)"
    echo "  Stop:    ./scripts/stop_router.sh"
    exit 0
fi

# =============================================================================
# Stage 4: Monitor Startup
# =============================================================================
echo -e "${CYAN}[4/4]${NC} Waiting for router to be ready..."
echo ""

if [ "$RAW_LOGS" = true ]; then
    echo -e "${DIM}Showing raw logs (Ctrl+C to stop)...${NC}"
    docker logs -f "$CONTAINER_NAME"
    exit 0
fi

# Smart monitoring with filtered output
START_TIME=$(date +%s)
LAST_MODEL=""
MODELS_DONE=false
CLASSIFIER_STARTED=false
SHOWN_MODELS=""

# Progress display function
show_progress() {
    local elapsed=$1
    local status=$2
    printf "\r      ${YELLOW}⏳${NC} %-50s ${DIM}[%ds]${NC}" "$status" "$elapsed"
}

while true; do
    ELAPSED=$(($(date +%s) - START_TIME))
    
    # Timeout after 15 minutes
    if [ $ELAPSED -gt 900 ]; then
        echo ""
        echo -e "\n${RED}✗${NC} Timeout after 15 minutes"
        echo "  Check logs: docker logs --tail 50 $CONTAINER_NAME"
        exit 1
    fi
    
    # Check if container crashed
    if ! docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
        echo ""
        echo -e "\n${RED}✗${NC} Container stopped unexpectedly"
        echo "  Check logs: docker logs $CONTAINER_NAME"
        exit 1
    fi
    
    # Check health endpoint
    if curl -s "http://localhost:${CLASSIFY_PORT}/health" 2>/dev/null | grep -q "healthy"; then
        echo -e "\r      ${GREEN}✓${NC} Router ready!                                          "
        break
    fi
    
    # Get router log for status
    ROUTER_LOG=$(docker exec "$CONTAINER_NAME" tail -30 /var/log/supervisor/router-error.log 2>/dev/null || echo "")
    
    # Check for model downloads
    if echo "$ROUTER_LOG" | grep -q "Downloading model:"; then
        CURRENT_MODEL=$(echo "$ROUTER_LOG" | grep "Downloading model:" | tail -1 | grep -oP 'models/\K[^"}\s]+' | head -1)
        if [ -n "$CURRENT_MODEL" ] && [ "$CURRENT_MODEL" != "$LAST_MODEL" ]; then
            LAST_MODEL="$CURRENT_MODEL"
            # Show download on new line if it's a new model
            if ! echo "$SHOWN_MODELS" | grep -q "$CURRENT_MODEL"; then
                echo -e "\r      ${CYAN}↓${NC} Downloading: ${CURRENT_MODEL}                    "
                SHOWN_MODELS="$SHOWN_MODELS $CURRENT_MODEL"
            fi
        fi
        show_progress $ELAPSED "Downloading models..."
    
    # Check for model loading complete
    elif echo "$ROUTER_LOG" | grep -q "Starting gRPC server\|gRPC server listening"; then
        if [ "$CLASSIFIER_STARTED" = false ]; then
            CLASSIFIER_STARTED=true
            echo -e "\r      ${GREEN}✓${NC} Models loaded                                       "
            echo -e "      ${CYAN}⟳${NC} Starting classifier..."
        fi
        show_progress $ELAPSED "Starting classifier service..."
    
    # Check for model loading
    elif echo "$ROUTER_LOG" | grep -q "Loading model\|Initializing"; then
        if [ "$MODELS_DONE" = false ]; then
            MODELS_DONE=true
            echo -e "\r      ${GREEN}✓${NC} Downloads complete                                  "
        fi
        show_progress $ELAPSED "Loading models..."
    
    else
        show_progress $ELAPSED "Initializing..."
    fi
    
    sleep 1
done

# =============================================================================
# Summary
# =============================================================================
echo ""
echo -e "${BOLD}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${BOLD}${GREEN}  ✓ ROUTER READY${NC}  ${DIM}(startup took ${ELAPSED}s)${NC}"
echo -e "${BOLD}═══════════════════════════════════════════════════════════════${NC}"
echo ""
echo -e "${BOLD}Endpoints:${NC}"
echo "  API:        http://localhost:${HTTP_PORT}/v1/chat/completions"
echo "  Health:     http://localhost:${CLASSIFY_PORT}/health"
echo "  Dashboard:  http://localhost:8700"
echo ""
echo -e "${BOLD}Commands:${NC}"
echo "  Stop:       ./scripts/stop_router.sh"
echo "  Logs:       docker logs -f $CONTAINER_NAME"
echo ""

# Quick routing test
echo -ne "${DIM}Testing routing...${NC}"
TEST=$(curl -s -m 10 -X POST "http://localhost:${HTTP_PORT}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d '{"model": "MoM", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 1}' 2>/dev/null || echo "")

if echo "$TEST" | grep -q '"model"'; then
    MODEL=$(echo "$TEST" | grep -oP '"model"\s*:\s*"\K[^"]+' | head -1)
    echo -e "\r${GREEN}✓${NC} Routing test passed (routed to: ${MODEL})           "
else
    echo -e "\r${YELLOW}⚠${NC} Routing test skipped (classifier still warming up) "
fi
echo ""
