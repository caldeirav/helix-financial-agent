#!/bin/bash
# =============================================================================
# vLLM Semantic Router Startup Script
# =============================================================================
# Starts the vLLM-SR container with proper monitoring of startup progress.
#
# USAGE:
#   ./scripts/start_router.sh           # Start with progress monitoring (default)
#   ./scripts/start_router.sh -f        # Start and follow all logs
#   ./scripts/start_router.sh --detach  # Start in background immediately
#   ./scripts/start_router.sh --quiet   # Minimal output
#
# The script monitors:
#   1. Container startup
#   2. Model downloads from HuggingFace (~1.5GB on first run)
#   3. ExtProc service readiness (gRPC classifier)
#   4. Health endpoint availability
# =============================================================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color
BOLD='\033[1m'

# Parse command line arguments
FOLLOW_LOGS=false
QUIET=false
DETACH=false

for arg in "$@"; do
    case $arg in
        -f|--follow)
            FOLLOW_LOGS=true
            shift
            ;;
        --detach)
            DETACH=true
            shift
            ;;
        -q|--quiet)
            QUIET=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  -f, --follow    Follow all container logs after starting"
            echo "  --detach        Start in background without waiting for ready"
            echo "  -q, --quiet     Minimal output (errors only)"
            echo "  -h, --help      Show this help message"
            exit 0
            ;;
    esac
done

# Helper functions
log_info() {
    [ "$QUIET" = true ] || echo -e "${BLUE}ℹ${NC} $1"
}

log_success() {
    [ "$QUIET" = true ] || echo -e "${GREEN}✓${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

log_error() {
    echo -e "${RED}✗${NC} $1"
}

log_stage() {
    [ "$QUIET" = true ] || echo -e "\n${BOLD}${CYAN}▶ $1${NC}"
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# Activate virtual environment if available
[ -f ".venv/bin/activate" ] && source .venv/bin/activate

# Load environment variables from .env
if [ -f "$PROJECT_ROOT/.env" ]; then
    set -a
    source "$PROJECT_ROOT/.env"
    set +a
fi

# Configuration
CONFIG_PATH="$PROJECT_ROOT/config/router_config.yaml"
VLLM_SR_DIR="$PROJECT_ROOT/config/.vllm-sr"
CONTAINER_NAME="vllm-sr-container"
NETWORK_NAME="vllm-sr-network"
IMAGE="ghcr.io/vllm-project/semantic-router/vllm-sr:latest"

# Ports
HTTP_PORT="${ROUTER_HTTP_PORT:-8801}"
CLASSIFY_PORT="${ROUTER_CLASSIFY_PORT:-8889}"
METRICS_PORT="${ROUTER_METRICS_PORT:-9190}"

# Timeouts
HEALTH_TIMEOUT=900  # 15 minutes max for model downloads (first run downloads ~1.5GB)

echo ""
echo -e "${BOLD}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${BOLD}           vLLM SEMANTIC ROUTER STARTUP${NC}"
echo -e "${BOLD}═══════════════════════════════════════════════════════════════${NC}"

# =============================================================================
# Stage 1: Preprocess Configuration
# =============================================================================
log_stage "Preprocessing configuration"

if [ ! -f "$CONFIG_PATH" ]; then
    log_info "Generating router configuration..."
    python -c "from helix_financial_agent.router.config import create_router_config; create_router_config()"
fi

mkdir -p "$VLLM_SR_DIR"

PROCESSED_CONFIG="$VLLM_SR_DIR/processed_config.yaml"
PREPROCESS_RESULT=$(python3 << 'PYEOF'
import os
import sys
import yaml

config_path = os.environ.get('CONFIG_PATH', 'config/router_config.yaml')
output_path = os.environ.get('PROCESSED_CONFIG', 'config/.vllm-sr/processed_config.yaml')

with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

resolved_count = 0
models = config.get('providers', {}).get('models', [])
for model in models:
    if 'access_key_env' in model:
        env_var = model.pop('access_key_env')
        value = os.environ.get(env_var)
        if value:
            model['access_key'] = value
            resolved_count += 1

with open(output_path, 'w') as f:
    yaml.dump(config, f, default_flow_style=False, sort_keys=False)

print(resolved_count)
PYEOF
)

if [ -f "$PROCESSED_CONFIG" ]; then
    CONFIG_PATH="$PROCESSED_CONFIG"
    log_success "Resolved $PREPROCESS_RESULT API key(s) from environment"
fi

# =============================================================================
# Stage 2: Cleanup and Network Setup
# =============================================================================
log_stage "Preparing Docker environment"

cleanup_container() {
    local name=$1
    if docker ps -a --format '{{.Names}}' | grep -q "^${name}$"; then
        docker stop "$name" 2>/dev/null || true
        docker rm "$name" 2>/dev/null || true
    fi
}

cleanup_container "$CONTAINER_NAME" 2>/dev/null
cleanup_container "vllm-sr-jaeger" 2>/dev/null
cleanup_container "vllm-sr-prometheus" 2>/dev/null
cleanup_container "vllm-sr-grafana" 2>/dev/null

if ! docker network ls --format '{{.Name}}' | grep -q "^${NETWORK_NAME}$"; then
    docker network create "$NETWORK_NAME" >/dev/null 2>&1
fi
log_success "Docker network ready"

# =============================================================================
# Stage 3: Start Observability Stack (optional)
# =============================================================================
log_stage "Starting observability stack"

docker run -d --name vllm-sr-jaeger \
    --network "$NETWORK_NAME" \
    -p 16686:16686 \
    -p 4318:4317 \
    jaegertracing/all-in-one:latest >/dev/null 2>&1 && log_success "Jaeger started" || log_info "Jaeger already running"

docker run -d --name vllm-sr-prometheus \
    --network "$NETWORK_NAME" \
    -p 9090:9090 \
    prom/prometheus:latest >/dev/null 2>&1 && log_success "Prometheus started" || log_info "Prometheus already running"

docker run -d --name vllm-sr-grafana \
    --network "$NETWORK_NAME" \
    -p 3000:3000 \
    grafana/grafana:latest >/dev/null 2>&1 && log_success "Grafana started" || log_info "Grafana already running"

# =============================================================================
# Stage 4: Start Router Container
# =============================================================================
log_stage "Starting vLLM-SR container"

ENV_VARS=""
[ -n "$HF_TOKEN" ] && ENV_VARS="$ENV_VARS -e HF_TOKEN=$HF_TOKEN"
[ -n "$HF_ENDPOINT" ] && ENV_VARS="$ENV_VARS -e HF_ENDPOINT=$HF_ENDPOINT"
[ -n "$GEMINI_API_KEY" ] && ENV_VARS="$ENV_VARS -e GEMINI_API_KEY=$GEMINI_API_KEY"

ENVOY_TEMPLATE="$VLLM_SR_DIR/envoy.template.yaml"
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
    $ENV_VARS \
    "$IMAGE" >/dev/null

if [ $? -ne 0 ]; then
    log_error "Failed to start container"
    exit 1
fi
log_success "Container started"

# =============================================================================
# Stage 5: Monitor Startup Progress
# =============================================================================

if [ "$FOLLOW_LOGS" = true ]; then
    echo ""
    echo -e "${BOLD}Following container logs (Ctrl+C to detach)...${NC}"
    echo ""
    docker logs -f "$CONTAINER_NAME"
    exit 0
fi

if [ "$DETACH" = true ]; then
    echo ""
    log_success "Router starting in background"
    echo ""
    echo "  Monitor progress:  docker logs -f $CONTAINER_NAME"
    echo "  Check health:      curl http://localhost:${CLASSIFY_PORT}/health"
    echo "  Stop router:       ./scripts/stop_router.sh"
    exit 0
fi

log_stage "Waiting for router to be ready"

# Track startup state
MODELS_DOWNLOADED=false
LAST_MODEL=""
DOWNLOAD_COUNT=0
TOTAL_MODELS=8  # Approximate number of models to download
START_TIME=$(date +%s)

echo ""

while true; do
    ELAPSED=$(($(date +%s) - START_TIME))
    
    # Timeout check
    if [ $ELAPSED -gt $HEALTH_TIMEOUT ]; then
        echo ""
        log_error "Startup timed out after ${HEALTH_TIMEOUT}s"
        echo ""
        echo "Check logs: docker logs --tail 100 $CONTAINER_NAME"
        exit 1
    fi
    
    # Check if container is still running
    if ! docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
        echo ""
        log_error "Container stopped unexpectedly"
        echo ""
        echo "Check logs: docker logs $CONTAINER_NAME"
        exit 1
    fi
    
    # Check health endpoint
    HEALTH=$(curl -s "http://localhost:${CLASSIFY_PORT}/health" 2>/dev/null || echo "")
    if echo "$HEALTH" | grep -q "healthy"; then
        echo ""
        log_success "Router is ready! (${ELAPSED}s)"
        break
    fi
    
    # Parse router logs for progress (only check every 2 seconds to reduce overhead)
    if [ $((ELAPSED % 2)) -eq 0 ]; then
        # Get recent router log entries
        LOG_OUTPUT=$(docker exec "$CONTAINER_NAME" tail -20 /var/log/supervisor/router-error.log 2>/dev/null || echo "")
        
        # Check for model download progress
        if echo "$LOG_OUTPUT" | grep -q "Downloading model:"; then
            CURRENT_MODEL=$(echo "$LOG_OUTPUT" | grep "Downloading model:" | tail -1 | sed 's/.*Downloading model: //' | tr -d '"' | tr -d '}' | tr -d '{')
            if [ "$CURRENT_MODEL" != "$LAST_MODEL" ] && [ -n "$CURRENT_MODEL" ]; then
                LAST_MODEL="$CURRENT_MODEL"
                MODEL_NAME=$(basename "$CURRENT_MODEL")
                echo -e "  ${CYAN}↓${NC} Downloading: ${MODEL_NAME}"
            fi
        fi
        
        # Check for successful downloads
        DOWNLOADED=$(echo "$LOG_OUTPUT" | grep -c "Successfully downloaded" 2>/dev/null || true)
        DOWNLOADED=${DOWNLOADED:-0}
        if [ "$DOWNLOADED" -gt "$DOWNLOAD_COUNT" ] 2>/dev/null; then
            DOWNLOAD_COUNT=$DOWNLOADED
        fi
        
        # Check for "Starting router" message
        if echo "$LOG_OUTPUT" | grep -q "Starting router"; then
            if [ "$MODELS_DOWNLOADED" = false ]; then
                MODELS_DOWNLOADED=true
                echo -e "  ${GREEN}✓${NC} Models downloaded"
                echo -e "  ${CYAN}⟳${NC} Starting classifier service..."
            fi
        fi
        
        # Check for gRPC server start
        if echo "$LOG_OUTPUT" | grep -q "gRPC server listening"; then
            echo -e "  ${GREEN}✓${NC} Classifier service started"
        fi
    fi
    
    # Show elapsed time every 10 seconds
    if [ $((ELAPSED % 10)) -eq 0 ] && [ $ELAPSED -gt 0 ]; then
        if [ "$MODELS_DOWNLOADED" = false ]; then
            echo -e "  ${YELLOW}⏳${NC} Downloading models... (${ELAPSED}s)"
        else
            echo -e "  ${YELLOW}⏳${NC} Starting services... (${ELAPSED}s)"
        fi
    fi
    
    sleep 1
done

# =============================================================================
# Stage 6: Print Summary
# =============================================================================
echo ""
echo -e "${BOLD}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${BOLD}${GREEN}           ROUTER READY${NC}"
echo -e "${BOLD}═══════════════════════════════════════════════════════════════${NC}"
echo ""
echo -e "${BOLD}Services:${NC}"
echo "  HTTP API:      http://localhost:${HTTP_PORT}/v1/chat/completions"
echo "  Health:        http://localhost:${CLASSIFY_PORT}/health"
echo "  Dashboard:     http://localhost:8700"
echo ""
echo -e "${BOLD}Observability:${NC}"
echo "  Jaeger:        http://localhost:16686"
echo "  Prometheus:    http://localhost:9090"
echo "  Grafana:       http://localhost:3000"
echo ""
echo -e "${BOLD}Commands:${NC}"
echo "  Stop router:   ./scripts/stop_router.sh"
echo "  View logs:     docker logs -f $CONTAINER_NAME"
echo ""

# Quick test
log_info "Running quick routing test..."
TEST_RESULT=$(curl -s -X POST "http://localhost:${HTTP_PORT}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d '{"model": "MoM", "messages": [{"role": "user", "content": "test"}], "max_tokens": 5}' 2>/dev/null || echo "")

if echo "$TEST_RESULT" | grep -q '"model"'; then
    ROUTED_MODEL=$(echo "$TEST_RESULT" | grep -o '"model":"[^"]*"' | head -1 | cut -d'"' -f4)
    log_success "Routing test passed (routed to: $ROUTED_MODEL)"
else
    log_warning "Routing test inconclusive - router may still be initializing"
fi

echo ""
