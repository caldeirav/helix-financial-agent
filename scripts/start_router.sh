#!/bin/bash
# =============================================================================
# vLLM Semantic Router Startup (Direct Docker Mode)
# =============================================================================
# This script starts the vLLM-SR container directly with correct port mappings.
# The vllm-sr CLI has port mapping issues, so we run docker directly.
#
# USAGE:
#   ./scripts/start_router.sh           # Start and show status, then exit
#   ./scripts/start_router.sh -f        # Start and follow logs (recommended)
#   ./scripts/start_router.sh --follow  # Same as -f
#   ./scripts/start_router.sh --detach  # Start in background (no log follow)
#
# SSH PORT FORWARDING (for remote access from main computer):
# If running on a remote machine (e.g., DGX Spark, ZGX Nano), you can access
# the dashboard and services from your main computer by setting up SSH tunnels.
#
# Run this on your MAIN COMPUTER (not the server):
#   ssh -L 8801:localhost:8801 \
#       -L 8889:localhost:8889 \
#       -L 8700:localhost:8700 \
#       -L 16686:localhost:16686 \
#       -L 9090:localhost:9090 \
#       -L 3000:localhost:3000 \
#       -L 9190:localhost:9190 \
#       user@your-server-ip
#
# Or use the helper script: ./scripts/ssh_port_forward.sh user@your-server-ip
# =============================================================================

set -e

# Parse command line arguments
FOLLOW_LOGS=false
for arg in "$@"; do
    case $arg in
        -f|--follow)
            FOLLOW_LOGS=true
            shift
            ;;
        --detach)
            FOLLOW_LOGS=false
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  -f, --follow    Follow container logs after starting (recommended)"
            echo "  --detach        Start in background without following logs"
            echo "  -h, --help      Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0 -f           Start router and follow logs"
            echo "  $0 --detach     Start router in background"
            exit 0
            ;;
    esac
done

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

# Ports (from .env or defaults)
HTTP_PORT="${ROUTER_HTTP_PORT:-8801}"
CLASSIFY_PORT="${ROUTER_CLASSIFY_PORT:-8889}"
METRICS_PORT="${ROUTER_METRICS_PORT:-9190}"

echo "Starting vLLM Semantic Router..."
echo "  HTTP API:    http://localhost:${HTTP_PORT}"
echo "  Classify:    http://localhost:${CLASSIFY_PORT}"
echo "  Metrics:     http://localhost:${METRICS_PORT}"
echo ""

# Generate config if not exists
if [ ! -f "$CONFIG_PATH" ]; then
    echo "Generating router configuration..."
    python -c "from helix_financial_agent.router.config import create_router_config; create_router_config()"
fi

# Create .vllm-sr directory for state
mkdir -p "$VLLM_SR_DIR"

# Preprocess config to resolve access_key_env references
# This converts access_key_env: "VAR_NAME" to access_key: "actual_value"
PROCESSED_CONFIG="$VLLM_SR_DIR/processed_config.yaml"
echo "Preprocessing config to resolve access_key_env..."
python3 << 'PYEOF'
import os
import sys
import yaml

config_path = os.environ.get('CONFIG_PATH', 'config/router_config.yaml')
output_path = os.environ.get('PROCESSED_CONFIG', 'config/.vllm-sr/processed_config.yaml')

with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

# Process providers.models to resolve access_key_env
models = config.get('providers', {}).get('models', [])
for model in models:
    if 'access_key_env' in model:
        env_var = model.pop('access_key_env')
        value = os.environ.get(env_var)
        if value:
            model['access_key'] = value
            print(f"  Resolved {env_var} for model '{model.get('name')}'")
        else:
            print(f"  WARNING: {env_var} not set for model '{model.get('name')}'", file=sys.stderr)

# Write processed config
with open(output_path, 'w') as f:
    yaml.dump(config, f, default_flow_style=False, sort_keys=False)

print(f"  Wrote processed config to {output_path}")
PYEOF

# Use processed config if preprocessing succeeded
if [ -f "$PROCESSED_CONFIG" ]; then
    CONFIG_PATH="$PROCESSED_CONFIG"
    echo "  Using processed config: $CONFIG_PATH"
fi

# Cleanup existing containers
cleanup_container() {
    local name=$1
    if docker ps -a --format '{{.Names}}' | grep -q "^${name}$"; then
        echo "Stopping and removing existing container: $name"
        docker stop "$name" 2>/dev/null || true
        docker rm "$name" 2>/dev/null || true
    fi
}

cleanup_container "$CONTAINER_NAME"
cleanup_container "vllm-sr-jaeger"
cleanup_container "vllm-sr-prometheus"
cleanup_container "vllm-sr-grafana"

# Create network if not exists
if ! docker network ls --format '{{.Name}}' | grep -q "^${NETWORK_NAME}$"; then
    echo "Creating Docker network: $NETWORK_NAME"
    docker network create "$NETWORK_NAME"
fi

# Start observability stack (optional - comment out if not needed)
echo ""
echo "Starting observability stack..."

# Jaeger
echo "  Starting Jaeger..."
docker run -d --name vllm-sr-jaeger \
    --network "$NETWORK_NAME" \
    -p 16686:16686 \
    -p 4318:4317 \
    jaegertracing/all-in-one:latest 2>/dev/null || echo "    (already running or port in use)"

# Prometheus  
echo "  Starting Prometheus..."
docker run -d --name vllm-sr-prometheus \
    --network "$NETWORK_NAME" \
    -p 9090:9090 \
    prom/prometheus:latest 2>/dev/null || echo "    (already running or port in use)"

# Grafana
echo "  Starting Grafana..."
docker run -d --name vllm-sr-grafana \
    --network "$NETWORK_NAME" \
    -p 3000:3000 \
    grafana/grafana:latest 2>/dev/null || echo "    (already running or port in use)"

echo ""
echo "Starting vLLM Semantic Router container..."

# Build environment variables to pass
ENV_VARS=""
[ -n "$HF_TOKEN" ] && ENV_VARS="$ENV_VARS -e HF_TOKEN=$HF_TOKEN"
[ -n "$HF_ENDPOINT" ] && ENV_VARS="$ENV_VARS -e HF_ENDPOINT=$HF_ENDPOINT"
[ -n "$GEMINI_API_KEY" ] && ENV_VARS="$ENV_VARS -e GEMINI_API_KEY=$GEMINI_API_KEY"

# Start the main vLLM-SR container
# Port mapping:
#   - HTTP_PORT -> 8801 (Envoy HTTP)
#   - CLASSIFY_PORT -> 8080 (Classification API inside container)
#   - METRICS_PORT -> 9190 (Prometheus metrics)
#   - 8700 -> 8700 (Dashboard)
#   - 50051 -> 50051 (gRPC)
#
# Custom Envoy template mount:
#   The custom envoy.template.yaml in config/.vllm-sr/ adds path_prefix support
#   for external APIs like Gemini that require path rewrites.
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
    "$IMAGE"

if [ $? -eq 0 ]; then
    echo ""
    echo "╔════════════════════════════════════════════════════════════╗"
    echo "║         vLLM SEMANTIC ROUTER STARTED                       ║"
    echo "╚════════════════════════════════════════════════════════════╝"
    echo ""
    echo "Services:"
    echo "  HTTP API (OpenAI-compatible): http://localhost:${HTTP_PORT}/v1"
    echo "  Classification API:           http://localhost:${CLASSIFY_PORT}/health"
    echo "  Metrics:                      http://localhost:${METRICS_PORT}/metrics"
    echo "  Dashboard:                    http://localhost:8700"
    echo ""
    echo "Observability:"
    echo "  Jaeger UI:                    http://localhost:16686"
    echo "  Prometheus:                   http://localhost:9090"
    echo "  Grafana:                      http://localhost:3000"
    echo ""
    echo "Commands:"
    echo "  View logs:   docker logs -f $CONTAINER_NAME"
    echo "  Stop:        docker stop $CONTAINER_NAME"
    echo "  Stop all:    ./scripts/stop_router.sh"
    echo ""
    echo "═══════════════════════════════════════════════════════════════"
    echo "SSH PORT FORWARDING (run on your LOCAL machine to access UIs):"
    echo "═══════════════════════════════════════════════════════════════"
    echo ""
    echo "Or copy-paste this command on your local machine:"
    LOCAL_IP=$(hostname -I 2>/dev/null | awk '{print $1}' || echo "YOUR_SERVER_IP")
    echo "ssh -L ${HTTP_PORT}:localhost:${HTTP_PORT} -L ${CLASSIFY_PORT}:localhost:${CLASSIFY_PORT} -L 8700:localhost:8700 -L 16686:localhost:16686 -L 9090:localhost:9090 -L 3000:localhost:3000 -L ${METRICS_PORT}:localhost:${METRICS_PORT} -N vincent@${LOCAL_IP}"
    echo ""
    
    # Follow logs or wait for ready
    if [ "$FOLLOW_LOGS" = true ]; then
        echo "═══════════════════════════════════════════════════════════════"
        echo "FOLLOWING CONTAINER LOGS"
        echo "═══════════════════════════════════════════════════════════════"
        echo ""
        echo "  Press Ctrl+C to detach from logs (container keeps running)"
        echo "  To stop the router completely: ./scripts/stop_router.sh"
        echo ""
        echo "═══════════════════════════════════════════════════════════════"
        echo ""
        # Follow logs - Ctrl+C will exit but container keeps running
        docker logs -f "$CONTAINER_NAME"
    else
        # Wait for container to be healthy
        echo "Waiting for router to be ready (first start may take a few minutes to download models)..."
        for i in {1..60}; do
            if curl -s "http://localhost:${CLASSIFY_PORT}/health" > /dev/null 2>&1; then
                echo ""
                echo "✓ Router is ready!"
                echo ""
                echo "To follow logs:  ./scripts/start_router.sh -f"
                echo "To stop router:  ./scripts/stop_router.sh"
                exit 0
            fi
            sleep 1
            echo -n "."
        done
        echo ""
        echo ""
        echo "Note: Router is still starting (downloading ML models on first run)."
        echo "This is normal - models are ~1.5GB and take a few minutes to download."
        echo ""
        echo "To monitor progress:  docker logs -f $CONTAINER_NAME"
        echo "To stop router:       ./scripts/stop_router.sh"
    fi
else
    echo "ERROR: Failed to start container"
    echo "Check Docker logs for details"
    exit 1
fi
