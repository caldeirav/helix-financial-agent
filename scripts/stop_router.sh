#!/bin/bash
# =============================================================================
# Stop vLLM Semantic Router and Observability Stack
# =============================================================================

echo "Stopping vLLM Semantic Router..."

# Stop main router container
if docker ps -q --filter "name=vllm-sr-container" | grep -q .; then
    echo "  Stopping vllm-sr-container..."
    docker stop vllm-sr-container
    docker rm vllm-sr-container 2>/dev/null
    echo "  ✓ Router stopped"
else
    echo "  Router not running"
fi

# Stop observability containers
for container in vllm-sr-jaeger vllm-sr-prometheus vllm-sr-grafana; do
    if docker ps -q --filter "name=$container" | grep -q .; then
        echo "  Stopping $container..."
        docker stop "$container"
        docker rm "$container" 2>/dev/null
    fi
done

echo ""
echo "All vLLM-SR components stopped."
echo ""
echo "To remove the Docker network: docker network rm vllm-sr-network"
