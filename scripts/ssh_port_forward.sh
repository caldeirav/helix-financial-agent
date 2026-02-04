#!/bin/bash
# =============================================================================
# SSH Port Forwarding Helper for Remote vLLM Semantic Router Access
# =============================================================================
# Run this script on your LOCAL machine (not the server) to access
# the router dashboard and services from your browser.
#
# Usage:
#   ./scripts/ssh_port_forward.sh user@server-ip
#   ./scripts/ssh_port_forward.sh vincent@192.168.1.100
#
# This will forward all necessary ports so you can access:
#   - Router HTTP API: http://localhost:8801
#   - Classification API: http://localhost:8889
#   - Dashboard: http://localhost:8700
#   - Jaeger UI: http://localhost:16686
#   - Prometheus: http://localhost:9090
#   - Grafana: http://localhost:3000
#   - Metrics: http://localhost:9190
# =============================================================================

if [ -z "$1" ]; then
    echo "Usage: $0 user@server-ip"
    echo ""
    echo "Example: $0 vincent@192.168.1.100"
    echo ""
    echo "This will set up SSH port forwarding to access the router services."
    exit 1
fi

SSH_TARGET="$1"

echo "╔════════════════════════════════════════════════════════════╗"
echo "║     SSH Port Forwarding for vLLM Semantic Router           ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""
echo "Connecting to: $SSH_TARGET"
echo ""
echo "Once connected, you can access:"
echo "  HTTP API:      http://localhost:8801/v1"
echo "  Classify API:  http://localhost:8889/health"
echo "  Dashboard:     http://localhost:8700"
echo "  Jaeger UI:     http://localhost:16686"
echo "  Prometheus:    http://localhost:9090"
echo "  Grafana:       http://localhost:3000 (admin/admin)"
echo "  Metrics:       http://localhost:9190/metrics"
echo ""
echo "Press Ctrl+C to close the tunnel."
echo ""

# Set up SSH tunnel with all necessary ports
ssh -L 8801:localhost:8801 \
    -L 8889:localhost:8889 \
    -L 8700:localhost:8700 \
    -L 16686:localhost:16686 \
    -L 9090:localhost:9090 \
    -L 3000:localhost:3000 \
    -L 9190:localhost:9190 \
    -L 8080:localhost:8080 \
    -N "$SSH_TARGET"
