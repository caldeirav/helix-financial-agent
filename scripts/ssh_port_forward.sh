#!/bin/bash
# =============================================================================
# SSH Port Forwarding for Web UIs
# =============================================================================
# Run this script on your LOCAL machine to access web interfaces running
# on a remote ZGX Nano or DGX Spark.
#
# Usage:
#   ./scripts/ssh_port_forward.sh user@server-ip
#   ./scripts/ssh_port_forward.sh vincent@zgx-08af
#
# Forwards web UI ports:
#   - Semantic Router Hub UI:  http://localhost:8080
#   - MLflow UI:               http://localhost:5000
#
# Note: Core services (llama.cpp, MCP, Router API) run locally on ZGX
# and don't need port forwarding.
# =============================================================================

if [ -z "$1" ]; then
    echo "Usage: $0 user@server-ip"
    echo ""
    echo "Example: $0 vincent@zgx-08af"
    echo ""
    echo "Forwards web UI ports for remote access."
    exit 1
fi

SSH_TARGET="$1"

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║     SSH Port Forwarding for Web UIs                            ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""
echo "Connecting to: $SSH_TARGET"
echo ""
echo "Once connected, access these web interfaces locally:"
echo ""
echo "  Semantic Router Hub UI:  http://localhost:8080"
echo "  MLflow UI:               http://localhost:5000"
echo ""
echo "To start MLflow UI on the server:"
echo "  cd /home/vincent/Code/helix-financial-agent"
echo "  source .venv/bin/activate"
echo "  mlflow ui --host 0.0.0.0"
echo ""
echo "Press Ctrl+C to close the tunnel."
echo ""

# Set up SSH tunnel for web UIs only
ssh -L 8080:localhost:8080 \
    -L 5000:localhost:5000 \
    -N "$SSH_TARGET"
