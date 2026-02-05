#!/bin/bash
# =============================================================================
# SSH Port Forwarding for Web UIs
# =============================================================================
# Run this script on your LOCAL machine (laptop/desktop) to access web
# interfaces running on the remote server (ZGX Nano / DGX Spark).
#
# Usage:
#   ./scripts/ssh_port_forward.sh <user>@<host>
#
# Forwards web UI ports:
#   - Semantic Router Hub UI:  http://localhost:8180 (remote 8080 → local 8180)
#   - MLflow UI:               http://localhost:5000
#
# Note: Uses local port 8180 instead of 8080 to avoid conflicts with
# Cursor IDE, which uses port 8080 for its internal features.
# =============================================================================

if [ -z "$1" ]; then
    echo "Usage: $0 <user>@<host>"
    echo ""
    echo "Example: $0 john@my-server.local"
    echo ""
    echo "Run this on your LOCAL machine to forward web UI ports."
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
echo "  Semantic Router Hub UI:  http://localhost:8180"
echo "  MLflow UI:               http://localhost:5000"
echo ""
echo "To start MLflow UI on the server:"
echo "  mlflow ui --host 0.0.0.0"
echo ""
echo "Press Ctrl+C to close the tunnel."
echo ""

# Set up SSH tunnel for web UIs
# Uses local port 8180 for Router Hub to avoid conflict with Cursor IDE (port 8080)
ssh -L 8180:localhost:8080 \
    -L 5000:localhost:5000 \
    -N "$SSH_TARGET"
