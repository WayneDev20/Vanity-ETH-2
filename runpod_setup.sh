#!/bin/bash
# Full RunPod setup: build vanity-eth-address + install Python deps
# Run this once when you start a new RunPod instance

set -e

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
VANITY_BIN="$REPO_DIR/vanity-eth-address/vanity-eth-address"

echo "=== RunPod Vanity-ETH Setup ==="
echo "Project dir: $REPO_DIR"
echo ""

# 0. Install nano for editing addresses.txt
echo ">>> Installing nano..."
apt-get update -qq && apt-get install -y nano >/dev/null 2>&1 || true

# 1. Build vanity-eth-address
echo ">>> Building vanity-eth-address (CUDA)..."
cd "$REPO_DIR/vanity-eth-address"
chmod +x runpod_build.sh
./runpod_build.sh
cd "$REPO_DIR"

# 2. Verify binary
if [ ! -f "$VANITY_BIN" ]; then
    echo "Error: Build failed - binary not found"
    exit 1
fi

# 3. Python deps (if not in venv)
echo ""
echo ">>> Installing Python dependencies..."
pip install -q ecdsa pycryptodome tqdm 2>/dev/null || pip3 install -q ecdsa pycryptodome tqdm

# 4. Create addresses.txt if missing
if [ ! -f "$REPO_DIR/addresses.txt" ]; then
    echo ""
    echo ">>> Creating addresses.txt (add your target addresses)"
    cat > "$REPO_DIR/addresses.txt" << 'EOF'
# One address per line - script matches first 4 + last 4 hex chars (after 0x)
# Replace the example below with your target addresses
0x1234abcdef1234567890abcdef1234567890abcd
EOF
    echo "   Edit with: nano addresses.txt"
fi

echo ""
echo "=== Setup complete ==="
echo ""
echo "To run:"
echo "  cd $REPO_DIR"
echo "  export VANITY_GPU=1"
echo "  export VANITY_GPU_CUDA_BIN=$VANITY_BIN"
echo "  python3 vanity_address.py"
echo ""
echo "Or use: ./runpod_run.sh"
