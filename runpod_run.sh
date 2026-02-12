#!/bin/bash
# Run vanity address generation on RunPod with GPU
# Usage: ./runpod_run.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Resolve CUDA binary path
VANITY_BIN="${VANITY_GPU_CUDA_BIN:-$SCRIPT_DIR/vanity-eth-address/vanity-eth-address}"

if [ ! -f "$VANITY_BIN" ]; then
    echo "Error: vanity-eth-address not found at $VANITY_BIN"
    echo "Run ./runpod_setup.sh first to build."
    exit 1
fi

if [ ! -f "addresses.txt" ]; then
    echo "Error: addresses.txt not found. Add target addresses (one per line)."
    exit 1
fi

export VANITY_GPU=1
export VANITY_GPU_CUDA_BIN="$VANITY_BIN"
export VANITY_GPU_DELAY_SEC=2
export VANITY_GPU_RETRIES=2

echo "Using GPU: $VANITY_BIN"
echo "Input: addresses.txt"
echo "Output: output.csv"
echo ""

python3 vanity_address.py
