#!/bin/bash
# Build vanity-eth-address on RunPod (or any Linux with CUDA)
set -e

echo "=== Checking environment ==="
nvidia-smi || { echo "Error: nvidia-smi not found. Ensure you're on a GPU instance."; exit 1; }
nvcc --version || { echo "Error: nvcc not found. Install CUDA toolkit."; exit 1; }

echo ""
echo "=== Building vanity-eth-address ==="
cd "$(dirname "$0")"

# Try Makefile first
if [ -f Makefile ]; then
    make clean
    make
else
    # Fallback: direct nvcc (from repo root)
    nvcc src/main.cu -o vanity-eth-address -O3 -Xptxas -v \
        -Xcompiler -static-libgcc -Xcompiler -static-libstdc++ \
        -gencode arch=compute_52,code=compute_52 \
        -gencode arch=compute_75,code=compute_75 \
        -gencode arch=compute_86,code=compute_86 \
        -gencode arch=compute_89,code=compute_89
fi

echo ""
echo "=== Build complete ==="
./vanity-eth-address --help 2>/dev/null | head -5 || true
echo "Binary: $(pwd)/vanity-eth-address"
