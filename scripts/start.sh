#!/bin/bash
set -e

REPO_DIR="/workspace/themepark-microserivce"

# Clone or pull latest code
if [ ! -d "$REPO_DIR" ]; then
    echo "[start] Cloning repo..."
    cd /workspace
    git clone https://github.com/texhy/themepark-microserivce.git
fi
cd "$REPO_DIR"
echo "[start] Pulling latest code..."
git pull

# Install dependencies
echo "[start] Installing GPU dependencies..."
pip install --no-cache-dir -r requirements-gpu.txt

# Set LD_LIBRARY_PATH so onnxruntime can find cuDNN 9 + cuBLAS from pip packages
CUDNN_LIB=$(python -c "import nvidia.cudnn; import os; print(os.path.join(os.path.dirname(nvidia.cudnn.__file__), 'lib'))" 2>/dev/null || echo "")
CUBLAS_LIB=$(python -c "import nvidia.cublas; import os; print(os.path.join(os.path.dirname(nvidia.cublas.__file__), 'lib'))" 2>/dev/null || echo "")

if [ -n "$CUDNN_LIB" ]; then
    export LD_LIBRARY_PATH="${CUDNN_LIB}:${CUBLAS_LIB}:${LD_LIBRARY_PATH:-}"
    echo "[start] LD_LIBRARY_PATH set: $LD_LIBRARY_PATH"
fi

# Ensure persistent dirs exist
export MODEL_DIR="${MODEL_DIR:-/workspace/models}"
export FAISS_INDEX_DIR="${FAISS_INDEX_DIR:-/workspace/faiss_indexes}"
mkdir -p "$MODEL_DIR" "$FAISS_INDEX_DIR"

echo "[start] MODEL_DIR=$MODEL_DIR"
echo "[start] FAISS_INDEX_DIR=$FAISS_INDEX_DIR"
echo "[start] Starting uvicorn..."

exec uvicorn app.main:app --host 0.0.0.0 --port 8000
