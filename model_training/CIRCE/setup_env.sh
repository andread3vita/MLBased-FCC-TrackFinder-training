#!/bin/bash
# Run once on the login node to create the cgatr_fcc conda environment.
set -uo pipefail

ENV_NAME=${CONDA_ENV:-cgatr_fcc}
TORCH_VERSION=2.5.1
CUDA_TAG=cu121

echo "[setup] Creating conda env '$ENV_NAME' with Python 3.10..."
conda create -y -n "$ENV_NAME" python=3.10

echo "[setup] Installing PyTorch $TORCH_VERSION+$CUDA_TAG..."
conda run -n "$ENV_NAME" pip install \
    "torch==${TORCH_VERSION}" \
    --index-url "https://download.pytorch.org/whl/${CUDA_TAG}"

echo "[setup] Installing other requirements..."
conda run -n "$ENV_NAME" pip install -r requirements.txt

echo "[setup] Installing torch_scatter (must match torch/CUDA build)..."
conda run -n "$ENV_NAME" pip install torch_scatter \
    -f "https://data.pyg.org/whl/torch-${TORCH_VERSION}+${CUDA_TAG}.html"

echo "[setup] Verifying..."
conda run -n "$ENV_NAME" python -c "
import torch, torch_scatter, lightning, polars, einops
print('torch:', torch.__version__)
print('torch_scatter: OK')
print('lightning:', lightning.__version__)
print('polars:', polars.__version__)
print('CUDA available:', torch.cuda.is_available())
print('env OK')
"
echo "[setup] Done. Activate with: conda activate $ENV_NAME"
