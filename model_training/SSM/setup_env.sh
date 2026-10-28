#!/usr/bin/env bash
# Create the SSM python environment. Usage:
#   ./setup_env.sh [cpu|cu126]   (default: cpu)
set -euo pipefail
FLAVOR="${1:-cpu}"
cd "$(dirname "$0")"
python3 -m venv venv
venv/bin/pip install --upgrade pip
if [[ "$FLAVOR" == "cpu" ]]; then
  venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
else
  venv/bin/pip install torch --index-url "https://download.pytorch.org/whl/${FLAVOR}"
fi
venv/bin/pip install onnx onnxruntime polars numpy
venv/bin/python -c "import torch, onnx, onnxruntime, polars, numpy; \
print('SSM env ready:', torch.__version__, '| CUDA:', torch.cuda.is_available())"
