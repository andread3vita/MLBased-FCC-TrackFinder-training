#!/usr/bin/env bash
# Run THIS on the offline GPU machine after transferring SSM/ (incl. offline_env/).
# Creates a venv and installs everything from the local wheelhouse.
# Activation afterwards:  source venv/bin/activate
set -euo pipefail
cd "$(dirname "$0")"
WHEELS="$(pwd)/offline_env/wheels"
[[ -d "$WHEELS" ]] || { echo "offline_env/wheels missing — run download_wheels.sh on the online machine first"; exit 1; }

rm -rf venv        # never reuse a (possibly transferred) venv: shebangs are absolute
python3 -m venv venv
# --no-index: never touch the network
venv/bin/pip install --no-index --find-links "$WHEELS" --upgrade pip setuptools wheel
venv/bin/pip install --no-index --find-links "$WHEELS" \
  torch onnx onnxruntime polars numpy wandb matplotlib tqdm

venv/bin/python -c "import torch, onnx, onnxruntime, polars, numpy; \
print('SSM offline env ready:', torch.__version__, '| CUDA:', torch.cuda.is_available())"
echo
echo "activate with:  source $(pwd)/venv/bin/activate"
