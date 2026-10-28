#!/usr/bin/env bash
# Run THIS on the online machine. Downloads wheels into offline_env/wheels.
# Transfer the whole SSM/ folder (with offline_env/) to the offline GPU machine.
#
# Usage:  ./download_wheels.sh <target_python X.Y> <torch_flavor> [torch_version]
#   target_python : python on the GPU machine, e.g. 3.9 or 3.11
#   torch_flavor  : cpu | cu118 | cu121 | cu124 | cu126  (match GPU machine CUDA)
#   torch_version : optional; py<=3.9 defaults to 2.5.1 (cp39 linux wheels stop
#                   there on the pytorch index), otherwise 2.8.0
# Example: ./download_wheels.sh 3.11 cu126
set -euo pipefail
PYV="${1:?target python version e.g. 3.9}"
FLAVOR="${2:-cpu}"
TORCH_VER="${3:-}"
cd "$(dirname "$0")"
mkdir -p offline_env/wheels

if command -v pip >/dev/null; then PIP="pip";
elif [[ -x venv/bin/pip ]]; then PIP="venv/bin/pip";
else PIP="python3 -m pip"; fi

if [[ -z "$TORCH_VER" ]]; then
  if [[ "$PYV" == 3.9* || "$PYV" == 3.8* ]]; then TORCH_VER="2.5.1";
  else TORCH_VER="2.8.0"; fi
fi
if [[ "$PYV" == 3.9* || "$PYV" == 3.8* ]] && [[ "$TORCH_VER" != 2.[0-5].* ]]; then
  echo "ERROR: python $PYV supports torch only up to 2.5.1 (no cp39 wheels for >=2.6)."
  echo "       Re-run with torch_version 2.5.1 or upgrade the GPU machine python."; exit 1
fi

if [[ "$FLAVOR" == "cpu" ]]; then TORCH_INDEX="https://download.pytorch.org/whl/cpu";
else TORCH_INDEX="https://download.pytorch.org/whl/${FLAVOR}"; fi

# torch's OWN wheel platform: cp3[89] torches are linux_x86_64; <=2.6 linux_x86_64;
# >=2.7 needs manylinux_2_28_x86_64.
if [[ "$PYV" == 3.9* || "$PYV" == 3.8* || "$TORCH_VER" == 2.[0-6].* ]]; then
  TPLAT="linux_x86_64"
else
  TPLAT="manylinux_2_28_x86_64"
fi
# Wheels in this set carry mixed platform tags (manylinux_2_17 / 2_27 / 2_28 /
# linux). pip's --platform matching is EXACT per tag, so list them all.
PLATS=(--platform manylinux2014_x86_64 --platform manylinux_2_27_x86_64 \
       --platform manylinux_2_28_x86_64 --platform linux_x86_64)
MLATS=("${PLATS[@]}" --only-binary=:all: --python-version "$PYV")

echo "downloading for python $PYV, torch==$TORCH_VER ($FLAVOR)"

# 1) venv bootstrap
$PIP download "${MLATS[@]}" -d offline_env/wheels pip setuptools wheel

# 2) torch ISOLATED (its wheel tag conflicts with manylinux deps under one resolver run)
$PIP download --only-binary=:all: --python-version "$PYV" --platform "$TPLAT" \
  --no-deps -d offline_env/wheels "torch==$TORCH_VER" --index-url "$TORCH_INDEX"

# 3) torch runtime deps + everything else (plain manylinux)
SYMPY="sympy"; if [[ "$TORCH_VER" == 2.5.* ]]; then SYMPY="sympy==1.13.1"; fi
$PIP download "${MLATS[@]}" -d offline_env/wheels \
  filelock typing-extensions networkx jinja2 fsspec "$SYMPY" mpmath markupsafe \
  onnx onnxruntime polars numpy wandb

# 4) CUDA runtime wheels for GPU flavors.
# For torch >= 2.7 the closure is RESOLVED (manual lists risk missing wheels
# like nvidia-cuda-nvrtc-cu12): torch's own wheel is downloaded again here but
# pip skips existing identical files.
if [[ "$FLAVOR" != "cpu" ]]; then
  case "$TORCH_VER" in
    2.7.*|2.8.*|2.9.*)
      # both indexes: torch itself is on the pytorch index, newer nvidia pins
      # (e.g. cudnn 9.10) only exist on PyPI
      $PIP download "${PLATS[@]}" --only-binary=:all: --python-version "$PYV" \
        -d offline_env/wheels "torch==$TORCH_VER" \
        --index-url https://pypi.org/simple --extra-index-url "$TORCH_INDEX" ;;
    *) # older torches: manual list is what pip can see
      $PIP download --only-binary=:all: --python-version "$PYV" --platform linux_x86_64 \
        -d offline_env/wheels triton --index-url "$TORCH_INDEX" ;;
  esac
fi

echo "wheelhouse ready in SSM/offline_env/wheels:"
du -sh offline_env/wheels; ls offline_env/wheels | wc -l
