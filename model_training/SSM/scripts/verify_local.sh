#!/usr/bin/env bash
# Local verification battery (CPU): math equivalence, pipeline smoke,
# ONNX export parity, CPU latency benchmark vs quadratic attention.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-venv/bin/python}

echo "==[1/4]== SSD chunk form vs exact stepwise recurrence"
$PY -m tests.test_ssd_equiv

echo "==[2/4]== pipeline smoke (featurize -> sort -> pad -> model -> OC loss)"
$PY -m tests.test_pipeline_smoke

echo "==[3/4]== ONNX export + ORT parity at dynamic lengths"
$PY -m ssm.export_onnx --random-init --out /tmp/tracker_ssm.onnx

echo "==[4/4]== ONNX CPU benchmark (SSM vs quadratic attention reference)"
$PY -m ssm.benchmark_onnx --ssm-onnx /tmp/tracker_ssm.onnx --n 1000 3000 6000 10000
