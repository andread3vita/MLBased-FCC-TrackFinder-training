# GATr checkpoint conversion to ONNX and TorchScript

These scripts convert geometry-only tracking checkpoints to either ONNX or a
LibTorch-loadable TorchScript `.pt` file.

The model implementation is always imported directly from the repository's
training source tree:

```text
model_training/GATR/src/gatr_v142
```

Do **not** install `gatr`, `GATr`, or `ggtf` in the conversion environment.  The
exporter prepends the local source directory to `sys.path` and refuses to run
unless the imported package is version 1.4.2 and comes from the path above.

The exported model accepts a float32 tensor with shape `[num_hits, 7]`:

```text
x, y, z, hit_type, vector_x, vector_y, vector_z
```

Its output shape is `[num_hits, clustering_space_dim + 5]`.  The first hit
dimension remains dynamic.

## Create a dedicated Conda environment

Run these commands from the repository root.  This creates a named environment
in Conda's normal environment directory,

```bash
conda create --name ggtf-conversion-v142 python=3.10 pip -y
conda activate ggtf-conversion-v142
python -m pip install --upgrade pip
python -m pip install -r conversionModel/requirements.txt
```

Activate it in later shells with:

```bash
conda activate ggtf-conversion-v142
```

Confirm both the dependency versions and the local GATr import:

```bash
cd conversionModel
python -c 'import torch, onnx, onnxruntime; print(torch.__version__, onnx.__version__, onnxruntime.__version__)'
python -c 'import export_tracking_onnx as e; import gatr; print(gatr.__version__, gatr.__file__)'
```

The second command must print version `1.4.2` and a path below
`model_training/GATR/src/gatr_v142/gatr`.

## Convert and verify ONNX

Always provide a real parquet event during export.  The exporter checks the
ONNX graph, runs ONNX Runtime, reconstructs the seven inference features from
measured parquet columns only, and compares the result with the checkpoint
loaded through the local GATr v1.4.2 implementation.

```bash
cd conversionModel

CHECKPOINT=/path/to/checkpoint.ckpt
PARQUET=/path/to/real_event_file.parquet

python export_tracking_onnx.py \
  "$CHECKPOINT" model.onnx \
  --verify-parquet "$PARQUET" \
  --event-index 0
```

The default acceptance thresholds are `rtol=1e-4` and `atol=1e-4`.  A shape
check alone is not enough: an all-zero tracing tensor can hide numerical
problems.  Export exits unsuccessfully if the real-event outputs do not meet
the thresholds.  Do not deploy an ONNX file from a failed check and do not
loosen the thresholds merely to make a conversion pass.

Use the standalone comparison command to repeat the test or inspect the
maximum and mean absolute differences:

```bash
python comparePythonOnnx.py \
  "$CHECKPOINT" model.onnx "$PARQUET" \
  --event-index 0
```

Both commands use EMA weights by default because those are the weights used
during validation.  Select raw weights consistently with `--weights-source
raw` only when that is intentional.

`--skip-numerical-check` exists for exporter debugging only.  An export made
with that option is not considered verified.

## Convert and verify TorchScript (`.pt`)

```bash
python export_tracking_torchscript.py \
  "$CHECKPOINT" model_torchscript.pt \
  --verify-parquet "$PARQUET" \
  --event-index 0
```

The TorchScript exporter requires bit-for-bit equality between eager PyTorch,
the traced module, and the reloaded module.  With `--verify-parquet`, it also
requires bit-for-bit equality on the selected real event.

## Real-event feature construction

`event_features.py` implements only inference preprocessing; it does not load
truth particles or create a DGL graph.  It reproduces the training convention:

- vertex hits (`hit_type == 1`) use `(hit_x, hit_y, hit_z)` and a zero vector;
- drift-chamber hits (`hit_type == 0`) use the midpoint of the left/right
  candidate positions and half of their displacement;
- vertex hits are placed before drift-chamber hits;
- rows containing non-finite measured values are removed.

Consequently the conversion environment does not need Lightning, DGL,
torch-scatter, torchdata, or an installed GGTF/GATr package.

## Weight and checkpoint restrictions

- Geometry-only checkpoints are supported.
- Detector-feature checkpoints are rejected because dropping trained inputs
  would change the network.
- The checkpoint supplies clustering dimension, GATr width/depth, position
  scale, helix-head setting, and other model hyperparameters.
