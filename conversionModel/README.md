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