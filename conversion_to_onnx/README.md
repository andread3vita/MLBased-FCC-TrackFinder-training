# ONNX export & inference

This folder hosts the ONNX export scripts for both model variants:

| File | Model | Notes |
|---|---|---|
| `onnx_conversion.py` | upstream GATr (`Gatr_onnx.ExampleWrapper`) | original flow, input `(N, 7)` |
| `comparePythonOnnx.py` | upstream GATr | torch / ORT parity check |
| **`onnx_export_v35.py`** | **CGATr v35 (Lightning)** | **batched export, input `(B, N, 10)`** |
| **`onnx_parity_v35.py`** | **CGATr v35 (Lightning)** | **per-event torch / ORT parity check** |
| **`eval_fcc_metrics_v35_ort.py`** | **CGATr v35 (Lightning)** | **full FCC-style eval running on ONNX Runtime** |

## CGATr v35 (Lightning) flow

The v35-Lightning artifacts are exported as a **single batched ONNX file**
that accepts `B >= 1` events. The training code used multi-event packing
(`xformers.BlockDiagonalMask`, not exportable), so the deployment surface
mirrors that: one file, one code path. Single-event inference is just a
`B == 1` call with an all-True padding mask.

**Inputs:**
* `features      : float32 [B, N_max, 10]` (5 vertex + 5 drift-chamber slots, see `model_training/src/dataset/parquet_dataset.py::collate_idea_events`)
* `padding_mask  : bool    [B, N_max]`   (`True` = real hit, `False` = padding)

**Output:**
* `coords_and_beta : float32 [B, N_max, embed_dim + 1]` — first `embed_dim` columns are clustering coordinates, last column is the beta logit (caller applies sigmoid). Padded rows carry garbage; slice with `padding_mask`.

`embed_dim` is set at training time and must be passed to the export
(default `4`).

### 0. Prerequisites

Same environment as `model_training/` (PyTorch, `onnx`, `onnxruntime`,
`polars`). On Hotdog use the `hotdog-ml` conda env. The container in
the original GATr flow is **not** required — the v35 scripts run on the
host Python env.

### 1. Export

From the repo root:

```bash
PYTHONPATH=model_training python conversion_to_onnx/onnx_export_v35.py \
    --checkpoint <path/to/cgatr_best.ckpt> \
    --out        <path/to/cgatr_v35.onnx> \
    --embed_dim  4
```

Accepts both Lightning checkpoints (`*.ckpt`, prefers the EMA shadow if
present) and the legacy v35 flat state dict (`cgatr_best.pt`). The export
runs at a sample shape (`B=4`, `N=512`) but `B` and `N` are exposed as
dynamic axes, so the same `.onnx` file works at any shape.

### 2. Parity check (smoke / numerical regression)

```bash
PYTHONPATH=model_training python conversion_to_onnx/onnx_parity_v35.py \
    --checkpoint <ckpt> --onnx <onnx> --embed_dim 4 \
    --data_dir <parquet_root> --val_seeds 181-181 \
    --max_events 50 --max_hits 5000
```

Compares torch (CPU) vs `onnxruntime` (CPU) per-event. Typical observed
diff is ~5e-7 (numerical noise); acceptance threshold is 1e-4.

### 3. Full FCC-style eval via ONNX Runtime

```bash
PYTHONPATH=model_training python conversion_to_onnx/eval_fcc_metrics_v35_ort.py \
    --data_dir <parquet_root> \
    --onnx <onnx> --embed_dim 4 \
    --eval_seeds 181-200 \
    --output_dir eval_results/v35_onnx_fcc \
    --tbeta 0.1 --td 0.05
```

Computes the same metrics as `model_training/src/eval_user_greedy_sweep.py`
(per-track efficiency, per-cluster purity, loose / strict-T match rate,
IDEA / CLD / displaced reconstructable masks, fake rate) but feeds the
forward pass through ORT instead of PyTorch — closes the loop on the
deployment artifact.

### Gotchas the wrapper handles for you

1. **Attention mask.** We build a `[B, 1, N, N]` mask from `padding_mask` and forward it through `torch.nn.functional.scaled_dot_product_attention`, replacing the `xformers.BlockDiagonalMask` path used in training.
2. **Per-event dual reference.** Stock `CGATr._construct_dual_reference` averages over **all** leading dims, mixing events. We replace it with a masked per-event mean so each event matches the single-event torch result.
3. **NaN-safe padded rows.** ORT's decomposed SDPA produces NaN on fully-padded query rows (`0/0` in softmax). We OR-augment the mask with the identity (so every query has at least itself in scope) and `torch.where`-zero padded outputs after every sub-block.
4. **Tracer cleanliness.** Einsums use ellipsis leading dims and channel counts are read from config Python ints rather than tensor `.shape[-1]` SymInts — eliminates the `TracerWarning: Converting a tensor to a Python boolean` noise.

## Upstream GATr flow (legacy)

For the original GATr (input `(N, 7)`) the container-based flow is
preserved:

0. Ensure Apptainer is installed.
1. `singularity pull docker://justdrew/onnxconversion`
2. `apptainer shell onnxconversion_latest.sif`
3. `python onnx_conversion.py -w <ckpt> -o <out_dir>`
