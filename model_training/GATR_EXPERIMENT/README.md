# GATr with the shared CIRCE loss (inference-optimisation experiment)

This folder is GATR_CIRCE_LOSS plus inference optimisations; the network,
loss and checkpoints are the same, so checkpoints of either folder convert
with the same tools.

* GPU (training and validation): `--compile` (with
  `--compile-mode default|max-autotune-no-cudagraphs`) runs the GATr network
  through `torch.compile` with static shapes; see `src/models/compiled_gatr.py`.
  The hit count of a batch is padded up to one of
  `--compile-buckets-per-octave` (default 4) sizes per factor of two, each
  compiled once (torch 2.2 mis-compiles symbolic hit counts); padding hits
  form their own attention block and are dropped, so results are unchanged.
  Attention stays eager (xFormers); the block-diagonal mask bypasses the
  compiled graph, opt_einsum is disabled, and gatr's basis caches are filled
  by one eager warm-up. Needs `module load cuda/12.1` on MN5 (Triton's
  `ptxas`). From `../slurmTrain.sh GATR_EXPERIMENT ...` compile is on by
  default (`GATR_COMPILE=0` for eager).
* `--float32-matmul-precision high` runs FP32 matrix multiplications on TF32
  tensor cores (default `highest` = exact FP32; `../slurmTrain.sh
  GATR_EXPERIMENT` uses `high`, `GATR_MATMUL_PRECISION=highest` to disable).
  Training-only: checkpoints and the CPU/ONNX export stay FP32.
* CPU (data production in a Gaudi functional): ONNX Runtime, which uses its
  own thread pool instead of OpenMP. `src/models/onnx_export.py` rewrites a
  loaded checkpoint for export; convert with
  `../../conversionModel/convert_to_onnx_fast.py`.
* `train_dummy_checkpoint.sh 'TRAIN_GLOB' 'VAL_GLOB' OUTPUT_DIR` submits a
  one-GPU, few-batch training (compiled by default, `COMPILE=0` for eager)
  that produces a checkpoint for testing the conversion.

The rest of this README describes the GATR_CIRCE_LOSS training it is based on.

This is the training-only GATr comparison copy. The original `../GATR` tree is
unchanged. This directory retains the GATr architecture, graph/data loader,
vendored runtime, and Lightning training/validation code needed by:

```bash
../train_circe_gatr_shared.sh GATR \
  '/path/train/Graphs_*_train.parquet' \
  '/path/validation/Graphs_*_test.parquet' \
  /path/output 0,1,2,3
```

The training step imports and calls CIRCE's object-condensation loss directly
from `../shared_training/circe_loss.py`; it does not call GATr's composite loss
with selected terms disabled. The auxiliary helix output head of the original
GATr tracking model is removed: the network outputs only the clustering
coordinates and the beta logit. Older checkpoints that still contain the
head resume normally (its tensors are dropped on load). Tracking metrics and plots are the current GATr implementation,
shared with CIRCE from `../shared_training/tracking_metrics.py`.

Both comparison arms use `../shared_training/logging_contract.py` and the
filtered logger in `../shared_training/wandb_logger.py`. Consequently their
W&B runs expose the same canonical configuration fields, scalar metric keys,
and operating-point sweep images; GATr-only diagnostics are not uploaded.

The common pT, displacement, and theta efficiency plots compare
double-majority matching with threshold-free one-to-one Hungarian matching.
The displacement axis uses uniform 20 mm bins over 0--2000 mm and the theta
axis uses uniform 5-degree bins over 10--170 degrees. Truth cuts
($N_\mathrm{hits} \geq 3$, 10° < $\theta$ < 170°, genStatus in {0,1}) are
post-reconstruction selections applied only when filling the curves.
