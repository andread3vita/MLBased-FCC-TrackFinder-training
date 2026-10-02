# Matched CIRCE/GATr training

Activate the existing GATr environment once, then select exactly one model per
invocation. Both invocations use the same Python executable and the same
physical train/validation Parquet files:

```bash
./train_circe_gatr_shared.sh CIRCE \
  '/path/train/Graphs_*_train.parquet' \
  '/path/validation/Graphs_*_test.parquet' \
  /path/to/output 0,1,2,3

./train_circe_gatr_shared.sh GATR \
  '/path/train/Graphs_*_train.parquet' \
  '/path/validation/Graphs_*_test.parquet' \
  /path/to/output 0,1,2,3
```

For a one-batch smoke training followed by one validation batch and a saved
checkpoint, use the model-selectable quick wrapper:

```bash
./train_circe_gatr_quick_checkpoint.sh CIRCE \
  '/path/train/Graphs_*.parquet' \
  '/path/validation/Graphs_*.parquet' \
  /path/to/output 0
```

Replace `CIRCE` with `GATR` for the other model. The wrapper creates a unique
run directory and prints the resulting checkpoint path on its final line.

All common architecture, optimizer, loss and metric values are environment
variables in the launcher. The comparison copy is `GATR_CIRCE_LOSS`; `GATR`
is untouched.

Set `LOG_WANDB=1`, and optionally `WANDB_PROJECT` and `WANDB_ENTITY`, to use
the single shared W&B logger implementation for either model. The logger
enforces one metric/config schema: identical loss-component, learning-rate,
validation-loss, working-point, and operating-point-plot keys are uploaded by
both runs. Architecture-specific diagnostics remain local and are deliberately
excluded from W&B so the two dashboards have the same columns and media.

Both models call `shared_training.circe_loss.object_condensation_loss` and
`shared_training.tracking_metrics`. The latter is copied from the current GATr
implementation and supplies greedy clustering, one-to-one double-majority
matching, count-weighted fake rate/tracking efficiency, and operating-point
selection. CIRCE alone turns wire/radius/angle into a full CGA circle; GATr
continues to turn wire plus left/right positions into its PGA inputs.

The shared evaluator supports `idea`, `double_majority`, and `hungarian`
matching. `hungarian` performs a global one-to-one assignment that maximizes
shared hits and requires only positive overlap; it has no 50% efficiency or
purity thresholds. At both selected working points, the pT and displacement
plots contain double-majority and Hungarian curves. Displacement uses uniform
50 mm bins from 0 to 2000 mm. Set `SWEEP_MATCH_METRIC=hungarian` to select
operating points with the Hungarian criterion instead of the default
double-majority criterion.

`circe_parquet_dataset.py` is only CIRCE's view of the canonical rows; keeping
it here avoids changing CIRCE's source-tree structure. GATr continues to use
its existing dataset and graph preprocessing code.
