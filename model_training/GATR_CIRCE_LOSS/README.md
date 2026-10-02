# GATr with the shared CIRCE loss

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
with selected terms disabled. The unchanged GATr architecture still contains
its auxiliary helix output head, but the comparison launcher freezes that
unused head. Tracking metrics and plots are the current GATr implementation,
shared with CIRCE from `../shared_training/tracking_metrics.py`.

Both comparison arms use `../shared_training/logging_contract.py` and the
filtered logger in `../shared_training/wandb_logger.py`. Consequently their
W&B runs expose the same canonical configuration fields, scalar metric keys,
and operating-point sweep images; GATr-only diagnostics are not uploaded.

The common pT and displacement efficiency plots compare double-majority
matching with threshold-free one-to-one Hungarian matching. The displacement
axis uses uniform 50 mm bins over 0--2000 mm.
