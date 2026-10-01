# IDEA no-background Parquet production

This is an independent copy of `noBackground` that writes the final graph
dataset directly as Parquet. The EDM4hep simulation and digitization products
remain ROOT files because those applications require ROOT; only
`graph/Graphs_<seed>_<train-or-test>.parquet` is the training artifact.

`src/process_tree.py` writes incrementally in 25-event row groups using
Zstandard level 1. Incremental output bounds conversion memory and the row
groups allow the unified `model_training/GATR` loader to skip unrelated events
during fractional reads.
Each event also stores `file_number`, which the tracking graph builder needs.

Example submission:

```bash
cd data_creation/condor_pipeline/IDEA/noBackground_parquet
python src/submit_jobs.py \
  --outdir /path/to/output \
  --mainDir "$PWD" \
  --minseed 1 \
  --maxseed 100 \
  --type train \
  --key4hep_version 2026-07-29
```

The submission scripts consider only the exact `.parquet` output complete, so
a stale ROOT graph file or interrupted `.parquet.tmp` file will be resubmitted.
