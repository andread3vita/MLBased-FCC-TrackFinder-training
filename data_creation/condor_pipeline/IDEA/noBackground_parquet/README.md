# IDEA shared CIRCE/GATr no-background Parquet production

The produced `Graphs_*.parquet` file is the canonical input for both models.
It stores measured planar positions, drift-wire position/radius/angles,
left/right drift ambiguity points, truth associations and particle metadata.
CIRCE constructs full circles in its loader; GATr constructs its existing
midpoint plus left/right-vector representation in its loader. Do not generate
separate model-specific files for a matched comparison.

The schema is tagged with scalar column `shared_schema_version=1`. Training
and validation file lists must be disjoint and are passed unchanged to both
arms of `model_training/train_circe_gatr_shared.sh`.

This is an independent copy of `noBackground` that writes the final graph
dataset directly as Parquet. The EDM4hep simulation and digitization products
remain ROOT files because those applications require ROOT; only
`graph/Graphs_<seed>_<train-val-test>.parquet` is the training artifact.

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
  --key4hep_version 2026-10-20 \
  --accounting-group group_u_FCC.local_gen
```

## Key4hep stacks

- `--key4hep_mode spack` (default): the frozen spack nightlies.
  `--key4hep_version` is a date; available dates are listed by `source
  /cvmfs/sw-nightlies.hsf.org/key4hep/setup.sh --spack --list-releases`
  (newest is `2026-10-20`; spack nightlies are no longer built and this mode
  will eventually stop working).
- `--key4hep_mode lcg`: the rolling LCG `devkey-head` view (the setup default
  on AlmaLinux 9). `--key4hep_version` is a weekday `Mon`..`Sat` or `latest`;
  there are no Sunday builds and no release listing. Updated continuously, so
  reproducible frozen production prefers `spack` while it lasts.

If the requested release does not exist the setup prints the available ones
and the job exits with code 75, which Condor retries.

The submission scripts consider only the exact `.parquet` output complete, so
a stale ROOT graph file or interrupted `.parquet.tmp` file will be resubmitted.
