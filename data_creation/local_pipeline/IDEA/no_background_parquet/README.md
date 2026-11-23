# IDEA local no-background Parquet production

Local (non-Condor) variant of
`data_creation/condor_pipeline/IDEA/noBackground_parquet`. The per-seed
production sequence is identical, so the output
`graph/Graphs_<seed>_<train-or-test>.parquet` files carry the same shared
CIRCE/GATr schema (`shared_schema_version=1`, `file_number` per seed) and can
be used interchangeably with Condor-produced files. As in the Condor variant,
the EDM4hep simulation and digitization products remain ROOT files; only the
Parquet graph file is the training artifact.

Differences from the Condor pipeline:

- `src/run_jobs.py` runs `src/runSequence.sh` directly on this machine instead
  of submitting it with `condor_submit`. There is no accounting group or job
  flavour; `--nprocs` controls how many seeds run in parallel.
- Per-seed logs go to `std/local_<seed>.out` / `std/local_<seed>.err` in the
  pipeline directory.
- Exit code 75 (failed Key4hep/CVMFS setup) is not retried automatically;
  rerun the same command and only missing outputs are produced.
- The unused legacy `IDEA_o1_v03` steering/digitizer files and the orphaned
  `tools_tree.py` of the Condor copy are not carried over.

`src/process_tree.py`, `utils/SteeringFile_IDEA_o1_v04.py` and
`utils/runIDEA_v4o1_trackerDigitizer.py` are copies of the Condor pipeline
files. Keep them in sync when the production sequence changes.

## Prerequisites

A machine with CVMFS access to the Key4hep nightlies (the sequence sources
`/cvmfs/sw-nightlies.hsf.org/key4hep/setup.sh` itself). No HTCondor needed.

## Example run

```bash
cd data_creation/local_pipeline/IDEA/no_background_parquet
python src/run_jobs.py \
  --outdir /path/to/output \
  --minseed 1 \
  --maxseed 100 \
  --type train \
  --key4hep_mode lcg \
  --key4hep_version latest \
  --nprocs 4
```

## Key4hep stacks

- `--key4hep_mode spack` (default, identical to the Condor pipeline): the
  frozen spack nightlies. `--key4hep_version` is a date; available dates are
  listed by `source /cvmfs/sw-nightlies.hsf.org/key4hep/setup.sh --spack
  --list-releases` (newest is `2026-10-20`; spack nightlies are no longer
  built and this mode will eventually stop working).
- `--key4hep_mode lcg`: the rolling LCG `devkey-head` view (the setup default
  on AlmaLinux 9). `--key4hep_version` is a weekday `Mon`..`Sat` or `latest`;
  there are no Sunday builds and no release listing. Updated continuously, so
  reproducible frozen production prefers `spack` while it lasts.

If the requested release does not exist the setup prints the available ones
and the job exits with code 75.

The driver creates `graph/` and `digi/` under `--outdir` and considers only
the exact `graph/Graphs_<seed>_<type>.parquet` output complete, so a stale ROOT
graph file or an interrupted `.parquet.tmp` file is rerun. Seeds are
independent: a failure does not stop the batch, and rerunning the same command
retries only the seeds whose Parquet file is missing.
