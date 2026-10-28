# SSM track finder (Mamba-2 / SSD, ONNX-CPU-first)

Independent re-implementation of the GATr tracking arm with a structured
state-space backbone. Same Parquet inputs, same object-condensation loss
semantics, same `{embedding, beta}` per-hit output contract as the GATr/CIRCE
arms — but no geometric algebra, no graphs, no xFormers, and a circuit that
exports cleanly to ONNX Runtime CPU.

## Layout

```
ssm/model.py        TrackerSSM: SSD chunk-form blocks -> per-hit [embedding(4), beta_logit(1)]
ssm/oc_loss.py      OC loss, torch-native port (identical math to shared GATr/CIRCE loss,
                    no torch_scatter dependency)
ssm/dataset.py      polars Parquet loader + featurization + ordering (Morton spatial or phi)
ssm/train.py        plain-PyTorch training loop (AdamW, bf16 autocast, plateau LR)
ssm/export_onnx.py  dynamic-N ONNX export + ORT parity check
ssm/benchmark_onnx.py  ORT CPU latency vs quadratic-attention reference
tests/              SSD equivalence + pipeline smoke tests
setup_env.sh        venv creation (cpu or CUDA wheel); requirements.txt lists deps
scripts/verify_local.sh  full local verification battery
```

## Environment (no singularity; plain venv)

Connected machine: `./setup_env.sh cpu` (or `cu126`).

**Offline GPU machine (no internet):**

```bash
# 1) on a machine WITH internet, build the wheelhouse:
./download_wheels.sh <python_X.Y> <cpu|cu124|cu126>     # example: ./download_wheels.sh 3.9 cpu
#    target python 3.9/3.8 => torch pinned to 2.5.1 (no newer cp39 linux wheels);
#    python >=3.10 => torch 2.8.0
#    FOR THIS PROJECT'S GPU MACHINE (python 3.10.14, driver CUDA 13.2):
#    ./download_wheels.sh 3.10 cu126   (cu126 binaries are driver-13-compatible)

# 2) transfer SSM/ to the GPU machine EXCLUDING venv/ (venvs are not relocatable):
rsync -a --exclude venv --exclude __pycache__ SSM/ target:/path/SSM/
./setup_env_offline.sh   # wipes any stale venv/, creates a fresh one offline
source venv/bin/activate
```

The wheelhouse (269 MB for cpu, ~4 GB for CUDA flavors) is validated here by an
offline install + the full test battery (SSD equivalence, smoke, ONNX parity).

## Training (on the GPU machine)

```bash
venv/bin/python -m ssm.train \
  --train-data /path/to/train_folder \
  --val-data /path/to/val_folder \
  --gpus 0,1 --out /path/to/run_dir
```

Data folders: `--train-data`/`--val-data` point directly at folders of parquet
files; layout is auto-detected per file (supported: flat per-hit tables with
`event_id` — CIRCE/CGA-style; and per-event tables with list columns
`hit_x/y/z, leftPosition_*, rightPosition_*, hit_type, hit_particle_index` —
GATr Graphs-style). File names are irrelevant; `--split train` only filters
file names containing `_train.` and falls back to ALL files if none match
(pass `--split '' --val-split ''` for tag-less folders). Diagnose any load
failure with `venv/bin/python -m ssm.inspect_schema <folder>`.

Useful flags: `--order-mode {spatial,type_phi}`, `--bidirectional`,
`--chunk-size`, `--d-model --n-layer --n-heads --d-state`, OC knobs
(`--qmin --s-b --oc-mode ...`). Checkpoints: `best.ckpt`, `last.ckpt`,
metrics `history.json`.

## Slurm + wandb (offline)

```bash
# submit (env-configurable; see header of slurmTrain.sh)
TRAIN_DATA=/path/train_folder VAL_DATA=/path/val_folder sbatch SSM/slurmTrain.sh
```

The sbatch script uses ONLY `SSM/venv` (no conda/other environments), sets
`WANDB_MODE=offline`, and logs runs under `$OUTPUT_DIR/wandb/`. Project defaults
to `IDEA_v4_o1_SSM` — override `WANDB_PROJECT`/`WANDB_ENTITY` (leave ENTITY
unset for your default team). Sync later from a connected machine with
`wandb sync $OUTPUT_DIR/wandb/offline-run-*`. Disable logging entirely with
`LOG_WANDB=0`.

## Export + inference

```bash
venv/bin/python -m ssm.export_onnx --ckpt run_dir/last.ckpt --out tracker.onnx
```

Graph contract: input `features (1, N_pad, 12) fp32` → outputs
`embedding (1, N_pad, 4)`, `beta_logit (1, N_pad, 1)`. N_pad must be a
multiple of 64 (pad with zeros AT THE END; slice outputs to the real hit
count downstream). Inference-side preprocessing (featurize + Morton sort +
pad) runs host-side and is outside the graph.

## Design decisions and why

- **SSD chunk form (two-level)**: mathematically identical to the Mamba-2
  recurrence (unit-tested to 2e-5 against an exhaustive stepwise scan in
  `tests/test_ssd_equiv.py`), but expressed as batched GEMMs + constant
  causal masks — the only ops ORT CPU runs near BLAS peak.
- **Morton (3D spatial) ordering** default: sequence locality for ALL track
  topologies, including displaced tracks (no origin circle) and
  loopers (phi wraps). Deterministic bit-interleave on (x,y,z); no learned
  hash, no LSH randomness, host-side. `--order-mode type_phi` keeps the
  primary-optimal azimuthal sort for ablation.
- **End-padding contract**: causality makes trailing zeros invisible to
  real-hit outputs, so no attention-style masks anywhere (fewer ops,
  simpler graph, dynamic N trivially).
- **fp32 state paths** inside the model regardless of autocast: selective
  scan cumsums are numerically fragile in half precision; projections may
  later be INT8-quantized with the state left fp32.
- **`--bidirectional`**: second scan over the reversed sequence (set data
  has no natural direction). Implemented bias-free so padding stays
  exactly zero on the reversed front and outputs stay exact.

## Measured ONNX-CPU latency (ORT 1.19, 8 threads, single fp32 event)

| hits/event | SSM ms/event | quadratic-attention reference (same width/params) | speedup |
|---|---|---|---|
| 1,000  |  35 |   47 | 1.3× |
| 3,000  |  94 |  226 | 2.4× |
| 6,000  | 207 |  827 | 4.0× |
| 10,000 | 327 | 2,138 | 6.5× |
| 15,000 | 507 | 3,780 | 7.5× |
| 20,000 | 567 | 11,102 | 19.6× |

SSM is linear in N; the reference is quadratic + memory-bound. GATr's own
ONNX graph (PGA ops on top of attention) is strictly worse than this
attention baseline.

## Verification status (this machine, CPU-only)

- [x] chunked SSD == stepwise recurrence (2e-5)
- [x] featurize/sort/pad/model/loss pipeline smoke
- [x] ONNX export, dynamic lengths verified vs PyTorch (<1e-4)
- [x] bidirectional variant exports + parity checked
- [ ] training on real data (run on the GPU machine)
- [ ] physics parity vs GATr arm (tracking efficiency / fake rate)
