# CGA (conformal) variant of the GGTF track finder

This adds a Conformal Geometric Algebra (Cl(4,1)) version of the transformer
that runs inside the existing GGTF pipeline. Nothing about the pipeline
changes: same data, same loss, same trainer - you just point the network
config at the conformal wrapper.

## Run it

Training, exactly like GATR but with one flag changed:

```bash
torchrun --nproc_per_node=4 -m src.train_lightning \
  --network-config src/models/wrapper/model_tracking_cgatr.py \
  ... everything else as in your usual GATR command ...
```

Works in the same docker image as GATR (`dologarcia/gatr:v9`). The CGA
product tables and equivariant bases are generated at first use, no extra
files needed.

## Reproduce our exact configuration

```bash
# Standard benchmark dataset (seeds 1-180 train, 181-190 val):
./train_circe.sh <parquet_dir> <output_dir> 4 1-180 181-190

# Official July 2026 production dataset (500k events, seeds 1-980 train, 981-1000 val):
./train_circe.sh <july2026_parquet_dir> <output_dir> 4 1-980 981-1000
```

runs the champion Pareto-optimal setup end to end (CIRCE compact hinge loss with
`qmin=3.0`, `repul_weight=2.0`, `beta_suppress=0.1`, `var_weight=0.3`, AdamW with
warm-up and plateau/cosine schedule, 16k-hit token batching). See the script
header for the full settings; `--recipe ggtf` and `--loss-backend ggtf` fall back to the
shared GATR-style training for A/B studies.

Or run directly via python:

```bash
python -u -m src.train_algebra_ab \
  --algebra conformal \
  --loss_backend circe \
  --recipe circe \
  --reference_width \
  --data_dir <parquet_dir> \
  --train_seeds 1-980 --val_seeds 981-1000 \
  --qmin 3.0 \
  --repul_weight 2.0 \
  --beta_suppress_weight 0.1 \
  --var_weight 0.3 \
  --epochs 16 \
  --start_lr 4e-4 \
  --max_tokens 16000 \
  --num_devices 4 \
  --output_dir <output_dir>
```

## Why conformal

Drift-chamber hits are circles (wire position, wire direction, drift radius),
and CGA represents circles, spheres and planes as native objects. Each drift
hit enters the network as its actual measured circle instead of a point or a
point pair, so the left/right ambiguity never has to be resolved upstream.
The encoding provably keeps the wire direction, which the two-point (left,
right) encoding discards.

## What is in this PR

- `src/cgatr/` - the CGA library (layers, primitives, interfaces, tests).
  Equivariance and geometry are covered by `src/cgatr/tests/test_cga.py`.
- `src/models/Cgatr_withModifications.py` - the conformal LightningModule,
  same structure as `Gatr_withModifications.py`. It ships at CIRCE's
  reference width (16 mv channels, ~2M params - a CGA multivector has 32
  components vs PGA's 16, so equal channels is not equal capacity). Pass
  `--capacity-matched` to size it to GATR's 924,488 parameters (within
  1.4%) for algebra A/B runs; `train_algebra_ab.py` does this itself.
- `src/models/wrapper/model_tracking_cgatr.py` - the network config to pass
  via `--network-config`.
- `src/dataset/parquet_ggtf_adapter.py` - feeds parquet datasets (see
  converters below) into the standard DGL graph contract, byte-identical
  inputs for both algebras. Includes a DDP-aware token-budget event sampler
  to prevent NCCL desyncs.
- `src/train_algebra_ab.py` + `src/models/smoke_algebra_ab.py` - a paired
  A/B trainer and a five-minute integration check. The smoke test runs both
  algebras forward and backward through the real loss on real events and
  verifies they see identical inputs:

  ```bash
  python -m src.models.smoke_algebra_ab --data_dir <parquet_dir> --cuda
  ```

- `data_creation/edm4hep_to_parquet.py` - converts digitised edm4hep ROOT to
  the parquet layout the adapter reads (per-seed directories with drift,
  vertex/silicon and MC particle tables, full circle geometry per drift hit).
- `data_creation/edm4hep_to_parquet_lowmem.py` - same output, streaming
  writer. Use this one for keepAllParticles samples: their MC tables are
  ~100k particles per event and the in-memory version needs tens of GB per
  worker (we found out the hard way). Also writes through a .tmp rename so
  interrupted conversions never leave a truncated file.

  ```bash
  python data_creation/edm4hep_to_parquet_lowmem.py \
    --input_dir <dir with seed_N/digi_edm4hep/*.root> \
    --output_dir <parquet_out> --split train
  ```

## Loss and Objective Dynamics

By default the conformal model trains with CIRCE's own objective
(`src/layers/losses_circe.py`): logarithmic attraction, compact Kieseler hinge
repulsion (`max(0, 1 - d)`), charge floor `qmin = 3.0`, non-alpha beta
suppression (`beta_suppress_weight = 0.1`), and within-cluster variance
regularization (`var_weight = 0.3`).

A systematic 5-way factorial ablation over 50,000 matched events established why
these parameters are optimal:
1. **Repulsion geometry dictates $\beta$-suppression:** GGTF uses Gaussian repulsion
   $\exp(-d^2/2)$, whose infinite-range tails provide continuous pushback across
   the entire event. In contrast, Kieseler hinge repulsion is strictly compact
   ($V_\text{rep} \equiv 0$ when $d \ge 1.0$). In drift chamber tracks with 50-150 hits,
   multiple hits along a helix naturally predict large $\beta$. Without suppression
   ($\beta_\text{suppress} = 0.0$), they pull hits into separate sub-clusters: latent
   track spread inflates $2.2\times$ ($0.060 \to 0.131$), separation-to-compactness
   crashes from $17.7 \to 6.9$, cluster collisions surge to $59.5\%$, and tracking
   efficiency drops by $-7.47\%$. $\beta_\text{suppress} = 0.1$ prevents this clone
   pressure without over-penalizing legitimate hit charges.
2. **Charge floor $q_\text{min} = 3.0$:** Baseline $q_\text{min}=0.1$ is too weak for
   curling tracks, while $q_\text{min}=4.5$ pulls hits too far outward, merging
   parallel tracks in dense jet cores. $q_\text{min}=3.0$ yields optimal compactness
   ($p_{50}=0.060$) and separation ratio ($17.7$).
3. **Repulsive weight $2.0$:** Balances the compact-support boundary against the
   logarithmic attraction.

Switch to the GGTF shared loss from the command line for algebra-only comparisons:

```bash
--loss-backend circe   # default: CIRCE's champion objective
--loss-backend ggtf    # shared object-condensation loss, for algebra A/B runs
```

(`train_algebra_ab.py` pins `ggtf` when `--loss_backend ggtf` is specified; the
flag is ignored by the GATR model.)

## Benchmark Tracking Performance

Evaluated on the full `eval-keepall` holdout (100 seeds, 50,000 events, 1.67M targets) under exact benchmark matching definitions ($purity > 75\%$, $15^\circ < \theta < 165^\circ$, $p_\mathrm{T} > 0.1$ GeV at champion operating point $t_\beta=0.6, t_d=0.10$):

| Metric | Selection / Condition | CIRCE (July Production, Ep 4) | Benchmark Target | Status |
|---|---|:---:|:---:|:---:|
| **Tracking Efficiency ($N_\mathrm{hits} > 10$)** | Standard IDEA benchmark tracks | **97.28%** | $> 90.0\%$ | **Exceeded (+7.28%)** |
| **Tracking Efficiency ($N_\mathrm{hits} > 3$)** | Inclusive track recovery down to 4 hits | **96.12%** | — | High inclusive recovery |
| **All-Track Efficiency ($N_\mathrm{hits} > 10$)** | All reconstructable tracks across detector volume | **93.88%** | $> 90.0\%$ | **Exceeded (+3.88%)** |
| **All-Track Efficiency ($N_\mathrm{hits} > 3$)** | Inclusive tracks across detector volume | **91.58%** | — | Robust recovery |
| **Fake Rate** | Unmatched non-merged candidates / all candidates | **3.74%** | $< 8.0\%$ | **Exceeded (2.1x lower)** |
| **Merge Rate** | Multi-track candidate coverage ($>75\%$ purity) | **12.50%** | — | Clean separation |
| **Candidates / Event** | Full detector acceptance | **36.14** | — | Clean multiplicity |

## Checkpoint

A trained checkpoint (IDEA v4 o1, 91 GeV Zqq) is available - it is too large
for the repo, ask us for the link and drop it wherever you like; the wrapper
loads it through the normal Lightning mechanisms.
