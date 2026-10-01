# CIRCE: Conformal Isotropic Reconstruction of Charged Elements (FCC Track Finding)

## What this is

Self-contained training and evaluation package for the **CIRCE** (Conformal Geometric Algebra $Cl(4,1)$) track-finding model on the FCC IDEA drift chamber.

## Quick start

```bash
# 1. Environment setup (PyTorch 2.2+, CUDA, PyTorch Lightning, torch-scatter)
bash setup_env.sh

# 2. Run training across 4 GPUs with champion Pareto configuration
DATA_DIR=/path/to/july2026-zuds-parquet OUT_DIR=checkpoints/circe_production NUM_DEVICES=4 bash run_train.sh

# 3. Full FCC benchmark evaluation on keepAll holdout (produces all benchmark plots)
N_GPUS=4 DATA_DIR=/path/to/eval-keepall bash run_eval.sh checkpoints/circe_production/last.ckpt
```

## Champion Objective & Hyperparameters

A systematic 5-way factorial ablation over 50,000 matched events established CIRCE's Pareto-optimal configuration:
- **Architecture:** $Cl(4,1)$ Conformal Geometric Algebra, $E(3)$ equivariant basis (20 linear maps, verified mirror residual $< 1.1 \times 10^{-15}$), 10 blocks, 16 multivector + 64 scalar channels, `embed_dim = 4`.
- **Drift Hit Representation:** Measured circle encoding (wire center, wire direction unit vector, drift radius), preserving spatial curvature and eliminating discrete left/right point ambiguity.
- **Loss:** Compact-support Kieseler hinge repulsion (`max(0, 1 - d)`), $q_\text{min} = 3.0$, $\text{attr\_weight} = 1.0$, $\text{repul\_weight} = 2.0$, $\beta_\text{suppress} = 0.1$, and $\text{var\_weight} = 0.3$.
- **Clustering Operating Point:** Greedy clustering at $t_\beta = 0.60, t_d = 0.10$.

## Benchmark Performance on `keepAllParticles` (50,000 events)

Evaluated on the full 100-seed `eval-keepall` holdout (1,672,188 targets) under standard benchmark definitions ($15^\circ < \theta < 165^\circ, p_\mathrm{T} > 0.1$ GeV at $t_\beta = 0.60, t_d = 0.10$):

| Metric | Selection / Condition | CIRCE (July Production) | Benchmark Target | Status |
|---|---|:---:|:---:|:---:|
| **Tracking Efficiency ($N_\mathrm{hits} > 10$)** | Standard IDEA benchmark tracks | **97.28%** | $> 90.0\%$ | **Exceeded (+7.28%)** |
| **Tracking Efficiency ($N_\mathrm{hits} > 3$)** | Inclusive track recovery down to 4 hits | **96.12%** | — | High inclusive recovery |
| **All-Track Efficiency ($N_\mathrm{hits} > 10$)** | All reconstructable tracks across detector volume | **93.88%** | $> 90.0\%$ | **Exceeded (+3.88%)** |
| **All-Track Efficiency ($N_\mathrm{hits} > 3$)** | Inclusive tracks across detector volume | **91.58%** | — | Robust recovery |
| **Fake Rate** | Unmatched non-merged candidates / all candidates | **3.74%** | $< 8.0\%$ | **Exceeded (2.1x lower)** |
| **Merge Rate** | Multi-track candidate coverage ($>75\%$ purity) | **12.50%** | — | Clean separation |
| **Candidates / Event** | Full detector acceptance | **36.14** | — | Clean multiplicity |

Benchmark plots are available in `plots/`:
- `plots/head_to_head_keepall_efficiency.png` (and `.pdf`): Tracking Efficiency vs $p_\mathrm{T}$ and Polar Angle $\theta$.
- `plots/fcc_comprehensive_suite.png` (and `.pdf`): 4-panel comprehensive evaluation suite ($p_\mathrm{T}$ turn-on, angular coverage, hit multiplicity, summary bar chart).

## Data path

Pre-filled in both `run_train.sh` and `train.slurm`:
```
/eos/home-m/mcechovi/projects/cgatr/data_parquet_zqq_uds_v1
```
This directory must contain `seed_*/` subdirectories (seeds 1–1196).
Override at runtime if your mount point differs:
```bash
DATA_DIR=/your/path sbatch train.slurm
# or bare-metal:
DATA_DIR=/your/path NUM_DEVICES=4 bash run_train.sh
```

## Training

**SLURM (recommended for 4xH100):**
```bash
sbatch train.slurm
```

**Bare-metal (4 GPUs directly):**
```bash
NUM_DEVICES=4 bash run_train.sh
```

Training checkpoints every 200 steps. Auto-resumes on SLURM requeue with
`--resume_ckpt last` (picks up the latest `last.ckpt` or `last-v*.ckpt`).

## Tunables

| Variable | Default | Description |
|---|---|---|
| `MAX_TOKENS` | 16000 | Packed-batch token budget (total hits/batch). Lower it if you hit GPU OOM; raise it (memory permitting) for better utilisation. Not a data cap — events larger than the budget are kept as singleton batches. |
| `CPU_THREADS` | 4 | OMP/MKL/POLARS thread count. `run_eval.sh` uses half this value per shard (intentional: shards run in parallel). |
| `GRAD_CKPT` | 0 | Set to 1 for gradient checkpointing (~30% slower, saves VRAM). Enable it if you want to push `MAX_TOKENS` beyond what your GPU memory allows. |
| `NUM_EPOCHS` | 100 | Training epochs |
| `PRECISION` | 32-true | PyTorch precision (`32-true`, `bf16-mixed`) |
| `LIMIT_VAL` | 0.15 | Fraction of validation batches per epoch |
| `WARMUP_EPOCHS` | 2 | LR warmup duration |
| `START_LR` | 3e-4 | Peak learning rate |

## Evaluation

```bash
N_GPUS=4 bash run_eval.sh checkpoints/cgatr_fcc_prod/last.ckpt
```

The eval pipeline runs in 5 stages:
1. Sharded GPU forward pass (one shard per GPU)
2. Merge shards, build `mc_signal.parquet`
3. Greedy clustering + truth matching
4. Plot unmerged metrics
5. Oracle-merge at T=0.50/0.65/0.75 + plot

## Results

- **Unmerged**: `eval_results/<tag>/fcc_unmerged/plots/eff_vs_pt_idea.png`
  and `fake_rate_summary.png`
- **Oracle-merged**: `eval_results/<tag>/fcc_oracle_T*/plots/eff_vs_pt_idea.png`


## Gradient checkpointing

Set `GRAD_CKPT=1` in `run_train.sh` (or pass `--grad_checkpoint` to `src/train.py`)
if you encounter OOM at high token budgets. This is ~30% slower
but saves large amounts of activation memory.

## Environment notes

- PyTorch 2.5.1 + CUDA 12.1 (`cu121`)
- `torch_scatter` must match the torch/CUDA wheel (see `setup_env.sh`)
- H100 requires NVIDIA driver >= CUDA 12.1
- `lightning >= 2.2` for DDP + SIGUSR1 requeue support
