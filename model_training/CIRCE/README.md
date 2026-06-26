# C-GATr FCC Track-Finding — Production Package

## What this is

Self-contained training and evaluation package for the C-GATr FCC track-finding model.

## Quick start

```bash
# 1. Unzip
unzip cgatr_fcc_pkg.zip && cd cgatr_fcc_pkg

# 2. Create conda environment (run once on login node, needs internet)
bash setup_env.sh

# 3. Submit training
./run_train.sh --data_dir /eos/experiment/fcc/ee/simulation/key4hep_2026_06_16/IDEA_v4_o1/Loopers/parquet --out_dir . --train_seed 1-800 --val_seed 801-1000

# 4. Monitor
tail -f logs/slurm/cgatr_fcc_<JOBID>.out

# 5. After training: run full evaluation (produces all plots)
N_GPUS=4 bash run_eval.sh checkpoints/cgatr_fcc_prod/last.ckpt
```

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
- `lightning >= 2.2` for DDP + SIGUSR1 requeue support
