#!/bin/bash
# Central training launcher. Edit the variables at the top, then:
#   bash run_train.sh           (bare-metal, 4 GPUs)
#   sbatch train.slurm          (SLURM)
set -uo pipefail
cd "$(dirname "$0")"

# ---- Colleague-tunable settings ----
DATA_DIR=${DATA_DIR:-/eos/home-m/mcechovi/projects/cgatr/data_parquet_zqq_uds_v1}   # override via env if needed
OUT_DIR=${OUT_DIR:-checkpoints/cgatr_fcc_prod}
TRAIN_SEEDS=${TRAIN_SEEDS:-1-1000}
VAL_SEEDS=${VAL_SEEDS:-1001-1196}
NUM_EPOCHS=${NUM_EPOCHS:-100}
NUM_DEVICES=${NUM_DEVICES:-4}
MAX_TOKENS=${MAX_TOKENS:-16000}
PRECISION=${PRECISION:-32-true}
GRAD_CKPT=${GRAD_CKPT:-0}
CPU_THREADS=${CPU_THREADS:-4}
NUM_WORKERS=${NUM_WORKERS:-8}
PREFETCH=${PREFETCH:-4}
CKPT_EVERY=${CKPT_EVERY:-200}
LIMIT_VAL=${LIMIT_VAL:-0.15}
WARMUP_EPOCHS=${WARMUP_EPOCHS:-2}
START_LR=${START_LR:-3e-4}

# ---- CPU thread env (propagated to all child processes) ----
export OMP_NUM_THREADS=$CPU_THREADS
export POLARS_MAX_THREADS=$CPU_THREADS
export MKL_NUM_THREADS=$CPU_THREADS
export CGATR_DATALOADER_MP_CTX=spawn
export CGATR_PIN_MEMORY=${CGATR_PIN_MEMORY:-1}
export CGATR_PARQUET_CACHE_SIZE=${CGATR_PARQUET_CACHE_SIZE:-256}
export NCCL_P2P_LEVEL=${NCCL_P2P_LEVEL:-NVL}
export PYTHONPATH=.

GC_FLAG=""
[[ "${GRAD_CKPT}" == "1" ]] && GC_FLAG="--grad_checkpoint"

exec python -u src/train.py \
  --data_dir "$DATA_DIR" \
  --train_seeds "$TRAIN_SEEDS" \
  --val_seeds "$VAL_SEEDS" \
  --num_epochs "$NUM_EPOCHS" \
  --num_devices "$NUM_DEVICES" \
  --max_tokens "$MAX_TOKENS" \
  --max_hits 0 \
  --precision "$PRECISION" \
  --embed_dim 4 \
  --num_blocks 10 \
  --beta_suppress_weight 0.1 \
  --num_workers "$NUM_WORKERS" \
  --prefetch_factor "$PREFETCH" \
  --persistent_workers \
  --cpu_threads "$CPU_THREADS" \
  --start_lr "$START_LR" \
  --warmup_epochs "$WARMUP_EPOCHS" \
  --output_dir "$OUT_DIR" \
  --run_tag cgatr_fcc_prod \
  --ckpt_every_n_train_steps "$CKPT_EVERY" \
  --auto_requeue \
  --resume_ckpt last \
  --init_weights none \
  --limit_val_batches "$LIMIT_VAL" \
  $GC_FLAG \
  "$@"
