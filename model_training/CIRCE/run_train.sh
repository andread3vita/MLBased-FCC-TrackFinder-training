#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"

# ---- Defaults ----
DATA_DIR=""
OUT_DIR=""
TRAIN_SEEDS=1-1000
VAL_SEEDS=1001-1196
NUM_EPOCHS=100
NUM_DEVICES=4
MAX_TOKENS=8000
PRECISION=32-true
GRAD_CKPT=1
CPU_THREADS=4
NUM_WORKERS=8
PREFETCH=4
CKPT_EVERY=200
LIMIT_VAL=1
WARMUP_EPOCHS=2
START_LR=3e-4

# ---- Argument parsing ----
while [[ $# -gt 0 ]]; do
  case "$1" in
    --data_dir)       DATA_DIR="$2";       shift 2 ;;
    --out_dir)        OUT_DIR="$2";        shift 2 ;;
    --train_seeds)    TRAIN_SEEDS="$2";    shift 2 ;;
    --val_seeds)      VAL_SEEDS="$2";      shift 2 ;;
    --num_epochs)     NUM_EPOCHS="$2";     shift 2 ;;
    --num_devices)    NUM_DEVICES="$2";    shift 2 ;;
    --max_tokens)     MAX_TOKENS="$2";     shift 2 ;;
    --precision)      PRECISION="$2";      shift 2 ;;
    --grad_checkpoint) GRAD_CKPT=1;        shift   ;;
    --cpu_threads)    CPU_THREADS="$2";    shift 2 ;;
    --num_workers)    NUM_WORKERS="$2";    shift 2 ;;
    --prefetch)       PREFETCH="$2";       shift 2 ;;
    --ckpt_every)     CKPT_EVERY="$2";     shift 2 ;;
    --limit_val)      LIMIT_VAL="$2";      shift 2 ;;
    --warmup_epochs)  WARMUP_EPOCHS="$2";  shift 2 ;;
    --start_lr)       START_LR="$2";       shift 2 ;;
    *)                break ;;             # remaining args passed to train.py
  esac
done

# ---- CPU thread env ----
export OMP_NUM_THREADS=$CPU_THREADS
export POLARS_MAX_THREADS=$CPU_THREADS
export MKL_NUM_THREADS=$CPU_THREADS
export CGATR_DATALOADER_MP_CTX=spawn
export CGATR_PIN_MEMORY=${CGATR_PIN_MEMORY:-1}
export CGATR_PARQUET_CACHE_SIZE=${CGATR_PARQUET_CACHE_SIZE:-256}
export NCCL_P2P_LEVEL=${NCCL_P2P_LEVEL:-NVL}
export PYTHONPATH=.

GC_FLAG=""
[[ "$GRAD_CKPT" == "1" ]] && GC_FLAG="--grad_checkpoint"

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