#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"

# ---- Defaults ----
DATA_DIR=""
OUT_DIR=""
TRAIN_SEEDS=1-980
VAL_SEEDS=981-1000
NUM_EPOCHS=16
NUM_DEVICES=4
MAX_TOKENS=16000
PRECISION=32-true
GRAD_CKPT=0
CPU_THREADS=4
NUM_WORKERS=4
PREFETCH=2
CKPT_EVERY=0
LIMIT_VAL=40
WARMUP_EPOCHS=2
START_LR=4e-4
LR_SCHEDULE=plateau

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
    --lr_schedule)    LR_SCHEDULE="$2";    shift 2 ;;
    *)                break ;;             # remaining args passed to train.py
  esac
done

# ---- CPU thread env ----
export OMP_NUM_THREADS=$CPU_THREADS
export POLARS_MAX_THREADS=$CPU_THREADS
export MKL_NUM_THREADS=$CPU_THREADS
export CGATR_DATALOADER_MP_CTX=spawn
export CGATR_PIN_MEMORY=${CGATR_PIN_MEMORY:-1}
export CGATR_PARQUET_CACHE_SIZE=${CGATR_PARQUET_CACHE_SIZE:-16}
export NCCL_P2P_LEVEL=${NCCL_P2P_LEVEL:-NVL}
export PYTHONPATH=.
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

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
  --gradient_clip_val 1.0 \
  --embed_dim 4 \
  --num_blocks 10 \
  --hidden_mv_channels 16 \
  --hidden_s_channels 64 \
  --qmin 3.0 \
  --attr_weight 1.0 \
  --repul_weight 2.0 \
  --beta_suppress_weight 0.1 \
  --var_weight 0.3 \
  --var_warmup_epochs 1 \
  --oc_mode paper_hinge \
  --num_workers "$NUM_WORKERS" \
  --prefetch_factor "$PREFETCH" \
  --persistent_workers \
  --cpu_threads "$CPU_THREADS" \
  --optimizer adamw \
  --weight_decay 1e-4 \
  --start_lr "$START_LR" \
  --min_lr 1e-5 \
  --warmup_epochs "$WARMUP_EPOCHS" \
  --lr_schedule "$LR_SCHEDULE" \
  --plateau_patience 3 \
  --plateau_factor 0.5 \
  --terminal_anneal_epochs 6 \
  --ema_decay 0.999 \
  --seed 42 \
  --output_dir "$OUT_DIR" \
  --run_tag circe_production \
  --ckpt_every_n_train_steps "$CKPT_EVERY" \
  --auto_requeue \
  --resume_ckpt last \
  --init_weights none \
  --limit_val_batches "$LIMIT_VAL" \
  --fix_particle_zero \
  --min_target_hits 3 \
  --fix_cga_null \
  --fix_wire_dir \
  --cga_hit_encoding sphere_circle \
  --physical_drift_geometry \
  --no-normalize_mv_inputs \
  --equivariance_group e3 \
  --invariant_output_head \
  --equi_init identity_algebra \
  $GC_FLAG \
  "$@"