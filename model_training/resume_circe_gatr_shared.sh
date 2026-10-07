#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

usage() {
    echo "Usage: $0 CIRCE|GATR TRAIN_GLOB VAL_GLOB OUTPUT_DIR CHECKPOINT [GPU_IDS]"
    echo "Quote the two globs so the identical file specifications reach both loaders."
    echo "CHECKPOINT must be a full Lightning checkpoint from the selected model."
}
if [[ $# -lt 5 || $# -gt 6 ]]; then
    echo "Expected 5 or 6 arguments; quote TRAIN_GLOB and VAL_GLOB." >&2
    usage >&2
    exit 2
fi

MODEL="${1,,}"
TRAIN_FILES="$2"
VAL_FILES="$3"
OUTPUT_DIR="$4"
CHECKPOINT="$5"
GPU_IDS="${6:-${TRAIN_GPUS:-0}}"
case "$MODEL" in circe|gatr) ;; *) usage >&2; exit 2 ;; esac
if [[ ! -f "$CHECKPOINT" ]]; then
    echo "Checkpoint does not exist: $CHECKPOINT" >&2
    exit 2
fi
CHECKPOINT="$(readlink -f "$CHECKPOINT")"
if [[ "$TRAIN_FILES" == "$VAL_FILES" ]]; then
    echo "Training and validation specifications must differ." >&2; exit 2
fi

mapfile -t TRAIN_PATHS < <(compgen -G "$TRAIN_FILES" | sort)
mapfile -t VAL_PATHS < <(compgen -G "$VAL_FILES" | sort)
if [[ "${#TRAIN_PATHS[@]}" -eq 0 || "${#VAL_PATHS[@]}" -eq 0 ]]; then
    echo "The training or validation file specification matched no files." >&2
    exit 2
fi
for index in "${!TRAIN_PATHS[@]}"; do
    TRAIN_PATHS[$index]="$(readlink -f "${TRAIN_PATHS[$index]}")"
done
for index in "${!VAL_PATHS[@]}"; do
    VAL_PATHS[$index]="$(readlink -f "${VAL_PATHS[$index]}")"
done

IFS=',' read -r -a GPU_ARRAY <<< "$GPU_IDS"
NUM_DEVICES="${#GPU_ARRAY[@]}"
LOGICAL_GPUS="$(seq -s, 0 $((NUM_DEVICES - 1)))"
export CUDA_VISIBLE_DEVICES="$GPU_IDS"
export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONHASHSEED="${TRAIN_SEED:-42}"
PYTHON_BIN="${PYTHON_BIN:-python}"

export LOG_WANDB=1
export WANDB_ENTITY='ml4hep'
export WANDB_PROJECT='IDEA_v4_o1_CIRCE_GATR_comparison'

EPOCHS="${NUM_EPOCHS:-16}"
BATCH_SIZE="${BATCH_SIZE:-8}"
WORKERS="${NUM_WORKERS:-4}"
MAX_TOKENS="${MAX_TOKENS:-16000}"
TRAIN_PRECISION="${TRAIN_PRECISION:-32-true}"
CPU_THREADS="${CPU_THREADS:-4}"
PREFETCH="${PREFETCH_FACTOR:-2}"
GRAD_CHECKPOINTING="${GRAD_CHECKPOINTING:-0}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY_N_TRAIN_STEPS:-0}"
TRAIN_BATCH_LIMIT="${LIMIT_TRAIN_BATCHES:-}"
VAL_BATCH_LIMIT="${LIMIT_VAL_BATCHES:-40}"
EMBED_DIM="${EMBED_DIM:-4}"
BLOCKS="${NUM_BLOCKS:-10}"
HIDDEN_MV="${HIDDEN_MV_CHANNELS:-16}"
HIDDEN_S="${HIDDEN_S_CHANNELS:-64}"
START_LR="${START_LR:-4e-4}"
MIN_LR="${MIN_LR:-1e-5}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
LR_WARMUP="${WARMUP_EPOCHS:-2}"
LR_SCHEDULE="${LR_SCHEDULE:-plateau}"
PLATEAU_PATIENCE="${PLATEAU_PATIENCE:-3}"
PLATEAU_FACTOR="${PLATEAU_FACTOR:-0.5}"
TERMINAL_ANNEAL="${TERMINAL_ANNEAL_EPOCHS:-6}"
CLIP="${GRADIENT_CLIP_VAL:-1.0}"
QMIN="${QMIN:-3.0}"
ATTR="${L_ATTRACTIVE_WEIGHT:-1.0}"
REPUL="${L_REPULSIVE_WEIGHT:-2.0}"
BETA_SUPPRESS="${BETA_SUPPRESS_WEIGHT:-0.1}"
VAR_WEIGHT="${VAR_WEIGHT:-0.3}"
VAR_WARMUP="${VAR_WARMUP_EPOCHS:-1}"
EMA="${EMA_DECAY:-0.999}"
SWEEP_TBETA="${SWEEP_TBETA_GRID:-0.2,0.35,0.5,0.6,0.7,0.75,0.8,0.85,0.9,0.95}"
SWEEP_TD="${SWEEP_TD_GRID:-0.1,0.15,0.2,0.25,0.3,0.4,0.5,0.55,0.6}"
SWEEP_MIN_HITS="${SWEEP_MIN_HITS_GRID:-3}"
SWEEP_EVENTS="${SWEEP_MAX_EVENTS:-1000}"
REJECTED_POLICY="${REJECTED_SEED_POLICY:-attach-after-accept}"
MATCHING_METRIC="${SWEEP_MATCH_METRIC:-double_majority}"
case "$TRAIN_PRECISION" in
    32-true|16-mixed|bf16-mixed) ;;
    *) echo "TRAIN_PRECISION must be 32-true, 16-mixed, or bf16-mixed." >&2; exit 2 ;;
esac
case "$LR_SCHEDULE" in
    plateau) GATR_LR_SCHEDULE="reduceplateau" ;;
    cosine) GATR_LR_SCHEDULE="flat+decay" ;;
    *) echo "LR_SCHEDULE must be plateau or cosine for matched training." >&2; exit 2 ;;
esac
if (( TERMINAL_ANNEAL >= EPOCHS )); then
    TERMINAL_ANNEAL=0
fi
case "$MATCHING_METRIC" in
    idea|double_majority|hungarian) ;;
    *) echo "SWEEP_MATCH_METRIC must be idea, double_majority, or hungarian." >&2; exit 2 ;;
esac
LOG_WANDB="${LOG_WANDB:-0}"
WANDB_PROJECT="${WANDB_PROJECT:-}"
WANDB_ENTITY="${WANDB_ENTITY:-}"

mkdir -p "$OUTPUT_DIR"
OUTPUT_DIR="$(cd "$OUTPUT_DIR" && pwd)"

run_circe() {
    local wandb_options=()
    local batch_limit_options=()
    local checkpoint_options=()
    [[ -n "$TRAIN_BATCH_LIMIT" ]] && batch_limit_options+=(--limit_train_batches "$TRAIN_BATCH_LIMIT")
    [[ "$VAL_BATCH_LIMIT" != "-1" ]] && batch_limit_options+=(--limit_val_batches "$VAL_BATCH_LIMIT")
    [[ "$GRAD_CHECKPOINTING" == "1" ]] && checkpoint_options+=(--grad_checkpoint)
    if [[ "$LOG_WANDB" == "1" ]]; then
        wandb_options=(--log_wandb --wandb_displayname circe_shared_resume)
        [[ -n "$WANDB_PROJECT" ]] && wandb_options+=(--wandb_projectname "$WANDB_PROJECT")
        [[ -n "$WANDB_ENTITY" ]] && wandb_options+=(--wandb_entity "$WANDB_ENTITY")
    fi
    "$PYTHON_BIN" "$ROOT_DIR/shared_training/validate_setup.py" \
        --train "${TRAIN_PATHS[@]}" --val "${VAL_PATHS[@]}"
    (
    cd "$ROOT_DIR/CIRCE"
    "$PYTHON_BIN" src/train.py \
        --train_files "${TRAIN_PATHS[@]}" --val_files "${VAL_PATHS[@]}" \
        --output_dir "$OUTPUT_DIR/circe" --run_tag circe_shared_resume \
        --resume_ckpt "$CHECKPOINT" \
        --num_epochs "$EPOCHS" --num_devices "$NUM_DEVICES" \
        --batch_size "$BATCH_SIZE" --max_tokens "$MAX_TOKENS" \
        --num_workers "$WORKERS" --prefetch_factor "$PREFETCH" \
        --persistent_workers --cpu_threads "$CPU_THREADS" \
        "${batch_limit_options[@]}" \
        --precision "$TRAIN_PRECISION" --gradient_clip_val "$CLIP" \
        --start_lr "$START_LR" --min_lr "$MIN_LR" \
        --weight_decay "$WEIGHT_DECAY" --warmup_epochs "$LR_WARMUP" \
        --optimizer adamw --lr_schedule "$LR_SCHEDULE" \
        --plateau_patience "$PLATEAU_PATIENCE" \
        --plateau_factor "$PLATEAU_FACTOR" \
        --terminal_anneal_epochs "$TERMINAL_ANNEAL" \
        --ckpt_every_n_train_steps "$CHECKPOINT_EVERY" \
        --embed_dim "$EMBED_DIM" \
        --num_blocks "$BLOCKS" --hidden_mv_channels "$HIDDEN_MV" \
        --hidden_s_channels "$HIDDEN_S" --algebra conformal \
        --cga_hit_encoding sphere_circle --physical_drift_geometry \
        --separate_hit_metadata --no-normalize_mv_inputs \
        --equivariance_group e3 --invariant_output_head \
        --fix_cga_null --fix_wire_dir --equi_init identity_algebra \
        --fix_particle_zero --min_target_hits 3 --oc_mode paper_hinge \
        --qmin "$QMIN" --attr_weight "$ATTR" --repul_weight "$REPUL" \
        --beta_suppress_weight "$BETA_SUPPRESS" --var_weight "$VAR_WEIGHT" \
        --var_warmup_epochs "$VAR_WARMUP" --ema_decay "$EMA" \
        --sweep_tbeta_grid "$SWEEP_TBETA" --sweep_td_grid "$SWEEP_TD" \
        --sweep_min_hits_grid "$SWEEP_MIN_HITS" \
        --validation_sweep_max_events "$SWEEP_EVENTS" \
        --sweep_match_metric "$MATCHING_METRIC" --sweep_truth_min_hits 3 \
        --rejected_seed_policy "$REJECTED_POLICY" --seed "${TRAIN_SEED:-42}" \
        "${checkpoint_options[@]}" \
        "${wandb_options[@]}"
    )
}

run_gatr() {
    local wandb_options=()
    local train_limit_options=()
    local checkpoint_options=()
    [[ -n "$TRAIN_BATCH_LIMIT" ]] && train_limit_options+=(--limit-train-batches "$TRAIN_BATCH_LIMIT")
    [[ "$GRAD_CHECKPOINTING" == "1" ]] && checkpoint_options+=(--gradient-checkpointing)
    if [[ "$LOG_WANDB" == "1" ]]; then
        wandb_options=(--log-wandb --wandb-displayname gatr_circe_loss_shared)
        [[ -n "$WANDB_PROJECT" ]] && wandb_options+=(--wandb-projectname "$WANDB_PROJECT")
        [[ -n "$WANDB_ENTITY" ]] && wandb_options+=(--wandb-entity "$WANDB_ENTITY")
    fi
    "$PYTHON_BIN" "$ROOT_DIR/shared_training/validate_setup.py" \
        --train "${TRAIN_PATHS[@]}" --val "${VAL_PATHS[@]}" --gatr
    cd "$ROOT_DIR/GATR_CIRCE_LOSS"
    "$PYTHON_BIN" -m src.train_lightning \
        --data-train "${TRAIN_PATHS[@]}" --data-val "${VAL_PATHS[@]}" \
        --data-config config_files/config_tracking_parquet.yaml \
        --network-config src/models/wrapper/model_tracking_gatr.py \
        --model-prefix "$OUTPUT_DIR/gatr_circe_loss/" \
        --clustering_loss_only --clustering_space_dim "$EMBED_DIM" \
        --gatr-blocks "$BLOCKS" --hidden-mv-channels "$HIDDEN_MV" \
        --hidden-s-channels "$HIDDEN_S" --gpus "$LOGICAL_GPUS" \
        --num-workers "$WORKERS" --prefetch-factor "$PREFETCH" \
        --cpu-threads "$CPU_THREADS" \
        --batch-size "$BATCH_SIZE" --max-tokens "$MAX_TOKENS" \
        --shared-indexed-loader \
        "${train_limit_options[@]}" \
        --accumulate-grad-batches 1 --num-epochs "$EPOCHS" \
        --limit-val-batches "$VAL_BATCH_LIMIT" --seed "${TRAIN_SEED:-42}" \
        --precision "$TRAIN_PRECISION" \
        --checkpoint-every-n-train-steps "$CHECKPOINT_EVERY" \
        --optimizer adamW --weight-decay "$WEIGHT_DECAY" \
        --start-lr "$START_LR" --min-lr "$MIN_LR" \
        --lr-scheduler "$GATR_LR_SCHEDULE" --warmup-epochs "$LR_WARMUP" \
        --plateau-patience "$PLATEAU_PATIENCE" \
        --plateau-factor "$PLATEAU_FACTOR" \
        --terminal-anneal-epochs "$TERMINAL_ANNEAL" \
        --gradient-clip-val "$CLIP" --ema-decay "$EMA" --condensation \
        --qmin "$QMIN" --L_attractive_weight "$ATTR" \
        --L_repulsive_weight "$REPUL" --beta-suppress-weight "$BETA_SUPPRESS" \
        --var-weight "$VAR_WEIGHT" --var-warmup-epochs "$VAR_WARMUP" \
        --helix-loss-weight 0 --sweep-tbeta-grid "$SWEEP_TBETA" \
        --sweep-td-grid "$SWEEP_TD" --sweep-min-hits-grid "$SWEEP_MIN_HITS" \
        --validation-sweep-max-events "$SWEEP_EVENTS" \
        --sweep-match-metric "$MATCHING_METRIC" --sweep-truth-min-hits 3 \
        --rejected-seed-policy "$REJECTED_POLICY" \
        --load-model-weights "$CHECKPOINT" --checkpoint-mode resume \
        "${checkpoint_options[@]}" \
        "${wandb_options[@]}"
}

case "$MODEL" in
    circe) run_circe ;;
    gatr) run_gatr ;;
esac
