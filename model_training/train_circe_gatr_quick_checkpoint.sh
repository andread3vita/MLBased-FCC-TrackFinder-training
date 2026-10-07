#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

usage() {
    echo "Usage: $0 CIRCE|GATR TRAIN_GLOB VAL_GLOB OUTPUT_DIR [GPU_ID]"
    echo "Runs one training batch and one validation batch, then prints a checkpoint path."
}

if [[ $# -lt 4 || $# -gt 5 ]]; then
    usage >&2
    exit 2
fi

MODEL="${1,,}"
case "$MODEL" in
    circe|gatr) ;;
    *) usage >&2; exit 2 ;;
esac

TRAIN_GLOB="$2"
VAL_GLOB="$3"
OUTPUT_ROOT="$4"
GPU_ID="${5:-${TRAIN_GPUS:-0}}"
if [[ "$GPU_ID" == *,* ]]; then
    echo "Quick validation accepts exactly one GPU ID; received '$GPU_ID'." >&2
    exit 2
fi

mkdir -p "$OUTPUT_ROOT"
OUTPUT_ROOT="$(cd "$OUTPUT_ROOT" && pwd)"
RUN_ID="$(date +%Y%m%d_%H%M%S)_$$"
RUN_DIR="$OUTPUT_ROOT/quick_${MODEL}_${RUN_ID}"

export NUM_EPOCHS=1
export LIMIT_TRAIN_BATCHES=1
export LIMIT_VAL_BATCHES=1
export BATCH_SIZE="${QUICK_BATCH_SIZE:-1}"
export MAX_TOKENS=0
export NUM_WORKERS="${QUICK_NUM_WORKERS:-0}"
export FETCH_FILES=1
export SWEEP_MAX_EVENTS=1
export SWEEP_TBETA_GRID="${QUICK_TBETA:-0.7}"
export SWEEP_TD_GRID="${QUICK_TD:-0.3}"
export SWEEP_MIN_HITS_GRID="${QUICK_MIN_HITS:-3}"
export LOG_WANDB=0

echo "Quick ${MODEL^^} smoke training: one train batch, one validation batch."
echo "Output directory: $RUN_DIR"

"$ROOT_DIR/train_circe_gatr_shared.sh" \
    "$MODEL" "$TRAIN_GLOB" "$VAL_GLOB" "$RUN_DIR" "$GPU_ID"

case "$MODEL" in
    circe)
        CHECKPOINT=""
        shopt -s nullglob
        CIRCE_CHECKPOINTS=(
            "$RUN_DIR"/circe/validation_epoch=0_step=*.ckpt
        )
        shopt -u nullglob
        if [[ "${#CIRCE_CHECKPOINTS[@]}" -gt 0 ]]; then
            CHECKPOINT="${CIRCE_CHECKPOINTS[0]}"
        fi
        ;;
    gatr)
        CHECKPOINT=""
        shopt -s nullglob
        GATR_CHECKPOINTS=(
            "$RUN_DIR"/gatr_circe_loss/validation_epoch=0_step=*.ckpt
        )
        shopt -u nullglob
        if [[ "${#GATR_CHECKPOINTS[@]}" -gt 0 ]]; then
            CHECKPOINT="${GATR_CHECKPOINTS[0]}"
        fi
        ;;
esac

if [[ -z "${CHECKPOINT:-}" || ! -f "$CHECKPOINT" ]]; then
    echo "Training completed, but the expected validation checkpoint was not found in $RUN_DIR." >&2
    exit 1
fi

echo "Checkpoint: $CHECKPOINT"
