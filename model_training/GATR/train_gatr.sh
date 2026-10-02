#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
USE_DETECTOR_FEATURES=0
DEFAULT_LAYERS_PER_SUPERLAYER=(8 8 8 8 8 8 8 8 8 8 8 8 8 8)
if [[ -n "${LAYERS_PER_SUPERLAYER_CSV:-}" ]]; then
    IFS=',' read -r -a LAYERS_PER_SUPERLAYER <<< "$LAYERS_PER_SUPERLAYER_CSV"
else
    LAYERS_PER_SUPERLAYER=("${DEFAULT_LAYERS_PER_SUPERLAYER[@]}")
fi
if [[ "${#LAYERS_PER_SUPERLAYER[@]}" -ne 14 ]]; then
    echo "LAYERS_PER_SUPERLAYER_CSV must contain exactly 14 comma-separated entries." >&2
    exit 2
fi
for layers in "${LAYERS_PER_SUPERLAYER[@]}"; do
    if [[ ! "$layers" =~ ^[1-9][0-9]*$ ]]; then
        echo "Invalid layers-per-superlayer entry '$layers'; use positive integers." >&2
        exit 2
    fi
done

case "$USE_DETECTOR_FEATURES" in
    0)
        DATA_CONFIG="config_files/config_tracking_parquet.yaml"
        FEATURE_OPTIONS=()
        MODEL_INPUT_DESCRIPTION="seven geometry-only features"
        ;;
    1)
        DATA_CONFIG="config_files/config_tracking_parquet_detector.yaml"
        FEATURE_OPTIONS=(
            --use-detector-features
            --layers-per-superlayer "${LAYERS_PER_SUPERLAYER[@]}"
        )
        MODEL_INPUT_DESCRIPTION="seven geometric values plus four detector scalars; layers/superlayer: ${LAYERS_PER_SUPERLAYER[*]}"
        ;;
esac
GATR_V142_ROOT="${SCRIPT_DIR}/src/gatr_v142"
if [ ! -f "${GATR_V142_ROOT}/gatr/__init__.py" ]; then
    echo "Vendored gatr_v142 package not found at ${GATR_V142_ROOT}" >&2
    exit 1
fi
export PYTHONPATH="${GATR_V142_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <output_dir> [wandb_project] [wandb_entity] [gpu_ids]"
    echo "Example: $0 results/run1 my_project ml4hep 0,1,2,3"
    echo "Set TRAIN_GPUS to choose the GPU list when gpu_ids is omitted."
    echo "Set DATA_TRAIN and optionally DATA_VAL to choose the Parquet input files."
    echo "Set GRADIENT_CLIP_VAL to the global L2 gradient clipping threshold (default: 1.0; 0 disables clipping)."
    echo "Detector scalar feature mode: $USE_DETECTOR_FEATURES."
    echo "Set LAYERS_PER_SUPERLAYER_CSV to 14 comma-separated positive integers (default: fourteen 8s)."
    echo "Override NUM_EPOCHS, STEPS_PER_EPOCH, or the loss/sweep variables through the environment."
    echo "Set REJECTED_SEED_POLICY to discard, keep, or attach-after-accept (default: attach-after-accept)."
    echo "This launcher keeps pT-binned truth-track weighting off and defaults to LR/variance warmup for stability."
    echo "Its defaults use the September fixedDataset with attach-after-accept clustering."
    exit 1
fi

OUTPUT_DIR="$1"
WANDB_PROJECT="${2:-IDEA_v3_o1_tracking_andrea}"
WANDB_ENTITY="${3:-ml4hep}"
GPU_IDS="${4:-${TRAIN_GPUS:-0,1,2,3}}"

DEFAULT_DATA_TRAIN='/eos/experiment/fcc/ee/simulation/key4hep_2026_09_10/91GeV/IDEA_o1_v4/fixedDataset/Zuds/graph/Graphs_*_train.parquet'
DATA_TRAIN="${DATA_TRAIN:-$DEFAULT_DATA_TRAIN}"
DATA_VAL="${DATA_VAL-/eos/experiment/fcc/ee/simulation/key4hep_2026_09_10/91GeV/IDEA_o1_v4/fixedDataset/Zuds_validation/graph/Graphs_*_test.parquet}"
EMBED_DIM="${EMBED_DIM:-5}"
GATR_BLOCKS="${GATR_BLOCKS:-10}"
HIDDEN_MV_CHANNELS="${HIDDEN_MV_CHANNELS:-16}"
HIDDEN_S_CHANNELS="${HIDDEN_S_CHANNELS:-64}"
NUM_WORKERS="${NUM_WORKERS:-4}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-1}"
FETCH_FILES="${FETCH_FILES:-2}"
BATCH_SIZE="${BATCH_SIZE:-8}"
ACCUMULATE_GRAD_BATCHES="${ACCUMULATE_GRAD_BATCHES:-1}"
CHECKPOINT_EVERY_N_STEPS="${CHECKPOINT_EVERY_N_STEPS:-12000}"
NUM_EPOCHS="${NUM_EPOCHS:-20}"
STEPS_PER_EPOCH="${STEPS_PER_EPOCH:-30000}"
LIMIT_VAL_BATCHES="${LIMIT_VAL_BATCHES:-125}"
VALIDATE_BEFORE_TRAINING="${VALIDATE_BEFORE_TRAINING:-0}"
TRAIN_SEED="${TRAIN_SEED:-42}"

START_LR="${START_LR:-4e-4}"
GRADIENT_CLIP_VAL="${GRADIENT_CLIP_VAL:-5.0}"
LR_SCHEDULER="${LR_SCHEDULER:-flat+decay}"
PLATEAU_FACTOR="${PLATEAU_FACTOR:-0.5}"
PLATEAU_PATIENCE="${PLATEAU_PATIENCE:-1}"
PLATEAU_THRESHOLD="${PLATEAU_THRESHOLD:-1e-3}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-2}"
MIN_LR="${MIN_LR:-1e-6}"
EMA_DECAY="${EMA_DECAY:-0.999}"
ATTENTION_PHI_SECTORS="${ATTENTION_PHI_SECTORS:-1}"

L_ATTRACTIVE_WEIGHT="${L_ATTRACTIVE_WEIGHT:-1.0}"
L_REPULSIVE_WEIGHT="${L_REPULSIVE_WEIGHT:-1.0}"
BETA_SUPPRESS_WEIGHT="${BETA_SUPPRESS_WEIGHT:-0.1}"
BETA_SECOND_WEIGHT="${BETA_SECOND_WEIGHT:-0.2}"
VAR_WEIGHT="${VAR_WEIGHT:-0.2}"
VAR_WARMUP_EPOCHS="${VAR_WARMUP_EPOCHS:-3}"
HARD_NEGATIVE_WEIGHT="${HARD_NEGATIVE_WEIGHT:-1.0}"
HARD_NEGATIVE_MAX_WEIGHT="${HARD_NEGATIVE_MAX_WEIGHT:-100}"
PT_TRACK_WEIGHTING="0"
PT_TRACK_WEIGHT_BIN_EDGES="${PT_TRACK_WEIGHT_BIN_EDGES:-0.4,0.9,5.0}"
PT_TRACK_WEIGHT_BIN_WEIGHTS="${PT_TRACK_WEIGHT_BIN_WEIGHTS:-1.5,1.2,0.75,2.0}"
HELIX_LOSS_WEIGHT="${HELIX_LOSS_WEIGHT:-0.5}"

SWEEP_MAX_EVENTS="${SWEEP_MAX_EVENTS:-1000}"
SWEEP_TBETA_GRID="${SWEEP_TBETA_GRID:-0.2,0.35,0.5,0.6,0.7,0.75,0.8,0.85,0.9,0.95}"
SWEEP_TD_GRID="${SWEEP_TD_GRID:-0.1,0.15,0.2,0.25,0.3,0.4,0.5,0.55,0.6}"
SWEEP_MIN_HITS_GRID="${SWEEP_MIN_HITS_GRID:-3}"
REJECTED_SEED_POLICY="${REJECTED_SEED_POLICY:-attach-after-accept}"

if [[ "$VALIDATE_BEFORE_TRAINING" != "0" && "$VALIDATE_BEFORE_TRAINING" != "1" ]]; then
    echo "Invalid VALIDATE_BEFORE_TRAINING '$VALIDATE_BEFORE_TRAINING'; use 0 or 1." >&2
    exit 2
fi
if [[ "$PT_TRACK_WEIGHTING" != "0" && "$PT_TRACK_WEIGHTING" != "1" ]]; then
    echo "Invalid PT_TRACK_WEIGHTING '$PT_TRACK_WEIGHTING'; use 0 or 1." >&2
    exit 2
fi
case "$REJECTED_SEED_POLICY" in
    discard|keep|attach-after-accept) ;;
    *)
        echo "Invalid REJECTED_SEED_POLICY '$REJECTED_SEED_POLICY'; use discard, keep, or attach-after-accept." >&2
        exit 2
        ;;
esac
if ! [[ "$TRAIN_SEED" =~ ^[0-9]+$ ]] || (( ${#TRAIN_SEED} > 10 )) \
    || { (( ${#TRAIN_SEED} == 10 )) && [[ "$TRAIN_SEED" > "4294967295" ]]; }; then
    echo "Invalid TRAIN_SEED '$TRAIN_SEED'; use an integer from 0 to 4294967295." >&2
    exit 2
fi
for POSITIVE_INTEGER_NAME in EMBED_DIM GATR_BLOCKS HIDDEN_MV_CHANNELS HIDDEN_S_CHANNELS \
    BATCH_SIZE ACCUMULATE_GRAD_BATCHES CHECKPOINT_EVERY_N_STEPS NUM_EPOCHS STEPS_PER_EPOCH; do
    POSITIVE_INTEGER_VALUE="${!POSITIVE_INTEGER_NAME}"
    if ! [[ "$POSITIVE_INTEGER_VALUE" =~ ^[1-9][0-9]*$ ]]; then
        echo "Invalid $POSITIVE_INTEGER_NAME '$POSITIVE_INTEGER_VALUE'; use a positive integer." >&2
        exit 2
    fi
done
if ! [[ "$NUM_WORKERS" =~ ^[0-9]+$ ]]; then
    echo "Invalid NUM_WORKERS '$NUM_WORKERS'; use a non-negative integer." >&2
    exit 2
fi
if ! [[ "$PREFETCH_FACTOR" =~ ^[1-9][0-9]*$ ]]; then
    echo "Invalid PREFETCH_FACTOR '$PREFETCH_FACTOR'; use a positive integer." >&2
    exit 2
fi

GPU_IDS="${GPU_IDS//[[:space:]]/}"
if ! [[ "$GPU_IDS" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
    echo "Invalid GPU list '$GPU_IDS'; use comma-separated IDs such as 0,1 or 2,3." >&2
    exit 2
fi
IFS=',' read -r -a GPU_ARRAY <<< "$GPU_IDS"
SEEN_GPU_IDS=","
for GPU_ID in "${GPU_ARRAY[@]}"; do
    if [[ "$SEEN_GPU_IDS" == *",${GPU_ID},"* ]]; then
        echo "GPU ID '$GPU_ID' was selected more than once." >&2
        exit 2
    fi
    SEEN_GPU_IDS="${SEEN_GPU_IDS}${GPU_ID},"
done

expand_data_spec() {
    local spec="$1"
    if [[ "$spec" =~ ^(.*)\{([0-9]+)\.\.([0-9]+)\}(.*)$ ]]; then
        local prefix="${BASH_REMATCH[1]}"
        local first="${BASH_REMATCH[2]}"
        local last="${BASH_REMATCH[3]}"
        local suffix="${BASH_REMATCH[4]}"
        if (( first > last )); then
            echo "Invalid descending data range in '$spec'." >&2
            exit 2
        fi
        local index
        for (( index=first; index<=last; index++ )); do
            DATA_ARGUMENTS+=("${prefix}${index}${suffix}")
        done
    else
        DATA_ARGUMENTS+=("$spec")
    fi
}

DATA_ARGUMENTS=(--data-train)
expand_data_spec "$DATA_TRAIN"
if [ -n "$DATA_VAL" ]; then
    DATA_ARGUMENTS+=(--data-val)
    expand_data_spec "$DATA_VAL"
    TRAIN_VAL_SPLIT="1.0"
else
    TRAIN_VAL_SPLIT="0.8"
fi

TRAINING_OPTIONS=(
    --num-epochs "$NUM_EPOCHS"
    --train-val-split "$TRAIN_VAL_SPLIT"
    --steps-per-epoch "$STEPS_PER_EPOCH"
)
if [[ "$VALIDATE_BEFORE_TRAINING" == "1" ]]; then
    TRAINING_OPTIONS+=(--validate-before-training)
fi
TRACK_WEIGHTING_OPTIONS=(
    --pt-track-weight-bin-edges "$PT_TRACK_WEIGHT_BIN_EDGES"
    --pt-track-weight-bin-weights "$PT_TRACK_WEIGHT_BIN_WEIGHTS"
)
if [[ "$PT_TRACK_WEIGHTING" == "1" ]]; then
    TRACK_WEIGHTING_OPTIONS+=(--pt-track-weighting)
fi

export PYTHONHASHSEED="$TRAIN_SEED"
TRAINING_NAME="$(basename "${OUTPUT_DIR%/}")"
mkdir -p "$OUTPUT_DIR"
OUTPUT_DIR="$(cd "$OUTPUT_DIR" && pwd)"

echo "Starting unweighted GATr training with LR and embedding-variance warmup"
echo "Output: $OUTPUT_DIR"
echo "GPU(s): $GPU_IDS; embedding dimension: $EMBED_DIM; GATr blocks: $GATR_BLOCKS"
echo "Training data: $DATA_TRAIN"
echo "Validation data: ${DATA_VAL:-internal 80/20 split}"
echo "Model input: $MODEL_INPUT_DESCRIPTION"
echo "Schedule: $LR_SCHEDULER, start_lr=$START_LR, epochs=$NUM_EPOCHS, steps/epoch=$STEPS_PER_EPOCH"
echo "Global L2 gradient clipping threshold: $GRADIENT_CLIP_VAL (0 disables clipping)"
echo "Warmup: LR=$WARMUP_EPOCHS epochs; embedding variance=$VAR_WARMUP_EPOCHS epochs"
echo "Embedding variance loss: var_weight=$VAR_WEIGHT"
echo "Rejected seed policy: $REJECTED_SEED_POLICY"
echo "pT track weighting: disabled"

cd "$SCRIPT_DIR"
python -m src.train_lightning \
    "${DATA_ARGUMENTS[@]}" \
    --data-config "$DATA_CONFIG" \
    "${FEATURE_OPTIONS[@]}" \
    --clustering_loss_only \
    --clustering_space_dim "$EMBED_DIM" \
    --network-config src/models/wrapper/model_tracking_gatr.py \
    --model-prefix "${OUTPUT_DIR}/" \
    --gatr-blocks "$GATR_BLOCKS" \
    --hidden-mv-channels "$HIDDEN_MV_CHANNELS" \
    --hidden-s-channels "$HIDDEN_S_CHANNELS" \
    --num-workers "$NUM_WORKERS" \
    --prefetch-factor "$PREFETCH_FACTOR" \
    --gpus "$GPU_IDS" \
    --seed "$TRAIN_SEED" \
    --batch-size "$BATCH_SIZE" \
    --accumulate-grad-batches "$ACCUMULATE_GRAD_BATCHES" \
    --checkpoint-every-n-train-steps "$CHECKPOINT_EVERY_N_STEPS" \
    --limit-val-batches "$LIMIT_VAL_BATCHES" \
    --start-lr "$START_LR" \
    --gradient-clip-val "$GRADIENT_CLIP_VAL" \
    "${TRAINING_OPTIONS[@]}" \
    --optimizer adamW \
    --weight-decay 1e-3 \
    --lr-scheduler "$LR_SCHEDULER" \
    --plateau-factor "$PLATEAU_FACTOR" \
    --plateau-patience "$PLATEAU_PATIENCE" \
    --plateau-threshold "$PLATEAU_THRESHOLD" \
    --warmup-epochs "$WARMUP_EPOCHS" \
    --min-lr "$MIN_LR" \
    --ema-decay "$EMA_DECAY" \
    --fetch-by-files \
    --fetch-step "$FETCH_FILES" \
    --condensation \
    --log-wandb \
    --wandb-displayname "$TRAINING_NAME" \
    --wandb-projectname "$WANDB_PROJECT" \
    --wandb-entity "$WANDB_ENTITY" \
    --qmin 0.1 \
    --L_attractive_weight "$L_ATTRACTIVE_WEIGHT" \
    --L_repulsive_weight "$L_REPULSIVE_WEIGHT" \
    --beta-suppress-weight "$BETA_SUPPRESS_WEIGHT" \
    --beta-second-weight "$BETA_SECOND_WEIGHT" \
    --var-weight "$VAR_WEIGHT" \
    --var-warmup-epochs "$VAR_WARMUP_EPOCHS" \
    --hard-negative-weight "$HARD_NEGATIVE_WEIGHT" \
    --hard-negative-max-weight "$HARD_NEGATIVE_MAX_WEIGHT" \
    "${TRACK_WEIGHTING_OPTIONS[@]}" \
    --helix-loss-weight "$HELIX_LOSS_WEIGHT" \
    --attention-phi-sectors "$ATTENTION_PHI_SECTORS" \
    --validation-sweep-max-events "$SWEEP_MAX_EVENTS" \
    --sweep-tbeta-grid "$SWEEP_TBETA_GRID" \
    --sweep-td-grid "$SWEEP_TD_GRID" \
    --sweep-min-hits-grid "$SWEEP_MIN_HITS_GRID" \
    --rejected-seed-policy "$REJECTED_SEED_POLICY" \
    --sweep-match-metric double_majority
