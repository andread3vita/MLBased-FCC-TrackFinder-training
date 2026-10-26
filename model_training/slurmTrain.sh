#!/bin/bash
#SBATCH --job-name=gatr_4gpus_simplified
#SBATCH --nodes=1
#SBATCH --cpus-per-task=80
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --time=72:00:00
#SBATCH --qos=acc_ehpc
#SBATCH --account=ehpc1013
#SBATCH --mail-type=all
#SBATCH --mail-user=andrea.de.vita@cern.ch

# Detect the real script location before Slurm can stage a copy.
SCRIPT_PATH="$(readlink -f -- "${BASH_SOURCE[0]}")"
SCRIPT_DIR="$(dirname -- "$SCRIPT_PATH")"

# bash on a login node submits the job; sbatch and srun execute it.
# Require a node list too, so a stale SLURM_JOB_ID alone cannot bypass submission.
if [[ "${1:-}" != "--worker" && ( -z "${SLURM_JOB_ID:-}" || -z "${SLURM_JOB_NODELIST:-}" ) ]]; then
    # A direct bash invocation uses the directory of the actual file.
    if [[ ! -f "$SCRIPT_DIR/shared_training/validate_setup.py" ]]; then
        echo "ERROR: This script must be in the model_training directory: $SCRIPT_DIR" >&2
        exit 1
    fi
    if [[ $# -lt 4 ]]; then
        echo "Usage: $0 CIRCE|GATR TRAIN_GLOB VAL_GLOB OUTPUT_DIR [GPU_IDS]" >&2
        exit 2
    fi
    # Slurm logs go next to the run outputs, not in logs-torchrun.
    mkdir -p "$4"
    SUBMIT_OUTPUT_DIR="$(cd "$4" && pwd)"
    SLURM_LOG_DIR="$SUBMIT_OUTPUT_DIR/slurm_logs"
    mkdir -p "$SLURM_LOG_DIR"
    echo "Submitting Slurm job from: $SCRIPT_DIR"
    echo "Slurm logs: $SLURM_LOG_DIR"
    exec sbatch --chdir="$SCRIPT_DIR" \
        --output="$SLURM_LOG_DIR/%j.out" \
        --error="$SLURM_LOG_DIR/%j.err" \
        --export="ALL,PROJECT_ROOT=$SCRIPT_DIR,TRAINING_SCRIPT=$SCRIPT_PATH" \
        "$SCRIPT_PATH" "$@"
fi

set -eo pipefail
# MN5 environment initialization
# Conda/MPI scripts may reference unset environment variables.
set +u
module load miniforge/24.3.0-0
source "/gpfs/apps/MN5/ACC/MINIFORGE/24.3.0-0/etc/profile.d/conda.sh"
conda activate /gpfs/apps/MN5/ACC/MINIFORGE/24.3.0-0/envs/mlpf
source "/gpfs/scratch/ehpc1013/lena/envs/hitpf-overlay/bin/activate"
# Restore strict Bash error checking after environment setup.
set -u
echo "Environment initialized successfully"
echo "Python: $(which python)"
python --version
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export SLURM_CPU_BIND=none
# ============================================================
# Resolve project root without trusting a stale SLURM_SUBMIT_DIR.
# The directory containing a REAL (non-staged) copy of the script
# takes priority for direct bash execution.
# ============================================================
is_project_dir() {
    [[ -f "$1/shared_training/validate_setup.py" &&
       -f "$1/GATR_CIRCE_LOSS/src/train_lightning.py" ]]
}

if is_project_dir "$SCRIPT_DIR"; then
    ROOT_DIR="$SCRIPT_DIR"
elif [[ -n "${PROJECT_ROOT:-}" ]] && is_project_dir "$PROJECT_ROOT"; then
    ROOT_DIR="$(realpath "$PROJECT_ROOT")"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" ]] && is_project_dir "$SLURM_SUBMIT_DIR"; then
    ROOT_DIR="$(realpath "$SLURM_SUBMIT_DIR")"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" ]] && is_project_dir "$SLURM_SUBMIT_DIR/model_training"; then
    ROOT_DIR="$(realpath "$SLURM_SUBMIT_DIR/model_training")"
else
    echo "ERROR: Cannot locate model_training." >&2
    echo "Script directory: $SCRIPT_DIR" >&2
    echo "PROJECT_ROOT=${PROJECT_ROOT:-unset}" >&2
    echo "SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR:-unset}" >&2
    echo "Submit from model_training or set PROJECT_ROOT to its absolute path." >&2
    exit 1
fi
ROOT_DIR="$(realpath "$ROOT_DIR")"
export PROJECT_ROOT="$ROOT_DIR"

# Always run the source script, never Slurm's temporary copy.
# sbatch exports TRAINING_SCRIPT (the real file path); fall back to this file's own path.
TRAINING_SCRIPT="${TRAINING_SCRIPT:-$SCRIPT_PATH}"
export TRAINING_SCRIPT

cd "${WORK_DIR:-$ROOT_DIR}"

echo "========================================"
echo "ROOT_DIR=$ROOT_DIR"
echo "TRAINING_SCRIPT=$TRAINING_SCRIPT"
echo "Current directory=$(pwd)"
echo "Python=$(command -v python)"
echo "========================================"

for required_file in \
    "$ROOT_DIR/shared_training/validate_setup.py" \
    "$ROOT_DIR/GATR_CIRCE_LOSS/src/train_lightning.py" \
    "$TRAINING_SCRIPT"
do
    if [[ ! -f "$required_file" ]]; then
        echo "ERROR: Required file not found: $required_file" >&2
        exit 1
    fi
done
# srun starts one copy of this script per node. Only the batch copy
# should spawn srun; worker copies launch torchrun locally.
WORKER_MODE=0
if [[ "${1:-}" == "--worker" ]]; then
    WORKER_MODE=1
    shift
fi
usage() {
    echo "Usage: $0 CIRCE|GATR TRAIN_GLOB VAL_GLOB OUTPUT_DIR [GPU_IDS]"
    echo "Quote the two globs so the identical file specifications reach both loaders."
}
if [[ $# -lt 4 || $# -gt 5 ]]; then usage >&2; exit 2; fi
MODEL="${1,,}"
TRAIN_FILES="$2"
VAL_FILES="$3"
OUTPUT_DIR="$4"
GPUS_PER_NODE="${GPUS_PER_NODE:-4}"
GPU_IDS="${5:-${TRAIN_GPUS:-$(seq -s, 0 $((GPUS_PER_NODE - 1)))}}"
case "$MODEL" in circe|gatr) ;; *) usage >&2; exit 2 ;; esac
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
if (( NUM_DEVICES != GPUS_PER_NODE )); then
    echo "GPU_IDS ($GPU_IDS) must specify GPUS_PER_NODE=$GPUS_PER_NODE GPUs." >&2
    exit 2
fi
export CUDA_VISIBLE_DEVICES="$GPU_IDS"
export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONHASHSEED="${TRAIN_SEED:-42}"
PYTHON_BIN="${PYTHON_BIN:-python}"
# SLURM coordinates one torchrun agent per node, each launching one
# process per GPU. All ranks use the same rendezvous endpoint.
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "Submit this script with sbatch inside a SLURM allocation." >&2
    exit 2
fi
# NODES="${SLURM_NNODES:-1}"
NODES=1  # Single-node test even inside a larger allocation.
export GPUS_PER_NODE NODES
export NUM_PROCS=$((NODES * GPUS_PER_NODE))
export MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"
export MASTER_PORT="${MASTER_PORT:-29500}"
if (( WORKER_MODE == 0 )); then
    echo "NNODES=$NODES"
    echo "NUM_PROCS=$NUM_PROCS"
    echo "MASTER_ADDR=$MASTER_ADDR"
    echo "MASTER_PORT=$MASTER_PORT"

    srun --nodes=1 --ntasks=1 --ntasks-per-node=1 \
        --nodelist="$MASTER_ADDR" \
        --cpu-bind=none \
        --export=ALL \
        bash "$TRAINING_SCRIPT" --worker "$@"

    echo "Training succeeded."
    exit 0
fi
echo "Training worker on $(hostname), CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
export NODE_RANK="${SLURM_PROCID:-0}"
launch_torchrun() {
    "$PYTHON_BIN" -m torch.distributed.run \
        --nnodes="$NODES" \
        --nproc_per_node="$GPUS_PER_NODE" \
        --node_rank="$NODE_RANK" \
        --rdzv_backend=c10d \
        --rdzv_endpoint="$MASTER_ADDR:$MASTER_PORT" \
        --rdzv_id="$SLURM_JOB_ID" \
        "$@"
}
export WANDB_MODE=offline
export LOG_WANDB=1
export WANDB_ENTITY='ml4hep'
export WANDB_PROJECT='IDEA_v4_o1_CIRCE_GATR_comparison'
EPOCHS="${NUM_EPOCHS:-100}"
BATCH_SIZE="${BATCH_SIZE:-8}"
WORKERS="${NUM_WORKERS:-4}"
MAX_TOKENS="${MAX_TOKENS:-20000}"
TRAIN_PRECISION="${TRAIN_PRECISION:-32-true}"
CPU_THREADS="${CPU_THREADS:-4}"
PREFETCH="${PREFETCH_FACTOR:-2}"
GRAD_CHECKPOINTING="${GRAD_CHECKPOINTING:-0}"
CIRCE_COMPILE="${CIRCE_COMPILE:-0}"
CIRCE_COMPILE_MODE="${CIRCE_COMPILE_MODE:-default}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY_N_TRAIN_STEPS:-0}"
TRAIN_BATCH_LIMIT="${LIMIT_TRAIN_BATCHES:-}"
VAL_BATCH_LIMIT="${LIMIT_VAL_BATCHES:-100}"
EMBED_DIM="${EMBED_DIM:-4}"
BLOCKS="${NUM_BLOCKS:-6}"
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
SWEEP_TBETA="${SWEEP_TBETA_GRID:-0.05,0.1,0.15,0.2,0.35,0.5,0.6,0.7,0.75,0.8,0.85,0.9,0.95}"
SWEEP_TD="${SWEEP_TD_GRID:-0.05,0.1,0.15,0.2,0.25,0.3,0.4,0.5,0.55,0.6}"
SWEEP_MIN_HITS="${SWEEP_MIN_HITS_GRID:-3}"
SWEEP_EVENTS="${SWEEP_MAX_EVENTS:-2000}"
REJECTED_POLICY="${REJECTED_SEED_POLICY:-attach-after-accept}"
MATCHING_METRIC="${SWEEP_MATCH_METRIC:-double_majority}"
case "$TRAIN_PRECISION" in
    32-true|16-mixed|bf16-mixed) ;;
    *) echo "TRAIN_PRECISION must be 32-true, 16-mixed, or bf16-mixed." >&2; exit 2 ;;
esac
case "$CIRCE_COMPILE_MODE" in
    default|max-autotune-no-cudagraphs) ;;
    *) echo "CIRCE_COMPILE_MODE must be default or max-autotune-no-cudagraphs." >&2; exit 2 ;;
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
# W&B offline run files go next to the Slurm logs: OUTPUT_DIR/wandb/
export WANDB_DIR="$OUTPUT_DIR"
run_circe() {
    local wandb_options=()
    local batch_limit_options=()
    local checkpoint_options=()
    local compile_options=()
    [[ -n "$TRAIN_BATCH_LIMIT" ]] && batch_limit_options+=(--limit_train_batches "$TRAIN_BATCH_LIMIT")
    [[ "$VAL_BATCH_LIMIT" != "-1" ]] && batch_limit_options+=(--limit_val_batches "$VAL_BATCH_LIMIT")
    [[ "$GRAD_CHECKPOINTING" == "1" ]] && checkpoint_options+=(--grad_checkpoint)
    [[ "$CIRCE_COMPILE" == "1" ]] && compile_options+=(--compile --compile_mode "$CIRCE_COMPILE_MODE")
    if [[ "$LOG_WANDB" == "1" ]]; then
        wandb_options=(--log_wandb --wandb_displayname circe_shared)
        [[ -n "$WANDB_PROJECT" ]] && wandb_options+=(--wandb_projectname "$WANDB_PROJECT")
        [[ -n "$WANDB_ENTITY" ]] && wandb_options+=(--wandb_entity "$WANDB_ENTITY")
    fi
    if [[ "$NODE_RANK" == "0" ]]; then
        "$PYTHON_BIN" "$ROOT_DIR/shared_training/validate_setup.py" \
        --train "${TRAIN_PATHS[@]}" --val "${VAL_PATHS[@]}"
    fi
    (
    cd "$ROOT_DIR/CIRCE"
    # Match GATr: let torchrun manage DDP ranks and Lightning detect TorchElastic.
    # CIRCE/src/train.py must use the accompanying TorchElastic environment fix.
    launch_torchrun src/train.py \
        --train_files "${TRAIN_PATHS[@]}" --val_files "${VAL_PATHS[@]}" \
        --output_dir "$OUTPUT_DIR/circe" --run_tag circe_shared \
        --num_epochs "$EPOCHS" --num_devices "$NUM_DEVICES" \
        --batch_size "$BATCH_SIZE" --max_tokens "$MAX_TOKENS" \
        --num_workers "$WORKERS" --prefetch_factor "$PREFETCH" \
        --persistent_workers --cpu_threads "$CPU_THREADS" \
        "${compile_options[@]}" \
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
        wandb_options=(--log-wandb --wandb-displayname gatr_4gpus)
        [[ -n "$WANDB_PROJECT" ]] && wandb_options+=(--wandb-projectname "$WANDB_PROJECT")
        [[ -n "$WANDB_ENTITY" ]] && wandb_options+=(--wandb-entity "$WANDB_ENTITY")
    fi
    if [[ "$NODE_RANK" == "0" ]]; then
        "$PYTHON_BIN" "$ROOT_DIR/shared_training/validate_setup.py" \
        --train "${TRAIN_PATHS[@]}" --val "${VAL_PATHS[@]}" --gatr
    fi
    cd "$ROOT_DIR/GATR_CIRCE_LOSS"
    launch_torchrun -m src.train_lightning \
        --data-train "${TRAIN_PATHS[@]}" --data-val "${VAL_PATHS[@]}" \
        --data-config config_files/config_tracking_parquet.yaml \
        --network-config src/models/wrapper/model_tracking_gatr.py \
        --model-prefix "$OUTPUT_DIR/simplifiedModel/" \
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
        "${checkpoint_options[@]}" \
        "${wandb_options[@]}"
}
case "$MODEL" in
    circe) run_circe ;;
    gatr) run_gatr ;;
esac