#!/bin/bash
#SBATCH --job-name=ssm_train
#SBATCH --nodes=1
#SBATCH --cpus-per-task=40
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --time=72:00:00
#SBATCH --qos=acc_ehpc
#SBATCH --account=ehpc1013
#SBATCH --mail-type=all
#SBATCH --mail-user=andrea.de.vita@cern.ch

# sbatch submission:
#   sbatch slurmTrain.sh [extra ssm.train args...]
# Direct bash run (no gpu allocation):
#   bash slurmTrain.sh [extra ssm.train args...]
#
# Overridable via environment (defaults in brackets):
#   TRAIN_DATA, VAL_DATA                       [required]
#   OUTPUT_DIR                                 [slurm_runs/ssm_$SLURM_JOB_ID]
#   NUM_EPOCHS [16]  TOKEN_BUDGET [20000]  D_MODEL [256]  N_LAYER [6]
#   TRAIN_MAX_FILES [all]  TRAIN_NEVENTS [all]  VAL_MAX_FILES [all]
#   VAL_NEVENTS [100]  WORKERS [8]  ORDER_MODE [spatial]  BIDIRECTIONAL [0]
set -euo pipefail
SCRIPT_PATH="$(readlink -f -- "${BASH_SOURCE[0]}")"
SCRIPT_DIR="$(dirname -- "$SCRIPT_PATH")"          # = SSM/ project root
cd "$SCRIPT_DIR"

# --- SSM's own env only (no miniforge/conda/other models' environments) ---
source "$SCRIPT_DIR/venv/bin/activate"
echo "Python: $(which python)"; python --version

: "${TRAIN_DATA:?set TRAIN_DATA=/path/to/train_folder}"
: "${VAL_DATA:?set VAL_DATA=/path/to/val_folder}"
OUTPUT_DIR="${OUTPUT_DIR:-$SCRIPT_DIR/slurm_runs/ssm_${SLURM_JOB_ID:-manual}}"
mkdir -p "$OUTPUT_DIR"

GPUS="${GPUS:-0}"
export CUDA_VISIBLE_DEVICES="$GPUS"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export SLURM_CPU_BIND=none

# wandb: offline on this cluster; sync later from a connected machine with
#   wandb sync $OUTPUT_DIR/wandb/offline-run-*
export WANDB_MODE=offline
export WANDB_DIR="$OUTPUT_DIR"
export LOG_WANDB="${LOG_WANDB:-1}"
export WANDB_PROJECT="${WANDB_PROJECT:-IDEA_v4_o1_SSM}"
export WANDB_ENTITY="${WANDB_ENTITY:-}"

args=(--train-data "$TRAIN_DATA" --val-data "$VAL_DATA" --gpus "$GPUS"
      --out "$OUTPUT_DIR"
      --epochs "${NUM_EPOCHS:-16}"
      --token-budget "${TOKEN_BUDGET:-20000}"
      --d-model "${D_MODEL:-256}" --n-layer "${N_LAYER:-6}"
      --workers "${WORKERS:-8}"
      --order-mode "${ORDER_MODE:-spatial}"
      "--$( [[ "${BIDIRECTIONAL:-0}" == "1" ]] && echo "" || echo "no-" )bidirectional")
[[ -n "${TRAIN_MAX_FILES:-}" ]] && args+=(--train-max-files "$TRAIN_MAX_FILES")
[[ -n "${TRAIN_NEVENTS:-}" ]]   && args+=(--train-nevents "$TRAIN_NEVENTS")
[[ -n "${VAL_MAX_FILES:-}" ]]   && args+=(--val-max-files "$VAL_MAX_FILES")
[[ -n "${VAL_NEVENTS:-}" ]]     && args+=(--val-nevents "$VAL_NEVENTS")
[[ -n "${SWEEP_EVENTS:-}" ]]    && args+=(--sweep-events "$SWEEP_EVENTS")
args+=("$@")

IFS=',' read -r -a _gpu_ids <<< "$GPUS"
NGPU=${#_gpu_ids[@]}

echo "========================================"
echo "SSM train: $SCRIPT_DIR"
echo "python -m ssm.train ${args[*]}"
echo "gpus:    $GPUS (n=$NGPU)"
echo "output:  $OUTPUT_DIR"
echo "========================================"
if (( NGPU > 1 )); then
  # multi-GPU: one rank per device via DDP; validation runs only on rank 0
  torchrun --standalone --nproc_per_node="$NGPU" -m ssm.train "${args[@]}"
else
  python -m ssm.train "${args[@]}"
fi
