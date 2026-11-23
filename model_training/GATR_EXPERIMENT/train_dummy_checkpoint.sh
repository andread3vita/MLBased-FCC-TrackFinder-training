#!/bin/bash
#SBATCH --job-name=gatr_experiment_dummy
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --qos=acc_ehpc
#SBATCH --account=ehpc1013

# Dummy GATR_EXPERIMENT training: a few train batches and one validation
# batch on one GPU, only to produce a checkpoint for
# conversionModel/convert_to_onnx_fast.py. The weights are NOT meaningful.
#
# Usage (from a login node; the script submits itself):
#   bash train_dummy_checkpoint.sh 'TRAIN_GLOB' 'VAL_GLOB' OUTPUT_DIR
# Optional environment: QOS (e.g. acc_debug), TIME_LIMIT, MAX_TRAIN_FILES=1,
# MAX_VAL_FILES=1, DUMMY_TRAIN_BATCHES=2, NUM_BLOCKS=10, HIDDEN_MV_CHANNELS=16,
# HIDDEN_S_CHANNELS=64, EMBED_DIM=4, COMPILE=1 (torch.compile the GATr
# network; COMPILE=0 runs eager), COMPILE_MODE=default.

SCRIPT_PATH="$(readlink -f -- "${BASH_SOURCE[0]}")"
SCRIPT_DIR="$(dirname -- "$SCRIPT_PATH")"

usage() {
    echo "Usage: $0 TRAIN_GLOB VAL_GLOB OUTPUT_DIR" >&2
    echo "Quote the globs so they reach the job unexpanded." >&2
}

# bash on a login node submits; sbatch executes this same file with --worker.
if [[ "${1:-}" != "--worker" ]]; then
    if [[ ! -f "$SCRIPT_DIR/src/train_lightning.py" ]]; then
        echo "ERROR: this script must stay in GATR_EXPERIMENT: $SCRIPT_DIR" >&2
        exit 1
    fi
    if [[ $# -ne 3 ]]; then usage; exit 2; fi
    mkdir -p "$3"
    OUTPUT_DIR="$(cd "$3" && pwd)"
    mkdir -p "$OUTPUT_DIR/slurm_logs"
    submit_options=()
    [[ -n "${QOS:-}" ]] && submit_options+=(--qos="$QOS")
    [[ -n "${TIME_LIMIT:-}" ]] && submit_options+=(--time="$TIME_LIMIT")
    echo "Submitting dummy GATR_EXPERIMENT training; logs in $OUTPUT_DIR/slurm_logs"
    exec sbatch "${submit_options[@]}" --chdir="$SCRIPT_DIR" \
        --output="$OUTPUT_DIR/slurm_logs/%j.out" \
        --error="$OUTPUT_DIR/slurm_logs/%j.err" \
        --export=ALL \
        "$SCRIPT_PATH" --worker "$1" "$2" "$OUTPUT_DIR"
fi
shift

set -eo pipefail
# MN5 environment (same as model_training/slurmTrain.sh). Conda/MPI scripts
# may reference unset variables, so -u is enabled only afterwards.
module load miniforge/24.3.0-0
# CUDA 12.1 toolkit: torch.compile/Triton need its ptxas and nvcc on the GPU
# nodes (the torch wheel itself is cu118 and does not ship them).
module load cuda/12.1
source "/gpfs/apps/MN5/ACC/MINIFORGE/24.3.0-0/etc/profile.d/conda.sh"
conda activate /gpfs/apps/MN5/ACC/MINIFORGE/24.3.0-0/envs/mlpf
source "/gpfs/scratch/ehpc1013/lena/envs/hitpf-overlay/bin/activate"
set -u
if [[ -z "${CUDA_HOME:-}" || ! -x "$CUDA_HOME/bin/ptxas" ]]; then
    echo "ERROR: module cuda/12.1 did not provide \$CUDA_HOME/bin/ptxas" >&2
    exit 1
fi
export TRITON_PTXAS_PATH="$CUDA_HOME/bin/ptxas"
# Triton builds its kernel launchers with $CC; MN5 modules set CC=icc
# (deprecated, prints remark #10441 per kernel). Use gcc when available.
if command -v gcc >/dev/null 2>&1; then export CC="$(command -v gcc)"; fi
echo "Python: $(command -v python) ($(python --version 2>&1))"
echo "CUDA toolkit: $CUDA_HOME ($(nvcc --version | tail -n 1))"
python -c "import torch, triton; print(f'torch {torch.__version__} (CUDA {torch.version.cuda}), triton {triton.__version__}, GPU available: {torch.cuda.is_available()}')"

if [[ $# -ne 3 ]]; then usage; exit 2; fi
TRAIN_FILES="$1"
VAL_FILES="$2"
OUTPUT_DIR="$3"
if [[ "$TRAIN_FILES" == "$VAL_FILES" ]]; then
    echo "Training and validation specifications must differ." >&2; exit 2
fi
# sbatch stages a copy of the script, so resolve the folder from --chdir.
EXPERIMENT_DIR="$(pwd)"
ROOT_DIR="$(dirname "$EXPERIMENT_DIR")"
if [[ ! -f "$EXPERIMENT_DIR/src/train_lightning.py" || ! -f "$ROOT_DIR/shared_training/validate_setup.py" ]]; then
    echo "ERROR: job must start in model_training/GATR_EXPERIMENT (got $EXPERIMENT_DIR)" >&2
    exit 1
fi

mapfile -t TRAIN_PATHS < <(compgen -G "$TRAIN_FILES" | sort | head -n "${MAX_TRAIN_FILES:-1}")
mapfile -t VAL_PATHS < <(compgen -G "$VAL_FILES" | sort | head -n "${MAX_VAL_FILES:-1}")
if [[ "${#TRAIN_PATHS[@]}" -eq 0 || "${#VAL_PATHS[@]}" -eq 0 ]]; then
    echo "The training or validation file specification matched no files." >&2
    exit 2
fi

export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONHASHSEED=42
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=disabled
RUN_DIR="$OUTPUT_DIR/dummy_${SLURM_JOB_ID:-local}"
compile_options=()
[[ "${COMPILE:-1}" == "1" ]] && compile_options+=(--compile --compile-mode "${COMPILE_MODE:-default}")
mkdir -p "$RUN_DIR"

echo "Train files: ${TRAIN_PATHS[*]}"
echo "Validation files: ${VAL_PATHS[*]}"
echo "Run directory: $RUN_DIR"

python "$ROOT_DIR/shared_training/validate_setup.py" \
    --train "${TRAIN_PATHS[@]}" --val "${VAL_PATHS[@]}" --gatr

# Same model/loss options as run_gatr in train_circe_gatr_shared.sh, with
# minimal data, a single sweep point and no W&B.
python -m src.train_lightning \
    --data-train "${TRAIN_PATHS[@]}" --data-val "${VAL_PATHS[@]}" \
    --data-config config_files/config_tracking_parquet.yaml \
    --network-config src/models/wrapper/model_tracking_gatr.py \
    --model-prefix "$RUN_DIR/gatr_experiment/" \
    --clustering_loss_only --clustering_space_dim "${EMBED_DIM:-4}" \
    --gatr-blocks "${NUM_BLOCKS:-10}" --hidden-mv-channels "${HIDDEN_MV_CHANNELS:-16}" \
    --hidden-s-channels "${HIDDEN_S_CHANNELS:-64}" --gpus 0 \
    --num-workers 0 --prefetch-factor 2 --cpu-threads 4 \
    --batch-size 1 --max-tokens 0 --shared-indexed-loader \
    --limit-train-batches "${DUMMY_TRAIN_BATCHES:-2}" --limit-val-batches 1 \
    --accumulate-grad-batches 1 --num-epochs 1 --seed 42 \
    --precision 32-true --checkpoint-every-n-train-steps 0 \
    --optimizer adamW --weight-decay 1e-4 --start-lr 4e-4 --min-lr 1e-5 \
    --lr-scheduler reduceplateau --warmup-epochs 0 \
    --plateau-patience 3 --plateau-factor 0.5 --terminal-anneal-epochs 0 \
    --gradient-clip-val 1.0 --ema-decay 0.999 --condensation \
    --qmin 3.0 --L_attractive_weight 1.0 --L_repulsive_weight 2.0 \
    --beta-suppress-weight 0.1 --var-weight 0.3 --var-warmup-epochs 1 \
    --sweep-tbeta-grid 0.7 --sweep-td-grid 0.3 \
    --sweep-min-hits-grid 3 --validation-sweep-max-events 1 \
    --sweep-match-metric double_majority --sweep-truth-min-hits 3 \
    --rejected-seed-policy attach-after-accept \
    ${compile_options[@]+"${compile_options[@]}"}

shopt -s nullglob
CHECKPOINTS=("$RUN_DIR"/gatr_experiment/validation_epoch=0_step=*.ckpt)
[[ "${#CHECKPOINTS[@]}" -eq 0 ]] && CHECKPOINTS=("$RUN_DIR"/gatr_experiment/*.ckpt)
shopt -u nullglob
if [[ "${#CHECKPOINTS[@]}" -eq 0 ]]; then
    echo "Training finished, but no checkpoint was found in $RUN_DIR/gatr_experiment." >&2
    exit 1
fi
echo "Checkpoint: ${CHECKPOINTS[0]}"
echo "Convert with (conversion environment):"
echo "  python conversionModel/convert_to_onnx_fast.py -w '${CHECKPOINTS[0]}' -o gatr_fast.onnx"
