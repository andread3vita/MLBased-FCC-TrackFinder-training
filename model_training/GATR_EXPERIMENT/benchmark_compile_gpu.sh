#!/bin/bash
#SBATCH --job-name=gatr_compile_bench
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --qos=acc_ehpc
#SBATCH --account=ehpc1013

# Eager vs torch.compile'd GATr on one GPU (synthetic batches, no data).
#
# Usage (from a login node; the script submits itself):
#   bash benchmark_compile_gpu.sh OUTPUT_DIR [benchmark_compile_gpu.py options]
# e.g. bash benchmark_compile_gpu.sh ~/bench --blocks 10 --checkpoint
# Optional environment: QOS (e.g. acc_debug), TIME_LIMIT.

SCRIPT_PATH="$(readlink -f -- "${BASH_SOURCE[0]}")"
SCRIPT_DIR="$(dirname -- "$SCRIPT_PATH")"

if [[ "${1:-}" != "--worker" ]]; then
    if [[ $# -lt 1 ]]; then
        echo "Usage: $0 OUTPUT_DIR [benchmark options]" >&2
        exit 2
    fi
    mkdir -p "$1"
    OUTPUT_DIR="$(cd "$1" && pwd)"
    shift
    submit_options=()
    [[ -n "${QOS:-}" ]] && submit_options+=(--qos="$QOS")
    [[ -n "${TIME_LIMIT:-}" ]] && submit_options+=(--time="$TIME_LIMIT")
    echo "Submitting GATr compile benchmark; log in $OUTPUT_DIR/%j.out"
    exec sbatch "${submit_options[@]}" --chdir="$SCRIPT_DIR" \
        --output="$OUTPUT_DIR/%j.out" --error="$OUTPUT_DIR/%j.err" \
        --export=ALL "$SCRIPT_PATH" --worker "$@"
fi
shift

set -eo pipefail
# Same environment as train_dummy_checkpoint.sh.
module load miniforge/24.3.0-0
module load cuda/12.1
source "/gpfs/apps/MN5/ACC/MINIFORGE/24.3.0-0/etc/profile.d/conda.sh"
conda activate /gpfs/apps/MN5/ACC/MINIFORGE/24.3.0-0/envs/mlpf
source "/gpfs/scratch/ehpc1013/lena/envs/hitpf-overlay/bin/activate"
set -u
export TRITON_PTXAS_PATH="$CUDA_HOME/bin/ptxas"
# Triton builds its kernel launchers with $CC; MN5 modules set CC=icc
# (deprecated, prints remark #10441 per kernel). Use gcc when available.
if command -v gcc >/dev/null 2>&1; then export CC="$(command -v gcc)"; fi
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python benchmark_compile_gpu.py "$@"
