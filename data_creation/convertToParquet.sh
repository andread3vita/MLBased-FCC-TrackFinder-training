#!/bin/bash

set -euo pipefail

if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <input_dir> <output_dir> [split] [n_jobs]"
    exit 1
fi

INPUT_DIR="$1"
OUTPUT_DIR="$2"
SPLIT="${3:-train}"
N_JOBS="${4:-8}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_SCRIPT="${REPO_DIR}/model_training/CIRCE/src/dataset/edm4hep_to_parquet.py"

mkdir -p "${OUTPUT_DIR}"

find "${INPUT_DIR}" -type f -name "*.root" \
    | sort \o
    | parallel -j "${N_JOBS}" '
        file="{}"
        seed=$(basename "$file" | sed -n "s/.*DIGI_\([0-9]\+\)_train\.root/\1/p")

        if [ -z "$seed" ]; then
            echo "Could not extract seed from $file" >&2
            exit 1
        fi

        echo "[seed $seed] Processing $file" && \
        python '"${PYTHON_SCRIPT}"' \
            --input_file "$file" \
            --seed "$seed" \
            --output_dir '"${OUTPUT_DIR}"' \
            --split '"${SPLIT}"'
    '