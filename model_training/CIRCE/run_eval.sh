#!/bin/bash
# Full FCC evaluation pipeline: forward pass -> metrics -> plots (unmerged + oracle-merged).
# Usage: N_GPUS=4 DATA_DIR=/path/to/v1_zqq_uds bash run_eval.sh /path/to/checkpoint.ckpt
set -uo pipefail
cd "$(dirname "$0")"

CKPT=${1:?Usage: bash run_eval.sh /path/to/checkpoint.ckpt}
DATA_DIR=${DATA_DIR:-/path/to/eos/v1_zqq_uds}
EVAL_SEEDS=${EVAL_SEEDS:-1001-1196}
N_GPUS=${N_GPUS:-4}
EMB_DIM=${EMB_DIM:-4}
NUM_BLOCKS=${NUM_BLOCKS:-10}
CPU_THREADS=${CPU_THREADS:-8}
TB=${TB:-0.025}
TD=${TD:-0.10}
TAG=${TAG:-$(basename "$CKPT" .ckpt)}
BASE=${BASE:-eval_results/$TAG}

export PYTHONPATH=.
export MPLBACKEND=Agg
export OMP_NUM_THREADS=$((CPU_THREADS/2))
export POLARS_MAX_THREADS=$CPU_THREADS

mkdir -p "$BASE"

echo "[eval] Checkpoint: $CKPT"
echo "[eval] Eval seeds: $EVAL_SEEDS, GPUs: $N_GPUS, output: $BASE"

# ---- Parse seed range ----
SEED_A=$(echo "$EVAL_SEEDS" | cut -d- -f1)
SEED_B=$(echo "$EVAL_SEEDS" | cut -d- -f2)
TOTAL=$((SEED_B - SEED_A + 1))
PER_GPU=$(( (TOTAL + N_GPUS - 1) / N_GPUS ))

# ---- STAGE 1: Sharded forward pass ----
echo "[eval] Stage 1: forward pass ($N_GPUS GPU shards)..."
PIDS=()
for g in $(seq 0 $((N_GPUS-1))); do
    S_START=$((SEED_A + g * PER_GPU))
    S_END=$((S_START + PER_GPU - 1))
    [[ $S_START -gt $SEED_B ]] && break
    [[ $S_END -gt $SEED_B ]] && S_END=$SEED_B
    SHARD_SEEDS="${S_START}-${S_END}"
    mkdir -p "$BASE/emb_shard_$g"
    CUDA_VISIBLE_DEVICES=$g POLARS_MAX_THREADS=$((CPU_THREADS/N_GPUS)) \
    python -u src/eval/forward_pass.py \
        --data_dir "$DATA_DIR" \
        --checkpoint "$CKPT" \
        --eval_seeds "$SHARD_SEEDS" \
        --max_hits 0 \
        --embed_dim $EMB_DIM \
        --num_blocks $NUM_BLOCKS \
        --gpu 0 \
        --cache_path "$BASE/emb_shard_$g" \
        > "$BASE/forward_shard_${g}.log" 2>&1 &
    PIDS+=($!)
done
for pid in "${PIDS[@]}"; do wait "$pid" || { echo "[eval] WARN: shard pid $pid failed"; }; done

# ---- Merge shard forward caches ----
echo "[eval] Merging forward shards..."
EVAL_BASE="$BASE" N_GPUS="$N_GPUS" python -c "
import os, sys, polars as pl
base = os.environ['EVAL_BASE']
n_gpus = int(os.environ['N_GPUS'])
shards = [pl.read_parquet(f'{base}/emb_shard_{g}/forward_hits.parquet') for g in range(n_gpus) if os.path.exists(f'{base}/emb_shard_{g}/forward_hits.parquet')]
if not shards: sys.exit(1)
merged = pl.concat(shards)
os.makedirs(f'{base}/emb_all', exist_ok=True)
merged.write_parquet(f'{base}/emb_all/forward_hits.parquet')
print(f'Merged {len(merged)} hits into {base}/emb_all/forward_hits.parquet')
"

# ---- STAGE 2: Build mc_signal ----
echo "[eval] Stage 2: building mc_signal.parquet..."
python -u src/eval/build_mc_signal.py \
    --cache "$BASE/emb_all/forward_hits.parquet" \
    --data_dir "$DATA_DIR" \
    --seed_start "$SEED_A" \
    --seed_end "$SEED_B" \
    --out "$BASE/mc_signal.parquet" \
    > "$BASE/build_mc_signal.log" 2>&1

# ---- STAGE 3: Cluster + match ----
echo "[eval] Stage 3: clustering + truth matching (tbeta=$TB td=$TD)..."
python -u src/eval/fcc_cache_parallel.py \
    --cache_path "$BASE/emb_all" \
    --mc_signal "$BASE/mc_signal.parquet" \
    --embed_dim $EMB_DIM \
    --tbeta $TB --td $TD \
    --beta_mode sigmoid \
    --workers $CPU_THREADS \
    --output_dir "$BASE/fcc_unmerged" \
    --tag "$TAG" \
    > "$BASE/fcc_cache.log" 2>&1

# ---- STAGE 4: Plot unmerged ----
echo "[eval] Stage 4: plotting unmerged..."
mkdir -p "$BASE/fcc_unmerged/plots"
python -u src/eval/plot_fcc_metrics.py \
    --cache_dir "$BASE/fcc_unmerged" \
    --output_dir "$BASE/fcc_unmerged/plots" \
    --tag "$TAG"

echo "[eval] Unmerged plots: $BASE/fcc_unmerged/plots/"

# ---- STAGE 5: Oracle merge + plot ----
for T in 0.50 0.65 0.75; do
    TDIR="$BASE/fcc_oracle_T${T}"
    echo "[eval] Stage 5: oracle merge T=$T..."
    python -u src/eval/truth_merge_diagnostic.py \
        --in_dir "$BASE/fcc_unmerged" \
        --out_dir "$TDIR" \
        --threshold "$T" \
        > "$BASE/oracle_merge_T${T}.log" 2>&1
    mkdir -p "$TDIR/plots"
    python -u src/eval/plot_fcc_metrics.py \
        --cache_dir "$TDIR" \
        --output_dir "$TDIR/plots" \
        --tag "${TAG}_oracle_T${T}"
    echo "[eval] Oracle T=$T plots: $TDIR/plots/"
    echo "NOTE: Oracle-merged fake rate is inflated by construction (super-clusters span multiple MC particles). Interpret only the merged efficiency." >> "$TDIR/plots/ORACLE_NOTE.txt"
done

echo ""
echo "[eval] DONE. Results:"
echo "  Unmerged:  $BASE/fcc_unmerged/plots/eff_vs_pt_idea.png"
echo "  Oracle:    $BASE/fcc_oracle_T*/plots/eff_vs_pt_idea.png"
echo "  Fake rate: $BASE/fcc_unmerged/plots/fake_rate_summary.png"
