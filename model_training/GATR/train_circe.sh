#!/bin/bash
# CIRCE champion production configuration, runnable as-is inside the gatr:v9
# (or ggtf-gatr:v9-cgatr) docker image.
#
# Proven Pareto champion parameters (settled by systematic 5-way ablation over
# 50,000 matched events):
#   data      Official July 2026 production (key4hep_2026_07_29/91GeV/IDEA_o1_v4/Zuds,
#             seeds 1-980 train, 981-1000 val) OR Zqq_uds seeds 1-180 + Loopers 201-1000
#             (point --data_dir at your parquet directory; converters in data_creation/)
#   targets   >= 3 hits, stored secondaries kept (the adapter applies the
#             create_garbage_label-style relabel)
#   loss      circe backend: attr 1.0, repul 2.0, qmin 3.0,
#             beta_suppress 0.1, var 0.3
#   optim     AdamW 4e-4, weight decay 1e-4, 2 warm-up epochs, flat, then a
#             half-cosine anneal to 1e-5 over the last 6 epochs (EMA 0.999)
#   batching  token budget 16k hits/batch
set -euo pipefail

DATA=${1:?usage: train_circe.sh <parquet_dir> <output_dir> [num_devices] [train_seeds] [val_seeds]}
OUT=${2:?usage: train_circe.sh <parquet_dir> <output_dir> [num_devices] [train_seeds] [val_seeds]}
DEVICES=${3:-4}
TRAIN_SEEDS=${4:-"1-980"}
VAL_SEEDS=${5:-"981-1000"}

python -u -m src.train_algebra_ab \
  --algebra circe \
  --loss_backend circe \
  --recipe circe \
  --reference_width \
  --data_dir "$DATA" \
  --train_seeds "$TRAIN_SEEDS" \
  --val_seeds "$VAL_SEEDS" \
  --qmin 3.0 \
  --repul_weight 2.0 \
  --beta_suppress_weight 0.1 \
  --var_weight 0.3 \
  --epochs 16 \
  --start_lr 4e-4 \
  --max_tokens 16000 \
  --num_devices "$DEVICES" \
  --output_dir "$OUT"
