# Training the GATr model

Run the commands below from `model_training/GATR` in an environment containing
the project dependencies. They are examples: replace the input paths, output
directory, GPU IDs, run name, and W&B settings with values for your training.

## Geometry-only training

```bash
python -m src.train_lightning \
    --data-train "$DATA_TRAIN" \
    --data-val "$DATA_VAL" \
    --data-config config_files/config_tracking_parquet.yaml \
    --clustering_loss_only \
    --clustering_space_dim 5 \
    --network-config src/models/wrapper/model_tracking_gatr.py \
    --model-prefix "$OUTPUT_DIR/" \
    --gatr-blocks 10 \
    --hidden-mv-channels 16 \
    --hidden-s-channels 64 \
    --num-workers 4 \
    --prefetch-factor 1 \
    --gpus "$GPU_IDS" \
    --seed 42 \
    --batch-size 8 \
    --accumulate-grad-batches 1 \
    --checkpoint-every-n-train-steps 12000 \
    --limit-val-batches 125 \
    --start-lr 4e-4 \
    --num-epochs 20 \
    --train-val-split 1.0 \
    --steps-per-epoch 30000 \
    --optimizer adamW \
    --weight-decay 1e-3 \
    --lr-scheduler flat+decay \
    --plateau-factor 0.5 \
    --plateau-patience 1 \
    --plateau-threshold 1e-3 \
    --warmup-epochs 2 \
    --min-lr 1e-6 \
    --ema-decay 0.999 \
    --fetch-by-files \
    --fetch-step 2 \
    --condensation \
    --log-wandb \
    --wandb-displayname "$RUN_NAME" \
    --wandb-projectname "$WANDB_PROJECT" \
    --wandb-entity "$WANDB_ENTITY" \
    --qmin 0.1 \
    --L_attractive_weight 1.0 \
    --L_repulsive_weight 1.0 \
    --beta-suppress-weight 0.1 \
    --beta-second-weight 0.2 \
    --var-weight 0.2 \
    --var-warmup-epochs 3 \
    --hard-negative-weight 1.0 \
    --hard-negative-max-weight 100 \
    --pt-track-weight-bin-edges 0.4,0.9,5.0 \
    --pt-track-weight-bin-weights 1.5,1.2,0.75,2.0 \
    --helix-loss-weight 0.5 \
    --attention-phi-sectors 1 \
    --validation-sweep-max-events 1000 \
    --sweep-tbeta-grid 0.2,0.35,0.5,0.6,0.7,0.75,0.8,0.85,0.9,0.95 \
    --sweep-td-grid 0.1,0.15,0.2,0.25,0.3,0.4,0.5,0.55,0.6 \
    --sweep-min-hits-grid 3 \
    --rejected-seed-policy attach-after-accept \
    --sweep-match-metric double_majority
```