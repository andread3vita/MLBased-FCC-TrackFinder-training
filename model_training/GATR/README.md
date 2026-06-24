# How to train the model

## 🚀 Environment Setup

To begin, it's necessary to create a Python environment that supports both **GATr** and **Weights & Biases (wandb)**. We recommend using a Docker container for consistency and ease of setup.

You can retrieve and run the container using **Apptainer** (formerly Singularity) with the following commands:

```bash
singularity pull docker://dologarcia/gatr:v9
singularity shell -B /eos/ -B /afs/ --nv gatr_v9.sif

# Install any additional Python dependencies inside the container
pip install lightning
pip install plotly
```

## 🧠 Model Training

### 📌 Example Command

```bash
python -m src.train_lightning \
  --data-train Zcard_graphs_{1..100}.root \
  --data-config ../config_files/config_tracking.yaml \
  -clust -clust_dim 3 \
  --network-config src/models/wrapper/model_tracking_gatr.py \
  --model-prefix training_results/ \
  --num-workers 0 \
  --gpus 0,1,2,3 \
  --batch-size 4 \
  --start-lr 1e-3 \
  --num-epochs 100 \
  --optimizer ranger \
  --fetch-step 0.04 \
  --condensation \
  --log-wandb \
  --wandb-displayname GATr_example \
  --wandb-projectname <yourProject> \
  --wandb-entity <yourEntity> \
  --frac_cluster_loss 0 \
  --qmin 3 \
  --use-average-cc-pos 0.99
```

### ✅ Recommended Configuration

- `--data-config`: `config_files/config_tracking.yaml`  
- `--network-config`: `src/models/wrapper/model_tracking_gatr.py`

These provide a reliable starting point for training the GATr model effectively.