# How to create the dataset
### 📦 Dataset Preparation

Before training, it is necessary to create a dataset in a specific format. This task is handled by the folder [`data_creation`](data_creation/).

Given a set of files containing digitized hits from multiple events, they can be processed by adapting the code written for **IDEA v2** or **IDEA v3**, depending on the type of hits and the collection names.

For example, **IDEA v3** includes two types of hits:

- `edm4hep::SenseWire`
- `edm4hep::TrackerHitPlane`

The first type (`SenseWire`) is processed by the function [`store_hit_col_CDC`](data_creation/data_processing/IDEAv3/tools_tree.py), which converts the circles defined by the sense wires into left and right positions.

The second type (`TrackerHitPlane`) is handled by the function [`store_hit_col_VTX_SIW`](data_creation/data_processing/IDEAv3/tools_tree.py).

Both functions are used inside the script:  
[`process_tree.py`](data_creation/data_processing/IDEAv3/process_tree.py)

### 👨‍💻 What You Need to Do

To adapt the data creation for your use case:

1. Create your own custom functions to process the hit collections.
2. Integrate them into `process_tree.py`.
3. Run the script to convert your input files into `.root` files, where each event is stored as a `TTree`.

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

To train the model, go inside `model_training/GATR` and use the script [`src/train_lightning.py`](model_training/src/train_lightning.py). This script supports extensive configuration through command-line arguments, which are defined in [`src/utils/parser_args.py`](model_training/src/utils/parser_args.py).

### 📌 Example Command

```bash
torchrun --nproc_per_node=4 -m src.train_lightning \
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

- `--data-config`: `config_files/config_tracking_global_vector.yaml`  
- `--network-config`: `src/models/wrapper/example_model_tracking_gatr_v_plot.py`

These provide a reliable starting point for training the GATr model effectively.

# How to convert the model into ONNX

To run inference in C++, the `.ckpt` file may need to be converted into an `.onnx` file.  
This can be done inside the folder `conversion_to_onnx/`.