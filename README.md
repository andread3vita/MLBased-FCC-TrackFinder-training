# GGTF track finder training (IDEA)

Repository for training the GGTF track finder for the FCC-ee IDEA drift chamber.
It covers three steps: creating the training dataset, training the models
(**CIRCE** and **GATr**), and converting a trained checkpoint to ONNX for C++
inference.

# Environment setup

Two different environments are involved:

### Dataset creation (Key4hep + HTCondor)

The dataset pipeline runs on HTCondor against a Key4hep nightly release. No
local setup is needed beyond access to CVMFS and Condor:
[`data_creation/scriptDatasetCreation.sh`](data_creation/scriptDatasetCreation.sh)
sources the Key4hep nightly from `/cvmfs/sw-nightlies.hsf.org/key4hep/` itself
(the nightly version, e.g. `2026-10-02`, is passed as an argument).

### Training (GPU Python environment)

Training needs a GPU environment with PyTorch, PyTorch Lightning,
`torch_scatter`, DGL (GATr arm only), `polars`, `pyarrow` and `wandb`. Two
setup recipes are used in this repository:

- **Apptainer/Singularity container** (the "GATr environment"; the
  matched launcher runs both models with the same Python executable, so it must
  also have the CIRCE requirements):

  ```bash
  singularity pull docker://dologarcia/gatr:v9
  singularity shell -B /eos/ -B /afs/ --nv gatr_v9.sif

  # additional Python dependencies inside the container
  pip install lightning
  pip install plotly
  pip install polars
  ```

- **Conda environment** (CIRCE standalone training): run
  [`model_training/CIRCE/setup_env.sh`](model_training/CIRCE/setup_env.sh) on a
  login node. It installs PyTorch 2.5.1 + CUDA 12.1, the matching
  `torch_scatter` wheel, and the packages in
  [`model_training/CIRCE/requirements.txt`](model_training/CIRCE/requirements.txt).

# How to create the dataset

Dataset creation is handled by [`data_creation/`](data_creation/); see its
[README](data_creation/README.md).

In short: run
[`data_creation/scriptDatasetCreation.sh`](data_creation/scriptDatasetCreation.sh),
which takes 10 arguments (`TRAIN_OR_TEST PIPELINE DETECTOR MINSEED MAXSEED
OUTDIR KEY4HEP_VERSION K4GEO_PATH K4FWCORE_PATH ACCOUNTING_GROUP`) and submits
Condor jobs through
[`data_creation/runDatasetCreation.py`](data_creation/runDatasetCreation.py).
Each job runs EDM4hep simulation + digitization and converts the digitized hits
into graph files. The fully tested pipeline is
[`condor_pipeline/IDEA/noBackground_parquet`](data_creation/condor_pipeline/IDEA/noBackground_parquet),
which writes the canonical training artifact
`Graphs_<seed>_<train|test>.parquet` under `OUTDIR/graph/` — a single
model-independent format consumed by both CIRCE and GATr (see the pipeline's
own [README](data_creation/condor_pipeline/IDEA/noBackground_parquet/README.md)).
Train and validation file lists must be disjoint.

# How to train the model

The production path is the matched CIRCE/GATr training described in
[`model_training/README.md`](model_training/README.md). From `model_training/`,
select one model per invocation:

```bash
./train_circe_gatr_shared.sh CIRCE \
  '/path/train/Graphs_*_train.parquet' \
  '/path/validation/Graphs_*_test.parquet' \
  /path/to/output 0,1,2,3
```

Replace `CIRCE` with `GATR` to train the GATr comparison arm on the same files
and hyperparameters. A quick single-GPU smoke run is available via
`train_circe_gatr_quick_checkpoint.sh`, and `resume_circe_gatr_shared.sh`
resumes an interrupted run.

Each model also has its own documentation:

- `model_training/CIRCE` — standalone CIRCE production training
  (`run_train.sh`), environment setup (`setup_env.sh` +
  `requirements.txt`), and FCC benchmark evaluation (`run_eval.sh`).
- [`model_training/GATR/README.md`](model_training/GATR/README.md) — standalone
  GATr training examples (`src/train_lightning.py`, options in
  `src/utils/parser_args.py`).
- [`model_training/GATR_CIRCE_LOSS/README.md`](model_training/GATR_CIRCE_LOSS/README.md)
  — the GATr comparison arm with the CIRCE loss.

# How to convert the model into ONNX

To run inference in C++, the `.ckpt` file may need to be converted into an
`.onnx` file. This is done in [`conversionModel/`](conversionModel/README.md).
