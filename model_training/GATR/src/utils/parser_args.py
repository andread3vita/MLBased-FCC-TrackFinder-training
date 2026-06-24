import argparse

parser = argparse.ArgumentParser()

parser.add_argument(
    "--freeze-clustering",
    action="store_true",
    default=False,
    help="Freeze the clustering part of the model",
)

parser.add_argument(
    "--regression-mode",
    action="store_true",
    default=False,
    help="run in regression mode if this flag is set; otherwise run in classification mode",
)
parser.add_argument(
    "--class-edges",
    action="store_true",
    default=False,
    help="run in classification mode with edges",
)
parser.add_argument("-c", "--data-config", type=str, help="data config YAML file")
parser.add_argument(
    "--extra-selection",
    type=str,
    default=None,
    help="Additional selection requirement, will modify `selection` to `(selection) & (extra)` on-the-fly",
)
parser.add_argument(
    "--extra-test-selection",
    type=str,
    default=None,
    help="Additional test-time selection requirement, will modify `test_time_selection` to `(test_time_selection) & (extra)` on-the-fly",
)
parser.add_argument(
    "-i",
    "--data-train",
    nargs="*",
    default=[],
    help="training files; supported syntax:"
    " (a) plain list, `--data-train /path/to/a/* /path/to/b/*`;"
    " (b) (named) groups [Recommended], `--data-train a:/path/to/a/* b:/path/to/b/*`,"
    " the file splitting (for each dataloader worker) will be performed per group,"
    " and then mixed together, to ensure a uniform mixing from all groups for each worker.",
)
parser.add_argument(
    "-l",
    "--data-val",
    nargs="*",
    default=[],
    help="validation files; when not set, will use training files and split by `--train-val-split`",
)
parser.add_argument(
    "-t",
    "--data-test",
    nargs="*",
    default=[],
    help="testing files; supported syntax:"
    " (a) plain list, `--data-test /path/to/a/* /path/to/b/*`;"
    " (b) keyword-based, `--data-test a:/path/to/a/* b:/path/to/b/*`, will produce output_a, output_b;"
    " (c) split output per N input files, `--data-test a%10:/path/to/a/*`, will split per 10 input files",
)
parser.add_argument(
    "-plot",
    "--data-plot",
    type=str,
    default="",
    help="make plots - specify the output dir in which they will be saved",
)
parser.add_argument(
    "--data-fraction",
    type=float,
    default=1,
    help="fraction of events to load from each file; for training, the events are randomly selected for each epoch",
)
parser.add_argument(
    "--file-fraction",
    type=float,
    default=1,
    help="fraction of files to load; for training, the files are randomly selected for each epoch",
)
parser.add_argument(
    "--fetch-by-files",
    action="store_true",
    default=False,
    help="When enabled, will load all events from a small number (set by ``--fetch-step``) of files for each data fetching. "
    "Otherwise (default), load a small fraction of events from all files each time, which helps reduce variations in the sample composition.",
)
parser.add_argument(
    "--fetch-step",
    type=float,
    default=0.01,
    help="fraction of events to load each time from every file (when ``--fetch-by-files`` is disabled); "
    "Or: number of files to load each time (when ``--fetch-by-files`` is enabled). Shuffling & sampling is done within these events, so set a large enough value.",
)
parser.add_argument(
    "--in-memory",
    action="store_true",
    default=False,
    help="load the whole dataset (and perform the preprocessing) only once and keep it in memory for the entire run",
)
parser.add_argument(
    "--train-val-split",
    type=float,
    default=0.8,
    help="training/validation split fraction",
)
parser.add_argument(
    "--no-remake-weights",
    action="store_true",
    default=False,
    help="do not remake weights for sampling (reweighting), use existing ones in the previous auto-generated data config YAML file",
)
parser.add_argument(
    "--demo",
    action="store_true",
    default=False,
    help="quickly test the setup by running over only a small number of events",
)
parser.add_argument(
    "--lr-finder",
    type=str,
    default=None,
    help="run learning rate finder instead of the actual training; format: ``start_lr, end_lr, num_iters``",
)
parser.add_argument(
    "--tensorboard",
    type=str,
    default=None,
    help="create a tensorboard summary writer with the given comment",
)
parser.add_argument(
    "--tensorboard-custom-fn",
    type=str,
    default=None,
    help="the path of the python script containing a user-specified function `get_tensorboard_custom_fn`, "
    "to display custom information per mini-batch or per epoch, during the training, validation or test.",
)
parser.add_argument(
    "-n",
    "--network-config",
    type=str,
    help="network architecture configuration file; the path must be relative to the current dir",
)
parser.add_argument(
    "-o",
    "--network-option",
    nargs=2,
    action="append",
    default=[],
    help="options to pass to the model class constructor, e.g., `--network-option use_counts False`",
)
parser.add_argument(
    "-m",
    "--model-prefix",
    type=str,
    default="models/{auto}/networkss",
    help="path to save or load the model; for training, this will be used as a prefix, so model snapshots "
    "will saved to `{model_prefix}_epoch-%d_state.pt` after each epoch, and the one with the best "
    "validation metric to `{model_prefix}_best_epoch_state.pt`; for testing, this should be the full path "
    "including the suffix, otherwise the one with the best validation metric will be used; "
    "for training, `{auto}` can be used as part of the path to auto-generate a name, "
    "based on the timestamp and network configuration",
)
parser.add_argument(
    "-p",
    "--model-pretrained",
    type=str,
    default="",
    help="Path to load the model from when training. Useful if your training has crashed in the middle.",
)
parser.add_argument(
    "--load-model-weights",
    type=str,
    default=None,
    help="initialize model with pre-trained weights",
)
parser.add_argument("--num-epochs", type=int, default=20, help="number of epochs")
parser.add_argument(
    "--steps-per-epoch",
    type=int,
    default=None,
    help="number of steps (iterations) per epochs; "
    "if neither of `--steps-per-epoch` or `--samples-per-epoch` is set, each epoch will run over all loaded samples",
)
parser.add_argument(
    "--steps-per-epoch-val",
    type=int,
    default=None,
    help="number of steps (iterations) per epochs for validation; "
    "if neither of `--steps-per-epoch-val` or `--samples-per-epoch-val` is set, each epoch will run over all loaded samples",
)
parser.add_argument(
    "--samples-per-epoch",
    type=int,
    default=None,
    help="number of samples per epochs; "
    "if neither of `--steps-per-epoch` or `--samples-per-epoch` is set, each epoch will run over all loaded samples",
)
parser.add_argument(
    "--samples-per-epoch-val",
    type=int,
    default=None,
    help="number of samples per epochs for validation; "
    "if neither of `--steps-per-epoch-val` or `--samples-per-epoch-val` is set, each epoch will run over all loaded samples",
)
parser.add_argument(
    "--optimizer",
    type=str,
    default="ranger",
    choices=["adam", "adamW", "radam", "ranger"],  # TODO: add more
    help="optimizer for the training",
)
parser.add_argument(
    "--optimizer-option",
    nargs=2,
    action="append",
    default=[],
    help="options to pass to the optimizer class constructor, e.g., `--optimizer-option weight_decay 1e-4`",
)
parser.add_argument(
    "--lr-scheduler",
    type=str,
    default="flat+decay",
    choices=[
        "none",
        "steps",
        "flat+decay",
        "flat+linear",
        "flat+cos",
        "one-cycle",
        "reduceplateau",
    ],
    help="learning rate scheduler",
)
parser.add_argument(
    "--warmup-steps",
    type=int,
    default=0,
    help="number of warm-up steps, only valid for `flat+linear` and `flat+cos` lr schedulers",
)
parser.add_argument(
    "--load-epoch",
    type=int,
    default=None,
    help="used to resume interrupted training, load model and optimizer state saved in the `epoch-%d_state.pt` and `epoch-%d_optimizer.pt` files",
)
parser.add_argument("--start-lr", type=float, default=5e-3, help="start learning rate")
parser.add_argument("--batch-size", type=int, default=128, help="batch size")
parser.add_argument(
    "--use-amp",
    action="store_true",
    default=False,
    help="use mixed precision training (fp16)",
)
parser.add_argument(
    "--gpus",
    type=str,
    default="0",
    help='device for the training/testing; to use CPU, set to empty string (""); to use multiple gpu, set it as a comma separated list, e.g., `1,2,3,4`',
)
parser.add_argument(
    "--predict-gpus",
    type=str,
    default=None,
    help='device for the testing; to use CPU, set to empty string (""); to use multiple gpu, set it as a comma separated list, e.g., `1,2,3,4`; if not set, use the same as `--gpus`',
)
parser.add_argument(
    "--num-workers",
    type=int,
    default=1,
    help="number of threads to load the dataset; memory consumption and disk access load increases (~linearly) with this numbers",
)
parser.add_argument(
    "--predict",
    action="store_true",
    default=False,
    help="run prediction instead of training",
)
parser.add_argument(
    "--predict-output",
    type=str,
    help="path to save the prediction output, support `.root` and `.parquet` format",
)
parser.add_argument(
    "--export-onnx",
    action="store_true",
    default=False,
    help="export the PyTorch model to ONNX model and save it at the given path (path must ends w/ .onnx); "
    "needs to set `--data-config`, `--network-config`, and `--model-prefix` (requires the full model path)",
)
parser.add_argument(
    "--io-test",
    action="store_true",
    default=False,
    help="test throughput of the dataloader",
)
parser.add_argument(
    "--copy-inputs",
    action="store_true",
    default=False,
    help="copy input files to the current dir (can help to speed up dataloading when running over remote files, e.g., from EOS)",
)
parser.add_argument(
    "--log",
    type=str,
    default="",
    help="path to the log file; `{auto}` can be used as part of the path to auto-generate a name, based on the timestamp and network configuration",
)
parser.add_argument(
    "--print",
    action="store_true",
    default=False,
    help="do not run training/prediction but only print model information, e.g., FLOPs and number of parameters of a model",
)
parser.add_argument(
    "--profile", action="store_true", default=False, help="run the profiler"
)
parser.add_argument(
    "--backend",
    type=str,
    choices=["gloo", "nccl", "mpi"],
    default=None,
    help="backend for distributed training",
)
parser.add_argument(
    "--cross-validation",
    type=str,
    default=None,
    help="enable k-fold cross validation; input format: `variable_name%k`",
)
parser.add_argument(
    "--log-wandb", action="store_true", default=False, help="use wandb for loging"
)
parser.add_argument(
    "--wandb-displayname",
    type=str,
    help="give display name to wandb run, if not entered a random one is generated",
)
parser.add_argument(
    "--wandb-projectname", type=str, help="project where the run is stored inside wandb"
)
parser.add_argument(
    "--wandb-entity", type=str, help="username or team name where you are sending runs"
)
parser.add_argument(
    "--clustering_loss_only", "-clust", action="store_true", default=False
)
parser.add_argument(
    "--clustering_and_energy_loss", "-clust_en", action="store_true", default=False
)
parser.add_argument("--clustering_space_dim", "-clust_dim", type=int, default=2)
parser.add_argument(
    "--n-noise",
    "-n-noise",
    type=int,
    default=0,
    help="Number of random features that get added to the input",
)
parser.add_argument(
    "--energy-loss",
    action="store_true",
    default=False,
    help="use energy loss of dij for edge importance, for now only implemented for the edge classification problem",
)
parser.add_argument(
    "--laplace",
    action="store_true",
    default=False,
    help="use laplace eigenvects with graph transformer",
)
parser.add_argument(
    "--diffs",
    action="store_true",
    default=False,
    help="use model with edge information",
)

parser.add_argument(
    "--train_cap",
    "-train_cap",
    type=int,
    default=None,
    help="Cap the number of training events",
)
parser.add_argument(
    "--val_cap",
    "-val_cap",
    type=int,
    default=None,
    help="Cap the number of validation events",
)
parser.add_argument(
    "--qmin", type=float, default=0.1, help="define qmin for condensation"
)

parser.add_argument(
    "--L_attractive_weight",
    type=float,
    default=1.0,
    help="Attractitve term of the potential weight",
)
parser.add_argument(
    "--L_repulsive_weight",
    type=float,
    default=1.0,
    help="Repulsive term of the potential weight",
)

parser.add_argument(
    "--frac_cluster_loss",
    type=float,
    default=0.1,
    help="Fraction of total pairs to use for the clustering loss",
)
parser.add_argument(
    "--condensation",
    action="store_true",
    default=False,
    help="use condensation loss and training",
)

parser.add_argument(
    "--energy_loss_delay",
    "-energy_loss_delay",
    default=0,
    type=int,
    help="Number of epochs before energy loss is active",
)

parser.add_argument(
    "--fill_loss_weight",
    default=0.0,
    type=float,
    help="weight for the fill loss to try to prevent mode collapse",
)

parser.add_argument(
    "--synthetic-graph-npart-range",
    "-synthetic",
    type=str,
    default="",
    help="Range of number of particles to use for synthetic graph generation: e.g. '3, 5'",
)

parser.add_argument(
    "--use-average-cc-pos",
    default=0.0,
    type=float,
    help="push the alpha to the mean of the coordinates in the object by this value",
)

parser.add_argument(
    "--losstype",
    type=str,
    default="hgcalimplementation",
    help="use the hgcal loss",
)
parser.add_argument(
    "--loss-regularization",
    action="store_true",
    default=False,
    help="use the hgcal regularization losses",
)


parser.add_argument(
    "--use_heads",
    action="store_true",
    default=False,
    help="Use the model with separate heads for the beta and the coords",
)

parser.add_argument(
    "--freeze_beta",
    action="store_true",
    default=False,
    help="freeze the beta head",
)

parser.add_argument(
    "--freeze_core",
    action="store_true",
    default=False,
    help="freeze the core of the model",
)

parser.add_argument(
    "--freeze_coords",
    action="store_true",
    default=False,
    help="freeze the coordinates head of the model",
)

parser.add_argument(
    "--beta_zeros", action="store_true", default=False, help="Add beta zeros loss"
)


parser.add_argument(
    "--copy_core_for_beta",
    action="store_true",
    default=False,
    help="Copy the core of the model for the beta head (the clustering core remains frozen)",
)


parser.add_argument(
    "--alternate_steps_beta_clustering",
    default=None,
    type=int,
    help="Alternate the training of the beta and clustering heads every N steps",
)


parser.add_argument(
    "--output_dir_inference_summary",
    default="",
    type=str,
    help="For inference_summary.py, specify the output directory. Otherwise, leave empty.",
)

parser.add_argument(
    "--tracks",
    action="store_true",
    default=False,
    help="Are we using track information",
)
parser.add_argument(
    "--correction",
    action="store_true",
    default=False,
    help="Train correction only",
)

parser.add_argument(
    "--global-features",
    default=False,
    action="store_true",
    help="if toggled, also adds global features to the graphs for energy correction",
)

parser.add_argument(
    "--graph-level-features",
    default=False,
    action="store_true",
    help="if toggled, considers the 'high-level' features for energy corr. (energy of the hits, number of the hits etc.)",
)


parser.add_argument(
    "--use-gt-clusters",
    default=False,
    action="store_true",
    help="If toggled, uses ground-truth clusters instead of the predicted ones by the model. We can use this to simulate 'ideal' clustering.",
)
parser.add_argument(
    "--tau",
    action="store_true",
    default=False,
    help="using tau mode for predict to store tau variables",
)

parser.add_argument(
    "--loss-type",
    type=str,
    default="hgcalimplementation",
    choices=["hgcalimplementation", "weighted"], 
    help="loss for the training",
)

# parser.add_argument(
#     "--matching-criteria",
#     type=str,
#     default="CMS-criteria",
#     choices=["CMS-criteria", "Ratios-criteria"], 
#     help="Choose which matching criteria must be used for the evaluation.",
# )
