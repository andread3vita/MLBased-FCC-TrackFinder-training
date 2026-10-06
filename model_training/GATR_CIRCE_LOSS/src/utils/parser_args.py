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
parser.add_argument(
    "--weights-source",
    choices=("raw", "ema"),
    default="ema",
    help=(
        "weights used for non-resume checkpoint loads: 'ema' reproduces "
        "validation/inference weights and requires ema_state_dict; 'raw' uses "
        "the regular state_dict (ignored for a full checkpoint resume)"
    ),
)
parser.add_argument("--num-epochs", type=int, default=16, help="number of epochs")
parser.add_argument(
    "--steps-per-epoch",
    type=int,
    default=None,
    help="number of steps (iterations) per epochs; "
    "if neither of `--steps-per-epoch` or `--samples-per-epoch` is set, each epoch will run over all loaded samples",
)
parser.add_argument(
    "--limit-train-batches",
    type=float,
    default=None,
    help=(
        "shared-loader training limit: a fraction below 1 or an absolute "
        "batch count at least 1, matching CIRCE/Lightning semantics"
    ),
)
parser.add_argument(
    "--steps-per-epoch-val",
    type=int,
    default=None,
    help="number of steps (iterations) per epochs for validation; "
    "if neither of `--steps-per-epoch-val` or `--samples-per-epoch-val` is set, each epoch will run over all loaded samples",
)
parser.add_argument(
    "--limit-val-batches",
    type=int,
    default=40,
    help="maximum validation batches per rank; use -1 to process all events",
)
parser.add_argument(
    "--validate-before-training",
    action="store_true",
    help=(
        "run one standalone validation pass before trainer.fit(); disabled by "
        "default and independent of the regular epoch-end validation"
    ),
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
    default="adamW",
    choices=["adam", "adamW", "radam"],
    help="optimizer for training (default: adamW)",
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
    default="reduceplateau",
    choices=["none", "flat+decay", "reduceplateau"],
    help=(
        "learning-rate schedule: none, epoch warmup plus cosine decay "
        "(flat+decay), or validation-metric ReduceLROnPlateau"
    ),
)
parser.add_argument(
    "--plateau-factor",
    type=float,
    default=0.5,
    help="multiplicative LR reduction used by the reduceplateau scheduler",
)
parser.add_argument(
    "--plateau-patience",
    type=int,
    default=3,
    help="validation epochs without validation-loss improvement before reducing LR",
)
parser.add_argument(
    "--plateau-threshold",
    type=float,
    default=1e-4,
    help="relative validation-loss improvement required by the reduceplateau scheduler",
)
parser.add_argument(
    "--load-epoch",
    type=int,
    default=None,
    help="used to resume interrupted training, load model and optimizer state saved in the `epoch-%d_state.pt` and `epoch-%d_optimizer.pt` files",
)
parser.add_argument("--start-lr", type=float, default=4e-4, help="start learning rate")
parser.add_argument(
    "--gradient-clip-val",
    type=float,
    default=1.0,
    help=(
        "maximum global L2 gradient norm applied before each optimizer step; "
        "use 0 to disable gradient clipping"
    ),
)
parser.add_argument("--batch-size", type=int, default=128, help="batch size")
parser.add_argument(
    "--max-tokens",
    type=int,
    default=16000,
    help=(
        "maximum total graph nodes (hits) per batch; use 0 for "
        "fixed --batch-size batching"
    ),
)
parser.add_argument(
    "--shared-indexed-loader",
    action="store_true",
    default=False,
    help=(
        "use CIRCE's canonical map-style event index, batch packing, DDP "
        "division and validation order; only GATr's event-to-feature transform "
        "remains model-specific"
    ),
)
parser.add_argument(
    "--accumulate-grad-batches",
    type=int,
    default=1,
    help="mini-batches accumulated before each optimizer step",
)
parser.add_argument(
    "--checkpoint-every-n-train-steps",
    type=int,
    default=0,
    help="optimizer-step interval for weights-only checkpoints; 0 disables",
)
parser.add_argument(
    "--precision",
    choices=("32-true", "16-mixed", "bf16-mixed"),
    default="32-true",
    help="Lightning numerical precision shared with CIRCE",
)
parser.add_argument(
    "--seed",
    type=int,
    default=42,
    help="root seed used for model initialization and all DataLoader workers",
)
parser.add_argument(
    "--gpus",
    type=str,
    default="0",
    help=(
        "required comma-separated CUDA device IDs for training/testing, "
        "for example 0 or 0,1; CPU execution is not supported"
    ),
)
parser.add_argument(
    "--num-workers",
    type=int,
    default=1,
    help=(
        "number of spawned DataLoader subprocesses per rank; memory use and "
        "disk-access load increase approximately linearly"
    ),
)
parser.add_argument(
    "--prefetch-factor",
    type=int,
    default=2,
    help="batches prefetched by each DataLoader worker (used when --num-workers > 0)",
)
parser.add_argument(
    "--cpu-threads",
    type=int,
    default=4,
    help="shared PyTorch/OMP/MKL CPU thread count",
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
    choices=["nccl"],
    default=None,
    help=(
        "PyTorch distributed backend used for multi-GPU training "
        "(default: nccl); ignored for single-GPU training"
    ),
)
parser.add_argument(
    "--cross-validation",
    type=str,
    default=None,
    help="enable k-fold cross validation; input format: `variable_name%k`",
)
parser.add_argument(
    "--log-wandb", action="store_true", default=False,
    help="enable Weights & Biases logging",
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
parser.add_argument(
    "--clustering_space_dim", "--embedding-dim", "-clust_dim",
    type=int, default=4,
    help="number of learned object-condensation coordinates",
)
parser.add_argument("--gatr-blocks", type=int, default=10)
parser.add_argument("--hidden-mv-channels", type=int, default=16)
parser.add_argument("--hidden-s-channels", type=int, default=64)
parser.add_argument(
    "--use-detector-features",
    action="store_true",
    default=False,
    help=(
        "feed the four computed detector-specific scalar channels from the detector "
        "data config into GATr; omit this flag for geometry-only training"
    ),
)
parser.add_argument(
    "--layers-per-superlayer",
    type=int,
    nargs=14,
    default=[8] * 14,
    metavar="N",
    help=(
        "number of local layers in each of the 14 superlayers; used to compute "
        "global_layer from cumulative offsets"
    ),
)
parser.add_argument("--position-scale", type=float, default=1000.0)
parser.add_argument(
    "--attention-phi-sectors", type=int, default=1,
    help="experimental sparse mode: isolate this many phi sectors per event; 1 is full attention",
)
parser.add_argument("--gradient-checkpointing", action="store_true", default=False)
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
    "--qmin", type=float, default=3.0, help="define qmin for condensation"
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
    default=2.0,
    help="Repulsive term of the potential weight",
)

parser.add_argument(
    "--frac_cluster_loss",
    type=float,
    default=0.1,
    help="deprecated compatibility option; the active OC loss uses all object interactions",
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
    help="deprecated compatibility option; use --var-weight instead",
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
    dest="loss_type",
    type=str,
    default="hgcalimplementation",
    help="use the hgcal loss",
)

parser.add_argument(
    "--beta-suppress-weight", type=float, default=0.1,
    help="penalty on non-alpha signal betas to reduce duplicate seeds",
)
parser.add_argument(
    "--beta-second-weight", type=float, default=0.0,
    help="weight for the mean second-highest signal beta per truth object",
)
parser.add_argument(
    "--var-weight", type=float, default=0.3,
    help="within-truth-track embedding compactness weight",
)
parser.add_argument("--var-warmup-epochs", type=int, default=1)
parser.add_argument(
    "--hard-negative-weight", type=float, default=1.0,
    help="exponent for inverse nearest-truth-track delta-R repulsion weighting",
)
parser.add_argument("--hard-negative-max-weight", type=float, default=100.0)
parser.add_argument(
    "--pt-track-weighting",
    action="store_true",
    default=False,
    help="weight per-truth-track loss reductions using piecewise pT bins",
)
parser.add_argument(
    "--pt-track-weight-bin-edges",
    type=str,
    default="0.4,0.9,5.0",
    help="comma-separated pT bin edges in GeV for optional track weighting",
)
parser.add_argument(
    "--pt-track-weight-bin-weights",
    type=str,
    default="1.5,1.2,0.75,2.0",
    help="comma-separated positive track weights; requires len(edges)+1 values",
)
parser.add_argument(
    "--helix-loss-weight", type=float, default=0.0,
    help="auxiliary inverse-pT and direction regression weight",
)
parser.add_argument("--weight-decay", type=float, default=1e-4)
parser.add_argument("--min-lr", type=float, default=1e-5)
parser.add_argument("--warmup-epochs", type=float, default=2.0)
parser.add_argument(
    "--terminal-anneal-epochs",
    type=int,
    default=6,
    help=(
        "for reduceplateau, cap learning rate with a final half-cosine "
        "anneal to --min-lr; 0 disables"
    ),
)
parser.add_argument("--ema-decay", type=float, default=0.999)

# Cheap post-inference validation sweep. These are strings so a complete grid
# is recorded verbatim in CLI metadata and checkpoints.
parser.add_argument(
    "--sweep-tbeta-grid", type=str,
    default="0.35,0.45,0.5,0.6,0.7,0.8",
)
parser.add_argument(
    "--sweep-td-grid", type=str,
    default="0.15,0.2,0.3,0.4,0.5,0.6",
)
parser.add_argument("--sweep-min-hits-grid", type=str, default="3,4")
parser.add_argument(
    "--rejected-seed-policy",
    choices=("discard", "keep", "attach-after-accept"),
    default="discard",
    help=(
        "handling of a condensation-point seed whose candidate has fewer than "
        "min_hits: discard it, keep it available to later clusters, or attach "
        "it only after a later cluster independently reaches min_hits"
    ),
)
parser.add_argument(
    "--sweep-match-metric", choices=("idea", "double_majority", "hungarian"),
    default="double_majority",
)
parser.add_argument("--sweep-truth-min-hits", type=int, default=3)
parser.add_argument("--validation-sweep-max-events", type=int, default=500)
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
