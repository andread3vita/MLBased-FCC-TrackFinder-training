#!/usr/bin/env python

import os
import ast
import sys
import shutil
import glob
import argparse
import functools
import numpy as np
import math
import torch
import warnings
import random
from src.models.Gatr_withModifications import ExampleWrapper
from src.utils.detector_features import validate_layers_per_superlayer

from torch.utils.data import DataLoader
from src.utils.parser_args import parser


import lightning as L
from lightning.pytorch.strategies import DDPStrategy
from lightning.pytorch.callbacks import (
    Callback,
    TQDMProgressBar,
    ModelCheckpoint,
)
from lightning.pytorch.profilers import AdvancedProfiler

sys.path.append(os.path.join(os.path.dirname(__file__), "../"))

from src.utils.train_utils import (
    train_load,
    test_load,
)
from src.utils.import_tools import import_module
from src.utils.train_utils import (
    get_samples_steps_per_epoch,
    model_setup,
    get_rank_device,
)
from src.utils.validation import resolve_validation_batch_limit
from src.utils.token_batching import stable_steps_per_rank
from shared_training.wandb_logger import build_experiment_logger

import warnings
from dgl.base import DGLWarning

warnings.simplefilter("ignore", DGLWarning)

parser.add_argument(
    "--checkpoint-mode",
    choices=("auto", "resume", "weights"),
    default="auto",
    help=(
        "how to load --load-model-weights: 'resume' restores the complete "
        "training state, 'weights' starts a new run from model weights, and "
        "'auto' selects resume when optimizer state is present"
    ),
)
parser.add_argument(
    "--override-resume-lr",
    action="store_true",
    help=(
        "when resuming a full Lightning checkpoint, replace every restored "
        "optimizer parameter-group learning rate with --start-lr while "
        "retaining the remaining optimizer and scheduler state"
    ),
)

print("Using PyTorch version:", torch.__version__)


class ValidationSweepModelCheckpoint(ModelCheckpoint):
    """Save one full checkpoint containing both sweep working points."""

    def _save_checkpoint(self, trainer, filepath):
        model = trainer.lightning_module
        previous_value = getattr(
            model, "_saving_validation_sweep_checkpoint", False
        )
        model._saving_validation_sweep_checkpoint = True
        try:
            super()._save_checkpoint(trainer, filepath)
        finally:
            model._saving_validation_sweep_checkpoint = previous_value

    def on_validation_end(self, trainer, pl_module):
        # Do not serialize stale callback metrics if a validation loader
        # produced no sweep working point.
        if not getattr(pl_module, "_validation_working_points", None):
            return
        super().on_validation_end(trainer, pl_module)


class SharedBatchSamplerEpochCallback(Callback):
    """Advance the common CIRCE batch plan exactly once per train epoch."""

    def on_train_epoch_start(self, trainer, pl_module):
        loader = trainer.train_dataloader
        batch_sampler = getattr(loader, "batch_sampler", None)
        if batch_sampler is not None and hasattr(batch_sampler, "set_epoch"):
            batch_sampler.set_epoch(trainer.current_epoch)


class TrainingStabilityMonitor(Callback):
    """Log pre-clipping gradients and representation/normalization health.

    Lightning retains ownership of backward, gradient clipping, and optimizer
    stepping. Forward hooks only retain detached scalar summaries, so they do
    not keep the training graph alive.
    """

    _GRADIENT_MODULES = (
        ("gatr", "gatr"),
        ("clustering", "clustering"),
        ("beta", "beta"),
        ("helix", "helix_proxy"),
    )

    def __init__(self, beta_seed_threshold=0.85):
        super().__init__()
        self.beta_seed_threshold = float(beta_seed_threshold)
        self._hook_handles = []
        self._embedding_rms = None
        self._beta_mean = None
        self._beta_seed_fraction = None

    @staticmethod
    def _module_gradient_norm(module):
        gradient_norms = [
            parameter.grad.detach().float().norm(2)
            for parameter in module.parameters()
            if parameter.requires_grad and parameter.grad is not None
        ]
        if not gradient_norms:
            return None
        return torch.stack(gradient_norms).norm(2)

    def _capture_embedding(self, module, inputs, output):
        if not module.training:
            return
        values = output.detach().float()
        self._embedding_rms = values.square().mean().sqrt()

    def _capture_beta(self, module, inputs, output):
        if not module.training:
            return
        probabilities = torch.sigmoid(output.detach().float())
        self._beta_mean = probabilities.mean()
        self._beta_seed_fraction = (
            probabilities > self.beta_seed_threshold
        ).float().mean()

    def on_fit_start(self, trainer, pl_module):
        if self._hook_handles:
            return
        if hasattr(pl_module, "clustering"):
            self._hook_handles.append(
                pl_module.clustering.register_forward_hook(self._capture_embedding)
            )
        if hasattr(pl_module, "beta"):
            self._hook_handles.append(
                pl_module.beta.register_forward_hook(self._capture_beta)
            )

    def on_fit_end(self, trainer, pl_module):
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles = []

    def on_before_optimizer_step(self, trainer, pl_module, optimizer):
        total_norm = self._module_gradient_norm(pl_module)
        if total_norm is None:
            return

        clip_value = trainer.gradient_clip_val
        if clip_value is not None and float(clip_value) > 0.0:
            threshold = total_norm.new_tensor(float(clip_value))
            finite = torch.isfinite(total_norm)
            clipping_scale = torch.where(
                finite,
                torch.clamp(threshold / (total_norm + 1.0e-12), max=1.0),
                torch.zeros_like(total_norm),
            )
            was_clipped = (finite & (total_norm > threshold)).float()
        else:
            finite = torch.isfinite(total_norm)
            clipping_scale = finite.float()
            was_clipped = total_norm.new_zeros(())

        metrics = {
            "trainer/grad_norm_preclip": total_norm,
            "trainer/gradient_clipping_scale": clipping_scale,
            # Its epoch aggregate is the fraction of optimizer steps clipped.
            "trainer/gradient_was_clipped": was_clipped,
            "trainer/grad_norm_is_finite": finite.float(),
        }
        for metric_suffix, attribute_name in self._GRADIENT_MODULES:
            module = getattr(pl_module, attribute_name, None)
            if module is None:
                continue
            module_norm = self._module_gradient_norm(module)
            if module_norm is not None:
                metrics[f"trainer/grad_norm_{metric_suffix}"] = module_norm

        for name, value in metrics.items():
            pl_module.log(
                name,
                value,
                on_step=True,
                on_epoch=True,
                # DDP has already synchronized gradients at this hook. Keep
                # monitoring communication-free; W&B records rank zero.
                sync_dist=False,
                batch_size=1,
            )

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        metrics = {
            "trainer/embedding_rms": self._embedding_rms,
            "trainer/beta_mean": self._beta_mean,
            "trainer/beta_seed_fraction": self._beta_seed_fraction,
        }
        try:
            batch_size = len(batch[0].batch_num_nodes())
        except (AttributeError, IndexError, TypeError):
            batch_size = 1
        for name, value in metrics.items():
            if value is not None:
                pl_module.log(
                    name,
                    value,
                    on_step=True,
                    on_epoch=True,
                    # A rank-zero sample is sufficient for these diagnostics
                    # and avoids three extra reductions on every batch.
                    sync_dist=False,
                    batch_size=batch_size,
                )
        self._embedding_rms = None
        self._beta_mean = None
        self._beta_seed_fraction = None


def resolve_input_files(file_specs, option_name):
    """Expand input globs while preserving optional named-file groups."""
    resolved = []
    for file_spec in file_specs:
        if ":" in file_spec:
            group_name, pattern = file_spec.split(":", 1)
            prefix = f"{group_name}:"
        else:
            pattern = file_spec
            prefix = ""

        matches = sorted(glob.glob(pattern))
        valid_matches = [path for path in matches if os.path.isfile(path)]
        if not valid_matches:
            print(f"No valid files matched {option_name}: {file_spec}")
            continue
        resolved.extend(prefix + path for path in valid_matches)
    return resolved


def main():

    args = parser.parse_args()
    args.layers_per_superlayer = list(
        validate_layers_per_superlayer(args.layers_per_superlayer)
    )
    if not 0 <= args.seed <= 0xFFFFFFFF:
        parser.error("--seed must be between 0 and 4294967295")
    if args.batch_size < 1:
        parser.error("--batch-size must be a positive integer")
    if args.max_tokens < 0:
        parser.error("--max-tokens must be non-negative")
    if args.accumulate_grad_batches < 1:
        parser.error("--accumulate-grad-batches must be a positive integer")
    if not math.isfinite(args.gradient_clip_val) or args.gradient_clip_val < 0.0:
        parser.error("--gradient-clip-val must be a finite non-negative number")
    if args.checkpoint_every_n_train_steps < 0:
        parser.error("--checkpoint-every-n-train-steps must be non-negative")
    if args.terminal_anneal_epochs < 0:
        parser.error("--terminal-anneal-epochs must be non-negative")
    if args.terminal_anneal_epochs >= args.num_epochs:
        parser.error("--terminal-anneal-epochs must be smaller than --num-epochs")
    if args.num_workers < 0:
        parser.error("--num-workers must be a non-negative integer")
    if args.cpu_threads < 0:
        parser.error("--cpu-threads must be a non-negative integer")
    if args.prefetch_factor < 1:
        parser.error("--prefetch-factor must be a positive integer")
    if not args.gpus or not args.gpus.strip():
        parser.error(
            "--gpus must contain at least one CUDA device ID; "
            "CPU execution is not supported"
        )
    try:
        configured_gpus = [int(value) for value in args.gpus.split(",")]
    except ValueError:
        parser.error("--gpus must be a comma-separated list of integer device IDs")
    if any(device < 0 for device in configured_gpus):
        parser.error("--gpus device IDs must be non-negative")
    if len(set(configured_gpus)) != len(configured_gpus):
        parser.error("--gpus must not contain duplicate device IDs")
    if args.predict and len(configured_gpus) != 1:
        parser.error("--predict currently supports exactly one CUDA device")
    if args.backend is not None and len(configured_gpus) == 1:
        warnings.warn(
            "--backend is ignored because distributed training is disabled "
            "when using a single GPU",
            stacklevel=2,
        )
    if not args.log_wandb and any(
        value is not None
        for value in (
            args.wandb_displayname,
            args.wandb_projectname,
            args.wandb_entity,
        )
    ):
        parser.error(
            "--wandb-displayname, --wandb-projectname, and --wandb-entity "
            "require --log-wandb"
        )
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    if args.cpu_threads > 0:
        torch.set_num_threads(args.cpu_threads)
        os.environ.setdefault("OMP_NUM_THREADS", str(args.cpu_threads))
        os.environ.setdefault("MKL_NUM_THREADS", str(args.cpu_threads))
    L.seed_everything(args.seed, workers=True)
    
    validation_data_requested = bool(args.data_val)
    args.data_train = resolve_input_files(args.data_train, "--data-train")
    args.data_val = resolve_input_files(args.data_val, "--data-val")

    if validation_data_requested and not args.data_val:
        parser.error("--data-val was provided but did not match any valid files")

    if len(args.data_train) == 0 and len(args.data_test) == 0:
        print("No valid input files remaining. Quit.")
        sys.exit(1)
        
    args = get_samples_steps_per_epoch(args)
    if (
        args.max_tokens > 0
        and args.steps_per_epoch is None
        and not args.shared_indexed_loader
    ):
        (
            args.steps_per_epoch,
            token_event_count,
            min_global_batches,
            max_global_batches,
        ) = stable_steps_per_rank(
            args.data_train,
            args.max_tokens,
            len(configured_gpus),
        )
        print(
            "Token-budget epoch plan: "
            f"{token_event_count} events, max_tokens={args.max_tokens}, "
            f"{args.steps_per_epoch} steps/rank "
            f"(64-epoch global packing range "
            f"{min_global_batches}-{max_global_batches})",
            flush=True,
        )
    try:
        validation_batch_limit = resolve_validation_batch_limit(
            args.limit_val_batches
        )
    except ValueError as error:
        parser.error(str(error))
    if args.limit_train_batches is not None:
        if args.limit_train_batches <= 0:
            parser.error("--limit-train-batches must be positive")
        training_batch_limit = (
            int(args.limit_train_batches)
            if args.limit_train_batches >= 1
            else float(args.limit_train_batches)
        )
    else:
        training_batch_limit = (
            args.steps_per_epoch if args.steps_per_epoch is not None else 1.0
        )
    training_mode = not args.predict
    
    gpus, process_device = get_rank_device(args)
    torch.cuda.set_device(process_device)
    print(
        f"LOCAL_RANK={os.environ.get('LOCAL_RANK', '0')} uses "
        f"{process_device} from configured GPUs {gpus}"
    )

    model_output_dir = os.path.abspath(args.model_prefix)
    os.makedirs(model_output_dir, exist_ok=True)
    try:
        experiment_logger = build_experiment_logger(
            enabled=args.log_wandb,
            output_dir=model_output_dir,
            project=args.wandb_projectname,
            entity=args.wandb_entity,
            run_name=args.wandb_displayname,
            csv_name="",
        )
    except RuntimeError as error:
        parser.error(str(error))

    if training_mode:
        print("USING TRAINING MODE")

        validation_checkpoint_callback = ValidationSweepModelCheckpoint(
            dirpath=args.model_prefix,
            filename=(
                "validation_epoch={epoch}_step={step}_"
                "pareto_f1={val_pareto_f1:.4f}_"
                "max_eff={val_max_tracking_efficiency:.4f}"
            ),
            every_n_epochs=1,
            save_top_k=-1,
            save_weights_only=False,
            save_on_train_epoch_end=False,
            auto_insert_metric_name=False,
        )
        
        callbacks = [
            TQDMProgressBar(refresh_rate=50),
            validation_checkpoint_callback,
        ]
        if args.shared_indexed_loader:
            callbacks.append(SharedBatchSamplerEpochCallback())
        if args.checkpoint_every_n_train_steps > 0:
            callbacks.append(ModelCheckpoint(
                dirpath=args.model_prefix,
                filename="_{epoch}_{step}",
                every_n_train_steps=args.checkpoint_every_n_train_steps,
                save_top_k=-1,
                save_weights_only=True,
            ))

        if len(gpus) > 1:
            distributed_backend = args.backend or "nccl"
            if distributed_backend != "nccl":
                parser.error(
                    f"--backend {distributed_backend!r} is not supported for "
                    "CUDA training; use --backend nccl"
                )
            strategy = DDPStrategy(
                process_group_backend=distributed_backend,
                find_unused_parameters=False,
                broadcast_buffers=False,
                gradient_as_bucket_view=True,
                static_graph=True,
            )
        else:
            strategy = "auto"

        trainer = L.Trainer(
            callbacks=callbacks,
            accelerator="gpu",
            devices=gpus,
            default_root_dir=args.model_prefix,
            logger=experiment_logger,
            max_epochs=args.num_epochs,
            limit_train_batches=training_batch_limit,
            strategy=strategy,
            accumulate_grad_batches=args.accumulate_grad_batches,
            log_every_n_steps=50,
            limit_val_batches=validation_batch_limit,
            # Pre-training validation is explicitly opt-in below.
            num_sanity_val_steps=0,
            precision=args.precision,
            gradient_clip_val=args.gradient_clip_val,
            use_distributed_sampler=not args.shared_indexed_loader,
        )

        args.local_rank = trainer.global_rank
        train_loader, val_loader, data_config, train_input_names = train_load(args)

        resume_checkpoint_path = None
        checkpoint_mode = None
        if args.load_model_weights:
            checkpoint_mode = args.checkpoint_mode
            if checkpoint_mode in ("auto", "resume"):
                checkpoint_metadata = ExampleWrapper.checkpoint_training_metadata(
                    args.load_model_weights
                )
                contains_training_state = checkpoint_metadata[
                    "contains_training_state"
                ]
                if checkpoint_mode == "auto":
                    checkpoint_mode = (
                        "resume" if contains_training_state else "weights"
                    )
                elif not contains_training_state:
                    raise ValueError(
                        "Cannot resume from a weights-only checkpoint. "
                        "Use --checkpoint-mode weights to start a new run "
                        "from its model weights."
                    )

            if checkpoint_mode == "resume":
                saved_scheduler = checkpoint_metadata["lr_scheduler"]
                if saved_scheduler is not None and saved_scheduler != args.lr_scheduler:
                    print(
                        "Resume mode: restoring checkpoint scheduler "
                        f"{saved_scheduler!r}; ignoring requested scheduler "
                        f"{args.lr_scheduler!r}.",
                        flush=True,
                    )
                    args.lr_scheduler = saved_scheduler
                saved_lrs = checkpoint_metadata["learning_rates"]
                if saved_lrs:
                    if args.override_resume_lr:
                        print(
                            "Resume mode: replacing checkpoint optimizer LR(s) "
                            f"{saved_lrs} with --start-lr={args.start_lr}.",
                            flush=True,
                        )
                    else:
                        print(
                            "Resume mode: restoring checkpoint optimizer LR(s) "
                            f"{saved_lrs}; --start-lr is ignored.",
                            flush=True,
                        )
                print(
                    "Resuming model, optimizer, scheduler, epoch, and step from",
                    args.load_model_weights,
                )
                resume_checkpoint_path = args.load_model_weights
            args.checkpoint_mode = checkpoint_mode

        model = model_setup(args, data_config)
        if args.load_model_weights and checkpoint_mode == "weights":
            print(
                f"Loading {args.weights_source} model weights from",
                args.load_model_weights,
            )
            model.load_compatible_weights(
                args.load_model_weights, weights_source=args.weights_source
            )

        if args.validate_before_training:
            print("Running validation before training starts", flush=True)
            model.validation_output_tag = "before_training"
            try:
                trainer.validate(
                    model=model,
                    dataloaders=val_loader,
                    ckpt_path=resume_checkpoint_path,
                )
            finally:
                model.validation_output_tag = None
        else:
            print(
                "Skipping validation before training; regular epoch-end "
                "validation remains enabled.",
                flush=True,
            )

        trainer.fit(
            model=model,
            train_dataloaders=train_loader,
            val_dataloaders=val_loader,
            ckpt_path=resume_checkpoint_path,
        )

    elif args.data_test:
        trainer = L.Trainer(
            callbacks=[TQDMProgressBar(refresh_rate=1)],
            accelerator="gpu",
            devices=gpus,
            default_root_dir=args.model_prefix,
            logger=experiment_logger
        )
        
        test_loaders, data_config = test_load(args)

        model = model_setup(args, data_config)
        if args.load_model_weights:
            print(
                f"Loading {args.weights_source} model weights from",
                args.load_model_weights,
            )
            model.load_compatible_weights(
                args.load_model_weights, weights_source=args.weights_source
            )

        for name, get_test_loader in test_loaders.items():
            test_loader = get_test_loader()

            trainer.validate(
                model=model,
                # The selected state was loaded above. Passing the checkpoint
                # again would make Lightning overwrite EMA with its raw
                # checkpoint state_dict.
                ckpt_path=None,
                dataloaders=test_loader,
            )
        

if __name__ == "__main__":
    main()
