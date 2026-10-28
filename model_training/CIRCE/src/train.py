#!/usr/bin/env python
"""C-GATr FCC Lightning training driver.

Distributed-launch fix: match the working GATr torchrun/Lightning environment
selection without changing the C-GATr architecture, loss or ablation switches.
Based on model_training/CIRCE/src/train.py in the linked repository.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
import warnings
import signal as _signal
import multiprocessing as mp


def _silence_lightning_noise() -> None:
    class _DropLightningTips(logging.Filter):
        def filter(self, record):
            msg = record.getMessage()
            return not (
                msg.startswith("\U0001f4a1 Tip:")
                or "litlogger" in msg
                or "Lightning Experiments platform" in msg
            )
    for name in ("lightning.pytorch.utilities.rank_zero",
                 "lightning.fabric.utilities.rank_zero"):
        logging.getLogger(name).addFilter(_DropLightningTips())
    for msg in (
        r".*does not have many workers.*",
        r".*Trying to infer the `batch_size`.*",
        r".*Checkpoint directory .* exists and is not empty.*",
        r".*tensorboardX.*",
        r".*`weights_only=False`.*",
        r".*number of training batches.*is smaller than the logging interval.*",
        r".*It is recommended to use `self\.log\('.*', \.\.\., sync_dist=True\).*",
    ):
        warnings.filterwarnings("ignore", message=msg)


_silence_lightning_noise()
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import lightning as L
import torch
from lightning.pytorch.callbacks import Callback, ModelCheckpoint
from lightning.pytorch.plugins.environments import SLURMEnvironment
from lightning.pytorch.strategies import DDPStrategy
from torch.utils.data import DataLoader
from src.lightning_module import CGATrV35LightningModule
from src.dataset.parquet_dataset import (
    IDEAParquetDataset, collate_idea_events as collate_legacy_idea_events,
)
from shared_training.circe_parquet_dataset import SharedIDEAParquetDataset
from shared_training.collation import collate_shared_events
from shared_training.event_batching import (
    FixedEventBatchSampler, TokenBudgetBatchSampler,
)
from shared_training.wandb_logger import build_experiment_logger


class ValidationSweepModelCheckpoint(ModelCheckpoint):
    """Save one full checkpoint after each completed validation sweep."""
    def _save_checkpoint(self, trainer, filepath):
        model = trainer.lightning_module
        previous_value = getattr(model, "_saving_validation_sweep_checkpoint", False)
        model._saving_validation_sweep_checkpoint = True
        try:
            super()._save_checkpoint(trainer, filepath)
        finally:
            model._saving_validation_sweep_checkpoint = previous_value

    def on_validation_end(self, trainer, pl_module):
        if not getattr(pl_module, "_validation_working_points", None):
            return
        super().on_validation_end(trainer, pl_module)


def _parse_seed_range(s: str):
    a, b = s.split("-")
    return int(a), int(b) + 1


def _normalize_limit_batches(x):
    if x is None:
        return 1.0
    if x >= 1:
        return int(x)
    return float(x)


def _apply_dry_run(args):
    args.num_epochs = 1
    args.limit_train_batches = 20
    args.limit_val_batches = 5
    print("[cgatr_fcc] --dry_run: num_epochs=1, "
          "limit_train_batches=20, limit_val_batches=5", flush=True)


def parse_args():
    p = argparse.ArgumentParser(description="C-GATr FCC training (M1-M5 baked in)")
    p.add_argument("--data_dir", default=None)
    p.add_argument("--train_files", nargs="+", default=None)
    p.add_argument("--val_files", nargs="+", default=None)
    p.add_argument("--train_seeds", default="1-1000")
    p.add_argument("--val_seeds", default="1001-1196")
    p.add_argument("--max_hits", type=int, default=0)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--max_tokens", type=int, default=0)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--persistent_workers", action="store_true", default=False)
    p.add_argument("--prefetch_factor", type=int, default=2)
    p.add_argument("--num_epochs", type=int, default=100)
    p.add_argument("--num_devices", type=int, default=4)
    p.add_argument("--precision", default="32-true",
                   choices=["32-true", "16-mixed", "bf16-mixed"])
    p.add_argument("--gradient_clip_val", type=float, default=1.0)
    p.add_argument("--start_lr", type=float, default=3e-4)
    p.add_argument("--min_lr", type=float, default=1e-5)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--warmup_epochs", type=int, default=2)
    p.add_argument("--warmup_steps", type=int, default=None)
    p.add_argument("--lr_schedule", default="cosine",
                   choices=["cosine", "plateau", "step"])
    p.add_argument("--lr_step_epochs", type=int, default=4)
    p.add_argument("--lr_step_factor", type=float, default=0.1)
    p.add_argument("--optimizer", default="adamw", choices=["adamw", "adam"])
    p.add_argument("--plateau_patience", type=int, default=4)
    p.add_argument("--plateau_factor", type=float, default=0.5)
    p.add_argument("--terminal_anneal_epochs", type=int, default=0)
    p.add_argument("--num_blocks", type=int, default=10)
    p.add_argument("--hidden_mv_channels", type=int, default=16)
    p.add_argument("--hidden_s_channels", type=int, default=64)
    p.add_argument("--embed_dim", type=int, default=4)
    p.add_argument("--beta_mlp", action="store_true", default=False)
    p.add_argument("--cosine_norm", action="store_true", default=False)
    _boolean_optional = getattr(argparse, "BooleanOptionalAction", None)
    if _boolean_optional is not None:
        p.add_argument("--normalize_mv_inputs", action=_boolean_optional,
                       default=True)
    else:
        p.add_argument("--normalize_mv_inputs", dest="normalize_mv_inputs",
                       action="store_true", default=True)
        p.add_argument("--no-normalize_mv_inputs", dest="normalize_mv_inputs",
                       action="store_false")
    p.add_argument("--layernorm_epsilon_mode", choices=["clamp", "add"],
                   default="clamp")
    p.add_argument("--use_time", action="store_true", default=False)
    p.add_argument("--algebra", choices=["conformal", "projective"],
                   default="conformal")
    p.add_argument("--pga_hit_encoding",
                   choices=["line", "ggtf", "ggtf_wire", "point_line"],
                   default="line")
    p.add_argument("--cga_hit_encoding",
                   choices=["circle", "sphere_circle", "sphere_plane", "point_line"],
                   default="circle")
    p.add_argument("--physical_drift_geometry", action="store_true", default=False)
    p.add_argument("--separate_hit_metadata", action="store_true", default=False)
    p.add_argument("--separate_hit_type", action="store_true", default=False)
    p.add_argument("--legacy_equivariance", action="store_true", default=False)
    p.add_argument("--equivariance_group", choices=["e3", "se3"], default="se3")
    p.add_argument("--invariant_output_head", action="store_true", default=False)
    p.add_argument("--drop_loopers", action="store_true", default=False)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--fix_cga_null", action="store_true", default=False)
    p.add_argument("--fix_wire_dir", action="store_true", default=False)
    p.add_argument("--two_channel_dc", action="store_true", default=False)
    p.add_argument("--equi_init", default="default",
                   choices=["default", "identity_algebra"])
    p.add_argument("--merge_daughters", action="store_true", default=False)
    p.add_argument("--qmin", type=float, default=0.1)
    p.add_argument("--attr_weight", type=float, default=1.0)
    p.add_argument("--repul_weight", type=float, default=1.0)
    p.add_argument("--fill_loss_weight", type=float, default=0.0)
    p.add_argument("--use_average_cc_pos", type=float, default=0.0)
    p.add_argument("--beta_suppress_weight", type=float, default=0.1)
    p.add_argument("--var_weight", type=float, default=0.3)
    p.add_argument("--var_warmup_epochs", type=int, default=2)
    p.add_argument("--oc_mode", choices=["paper_hinge", "ggtf"],
                   default="paper_hinge")
    p.add_argument("--tbeta", type=float, default=0.1)
    p.add_argument("--td", type=float, default=0.2)
    p.add_argument("--sweep_tbeta_grid", default="0.35,0.45,0.5,0.6,0.7,0.8")
    p.add_argument("--sweep_td_grid", default="0.15,0.2,0.3,0.4,0.5,0.6")
    p.add_argument("--sweep_min_hits_grid", default="3,4")
    p.add_argument("--sweep_match_metric",
                   choices=("idea", "double_majority", "hungarian"),
                   default="double_majority")
    p.add_argument("--sweep_truth_min_hits", type=int, default=3,
                   help="truth N_hits cut for the binned efficiency curves "
                   "(post-reconstruction; also selects the clustering min_hits "
                   "working point). The scalar efficiency/fake-rate metrics "
                   "apply no truth cuts.")
    p.add_argument("--validation_sweep_max_events", type=int, default=500)
    p.add_argument("--rejected_seed_policy",
                   choices=("discard", "keep", "attach-after-accept"),
                   default="discard")
    p.add_argument("--ema_decay", type=float, default=0.999)
    p.add_argument("--output_dir", default="checkpoints/cgatr_fcc_prod")
    p.add_argument("--epoch_csv_path", default=None)
    p.add_argument("--run_tag", default="cgatr_fcc_prod")
    p.add_argument("--log_wandb", action="store_true", default=False)
    p.add_argument("--wandb_projectname", default=None)
    p.add_argument("--wandb_entity", default=None)
    p.add_argument("--wandb_displayname", default=None)
    p.add_argument("--resume_ckpt", default="none")
    p.add_argument("--init_weights", default="none")
    p.add_argument("--limit_train_batches", type=float, default=None)
    p.add_argument("--limit_val_batches", type=float, default=None)
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--legacy_variable_epoch_batches", action="store_true")
    p.add_argument("--fix_particle_zero", action="store_true")
    p.add_argument("--min_target_hits", type=int, default=0)
    p.add_argument("--secondaries_as_noise", action="store_true", default=False)
    p.add_argument("--max_time", default=None)
    p.add_argument("--ckpt_every_n_train_steps", type=int, default=200)
    p.add_argument("--auto_requeue", action="store_true", default=False)
    p.add_argument("--grad_checkpoint", action="store_true", default=False)
    p.add_argument("--compile", action="store_true", default=False)
    p.add_argument("--compile_mode", default="default",
                   choices=("default", "max-autotune-no-cudagraphs"))
    p.add_argument("--cpu_threads", type=int, default=4)
    return p.parse_args()


def make_loaders(args):
    """Build train and validation DataLoaders without rank-overlapping batches."""
    shared_files = args.train_files is not None or args.val_files is not None
    if shared_files:
        if args.data_dir is not None or not args.train_files or not args.val_files:
            raise ValueError("Use either --data_dir or both --train_files and --val_files.")
    elif args.data_dir is None:
        raise ValueError("Provide --data_dir or both shared file lists")
    tr_a, tr_b = _parse_seed_range(args.train_seeds)
    va_a, va_b = _parse_seed_range(args.val_seeds)

    max_hits = args.max_hits if args.max_hits > 0 else None
    print(("Loading shared training files" if shared_files else
           f"Loading training data: seeds {tr_a}-{tr_b - 1}")
          + f"  max_hits_per_event={max_hits}", flush=True)
    ggtf_targets = dict(
        drop_loopers=args.drop_loopers,
        merge_daughters=args.merge_daughters,
        with_drift_dir=(args.algebra == "projective" and
                        args.pga_hit_encoding in ("ggtf", "ggtf_wire")),
    )
    if any(ggtf_targets.values()):
        print(f"GGTF regime / inputs: {ggtf_targets}", flush=True)
    if shared_files:
        if args.drop_loopers or args.merge_daughters or args.oc_mode != "paper_hinge":
            raise ValueError(
                "The shared comparison path uses paper_hinge without "
                "truth-dependent hit deletion or daughter merging."
            )
        common = dict(
            max_hits_per_event=max_hits,
            with_time=args.use_time,
            with_drift_dir=ggtf_targets["with_drift_dir"],
        )
        train_ds = SharedIDEAParquetDataset(args.train_files, **common)
        val_ds = SharedIDEAParquetDataset(args.val_files, **common)
        collate_fn = collate_shared_events
    else:
        train_ds = IDEAParquetDataset(
            args.data_dir, seed_range=(tr_a, tr_b),
            max_hits_per_event=max_hits, with_time=args.use_time,
            ggtf_loss=args.oc_mode == "ggtf",
            min_target_hits=args.min_target_hits,
            secondaries_as_noise=args.secondaries_as_noise,
            **ggtf_targets,
        )
        print(f"Loading validation data: seeds {va_a}-{va_b - 1}"
              f"  max_hits_per_event={max_hits}", flush=True)
        val_ds = IDEAParquetDataset(
            args.data_dir, seed_range=(va_a, va_b),
            max_hits_per_event=max_hits, with_time=args.use_time,
            ggtf_loss=args.oc_mode == "ggtf",
            min_target_hits=args.min_target_hits,
            secondaries_as_noise=args.secondaries_as_noise,
            **ggtf_targets,
        )
        collate_fn = collate_legacy_idea_events

    pin = os.environ.get("CGATR_PIN_MEMORY", "1") not in ("0", "false", "False")
    base_kwargs = dict(
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=pin,
    )
    if args.num_workers > 0:
        base_kwargs["persistent_workers"] = args.persistent_workers
        base_kwargs["prefetch_factor"] = args.prefetch_factor
        mp_ctx_name = os.environ.get("CGATR_DATALOADER_MP_CTX", "spawn")
        if mp_ctx_name not in ("fork",):
            base_kwargs["multiprocessing_context"] = mp.get_context(mp_ctx_name)

    if args.max_tokens > 0:
        print(f"DataLoader: TokenBudgetBatchSampler  max_tokens={args.max_tokens}",
              flush=True)
        train_sampler = TokenBudgetBatchSampler(
            train_ds, max_tokens=args.max_tokens,
            shuffle=True, drop_last=True,
            stable_epoch_length=not args.legacy_variable_epoch_batches,
            seed=args.seed,
        )
        # Validation subset is deterministically shuffled over event sizes,
        # and is not advanced by the epoch callback.
        val_sampler = TokenBudgetBatchSampler(
            val_ds, max_tokens=args.max_tokens,
            shuffle=True, drop_last=False,
            stable_epoch_length=False,
            seed=args.seed,
        )
    else:
        print(f"DataLoader: shared fixed batch_size={args.batch_size}", flush=True)
        train_sampler = FixedEventBatchSampler(
            train_ds, args.batch_size, shuffle=True, drop_last=True,
            seed=args.seed,
        )
        val_sampler = FixedEventBatchSampler(
            val_ds, args.batch_size, shuffle=False, drop_last=False,
            seed=args.seed,
        )
    train_loader = DataLoader(train_ds, batch_sampler=train_sampler, **base_kwargs)
    val_loader = DataLoader(val_ds, batch_sampler=val_sampler, **base_kwargs)
    return train_loader, val_loader


class _BatchSamplerEpochCallback(Callback):
    def on_train_epoch_start(self, trainer, pl_module):
        loader = trainer.train_dataloader
        if loader is None:
            return
        bs = getattr(loader, "batch_sampler", None)
        if bs is not None and hasattr(bs, "set_epoch"):
            bs.set_epoch(trainer.current_epoch)


class _HeartbeatCallback(Callback):
    def __init__(self, every_n_steps: int = 50):
        super().__init__()
        self.every_n_steps = int(every_n_steps)
        self._t0 = time.perf_counter()

    def on_train_epoch_start(self, trainer, pl_module):
        self._t0 = time.perf_counter()

    def on_fit_start(self, trainer, pl_module):
        self._t0 = time.perf_counter()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if not trainer.is_global_zero:
            return
        if batch_idx % self.every_n_steps != 0:
            return
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs
        loss_val = float(loss.detach().item()) if hasattr(loss, "detach") else float(loss)
        lr = trainer.optimizers[0].param_groups[0]["lr"]
        dt = time.perf_counter() - self._t0
        print(
            f"  Epoch {pl_module.current_epoch + 1} | Batch {batch_idx} | "
            f"Loss {loss_val:.4f} | LR {lr:.2e} | Time {dt:.1f}s",
            flush=True,
        )


class _EpochCSVCallback(Callback):
    def __init__(self, csv_path: str, run_tag: str, world_size: int):
        super().__init__()
        self.csv_path = csv_path
        self.run_tag = run_tag
        self.world_size = world_size
        self._train_t0 = 0.0
        self._val_t0 = 0.0
        self._train_wall = 0.0
        self._val_wall = 0.0
        self._initialised = False
        self._validated_this_epoch = False

    def _init_file(self):
        os.makedirs(os.path.dirname(self.csv_path) or ".", exist_ok=True)
        fresh = not os.path.exists(self.csv_path) or os.path.getsize(self.csv_path) == 0
        if fresh:
            with open(self.csv_path, "w") as f:
                f.write(
                    "run,epoch,mean_train_loss,val_loss,val_match_loose,"
                    "val_match_strict50,wall_s_train,wall_s_val,lr,world_size,timestamp\n"
                )
        self._initialised = True

    def on_train_epoch_start(self, trainer, pl_module):
        self._train_t0 = time.perf_counter()
        self._validated_this_epoch = False

    def on_validation_epoch_start(self, trainer, pl_module):
        if trainer.sanity_checking:
            self._val_t0 = time.perf_counter()
            return
        self._train_wall = time.perf_counter() - self._train_t0
        self._val_t0 = time.perf_counter()

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return
        self._val_wall = time.perf_counter() - self._val_t0
        self._validated_this_epoch = True

    def on_train_epoch_end(self, trainer, pl_module):
        if not trainer.is_global_zero:
            return
        if not self._initialised:
            self._init_file()
        m = trainer.callback_metrics
        nan = float("nan")
        if self._validated_this_epoch:
            train_loss = float(m.get("train_loss", nan))
            val_loss = float(m.get("val_loss", nan))
            loose = float(m.get("val/match_rate", nan))
            strict50 = float(m.get("val/match_rate_strict50", nan))
            val_wall = self._val_wall
        else:
            # Avoid writing stale metrics from the previous epoch.
            train_loss = val_loss = loose = strict50 = nan
            val_wall = nan
            self._train_wall = time.perf_counter() - self._train_t0
        lr = trainer.optimizers[0].param_groups[0]["lr"]
        ts = time.strftime("%Y-%m-%dT%H:%M:%S")
        with open(self.csv_path, "a") as f:
            f.write(
                f"{self.run_tag},{pl_module.current_epoch + 1},"
                f"{train_loss:.6f},{val_loss:.6f},"
                f"{loose:.6f},{strict50:.6f},"
                f"{self._train_wall:.3f},{val_wall:.3f},"
                f"{lr:.6e},{self.world_size},{ts}\n"
            )
        print(
            f"[csv] epoch {pl_module.current_epoch + 1}: "
            f"train={train_loss:.4f} val={val_loss:.4f} "
            f"loose={loose:.4f} strict50={strict50:.4f} "
            f"train_s={self._train_wall:.1f} val_s={val_wall:.1f}"
            + ("" if self._validated_this_epoch else "   [no validation this epoch]"),
            flush=True,
        )


def main():
    args = parse_args()
    torch.backends.cudnn.benchmark = True
    # Preserve FP32 precision for geometric products.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    if args.cpu_threads > 0:
        torch.set_num_threads(args.cpu_threads)
        os.environ.setdefault("OMP_NUM_THREADS", str(args.cpu_threads))
        os.environ.setdefault("POLARS_MAX_THREADS", str(args.cpu_threads))
        os.environ.setdefault("MKL_NUM_THREADS", str(args.cpu_threads))
    L.seed_everything(args.seed, workers=True)

    if args.dry_run:
        _apply_dry_run(args)

    train_loader, val_loader = make_loaders(args)
    eff_max_hits = args.max_hits if args.max_hits > 0 else "ALL (uncapped)"
    print(f"[cgatr_fcc] effective max_hits per event = {eff_max_hits}", flush=True)

    module = CGATrV35LightningModule(args)
    if getattr(args, "compile", False):
        module.model.compile(dynamic=True, mode=args.compile_mode)
        print(
            "[cgatr_fcc] torch.compile enabled "
            f"(dynamic=True, mode={args.compile_mode}, attention excluded)",
            flush=True,
        )
    os.makedirs(args.output_dir, exist_ok=True)

    csv_path = args.epoch_csv_path or os.path.join(
        args.output_dir, "epoch_metrics.csv",
    )
    callbacks = [
        ValidationSweepModelCheckpoint(
            dirpath=args.output_dir,
            filename=(
                "validation_epoch={epoch}_step={step}_"
                "pareto_f1={val_pareto_f1:.4f}_"
                "max_eff={val_max_tracking_efficiency:.4f}"
            ),
            auto_insert_metric_name=False,
            every_n_epochs=1,
            save_top_k=-1,
            save_weights_only=False,
            save_on_train_epoch_end=False,
        ),
        _BatchSamplerEpochCallback(),
        _HeartbeatCallback(every_n_steps=50),
        _EpochCSVCallback(csv_path, args.run_tag, world_size=args.num_devices),
    ]
    if args.ckpt_every_n_train_steps and args.ckpt_every_n_train_steps > 0:
        callbacks.append(ModelCheckpoint(
            dirpath=args.output_dir,
            filename="cgatr_step{step:08d}",
            auto_insert_metric_name=False,
            every_n_train_steps=args.ckpt_every_n_train_steps,
            save_top_k=0,
            save_last=True,
            save_weights_only=False,
        ))

    if args.num_devices > 1:
        strategy = DDPStrategy(
            find_unused_parameters=False,
            broadcast_buffers=False,
            gradient_as_bucket_view=True,
            static_graph=True,
        )
    else:
        strategy = "auto"

    # The custom batch samplers already shard a shared global plan by rank.
    # As in GATr's shared-loader path, do not let Lightning replace them.
    use_distributed_sampler = False

    # DISTRIBUTED FIX: Under torchrun use the same Lightning auto-detection
    # approach as the working GATr driver. Never inject SLURMEnvironment into
    # torchrun-launched processes even if the SLURM allocation variables exist.
    # This preserves real native-SLURM support when torchrun is not used.
    plugins = []
    under_torchelastic = (
        torch.distributed.is_available()
        and torch.distributed.is_torchelastic_launched()
    )
    if under_torchelastic:
        print(
            "[cgatr_fcc] torchrun detected -> Lightning auto-selects "
            "TorchElasticEnvironment "
            f"(rank={os.environ.get('RANK')}, "
            f"local_rank={os.environ.get('LOCAL_RANK')}, "
            f"world_size={os.environ.get('WORLD_SIZE')})",
            flush=True,
        )
    elif "SLURM_JOB_ID" in os.environ:
        if args.auto_requeue:
            plugins.append(SLURMEnvironment(
                auto_requeue=True, requeue_signal=_signal.SIGUSR1,
            ))
        else:
            plugins.append(SLURMEnvironment(auto_requeue=False))
    else:
        print("[cgatr_fcc] automatic Lightning environment detection", flush=True)

    trainer = L.Trainer(
        default_root_dir=args.output_dir,
        max_epochs=args.num_epochs,
        max_time=args.max_time,
        devices=args.num_devices,
        accelerator="gpu",
        strategy=strategy,
        precision=args.precision,
        gradient_clip_val=args.gradient_clip_val,
        callbacks=callbacks,
        plugins=plugins or None,
        logger=build_experiment_logger(
            enabled=args.log_wandb,
            output_dir=args.output_dir,
            project=args.wandb_projectname,
            entity=args.wandb_entity,
            run_name=args.wandb_displayname or args.run_tag,
            csv_name="",
        ),
        log_every_n_steps=50,
        enable_progress_bar=False,
        enable_model_summary=True,
        num_sanity_val_steps=0,
        deterministic=False,
        use_distributed_sampler=use_distributed_sampler,
        limit_train_batches=_normalize_limit_batches(args.limit_train_batches),
        limit_val_batches=_normalize_limit_batches(args.limit_val_batches),
    )

    resume_path: str | None = None
    if args.resume_ckpt and args.resume_ckpt.lower() not in ("none", ""):
        if args.resume_ckpt == "last":
            import glob as _glob
            cands = (
                _glob.glob(os.path.join(args.output_dir, "last.ckpt"))
                + _glob.glob(os.path.join(args.output_dir, "last-v*.ckpt"))
            )
            best, best_step = None, -1
            for c in cands:
                try:
                    gs = torch.load(
                        c, map_location="cpu", weights_only=False,
                    ).get("global_step", -1)
                except Exception as exc:
                    print(f"[cgatr_fcc] WARN unreadable ckpt {c}: {exc}", flush=True)
                    gs = -1
                if gs is not None and gs > best_step:
                    best, best_step = c, gs
            resume_path = best
            if resume_path:
                print(
                    "[cgatr_fcc] resume=last -> newest ckpt "
                    f"{os.path.basename(resume_path)} (global_step={best_step})",
                    flush=True,
                )
        else:
            resume_path = args.resume_ckpt
        if resume_path:
            print(f"[cgatr_fcc] Resuming from {resume_path}", flush=True)

    t_total_start = time.perf_counter()
    if os.environ.get("CGATR_SKIP_VAL", "").strip() in ("1", "true", "True", "yes"):
        if trainer.is_global_zero:
            print("[cgatr_fcc] CGATR_SKIP_VAL=1 -> skipping validation entirely",
                  flush=True)
        val_loader = None
    if resume_path is None and args.init_weights and args.init_weights.lower() not in ("none", ""):
        state_dict = torch.load(
            args.init_weights, map_location="cpu", weights_only=False,
        )
        state_dict = state_dict.get("state_dict", state_dict)
        state_dict = {k[6:]: v for k, v in state_dict.items()
                      if k.startswith("model.")}
        module.model.load_state_dict(state_dict, strict=True)
        if trainer.is_global_zero:
            print(
                f"[cgatr_fcc] init_weights: loaded {len(state_dict)} model "
                f"tensors from {args.init_weights}", flush=True,
            )
    trainer.fit(module, train_loader, val_loader, ckpt_path=resume_path)
    total_wall = time.perf_counter() - t_total_start
    if trainer.is_global_zero:
        wall_path = os.path.join(args.output_dir, "total_wall_clock_s.txt")
        with open(wall_path, "w") as f:
            f.write(f"{total_wall:.3f}\n")
        print(f"[cgatr_fcc] total_wall_clock_s={total_wall:.1f}", flush=True)


if __name__ == "__main__":
    main()
