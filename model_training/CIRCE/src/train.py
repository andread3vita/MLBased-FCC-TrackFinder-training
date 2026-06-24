#!/usr/bin/env python
"""C-GATr FCC training driver via PyTorch Lightning.

M1-M5 improvements baked in (no env flags). SDPA-only attention for ONNX.
4xH100 SLURM production training.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
import warnings


def _silence_lightning_noise() -> None:
    class _DropLightningTips(logging.Filter):
        def filter(self, record):
            msg = record.getMessage()
            return not (
                msg.startswith("\U0001f4a1 Tip:")
                or "litlogger" in msg
                or "Lightning Experiments platform" in msg
            )

    for _name in (
        "lightning.pytorch.utilities.rank_zero",
        "lightning.fabric.utilities.rank_zero",
    ):
        logging.getLogger(_name).addFilter(_DropLightningTips())

    for _msg in (
        r".*does not have many workers.*",
        r".*Trying to infer the `batch_size`.*",
        r".*Checkpoint directory .* exists and is not empty.*",
        r".*tensorboardX.*",
        r".*`weights_only=False`.*",
        r".*number of training batches.*is smaller than the logging interval.*",
        r".*It is recommended to use `self\.log\('.*', \.\.\., sync_dist=True\).*",
    ):
        warnings.filterwarnings("ignore", message=_msg)


_silence_lightning_noise()

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import lightning as L
import torch
from lightning.pytorch.callbacks import (
    Callback, LearningRateMonitor, ModelCheckpoint,
)
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.plugins.environments import SLURMEnvironment
from lightning.pytorch.strategies import DDPStrategy
import multiprocessing as mp
from torch.utils.data import DataLoader

import signal as _signal

from src.lightning_module import CGATrV35LightningModule
from src.dataset.parquet_dataset import (
    IDEAParquetDataset, TokenBudgetBatchSampler, collate_idea_events,
)


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------
def _parse_seed_range(s: str):
    a, b = s.split("-")
    return int(a), int(b) + 1


def _normalize_limit_batches(x):
    if x is None:
        return 1.0
    if x >= 1:
        return int(x)
    return float(x)


def _apply_dry_run(args) -> None:
    args.num_epochs = 1
    args.limit_train_batches = 20
    args.limit_val_batches = 5
    print(
        "[cgatr_fcc] --dry_run: num_epochs=1, "
        "limit_train_batches=20, limit_val_batches=5",
        flush=True,
    )


def parse_args():
    p = argparse.ArgumentParser(description="C-GATr FCC training (M1-M5 baked in)")
    p.add_argument("--data_dir", required=True)
    p.add_argument("--train_seeds", default="1-1000")
    p.add_argument("--val_seeds", default="1001-1196")
    p.add_argument(
        "--max_hits", type=int, default=0,
        help="Per-event hard hit-count cap. 0 (default) = no cap.",
    )
    p.add_argument("--batch_size", type=int, default=2,
                   help="Used only when --max_tokens is 0.")
    p.add_argument(
        "--max_tokens", type=int, default=0,
        help="If >0, use TokenBudgetBatchSampler with this packed-batch budget.",
    )
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument(
        "--persistent_workers", action="store_true", default=False,
        help="Keep DataLoader workers alive across epochs.")
    p.add_argument(
        "--prefetch_factor", type=int, default=2,
        help="Number of batches each worker pre-builds.")

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

    p.add_argument("--num_blocks", type=int, default=10)
    p.add_argument("--hidden_mv_channels", type=int, default=16)
    p.add_argument("--hidden_s_channels", type=int, default=64)
    p.add_argument("--embed_dim", type=int, default=4,
                   help="Clustering-coord dimensionality.")
    p.add_argument("--beta_mlp", action="store_true", default=False)
    p.add_argument("--cosine_norm", action="store_true", default=False)
    p.add_argument("--normalize_mv_inputs",
                   action=argparse.BooleanOptionalAction, default=True)

    p.add_argument("--qmin", type=float, default=0.1)
    p.add_argument("--attr_weight", type=float, default=1.0)
    p.add_argument("--repul_weight", type=float, default=1.0)
    p.add_argument("--fill_loss_weight", type=float, default=0.0)
    p.add_argument("--use_average_cc_pos", type=float, default=0.0)
    p.add_argument("--beta_suppress_weight", type=float, default=0.1)
    p.add_argument("--var_weight", type=float, default=0.3)
    p.add_argument("--var_warmup_epochs", type=int, default=2)

    p.add_argument("--tbeta", type=float, default=0.1)
    p.add_argument("--td", type=float, default=0.2)
    p.add_argument("--ema_decay", type=float, default=0.999)

    p.add_argument("--output_dir", default="checkpoints/cgatr_fcc_prod")
    p.add_argument("--epoch_csv_path", default=None)
    p.add_argument("--run_tag", default="cgatr_fcc_prod")
    p.add_argument("--resume_ckpt", default="none",
                   help="'last' / explicit path / 'none' to disable.")
    p.add_argument("--init_weights", default="none",
                   help="Path to a checkpoint to load MODEL WEIGHTS ONLY.")

    p.add_argument(
        "--limit_train_batches", type=float, default=None,
        help="Fraction (<1) or absolute count (>=1) of train batches per epoch.",
    )
    p.add_argument(
        "--limit_val_batches", type=float, default=None,
        help="Same convention as --limit_train_batches.",
    )
    p.add_argument(
        "--dry_run", action="store_true",
        help="Shortcut: 1 epoch, 20 train batches, 5 val batches.",
    )
    p.add_argument(
        "--max_time", default=None,
        help="Pass-through to L.Trainer(max_time=...). 'HH:MM:SS'.",
    )
    p.add_argument(
        "--ckpt_every_n_train_steps", type=int, default=200,
        help="If >0, write a step-based checkpoint every N training steps.",
    )
    p.add_argument(
        "--auto_requeue", action="store_true", default=False,
        help="Install Lightning's SLURMEnvironment(auto_requeue=True).",
    )
    p.add_argument(
        "--grad_checkpoint", action="store_true", default=False,
        help="Enable activation checkpointing in CGATr blocks (~30%% slower, "
             "saves large activation memory; recommended when max_tokens > 8000).",
    )
    p.add_argument(
        "--cpu_threads", type=int, default=4,
        help="Set torch.set_num_threads and OMP/POLARS/MKL thread counts. "
             "Central knob for the whole pipeline.",
    )
    return p.parse_args()


def make_loaders(args):
    """Build (train, val) DataLoaders."""
    tr_a, tr_b = _parse_seed_range(args.train_seeds)
    va_a, va_b = _parse_seed_range(args.val_seeds)

    max_hits = args.max_hits if args.max_hits > 0 else None

    print(f"Loading training data: seeds {tr_a}-{tr_b - 1}"
          f"  max_hits_per_event={max_hits}", flush=True)
    train_ds = IDEAParquetDataset(args.data_dir, seed_range=(tr_a, tr_b),
                                  max_hits_per_event=max_hits)
    print(f"Loading validation data: seeds {va_a}-{va_b - 1}"
          f"  max_hits_per_event={max_hits}", flush=True)
    val_ds = IDEAParquetDataset(args.data_dir, seed_range=(va_a, va_b),
                                max_hits_per_event=max_hits)

    _pin = os.environ.get("CGATR_PIN_MEMORY", "1") not in ("0", "false", "False")
    base_kwargs = dict(
        num_workers=args.num_workers,
        collate_fn=collate_idea_events,
        pin_memory=_pin,
    )
    if args.num_workers > 0:
        base_kwargs["persistent_workers"] = args.persistent_workers
        base_kwargs["prefetch_factor"] = args.prefetch_factor
        mp_ctx_name = os.environ.get("CGATR_DATALOADER_MP_CTX", "spawn")
        if mp_ctx_name not in ("fork",):
            base_kwargs["multiprocessing_context"] = mp.get_context(mp_ctx_name)

    if args.max_tokens > 0:
        print(
            f"DataLoader: TokenBudgetBatchSampler  max_tokens={args.max_tokens}",
            flush=True,
        )
        train_sampler = TokenBudgetBatchSampler(
            train_ds, max_tokens=args.max_tokens,
            shuffle=True, drop_last=True,
        )
        val_sampler = TokenBudgetBatchSampler(
            val_ds, max_tokens=args.max_tokens,
            shuffle=False, drop_last=False,
        )
        train_loader = DataLoader(
            train_ds, batch_sampler=train_sampler, **base_kwargs,
        )
        val_loader = DataLoader(
            val_ds, batch_sampler=val_sampler, **base_kwargs,
        )
    else:
        print(f"DataLoader: fixed batch_size={args.batch_size}", flush=True)
        train_loader = DataLoader(
            train_ds, shuffle=True, drop_last=True,
            batch_size=args.batch_size, **base_kwargs,
        )
        val_loader = DataLoader(
            val_ds, shuffle=False, drop_last=False,
            batch_size=args.batch_size, **base_kwargs,
        )
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

    def _init_file(self):
        os.makedirs(os.path.dirname(self.csv_path) or ".", exist_ok=True)
        with open(self.csv_path, "w") as f:
            f.write(
                "run,epoch,mean_train_loss,val_loss,val_match_loose,"
                "val_match_strict50,wall_s_train,wall_s_val,lr,world_size\n"
            )
        self._initialised = True

    def on_train_epoch_start(self, trainer, pl_module):
        self._train_t0 = time.perf_counter()

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

    def on_train_epoch_end(self, trainer, pl_module):
        if not trainer.is_global_zero:
            return
        if not self._initialised:
            self._init_file()
        m = trainer.callback_metrics
        train_loss = float(m.get("train_loss", float("nan")))
        val_loss = float(m.get("val_loss", float("nan")))
        loose = float(m.get("val/match_rate", float("nan")))
        strict50 = float(m.get("val/match_rate_strict50", float("nan")))
        lr = trainer.optimizers[0].param_groups[0]["lr"]
        with open(self.csv_path, "a") as f:
            f.write(
                f"{self.run_tag},{pl_module.current_epoch + 1},"
                f"{train_loss:.6f},{val_loss:.6f},"
                f"{loose:.6f},{strict50:.6f},"
                f"{self._train_wall:.3f},{self._val_wall:.3f},"
                f"{lr:.6e},{self.world_size}\n"
            )
        print(
            f"[csv] epoch {pl_module.current_epoch + 1}: "
            f"train={train_loss:.4f} val={val_loss:.4f} "
            f"loose={loose:.4f} strict50={strict50:.4f} "
            f"train_s={self._train_wall:.1f} val_s={self._val_wall:.1f}",
            flush=True,
        )


def main():
    args = parse_args()

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    if args.cpu_threads > 0:
        torch.set_num_threads(args.cpu_threads)
        os.environ.setdefault("OMP_NUM_THREADS", str(args.cpu_threads))
        os.environ.setdefault("POLARS_MAX_THREADS", str(args.cpu_threads))
        os.environ.setdefault("MKL_NUM_THREADS", str(args.cpu_threads))

    L.seed_everything(42, workers=True)

    if args.dry_run:
        _apply_dry_run(args)

    train_loader, val_loader = make_loaders(args)
    eff_max_hits = args.max_hits if args.max_hits > 0 else "ALL (uncapped)"
    print(f"[cgatr_fcc] effective max_hits per event = {eff_max_hits}", flush=True)

    module = CGATrV35LightningModule(args)

    os.makedirs(args.output_dir, exist_ok=True)

    csv_path = args.epoch_csv_path or os.path.join(
        args.output_dir, "epoch_metrics.csv",
    )

    callbacks = [
        ModelCheckpoint(
            dirpath=args.output_dir,
            filename="cgatr_epoch{epoch:02d}",
            auto_insert_metric_name=False,
            every_n_epochs=1,
            save_top_k=-1,
            save_last=True,
            save_weights_only=False,
        ),
        ModelCheckpoint(
            dirpath=args.output_dir,
            filename="cgatr_best",
            auto_insert_metric_name=False,
            monitor="val_loss",
            mode="min",
            save_top_k=1,
            save_weights_only=False,
        ),
        LearningRateMonitor(logging_interval="epoch"),
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

    use_distributed_sampler = (args.max_tokens == 0)

    plugins = []
    _under_slurm = "SLURM_JOB_ID" in os.environ
    if not _under_slurm:
        print("[cgatr_fcc] no SLURM_JOB_ID -> direct DDP (no SLURMEnvironment)", flush=True)
    elif args.auto_requeue:
        plugins.append(SLURMEnvironment(
            auto_requeue=True, requeue_signal=_signal.SIGUSR1,
        ))
    else:
        plugins.append(SLURMEnvironment(auto_requeue=False))

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
        logger=CSVLogger(args.output_dir, name=""),
        log_every_n_steps=50,
        enable_progress_bar=False,
        enable_model_summary=True,
        deterministic=False,
        use_distributed_sampler=use_distributed_sampler,
        limit_train_batches=_normalize_limit_batches(args.limit_train_batches),
        limit_val_batches=_normalize_limit_batches(args.limit_val_batches),
    )

    resume_path: str | None = None
    if args.resume_ckpt and args.resume_ckpt.lower() not in ("none", ""):
        if args.resume_ckpt == "last":
            import glob as _glob
            cands = (_glob.glob(os.path.join(args.output_dir, "last.ckpt"))
                     + _glob.glob(os.path.join(args.output_dir, "last-v*.ckpt")))
            best, best_step = None, -1
            for c in cands:
                try:
                    gs = torch.load(c, map_location="cpu", weights_only=False).get("global_step", -1)
                except Exception as _e:
                    print(f"[cgatr_fcc] WARN unreadable ckpt {c}: {_e}", flush=True)
                    gs = -1
                if gs is not None and gs > best_step:
                    best, best_step = c, gs
            resume_path = best
            if resume_path:
                print(f"[cgatr_fcc] resume=last -> newest ckpt {os.path.basename(resume_path)} (global_step={best_step})", flush=True)
        else:
            resume_path = args.resume_ckpt
        if resume_path:
            print(f"[cgatr_fcc] Resuming from {resume_path}", flush=True)

    t_total_start = time.perf_counter()
    if os.environ.get("CGATR_SKIP_VAL", "").strip() in ("1", "true", "True", "yes"):
        if trainer.is_global_zero:
            print("[cgatr_fcc] CGATR_SKIP_VAL=1 -> skipping validation entirely", flush=True)
        val_loader = None
    if resume_path is None and args.init_weights and args.init_weights.lower() not in ("none", ""):
        _isd = torch.load(args.init_weights, map_location="cpu", weights_only=False)
        _isd = _isd.get("state_dict", _isd)
        _isd = {k[6:]: v for k, v in _isd.items() if k.startswith("model.")}
        module.model.load_state_dict(_isd, strict=True)
        if trainer.is_global_zero:
            print(f"[cgatr_fcc] init_weights: loaded {len(_isd)} model tensors from {args.init_weights}", flush=True)
    trainer.fit(module, train_loader, val_loader, ckpt_path=resume_path)
    total_wall = time.perf_counter() - t_total_start

    if trainer.is_global_zero:
        wall_path = os.path.join(args.output_dir, "total_wall_clock_s.txt")
        with open(wall_path, "w") as f:
            f.write(f"{total_wall:.3f}\n")
        print(f"[cgatr_fcc] total_wall_clock_s={total_wall:.1f}", flush=True)


if __name__ == "__main__":
    main()
