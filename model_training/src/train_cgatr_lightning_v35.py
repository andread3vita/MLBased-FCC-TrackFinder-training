#!/usr/bin/env python
"""v35 CGATr training driver via PyTorch Lightning (`v35lightning`).

Mirrors `src/train_cgatr_parquet.py` exactly under Lightning:
  - Same CGATrParquetModel + OC loss (imported from v35).
  - Same EMA(0.999), same OC weights, same var_weight ramp.
  - Same AdamW + LambdaLR (linear warmup + half-cosine).
  - Same DataLoader strategy: batch_size + DistributedSampler
    (NOT the TokenBudgetBatchSampler used by v37/v38 Lightning).
  - Same defaults for the smoke run: batch=2, max_hits=3000,
    train_seeds=1-3, val_seeds=181-181, 3 epochs, 4 GPUs.

Per-epoch metrics CSV at `<output_dir>/epoch_metrics.csv` mirrors the
schema written by Run A:
    run,epoch,mean_train_loss,val_loss,val_match_loose,
    val_match_strict50,wall_s_train,wall_s_val,lr,world_size

Launch:
    bash src/run_smoke_v35_lightning.sh
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
import warnings


# ---------------------------------------------------------------------------
# Silence Lightning's purely cosmetic warnings / tips. Must run BEFORE the
# `import lightning as L` so the filters are in place when Lightning sets
# up its loggers.
# ---------------------------------------------------------------------------
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
        # We deliberately use sync_dist=False on training-step logs to
        # avoid the divergent-codepath NCCL deadlock; the DDP-aggregated
        # train_loss is computed manually in on_train_epoch_end. So
        # Lightning's "recommended sync_dist=True" hint is just noise.
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
from lightning.pytorch.strategies import DDPStrategy
from torch.utils.data import DataLoader

from src.cgatr_v35_lightning_module import CGATrV35LightningModule
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
    """Lightning expects fractions as float<1 and absolute counts as int.

    Our CLI accepts a single number: <1 means fraction, >=1 means
    absolute. Translate to Lightning's convention; None means "no
    limit" (we pass 1.0 instead so the trainer is happy).
    """
    if x is None:
        return 1.0
    if x >= 1:
        return int(x)
    return float(x)


def _apply_dry_run(args) -> None:
    """In-place: shrink the run so it finishes in <30s for debugging."""
    args.num_epochs = 1
    args.limit_train_batches = 20
    args.limit_val_batches = 5
    print(
        "[v35lightning] --dry_run: num_epochs=1, "
        "limit_train_batches=20, limit_val_batches=5",
        flush=True,
    )


def parse_args():
    p = argparse.ArgumentParser(description="v35lightning CGATr training")
    p.add_argument("--data_dir", required=True)
    p.add_argument("--train_seeds", default="1-3")
    p.add_argument("--val_seeds", default="181-181")
    p.add_argument(
        "--max_hits", type=int, default=0,
        help="Per-event hard hit-count cap. 0 (default) = no cap; the "
             "model sees the full event. The legacy v35 smoke used "
             "3000, which truncated ~46%% of events at the median.",
    )
    p.add_argument("--batch_size", type=int, default=2,
                   help="Used only when --max_tokens is 0.")
    p.add_argument(
        "--max_tokens", type=int, default=0,
        help="If >0, use TokenBudgetBatchSampler with this packed-batch "
             "budget (total hits per batch across packed events). The "
             "largest single event in train+val (180+20 seeds) is 17745 "
             "hits, so 20000-24000 covers everything as singletons-or-"
             "better. Recommended for A100 80GB. When set, --batch_size "
             "is ignored.",
    )
    p.add_argument("--num_workers", type=int, default=0)

    p.add_argument("--num_epochs", type=int, default=3)
    p.add_argument("--num_devices", type=int, default=4)
    p.add_argument("--precision", default="32-true",
                   choices=["32-true", "16-mixed", "bf16-mixed"])
    p.add_argument("--gradient_clip_val", type=float, default=1.0)

    p.add_argument("--start_lr", type=float, default=1e-3)
    p.add_argument("--min_lr", type=float, default=1e-5)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--warmup_epochs", type=int, default=2)
    p.add_argument("--warmup_steps", type=int, default=None)

    p.add_argument("--num_blocks", type=int, default=10)
    p.add_argument("--hidden_mv_channels", type=int, default=16)
    p.add_argument("--hidden_s_channels", type=int, default=64)
    p.add_argument(
        "--embed_dim", type=int, default=4,
        help="Clustering-coord dimensionality. v34 PCA showed the "
             "useful subspace is rank-4 (`eval_results/v34_analysis_"
             "merged/phase_a_decision.md`); legacy v35 trained at 5 "
             "for safety, this Lightning rebuild defaults to 4. Note: "
             "an N-D checkpoint can only be loaded by a model built "
             "with the same N.",
    )
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

    p.add_argument("--output_dir", default="checkpoints/v35_lightning_smoke")
    p.add_argument("--epoch_csv_path", default=None,
                   help="Per-epoch metrics CSV. Defaults to "
                        "<output_dir>/epoch_metrics.csv.")
    p.add_argument("--run_tag", default="v35_lightning_smoke")
    p.add_argument("--resume_ckpt", default="none",
                   help="'last' / explicit path / 'none' to disable.")

    # Fast-iteration knobs for smoke / debugging.
    p.add_argument(
        "--limit_train_batches", type=float, default=None,
        help="If <1.0, fraction of train batches per epoch; if >=1, "
             "absolute number of batches. Default: full epoch.",
    )
    p.add_argument(
        "--limit_val_batches", type=float, default=None,
        help="Same convention as --limit_train_batches.",
    )
    p.add_argument(
        "--dry_run", action="store_true",
        help="Shortcut: 1 epoch, 20 train batches, 5 val batches. "
             "Useful to validate the training loop in ~30s.",
    )
    return p.parse_args()


def make_loaders(args):
    """Build (train, val) DataLoaders.

    Two regimes:
      * `args.max_tokens > 0` — `TokenBudgetBatchSampler` packs events
        until their hit-count sum hits the budget. Memory is then
        bounded by `max_tokens` regardless of how big the largest
        event is. The sampler is DDP-aware (reads LOCAL_RANK and
        WORLD_SIZE from env). `--batch_size` is ignored in this mode.
        Use this on A100 80GB.
      * `args.max_tokens == 0` — fixed `batch_size` per rank; Lightning
        auto-wraps the loader in `DistributedSampler` under DDP. Same
        path as v35. Use this on V100 32GB or for parity smoke runs.

    `args.max_hits == 0` disables per-event hit-count truncation
    (model sees the full event). v35 used 3000, which truncated ~46%
    of events at the median.
    """
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

    base_kwargs = dict(
        num_workers=args.num_workers,
        collate_fn=collate_idea_events,
        pin_memory=True,
    )

    if args.max_tokens > 0:
        print(
            f"DataLoader: TokenBudgetBatchSampler  max_tokens={args.max_tokens}"
            f"  (largest dataset event = {max(e[3] for e in train_ds._index)} hits)",
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
        print(f"DataLoader: fixed batch_size={args.batch_size}  "
              f"(Lightning will wrap with DistributedSampler under DDP)",
              flush=True)
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
    """Calls `set_epoch(epoch)` on the train DataLoader's batch_sampler.

    Lightning auto-handles this for vanilla `DistributedSampler`, but
    NOT for custom `batch_sampler`s like `TokenBudgetBatchSampler`. If
    we skip it, the sampler keeps producing the same shuffle every
    epoch — silent training-quality regression. No-op when the sampler
    doesn't expose `set_epoch` (i.e. the fixed-batch_size code path).
    """

    def on_train_epoch_start(self, trainer, pl_module):
        loader = trainer.train_dataloader
        if loader is None:
            return
        bs = getattr(loader, "batch_sampler", None)
        if bs is not None and hasattr(bs, "set_epoch"):
            bs.set_epoch(trainer.current_epoch)


class _HeartbeatCallback(Callback):
    """Per-batch stdout heartbeat (rank 0 only).

    Lightning's progress bar is disabled in DDP for cleaner logs; this
    keeps a v35-style "batch_idx Loss=... lr=... time=..." line going
    so we can see whether training is actually progressing.
    """

    def __init__(self, every_n_steps: int = 50):
        super().__init__()
        self.every_n_steps = int(every_n_steps)
        self._t0 = 0.0

    def on_train_epoch_start(self, trainer, pl_module):
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
    """Writes one CSV row per train+val epoch on rank 0.

    Lightning's hook order is:
        on_train_epoch_start
        ... training steps ...
        on_validation_epoch_start
        ... validation steps ...
        on_validation_epoch_end
        on_train_epoch_end   <-- runs LAST; this is where the CSV row goes

    So `on_train_epoch_end` is the only place where both train and val
    aggregated metrics (`train_loss`, `val_loss`, ...) are guaranteed
    to be in `trainer.callback_metrics` for the just-finished epoch.

    Schema (mirrors Run A's v35 trainer):
      run,epoch,mean_train_loss,val_loss,val_match_loose,
      val_match_strict50,wall_s_train,wall_s_val,lr,world_size
    """

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
        # Lightning calls this on the sanity-check pass before training
        # starts (current_epoch=0, _train_t0=0). Don't record train_wall
        # for that case.
        if trainer.sanity_checking:
            self._val_t0 = time.perf_counter()
            return
        # End of train, start of val.
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

    L.seed_everything(42, workers=True)

    if args.dry_run:
        _apply_dry_run(args)

    train_loader, val_loader = make_loaders(args)

    # `steps_per_epoch` is now resolved inside CGATrV35LightningModule's
    # `configure_optimizers` via `self.trainer.estimated_stepping_batches
    # // num_epochs`, which correctly accounts for DDP shard,
    # limit_train_batches, and grad accumulation. Passing it manually
    # here is no longer required (and was the source of the smoke-run
    # LR-schedule bug).
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

    # DDP knobs mirror v35: skip the unused-param walk and DO NOT sync
    # BatchNorm buffers each forward (v35 trains BN per-rank).
    if args.num_devices > 1:
        strategy = DDPStrategy(
            find_unused_parameters=False,
            broadcast_buffers=False,
        )
    else:
        strategy = "auto"

    # When we provide our own DDP-aware batch_sampler
    # (TokenBudgetBatchSampler), Lightning MUST NOT also wrap the
    # loader with its DistributedSampler — that would mean each rank
    # sees a shard of an already-sharded list and trains on 1/N^2 of
    # the data while silently desyncing the LR schedule.
    use_distributed_sampler = (args.max_tokens == 0)

    trainer = L.Trainer(
        max_epochs=args.num_epochs,
        devices=args.num_devices,
        accelerator="gpu",
        strategy=strategy,
        precision=args.precision,
        gradient_clip_val=args.gradient_clip_val,
        callbacks=callbacks,
        logger=CSVLogger(args.output_dir, name=""),
        log_every_n_steps=10,
        enable_progress_bar=False,
        enable_model_summary=True,
        deterministic=False,
        use_distributed_sampler=use_distributed_sampler,
        limit_train_batches=_normalize_limit_batches(args.limit_train_batches),
        limit_val_batches=_normalize_limit_batches(args.limit_val_batches),
    )

    # --resume_ckpt: "none"/"" disables; "last" auto-finds `last.ckpt`
    # under `output_dir`; any other value is taken as a literal path.
    resume_path: str | None = None
    if args.resume_ckpt and args.resume_ckpt.lower() not in ("none", ""):
        if args.resume_ckpt == "last":
            cand = os.path.join(args.output_dir, "last.ckpt")
            resume_path = cand if os.path.exists(cand) else None
        else:
            resume_path = args.resume_ckpt
        if resume_path:
            print(f"[v35lightning] Resuming from {resume_path}", flush=True)

    t_total_start = time.perf_counter()
    trainer.fit(module, train_loader, val_loader, ckpt_path=resume_path)
    total_wall = time.perf_counter() - t_total_start

    if trainer.is_global_zero:
        wall_path = os.path.join(args.output_dir, "total_wall_clock_s.txt")
        with open(wall_path, "w") as f:
            f.write(f"{total_wall:.3f}\n")
        print(f"[v35lightning] total_wall_clock_s={total_wall:.1f}", flush=True)


if __name__ == "__main__":
    main()
