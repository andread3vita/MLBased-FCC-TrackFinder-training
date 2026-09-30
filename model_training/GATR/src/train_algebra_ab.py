"""Train one arm of the algebra A/B inside GGTF's pipeline.

The point of this script is that it trains *their* LightningModule. `ExampleWrapper` in
`Gatr_withModifications.py` (projective) and `Cgatr_withModifications.py` (conformal) each
carry their own `training_step`, `validation_step` and `configure_optimizers`, so the recipe --
Adam, `ReduceLROnPlateau(factor=0.5, patience=3, threshold=1e-3)`, their loss weights, their
input assembly -- comes from their code and not from ours. What this script supplies is the
data, via the parquet adapter, and a Trainer to turn the crank.

That makes the arms differ in the algebra and nothing else. The two modules read an identical
set of eleven `self.args` fields, verified, and are capacity-matched to within 1.4%.

The repository's executable `train_gatr.sh` leaves `loss_type` and
`use_average_cc_pos` at their parser defaults, `"hgcalimplementation"` and
`0.0`; this harness does the same. The older README instead passes
`use_average_cc_pos=0.99`, so that setting remains publication-ambiguous.

Their `training_step` calls `log_losses_wandb_tracking(True, ...)` unconditionally, which
reaches `wandb.log`. Rather than edit their file, wandb is initialised in disabled mode, which
turns those calls into no-ops.

    python -m src.train_algebra_ab --algebra conformal \
        --data_dir /data/data-final/parquet --train_seeds 1-60 --val_seeds 181-190 \
        --output_dir /work/runs/ab_conformal
"""
from __future__ import annotations

import argparse
import csv
import os
import time
import types

import torch
from torch.utils.data import DataLoader

import dgl  # noqa: F401  (registers the backend before the models import it)
import lightning as L
from lightning.pytorch.callbacks import Callback, ModelCheckpoint

from src.dataset.parquet_ggtf_adapter import (
    ParquetGGTFDataset, TokenBudgetEventSampler, collate_ggtf, parse_seed_range,
)

# Their defaults, from parser_args.py where they exist.
THEIR_DEFAULTS = dict(
    qmin=0.1,
    L_attractive_weight=1.0,
    L_repulsive_weight=1.0,
    frac_cluster_loss=0.1,
    fill_loss_weight=0.0,
    # Defaults used when Andrea's current train_gatr.sh omits these switches.
    loss_type="hgcalimplementation",
    use_average_cc_pos=0.0,
    beta_suppress_weight=0.0,
    var_weight=0.0,
)

# CIRCE champion defaults from the systematic 5-way ablation:
CIRCE_DEFAULTS = dict(
    qmin=3.0,
    L_attractive_weight=1.0,
    L_repulsive_weight=2.0,
    frac_cluster_loss=0.0,
    fill_loss_weight=0.0,
    loss_type="paper_hinge",
    use_average_cc_pos=0.0,
    beta_suppress_weight=0.1,
    var_weight=0.3,
)

ARMS = {
    "projective": "src.models.Gatr_withModifications",
    "conformal": "src.models.Cgatr_withModifications",
}


class SamplerEpoch(Callback):
    """Reshuffle the token-budget batches each epoch.

    Without this every epoch replays one packing, which would quietly turn the run into
    repeated passes over identical batches.
    """

    def __init__(self, sampler):
        super().__init__()
        self.sampler = sampler

    def on_train_epoch_start(self, trainer, pl_module):
        if self.sampler is not None:
            self.sampler.set_epoch(trainer.current_epoch)


class EpochCSV(Callback):
    """Per-epoch validation loss and common clustering metrics.

    Writes only what was measured: a missing `val_loss` is recorded blank rather than carried
    over from the previous epoch, which is the failure our own trainer had for weeks (see
    FINDINGS.md M13). Existing loss-only rows are retained with blank metric columns when a
    run resumes after upgrading this callback.
    """

    FIELDS = [
        "run", "epoch", "val_loss", "val_match_loose",
        "val_match_strict50", "tbeta", "td", "wall_s_train", "lr",
        "world_size", "timestamp",
    ]

    def __init__(self, path: str, run_tag: str):
        super().__init__()
        self.path = path
        self.run_tag = run_tag
        self._t0 = 0.0
        self._init = False

    def on_train_epoch_start(self, trainer, pl_module):
        self._t0 = time.perf_counter()

    def on_train_epoch_end(self, trainer, pl_module):
        if not trainer.is_global_zero:
            return
        if not self._init:
            os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
            old_rows = []
            if os.path.exists(self.path) and os.path.getsize(self.path):
                with open(self.path, newline="") as f:
                    old_rows = list(csv.DictReader(f))
            with open(self.path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self.FIELDS)
                writer.writeheader()
                for row in old_rows:
                    writer.writerow({key: row.get(key, "") for key in self.FIELDS})
            self._init = True
        m = trainer.callback_metrics
        val = m.get("val_loss")
        val_s = "" if val is None else f"{float(val):.6f}"
        loose = m.get("val/match_rate")
        loose_s = "" if loose is None else f"{float(loose):.6f}"
        strict50 = m.get("val/match_rate_strict50")
        strict50_s = "" if strict50 is None else f"{float(strict50):.6f}"
        lr = trainer.optimizers[0].param_groups[0]["lr"]
        with open(self.path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.FIELDS)
            writer.writerow({
                "run": self.run_tag,
                "epoch": pl_module.current_epoch + 1,
                "val_loss": val_s,
                "val_match_loose": loose_s,
                "val_match_strict50": strict50_s,
                "tbeta": float(getattr(pl_module.args, "tbeta", 0.1)),
                "td": float(getattr(pl_module.args, "td", 0.2)),
                "wall_s_train": f"{time.perf_counter() - self._t0:.3f}",
                "lr": f"{lr:.6e}",
                "world_size": trainer.world_size,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            })
        print(f"[csv] epoch {pl_module.current_epoch + 1}: val={val_s or 'not measured'} "
              f"loose={loose_s or 'not measured'} "
              f"strict50={strict50_s or 'not measured'} "
              f"lr={lr:.2e} train_s={time.perf_counter() - self._t0:.1f}", flush=True)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--algebra", required=True, choices=sorted(ARMS))
    p.add_argument("--data_dir", default="/data/data-final/parquet")
    p.add_argument("--train_seeds", default="1-60")
    p.add_argument("--val_seeds", default="181-190")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_events", type=int, default=8,
                   help="events per batch, their recipe's unit; ignored unless "
                        "--max_tokens is 0")
    p.add_argument("--max_tokens", type=int, default=22000,
                   help="hits per batch. Preferred over --batch_events because our events "
                        "span 912-10540 hits, so a fixed event count makes the batch vary "
                        "tenfold in memory and OOM on an unlucky draw. 0 uses --batch_events.")
    p.add_argument("--start_lr", type=float, default=4e-4,
                   help="optimizer learning rate (default: 4e-4 for CIRCE, 3e-4 or 1e-3 for GGTF)")
    p.add_argument("--qmin", type=float, default=None,
                   help="object-condensation charge floor; defaults to 3.0 for circe, 0.1 for ggtf")
    p.add_argument("--L_repulsive_weight", "--repul_weight", type=float, default=None,
                   help="repulsive potential weight; defaults to 2.0 for circe, 1.0 for ggtf")
    p.add_argument("--L_attractive_weight", "--attr_weight", type=float, default=1.0,
                   help="attractive potential weight (default: 1.0)")
    p.add_argument("--beta_suppress_weight", type=float, default=0.1,
                   help="beta suppression weight on non-alpha signal hits (default: 0.1 for circe)")
    p.add_argument("--var_weight", type=float, default=0.3,
                   help="within-cluster variance regularizer weight (default: 0.3 for circe)")
    p.add_argument("--tbeta", type=float, default=0.1,
                   help="fixed beta threshold for online validation metrics")
    p.add_argument("--td", type=float, default=0.2,
                   help="fixed clustering radius for online validation metrics")
    p.add_argument("--metric_val_batches", type=int, default=40,
                   help="per-rank validation batches scored by online clustering metrics; "
                        "loss still uses --limit_val_batches")
    p.add_argument("--num_devices", type=int, default=1,
                   help="number of DDP ranks; token-budget batches are split evenly across "
                        "ranks")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--limit_train_batches", type=int, default=0,
                   help="0 means no limit; a small value gives a quick end-to-end check")
    p.add_argument("--limit_val_batches", type=int, default=0)
    p.add_argument("--max_events_per_seed", type=int, default=0)
    p.add_argument("--loss_backend", default="ggtf", choices=["circe", "ggtf"],
                   help="objective for the conformal arm; the A/B default is "
                        "'ggtf' so both arms share one loss. 'circe' trains "
                        "the conformal arm with its own objective.")
    p.add_argument(
        "--loss_computation",
        default="dense",
        choices=["dense", "event_block"],
        help="Implementation of GGTF's mathematically identical tracking loss. "
             "'dense' is the repository code; 'event_block' avoids computing "
             "masked cross-event hit/object pairs.",
    )
    p.add_argument("--recipe", default="ggtf", choices=["circe", "ggtf"],
                   help="optimizer recipe for the conformal arm; A/B default "
                        "'ggtf' keeps Adam+plateau on both arms.")
    p.add_argument("--reference_width", action="store_true",
                   help="run the conformal arm at CIRCE's reference width "
                        "(16 mv channels) instead of the capacity-matched "
                        "width; use for single-arm CIRCE training, never "
                        "for paired A/B runs.")
    p.add_argument("--num_epochs_hint", type=int, default=0,
                   help="epochs the circe schedule plans for; defaults to --epochs")
    p.add_argument("--resume", default="",
                   help="checkpoint to resume from, or 'last'")
    return p.parse_args()


def main():
    a = parse_args()

    # Disabled rather than absent: their training_step logs unconditionally.
    import wandb
    wandb.init(mode="disabled")

    os.makedirs(a.output_dir, exist_ok=True)
    defaults = CIRCE_DEFAULTS if a.loss_backend == "circe" else THEIR_DEFAULTS
    loss_args = dict(defaults)
    if a.qmin is not None:
        loss_args["qmin"] = a.qmin
    if a.L_repulsive_weight is not None:
        loss_args["L_repulsive_weight"] = a.L_repulsive_weight
    loss_args["L_attractive_weight"] = a.L_attractive_weight
    loss_args["beta_suppress_weight"] = a.beta_suppress_weight
    loss_args["var_weight"] = a.var_weight

    model_args = types.SimpleNamespace(
        loss_backend=a.loss_backend,
        loss_computation=a.loss_computation,
        recipe=a.recipe,
        capacity_matched=not a.reference_width,  # A/B default: match parameter count
        num_epochs=(a.num_epochs_hint or a.epochs),
        start_lr=a.start_lr,
        predict=False,          # keeps validation to the loss, no efficiency tables
        tau=False,
        tbeta=a.tbeta,
        td=a.td,
        metric_val_batches=a.metric_val_batches,
        model_prefix=os.path.join(a.output_dir, ""),
        **loss_args,
    )

    import importlib
    module = importlib.import_module(ARMS[a.algebra])
    model = module.ExampleWrapper(model_args)
    n_par = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[ab] {a.algebra} arm: {n_par:,} trainable parameters", flush=True)
    print(f"[ab] loss/optimizer: backend={a.loss_backend}, recipe={a.recipe}, "
          f"qmin={loss_args['qmin']:g}, repul_weight={loss_args['L_repulsive_weight']:g}, "
          f"beta_suppress={loss_args['beta_suppress_weight']:g}, var_weight={loss_args['var_weight']:g}, "
          f"start_lr={a.start_lr:g}", flush=True)

    kw = dict(max_events_per_seed=a.max_events_per_seed or None)
    train_ds = ParquetGGTFDataset(a.data_dir, parse_seed_range(a.train_seeds), **kw)
    val_ds = ParquetGGTFDataset(a.data_dir, parse_seed_range(a.val_seeds), **kw)
    print(f"[ab] {len(train_ds)} train events, {len(val_ds)} val events", flush=True)

    loader_kw = dict(
        collate_fn=collate_ggtf, num_workers=a.num_workers, pin_memory=True,
        persistent_workers=a.num_workers > 0,
    )
    if a.max_tokens > 0:
        train_sampler = TokenBudgetEventSampler(
            train_ds.sizes, a.max_tokens, shuffle=True,
            num_replicas=a.num_devices)
        val_sampler = TokenBudgetEventSampler(
            val_ds.sizes, a.max_tokens, shuffle=False, verbose=False,
            num_replicas=a.num_devices)
        train_loader = DataLoader(train_ds, batch_sampler=train_sampler, **loader_kw)
        val_loader = DataLoader(val_ds, batch_sampler=val_sampler, **loader_kw)
        sizes = train_ds.sizes
        print(f"[ab] token budget {a.max_tokens}: {len(train_sampler)} batches/rank/epoch; "
              f"events span {min(sizes)}-{max(sizes)} hits, median "
              f"{sorted(sizes)[len(sizes) // 2]}", flush=True)
    else:
        train_loader = DataLoader(train_ds, batch_size=a.batch_events, shuffle=True,
                                  drop_last=True, **loader_kw)
        val_loader = DataLoader(val_ds, batch_size=a.batch_events, shuffle=False,
                                drop_last=False, **loader_kw)
        train_sampler = None

    trainer = L.Trainer(
        default_root_dir=a.output_dir,
        max_epochs=a.epochs,
        accelerator="gpu",
        devices=a.num_devices,
        strategy="ddp" if a.num_devices > 1 else "auto",
        # the token-budget sampler does its own rank slicing; Lightning must
        # not try to wrap or replace it under DDP
        use_distributed_sampler=False,
        logger=False,           # wandb is disabled and their module logs through it
        enable_progress_bar=False,
        enable_model_summary=True,
        num_sanity_val_steps=0,
        limit_train_batches=a.limit_train_batches or 1.0,
        limit_val_batches=a.limit_val_batches or 1.0,
        callbacks=[
            ModelCheckpoint(dirpath=a.output_dir, filename="ab_{epoch:02d}",
                            auto_insert_metric_name=False, every_n_epochs=1,
                            save_top_k=-1, save_last=True),
            SamplerEpoch(train_sampler),
            EpochCSV(os.path.join(a.output_dir, "epoch_metrics.csv"),
                     f"ab_{a.algebra}"),
        ],
    )

    ckpt = None
    if a.resume:
        ckpt = (os.path.join(a.output_dir, "last.ckpt")
                if a.resume == "last" else a.resume)
        if not os.path.exists(ckpt):
            print(f"[ab] no checkpoint at {ckpt}, starting fresh", flush=True)
            ckpt = None

    trainer.fit(model, train_loader, val_loader, ckpt_path=ckpt)
    print(f"[ab] {a.algebra} done", flush=True)


if __name__ == "__main__":
    main()
