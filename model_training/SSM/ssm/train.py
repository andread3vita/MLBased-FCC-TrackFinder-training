"""Training entry point for TrackerSSM (plain PyTorch, no Lightning).

Usage:
    python -m ssm.train --train-data /path/to/train_folder \
        --val-data /path/to/val_folder --gpus 0,1 --out /path/to/run_dir

Notes:
- bf16 autocast on CUDA by default (SSM arm is precision-free); loss in fp32.
- Loss: object condensation, identical semantics to the GATr/CIRCE arms but
  from this package's torch-native copy (ssm.oc_loss).
- --gpus sets CUDA_VISIBLE_DEVICES before device init; single process by
  default, multi-GPU via:  torchrun --nproc_per_node=K -m ssm.train --gpus 0,1 ...
"""

from __future__ import annotations

import argparse
import json
from functools import partial
import math
import os
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .dataset import EventParquetDataset, PackedBatchSampler, collate_batch, augment_phi
from .oc_loss import object_condensation_loss, sequence_lengths_to_batch, variance_weight
from .model import TrackerSSM, count_parameters
from . import wandb_logger as wblog
from .sweeps import run_sweep, working_point_scalars, save_sweep_plots
from .metrics import DEFAULT_TBETA_GRID, DEFAULT_TD_GRID
import random


def contract_train_keys(row, lr, args):
    """Map history row to the exact GATR_CIRCE_LOSS train/* wandb schema."""
    out = {
        "train/loss": row["train_loss"],
        "train/learning_rate": lr,
    }
    comp = {
        "train/L_V_att": "train/loss_attractive_weighted",
        "train/L_V_rep": "train/loss_repulsive_weighted",
        "train/L_beta_sig": "train/loss_beta_signal",
        "train/L_beta_noise": "train/loss_beta_noise",
        "train/L_beta_suppress": "train/loss_beta_suppress_weighted",
        "train/L_var": "train/loss_embedding_variance_weighted",
    }
    weights = {
        "train/loss_attractive_weighted": args.attr_weight,
        "train/loss_repulsive_weighted": args.repul_weight,
        "train/loss_beta_suppress_weighted": 1.0,
        "train/loss_embedding_variance_weighted": row.get("train_var_weight", 0.0),
    }
    for src, dst in comp.items():
        if src in row:
            out[dst] = weights.get(dst, 1.0) * row[src]
    out["train/embedding_variance_weight"] = row.get("train_var_weight", 0.0)
    return out


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train-data", required=True,
                   help="folder with the full training dataset "
                        "(seed_*/{dc,vtx}_hits_<split>.parquet)")
    p.add_argument("--val-data", default=None,
                   help="folder with the full validation dataset (same layout). "
                        "If omitted, --train-data with --val-split is used.")
    p.add_argument("--split", default="train",
                   help="optional filename tag filter ('_<split>.'); "
                        "if no file matches, ALL parquets are used")
    p.add_argument("--val-split", default="test",
                   help="same for validation; pass --val-split '' for tag-less folders")
    p.add_argument("--val-nevents", type=int, default=None,
                   help="max validation events per file (None = all)")
    p.add_argument("--train-max-files", type=int, default=None,
                   help="cap number of training parquet files (None = all)")
    p.add_argument("--val-max-files", type=int, default=None,
                   help="cap number of validation parquet files (None = all)")
    p.add_argument("--train-nevents", type=int, default=None,
                   help="cap TOTAL training events across files (None = all)")
    p.add_argument("--gpus", default="0",
                   help="comma-separated GPU ids to use, e.g. '0,1,2,3' "
                        "(sets CUDA_VISIBLE_DEVICES; multi-GPU via torchrun)")
    p.add_argument("--data", default=None, help="DEPRECATED alias of --train-data")
    p.add_argument("--out", required=True, help="output directory")
    # model
    p.add_argument("--d-model", type=int, default=256)
    p.add_argument("--n-layer", type=int, default=6)
    p.add_argument("--expand", type=int, default=1)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--d-state", type=int, default=64)
    p.add_argument("--chunk-size", type=int, default=64)
    p.add_argument("--embedding-dim", type=int, default=4)
    # optimisation (matched-launcher defaults)
    p.add_argument("--epochs", type=int, default=16)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--clip", type=float, default=1.0)
    p.add_argument("--warmup-epochs", type=int, default=2)
    p.add_argument("--plateau-patience", type=int, default=3)
    p.add_argument("--plateau-factor", type=float, default=0.5)
    p.add_argument("--token-budget", type=int, default=20000)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--log-every-n-steps", type=int, default=50,
                   help="wandb train/*_step logging interval in batches (Lightning: 50)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    # OC loss knobs (matched-launcher defaults)
    p.add_argument("--qmin", type=float, default=0.1)
    p.add_argument("--attr-weight", type=float, default=1.0)
    p.add_argument("--repul-weight", type=float, default=1.0)
    p.add_argument("--sweep-events", type=int, default=100,
                   help="validation events used for the physics working-point sweep")
    p.add_argument("--sweep-start-epoch", type=int, default=1,
                   help="first epoch that runs the physics sweep")
    p.add_argument("--sweep-every", type=int, default=1,
                   help="run the physics sweep every N epochs (always also at the last epoch)")
    p.add_argument("--sweep-tbeta-grid",
                   default=",".join(str(v) for v in DEFAULT_TBETA_GRID))
    p.add_argument("--sweep-td-grid",
                   default=",".join(str(v) for v in DEFAULT_TD_GRID))
    p.add_argument("--physics-metrics", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--s-b", type=float, default=1.0)
    p.add_argument("--beta-suppress-weight", type=float, default=0.0)
    p.add_argument("--var-weight", type=float, default=0.0)
    p.add_argument("--var-warmup-epochs", type=int, default=2)
    p.add_argument("--oc-mode", choices=["paper_hinge", "ggtf"], default="paper_hinge")
    p.add_argument("--phi-augment", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--grad-ckpt", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--order-mode", choices=["spatial", "type_phi"], default="spatial")
    p.add_argument("--bidirectional", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--log-wandb", action=argparse.BooleanOptionalAction,
                   default=os.environ.get("LOG_WANDB", "0") == "1")
    p.add_argument("--wandb-project", default=os.environ.get("WANDB_PROJECT", None))
    p.add_argument("--wandb-entity", default=os.environ.get("WANDB_ENTITY", None))
    p.add_argument("--wandb-name", default=os.environ.get("WANDB_NAME", None))
    return p.parse_args()


def _worker_init_fs_sharing(worker_id):
    """File-backed tensor sharing: worker->main transfers avoid /dev/shm SIGBUS."""
    torch.multiprocessing.set_sharing_strategy("file_system")


def collate_train(events, chunk_size, phi_augment):
    """Top-level (picklable) collate with per-event phi rotation."""
    if phi_augment:
        events = [
            augment_phi({k: (v.clone() if torch.is_tensor(v) else v) for k, v in e.items()},
                        random.uniform(-math.pi, math.pi))
            for e in events
        ]
    return collate_batch(events, chunk_size)


def collate_eval(events, chunk_size):
    return collate_batch(events, chunk_size)


def batch_loss(outputs, batch_tensors, args, vw, device):
    emb, beta_logit = outputs
    B, N, D = emb.shape
    coords = emb.reshape(B * N, D)
    beta = torch.sigmoid(beta_logit.reshape(B * N)).float()
    mc = batch_tensors["mc_index"].reshape(B * N)
    lengths = batch_tensors["lengths"]
    batch_idx = sequence_lengths_to_batch([N] * B, device).to(device)
    return object_condensation_loss(
        coords.float(), beta, mc, batch_idx,
        qmin=args.qmin, s_B=args.s_b, attr_weight=args.attr_weight,
        repul_weight=args.repul_weight,
        beta_suppress_weight=args.beta_suppress_weight,
        var_weight=vw, return_components=True, oc_mode=args.oc_mode,
    )


_global_step = 0
_DDP_ACTIVE = False
_DIST = None


def run_epoch(model, loader, args, device, optimizer=None, scaler=None, epoch=1):
    global _global_step
    training = optimizer is not None
    model.train(training)
    vw = variance_weight(epoch, args.var_weight, args.var_warmup_epochs)
    tot, components_sum, nb = 0.0, {}, 0
    it = loader
    try:
        from tqdm import tqdm
        it = tqdm(loader, desc=f"Epoch {epoch}: ", miniters=50,
                  bar_format="{desc}{percentage:3.0f}%|{bar:10}| {n_fmt}/{total_fmt} "
                             "[{elapsed}<{remaining}, {rate_fmt}]")
    except ImportError:
        pass
    for batch in it:
        feats = batch["features"].to(device, non_blocking=True)
        mc = batch["mc_index"].to(device, non_blocking=True)
        if training and args.phi_augment:
            for i in range(feats.shape[0]):
                pass  # rotation happens in the worker: see collate wrapper below
        with torch.autocast("cuda", dtype=torch.bfloat16,
                            enabled=(training and args.bf16 and device.type == "cuda")):
            outputs = model(feats)
        loss, comp = batch_loss(outputs, {"mc_index": mc, "lengths": batch["lengths"]},
                                args, vw, device)
        if _DDP_ACTIVE:
            # all ranks must skip jointly, else DDP collectives desync
            flag = torch.tensor([torch.isfinite(loss).float()], device=device)
            _DIST.all_reduce(flag, op=_DIST.ReduceOp.MIN)
            if flag.item() < 1:
                continue
        elif not torch.isfinite(loss):
            continue
        if training:
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            scaler.step(optimizer)
            scaler.update()
        tot += float(loss.detach())
        for k, v in comp.items():
            components_sum[k] = components_sum.get(k, 0.0) + float(v)
        nb += 1
        if training:
            _global_step += 1
            if args.log_every_n_steps > 0 and _global_step % args.log_every_n_steps == 0:
                wblog.log({
                    "train/loss_step": float(loss.detach()),
                    "train/learning_rate_step": optimizer.param_groups[0]["lr"],
                    "train/loss_attractive_weighted_step": args.attr_weight * float(comp["L_V_att"]),
                    "train/loss_repulsive_weighted_step": args.repul_weight * float(comp["L_V_rep"]),
                    "train/loss_beta_signal_step": float(comp["L_beta_sig"]),
                    "train/loss_beta_noise_step": float(comp["L_beta_noise"]),
                    "train/loss_beta_suppress_weighted_step": float(comp["L_beta_suppress"]),
                    "train/loss_embedding_variance_weighted_step": vw * float(comp["L_var"]),
                    "train/embedding_variance_weight_step": vw,
                }, step=_global_step)
    n = max(nb, 1)
    return tot / n, {k: v / n for k, v in components_sum.items()}


def main():
    args = parse_args()
    args.train_data = args.train_data or args.data
    if not args.train_data:
        raise SystemExit("--train-data is required")
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    print(f"CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}", flush=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)

    global _DDP_ACTIVE, _DIST
    import torch.distributed as dist
    _DIST = dist
    ddp = int(os.environ.get("WORLD_SIZE", "1")) > 1
    _DDP_ACTIVE = ddp
    if ddp:
        dist.init_process_group("nccl")
        local = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(device := torch.device("cuda", local))
        rank, world = dist.get_rank(), dist.get_world_size()
    else:
        rank, world = 0, 1

    wblog.start(vars(args), str(out),
                args.wandb_name or out.name,
                args.wandb_project, args.wandb_entity,
                enabled=args.log_wandb and rank == 0)

    train_ds = EventParquetDataset(args.train_data, args.split, order_mode=args.order_mode,
                                   max_files=args.train_max_files,
                                   max_events=args.train_nevents)
    sizes = [train_ds.event_size(i) for i in range(len(train_ds))]
    sampler = PackedBatchSampler(sizes, args.token_budget, args.chunk_size, seed=args.seed,
                                 rank=rank, world_size=world)

    dl_kwargs = {}
    eff_workers = args.workers
    pin = True
    if ddp:
        eff_workers = max(1, args.workers // world)   # workers are PER rank
        pin = False                                    # avoid shm pressure with N ranks
    if eff_workers > 0:
        import multiprocessing as mp
        dl_kwargs["multiprocessing_context"] = mp.get_context("spawn")
        dl_kwargs["worker_init_fn"] = _worker_init_fs_sharing
    if rank == 0:
        print(f"DataLoader: {eff_workers} workers/rank, pin_memory={pin}, "
              f"sharing=file_system", flush=True)
    train_loader = DataLoader(
        train_ds, batch_sampler=sampler, num_workers=eff_workers,
        collate_fn=partial(collate_train, chunk_size=args.chunk_size,
                           phi_augment=args.phi_augment),
        pin_memory=pin, **dl_kwargs)

    val_loader = None
    val_root = args.val_data or args.train_data
    if rank == 0 and any(Path(val_root).rglob("*.parquet")):
        val_ds = EventParquetDataset(val_root, args.val_split,
                                     max_events_per_seed=args.val_nevents,
                                     order_mode=args.order_mode,
                                     max_files=args.val_max_files)
        val_sizes = [val_ds.event_size(i) for i in range(len(val_ds))]
        val_sampler = PackedBatchSampler(val_sizes, args.token_budget, args.chunk_size,
                                         shuffle=False)  # validation runs only on rank 0
        val_loader = DataLoader(
            val_ds, batch_sampler=val_sampler, num_workers=eff_workers,
            collate_fn=partial(collate_eval, chunk_size=args.chunk_size),
            pin_memory=pin, **dl_kwargs)

    model = TrackerSSM(
        d_model=args.d_model, n_layer=args.n_layer, expand=args.expand,
        n_heads=args.n_heads, d_state=args.d_state, chunk_size=args.chunk_size,
        embedding_dim=args.embedding_dim, bidirectional=args.bidirectional,
    ).to(device)
    n_params = count_parameters(model)
    print(f"TrackerSSM parameters: {n_params:,}", flush=True)
    if ddp:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local])

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=args.plateau_patience, factor=args.plateau_factor)
    scaler = torch.amp.GradScaler("cuda", enabled=(args.bf16 and device.type == "cuda"))

    history = []
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        sampler.set_epoch(epoch)
        if rank == 0:
            print(json.dumps({"phase": f"epoch {epoch}: training"}), flush=True)
        tr_loss, tr_comp = run_epoch(model, train_loader, args, device,
                                     optimizer=optimizer, scaler=scaler, epoch=epoch)
        print(json.dumps({"phase": f"epoch {epoch}: validation"}), flush=True)
        if ddp:
            # average the epoch train loss across ranks for the log
            ttr = torch.tensor([tr_loss], device=device)
            dist.all_reduce(ttr, op=dist.ReduceOp.AVG)
            tr_loss = ttr.item()
        lr_now = optimizer.param_groups[0]["lr"]
        row = {"epoch": epoch, "train_loss": tr_loss, "lr": lr_now,
               "wall_s": round(time.time() - t0, 1), **{f"train_{k}": v for k, v in tr_comp.items()}}
        # DDP waits here while rank 0 handles validation alone (as requested)
        if ddp:
            dist.barrier()
        if val_loader is not None and rank == 0:
            with torch.no_grad():
                va_loss, va_comp = run_epoch(model, val_loader, args, device, epoch=epoch)
            row.update({"val_loss": va_loss,
                        **{f"val_{k}": v for k, v in va_comp.items()}})
            sweep_due = (epoch >= args.sweep_start_epoch and
                         (epoch % args.sweep_every == 0 or epoch == args.epochs))
            if args.physics_metrics and args.sweep_events > 0 and sweep_due:
                print(json.dumps({"sweep": f"epoch {epoch}: physics sweep over "
                                  f"{args.sweep_events} events (this can take minutes)"}),
                      flush=True)
                tb = [float(v) for v in args.sweep_tbeta_grid.split(",") if v]
                td = [float(v) for v in args.sweep_td_grid.split(",") if v]
                acc = run_sweep(model, val_loader, device, args.sweep_events, tb, td)
                wp = acc.working_points()
                wblog.log(working_point_scalars(wp), step=epoch)
                wblog.log_media(save_sweep_plots(wp, str(out)), step=epoch)
                row.update({f"sweep_{k.split('/')[-1]}": v
                            for k, v in working_point_scalars(wp).items()})
                part, best = wp["pareto_f1"], wp["max_efficiency"]
                if part is not None and best is not None:
                    print(f"validation sweep over {args.sweep_events} events selected "
                          f"maximum-efficiency WP: tbeta={best['tbeta']}, td={best['td']}, "
                          f"eff={best['efficiency']:.4f}, fake={best['fake_rate']:.4f}, "
                          f"F1={best['f1']:.4f}; "
                          f"Pareto maximum-F1 WP: tbeta={part['tbeta']}, td={part['td']}, "
                          f"eff={part['efficiency']:.4f}, fake={part['fake_rate']:.4f}, "
                          f"F1={part['f1']:.4f}", flush=True)
        if ddp:
            # share rank-0's val_loss so every rank moves its LR consistently
            t = torch.tensor([row.get("val_loss", float("inf"))], device=device)
            dist.broadcast(t, src=0)
            va_loss = t.item()
        elif val_loader is not None:
            va_loss = row["val_loss"]
        else:
            va_loss = None
        if va_loss is not None:
            scheduler.step(va_loss)
        dist.barrier() if ddp else None
        if rank == 0:
            history.append(row)
            print(json.dumps(row), flush=True)
            wb = contract_train_keys(row, lr_now, args)
            if val_loader is not None:
                wb["validation/loss"] = row.get("val_loss", 0.0)
            wblog.log(wb, step=epoch)
            state = (model.module.state_dict() if ddp else model.state_dict())
            ckpt = {"model": state, "optimizer": optimizer.state_dict(),
                    "args": vars(args), "epoch": epoch, "history": history}
            torch.save(ckpt, out / "last.ckpt")
            if val_loader is not None or epoch == 1 or row.get("val_loss", 1e9) <= min(
                    (r.get("val_loss", 1e9) for r in history), default=1e9):
                torch.save(ckpt, out / "best.ckpt")
            pareto = row.get("sweep_pareto_f1", 0.0)
            max_eff = row.get("sweep_max_tracking_efficiency", 0.0)
            tag = (f"validation_epoch={epoch}_step={_global_step}"
                   f"_pareto_f1={pareto:.4f}_max_eff={max_eff:.4f}.ckpt")
            torch.save(ckpt, out / tag)
        if ddp:
            dist.barrier()
    with open(out / "history.json", "w") as fh:
        json.dump(history, fh, indent=1)
    wblog.finish()


if __name__ == "__main__":
    main()
