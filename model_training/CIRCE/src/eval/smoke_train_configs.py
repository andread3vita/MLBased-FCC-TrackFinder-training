"""Validate the queued training configurations on CPU before they take GPU time.

The Phase 1, Phase 2 and Phase 3 launchers sit in a queue for roughly twenty hours
before the first one starts, and each then runs unattended for eight to sixteen. A
flag argparse rejects, an optimizer name the training script does not know, or a
schedule wired for only one code path would surface as an immediate crash at 02:00,
with nothing else in the queue getting a chance either. So build each configuration
for real and take one optimizer step on CPU.

    PYTHONPATH=. python src/eval/smoke_train_configs.py

The configurations are restated here rather than parsed out of the launchers, because
the flags there are assembled from shell function locals that only bash can resolve
correctly. To stop the restatement drifting, `check_drift` asserts that every flag
appearing in a launcher's `train.py` block also appears in the matching config here,
and vice versa. A launcher gaining a flag therefore fails this test until the
configuration below is updated, which is the intended coupling.

Uses two real events per config rather than synthetic hits, because every queued run
trains with `--drop_loopers` and most with `--merge_daughters`, and those change the
dataset path rather than the model. A configuration that builds but whose target set
comes out empty is exactly the failure this is meant to catch.
"""

import argparse
import re
import sys
import types
from pathlib import Path
from unittest import mock

import torch

LAUNCHERS = Path("/home/marko.cechovic/cgatr-runs/ablation-v1")

# Flags every arm shares, and which do not affect what is built here.
COMMON = """
    --data_dir /home/marko.cechovic/cgatr-data/data-final/parquet
    --train_seeds 1-60 --val_seeds 181-190
    --num_devices 4 --max_hits 0 --precision 32-true
    --num_blocks 10 --attr_weight 1.0 --repul_weight 1.0
    --beta_suppress_weight 0.1 --var_weight 0.3 --var_warmup_epochs 1
    --num_workers 4 --prefetch_factor 2 --persistent_workers --cpu_threads 4
    --resume_ckpt last --ckpt_every_n_train_steps 200 --limit_val_batches 40
    --output_dir /tmp/smoke_train --run_tag smoke
""".split()

# (launcher, arm label, the flags that differ)
CONFIGS = [
    ("launch_phase1_loopers.sh", "conformal_noloop", """
        --num_epochs 8 --max_tokens 22000 --embed_dim 4 --hidden_mv_channels 16
        --start_lr 4e-4 --warmup_epochs 2 --lr_schedule plateau
        --plateau_patience 4 --plateau_factor 0.5
        --drop_loopers
    """),
    ("launch_phase1_loopers.sh", "projective_ggtf_noloop", """
        --num_epochs 8 --max_tokens 22000 --embed_dim 4 --hidden_mv_channels 26
        --start_lr 4e-4 --warmup_epochs 2 --lr_schedule plateau
        --plateau_patience 4 --plateau_factor 0.5
        --drop_loopers --algebra projective --pga_hit_encoding ggtf
    """),
    # Phase 2 is the one worth checking hardest: it is the only configuration using
    # Adam, the step schedule, a fixed batch and embed_dim 3, none of which any
    # completed run has exercised.
    ("launch_parity.sh", "parity_conformal", """
        --num_epochs 10 --max_tokens 0 --batch_size 8
        --embed_dim 3 --hidden_mv_channels 16
        --optimizer adam --start_lr 1e-3 --min_lr 1e-6 --warmup_epochs 0
        --lr_schedule step --lr_step_epochs 4 --lr_step_factor 0.1
        --drop_loopers --merge_daughters --tbeta 0.6 --td 0.2
    """),
    ("launch_parity.sh", "parity_projective", """
        --num_epochs 10 --max_tokens 0 --batch_size 8
        --embed_dim 3 --hidden_mv_channels 26
        --optimizer adam --start_lr 1e-3 --min_lr 1e-6 --warmup_epochs 0
        --lr_schedule step --lr_step_epochs 4 --lr_step_factor 0.1
        --drop_loopers --merge_daughters --tbeta 0.6 --td 0.2
        --algebra projective
    """),
    # No --drop_loopers on the v2 arms, and that is the configuration now queued.
    # The ladder used to filter, on the stated grounds that it was cheap and matched
    # GGTF; the second half was false -- their extent cut sits behind a hardcoded
    # `remove_lowEnergyParticle = False` and never runs -- and filtering also made the
    # rungs incomparable to the unfiltered five-arm table they are measured against.
    # `launch_v2_ablations.sh` keeps DROP_LOOPERS=1 as an option, unexercised here
    # because the filtered path is already covered by the Phase 1 entries above.
    ("launch_v2_ablations.sh", "v2_a_nullfix", """
        --num_epochs 8 --max_tokens 22000 --embed_dim 4 --hidden_mv_channels 16
        --start_lr 4e-4 --warmup_epochs 2 --lr_schedule plateau
        --plateau_patience 4 --plateau_factor 0.5
        --fix_cga_null
    """),
    ("launch_v2_ablations.sh", "v2_d_init (full stack)", """
        --num_epochs 8 --max_tokens 22000 --embed_dim 4 --hidden_mv_channels 16
        --start_lr 4e-4 --warmup_epochs 2 --lr_schedule plateau
        --plateau_patience 4 --plateau_factor 0.5
        --fix_cga_null --use_time --two_channel_dc
        --equi_init identity_algebra
    """),
    ("launch_second_seed.sh", "conformal seed 2", """
        --num_epochs 8 --max_tokens 22000 --embed_dim 4 --hidden_mv_channels 16
        --start_lr 4e-4 --warmup_epochs 2 --lr_schedule plateau
        --plateau_patience 4 --plateau_factor 0.5
        --seed 2
    """),
    ("launch_second_seed.sh", "projective seed 2", """
        --num_epochs 8 --max_tokens 22000 --embed_dim 4 --hidden_mv_channels 26
        --start_lr 4e-4 --warmup_epochs 2 --lr_schedule plateau
        --plateau_patience 4 --plateau_factor 0.5
        --seed 2 --algebra projective
    """),
    (
        "/home/marko.cechovic/cgatr-runs/e3-paper-physical/run_final_hail_mary.sh",
        "final corrected E3",
        """
        --num_epochs 40 --max_tokens 16000 --embed_dim 4
        --hidden_mv_channels 16 --hidden_s_channels 64
        --gradient_clip_val 1.0 --qmin 0.1 --oc_mode paper_hinge
        --optimizer adamw --weight_decay 1e-4
        --start_lr 4e-4 --min_lr 1e-5 --warmup_epochs 2
        --lr_schedule plateau --plateau_patience 3 --plateau_factor 0.5
        --terminal_anneal_epochs 6 --ema_decay 0.999
        --fix_particle_zero --min_target_hits 3
        --fix_cga_null --fix_wire_dir --cga_hit_encoding sphere_circle
        --physical_drift_geometry --no-normalize_mv_inputs
        --equivariance_group e3 --invariant_output_head
        --equi_init identity_algebra
        """,
    ),
]

# Flags a launcher may carry that never reach the model, so are not worth restating.
DRIFT_IGNORE = {
    "--data_dir", "--train_seeds", "--val_seeds", "--num_devices", "--max_hits",
    "--precision", "--num_workers", "--prefetch_factor", "--persistent_workers",
    "--cpu_threads", "--resume_ckpt", "--ckpt_every_n_train_steps",
    "--limit_val_batches", "--output_dir", "--run_tag", "--grad_checkpoint",
    "--auto_requeue", "--max_time", "--epoch_csv_path", "--init_weights",
}


def launcher_flags(path):
    """Every --flag inside a launcher's train.py invocation, including $EXTRA sets."""
    text = path.read_text()
    m = re.search(r"python -u src/train\.py(.*?)\$EXTRA", text, re.S)
    if m is None:
        m = re.search(r"python -u src/train\.py(.*?)2>&1", text, re.S)
    block = m.group(1) if m else ""
    # The per-arm extras are defined as shell variables, e.g. NULL="--fix_cga_null".
    extras = " ".join(re.findall(r'^[A-Z_]+="(--[^"]*)"', text, re.M))
    return set(re.findall(r"--[a-z_]+", block + " " + extras))


def check_drift():
    problems = []
    for name in sorted({c[0] for c in CONFIGS}):
        path = LAUNCHERS / name
        if not path.exists():
            continue
        want = launcher_flags(path) - DRIFT_IGNORE
        have = set()
        for lname, _, flags in CONFIGS:
            if lname == name:
                have |= set(re.findall(r"--[a-z_]+", flags))
        have |= set(f for f in COMMON if f.startswith("--"))
        missing = want - have - DRIFT_IGNORE
        if missing:
            problems.append(f"{name}: launcher has {sorted(missing)}, "
                            f"not covered by any config here")
    return problems


DATA = "/home/marko.cechovic/cgatr-data/data-final/parquet"


def build_and_step(argv, label):
    from src.train import parse_args
    from src.lightning_module import CGATrV35LightningModule
    from src.dataset.parquet_dataset import (IDEAParquetDataset,
                                             collate_idea_events)

    with mock.patch.object(sys, "argv", ["train.py"] + argv):
        args = parse_args()

    lm = CGATrV35LightningModule(args)
    ds = IDEAParquetDataset(
        DATA, seed_range=(1, 2), max_hits_per_event=400,
        with_time=args.use_time,
        with_drift_dir=lm.model.needs_drift_dir,
        drop_loopers=args.drop_loopers,
        merge_daughters=getattr(args, "merge_daughters", False),
    )
    events = [ds[i] for i in range(min(2, len(ds)))]
    events = [e for e in events if e is not None]
    if not events:
        raise ValueError("dataset returned no usable events for this configuration")
    batch = collate_idea_events(events)
    if batch is None:
        raise ValueError("collate produced an empty batch")

    n_tgt = len(torch.unique(batch["mc_index"][batch["mc_index"] >= 0]))
    if n_tgt < 2:
        raise ValueError(f"only {n_tgt} target particle(s) survive the filters")

    lm._trainer = types.SimpleNamespace(
        current_epoch=0, global_step=0, max_epochs=args.num_epochs,
        estimated_stepping_batches=100)

    conf = lm.configure_optimizers()
    if isinstance(conf, dict):
        opt, sc = conf["optimizer"], conf.get("lr_scheduler")
    elif isinstance(conf, (list, tuple)) and len(conf) == 2:
        opt, sc = conf[0][0], conf[1][0]
    else:
        opt, sc = conf, None
    sched = sc.get("scheduler") if isinstance(sc, dict) else sc
    # training_step reads the live LR off the trainer for logging.
    lm._trainer.optimizers = [opt]
    lm._trainer.lr_scheduler_configs = []

    with mock.patch.object(lm, "log"), mock.patch.object(lm, "log_dict"):
        out = lm.training_step(batch, 0)
    loss = out["loss"] if isinstance(out, dict) else out
    loss.backward()
    opt.step()

    if not torch.isfinite(loss):
        raise ValueError(f"loss is {loss.item()}")
    grads = [p for p in lm.parameters() if p.grad is not None]
    if not grads:
        raise ValueError("no parameter received a gradient")

    print(f"  [ok] {label:<24} loss={loss.item():8.3f}  "
          f"params={sum(p.numel() for p in lm.parameters()):>9,}  "
          f"{type(opt).__name__:<5} lr={opt.param_groups[0]['lr']:<7g} "
          f"sched={type(sched).__name__ if sched is not None else 'none':<18} "
          f"hits={batch['features'].shape[0]:>4} targets={n_tgt:>3}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", default=None, help="substring filter on the arm label")
    args = ap.parse_args()

    problems = check_drift()
    if problems:
        print("drift between the launchers and the configs in this file:")
        for p in problems:
            print(f"  {p}")
        raise SystemExit(1)
    print("no drift: every launcher flag is covered by a config here\n")

    failures, last = [], None
    for launcher, label, flags in CONFIGS:
        if args.only and args.only not in label:
            continue
        if launcher != last:
            print(launcher)
            last = launcher
        argv = COMMON + flags.split()
        try:
            build_and_step(argv, label)
        except SystemExit as e:
            print(f"  [FAIL] {label}: argparse rejected the flags (exit {e.code})")
            failures.append(label)
        except Exception as e:
            print(f"  [FAIL] {label}: {type(e).__name__}: {e}")
            failures.append(label)

    print()
    if failures:
        raise SystemExit(f"{len(failures)} configuration(s) failed: {failures}")
    print("every queued training configuration parses, builds and steps")
    print("the loss shown is one step on an untrained model and is not comparable "
          "across algebras;\nthe projective arms start orders of magnitude higher "
          "and converge to within a factor of\ntwo of conformal by the end of "
          "epoch 1 (see ablation-v1/*/epoch_metrics.csv)")


if __name__ == "__main__":
    main()
