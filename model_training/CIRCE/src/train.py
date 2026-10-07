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
    Callback, ModelCheckpoint,
)
from lightning.pytorch.plugins.environments import SLURMEnvironment
from lightning.pytorch.strategies import DDPStrategy
import multiprocessing as mp
from torch.utils.data import DataLoader

import signal as _signal

from src.lightning_module import CGATrV35LightningModule
from src.dataset.parquet_dataset import (
    IDEAParquetDataset, collate_idea_events as collate_legacy_idea_events,
)
from shared_training.circe_parquet_dataset import SharedIDEAParquetDataset
from shared_training.collation import collate_shared_events
from shared_training.event_batching import (
    FixedEventBatchSampler,
    TokenBudgetBatchSampler,
)
from shared_training.wandb_logger import build_experiment_logger


class ValidationSweepModelCheckpoint(ModelCheckpoint):
    """Save one full checkpoint after each completed validation sweep."""

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
        # Avoid writing a checkpoint with stale filename metrics when a
        # validation loader produced no operating-point sweep.
        if not getattr(pl_module, "_validation_working_points", None):
            return
        super().on_validation_end(trainer, pl_module)


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
    p.add_argument("--data_dir", default=None,
                   help="Legacy split CIRCE dataset directory.")
    p.add_argument("--train_files", nargs="+", default=None,
                   help="Canonical shared event-wise Parquet files or globs.")
    p.add_argument("--val_files", nargs="+", default=None,
                   help="Canonical shared validation Parquet files or globs.")
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
    p.add_argument("--lr_schedule", default="cosine",
                   choices=["cosine", "plateau", "step"],
                   help="cosine: warmup + half-cosine to min_lr over --num_epochs. "
                        "plateau: warmup + ReduceLROnPlateau on val_loss (adapts to "
                        "convergence; robust when run length is uncertain). "
                        "step: GGTF's schedule, multiply by --lr_step_factor every "
                        "--lr_step_epochs epochs down to --min_lr, no warmup.")
    p.add_argument("--lr_step_epochs", type=int, default=4,
                   help="Epochs between LR drops (step schedule).")
    p.add_argument("--lr_step_factor", type=float, default=0.1,
                   help="LR multiplier at each drop (step schedule).")
    p.add_argument("--optimizer", default="adamw", choices=["adamw", "adam"],
                   help="adam is for the GGTF parity runs, which use plain Adam; "
                        "decoupled weight decay is not a neutral difference when "
                        "the point is to match their recipe.")
    p.add_argument("--plateau_patience", type=int, default=4,
                   help="Epochs without val_loss improvement before LR drop (plateau).")
    p.add_argument("--plateau_factor", type=float, default=0.5,
                   help="LR multiplier on plateau (plateau schedule).")
    p.add_argument(
        "--terminal_anneal_epochs",
        type=int,
        default=0,
        help="For the plateau schedule, cap the LR with a half-cosine decay "
             "from start_lr to min_lr over the final N epochs. Zero disables "
             "it. This guarantees a refinement phase even while val_loss is "
             "still improving too quickly for ReduceLROnPlateau to fire.",
    )

    p.add_argument("--num_blocks", type=int, default=10)
    p.add_argument("--hidden_mv_channels", type=int, default=16)
    p.add_argument("--hidden_s_channels", type=int, default=64)
    p.add_argument("--embed_dim", type=int, default=4,
                   help="Clustering-coord dimensionality.")
    p.add_argument("--beta_mlp", action="store_true", default=False)
    p.add_argument("--cosine_norm", action="store_true", default=False)
    _boolean_optional = getattr(argparse, "BooleanOptionalAction", None)
    if _boolean_optional is not None:
        p.add_argument("--normalize_mv_inputs",
                   action=_boolean_optional, default=True,
                   help="Legacy per-token Euclidean L2 normalization. It "
                        "preserves rotations but breaks conformal translation "
                        "equivariance; paper-faithful C-GATr runs must pass "
                        "--no-normalize_mv_inputs.")
    else:
        p.add_argument("--normalize_mv_inputs", dest="normalize_mv_inputs",
                       action="store_true", default=True)
        p.add_argument("--no-normalize_mv_inputs", dest="normalize_mv_inputs",
                       action="store_false")
    p.add_argument(
        "--layernorm_epsilon_mode",
        choices=["clamp", "add"],
        default="clamp",
        help="Equivariant LayerNorm denominator: 'clamp' reproduces released "
             "GATr max(norm^2, epsilon); 'add' follows the literal C-GATr "
             "paper equation norm^2 + epsilon.",
    )
    p.add_argument("--use_time", action="store_true", default=False,
                   help="Feed normalized hit time as a scalar input channel.")
    p.add_argument("--algebra", choices=["conformal", "projective"],
                   default="conformal",
                   help="Geometric algebra: conformal Cl(4,1) (drift hits are "
                        "circles) or projective Cl(3,0,1) (the GGTF algebra, "
                        "where the drift radius becomes a scalar).")
    p.add_argument("--pga_hit_encoding",
                   choices=["line", "ggtf", "ggtf_wire", "point_line"],
                   default="line",
                   help="How a drift hit enters the projective algebra, which "
                        "cannot hold a circle. 'line': the wire as a Plucker "
                        "bivector, drift radius as a scalar channel. 'ggtf': the "
                        "two tangency points as GGTF does it, embed_point(left) + "
                        "embed_translation(right - left), so the radius stays "
                        "geometric but the wire direction is dropped. "
                        "'ggtf_wire': that pair plus the Plucker wire in a "
                        "second channel, so the projective arm is handed every "
                        "fact the data holds and the comparison cannot be "
                        "dismissed as starving it. 'point_line': a measured "
                        "point on the wire plus the Plucker wire in two channels; "
                        "use with --separate_hit_metadata to carry radius "
                        "without an arbitrary left/right gauge. Projective only.")
    p.add_argument("--cga_hit_encoding",
                   choices=[
                       "circle", "sphere_circle", "sphere_plane", "point_line"
                   ],
                   default="circle",
                   help="Conformal DC representation. 'circle' is the legacy "
                        "single IPNS circle; 'sphere_circle' adds a shared "
                        "grade-1 sphere channel; 'sphere_plane' exposes the two "
                        "grade-1 constraints whose wedge is the circle. "
                        "'point_line' carries the digitized wire point and full "
                        "CGA wire line in two channels, with drift radius and hit "
                        "type in separate scalars; this is the deployable "
                        "tube-like observation used by the corrected run.")
    p.add_argument("--physical_drift_geometry", action="store_true", default=False,
                   help="Scale drift radius by the same 1000 mm length as "
                        "positions. Legacy runs use /5 to make radius order one, "
                        "which inflates the claimed circle by exactly 200x.")
    p.add_argument("--separate_hit_metadata", action="store_true", default=False,
                   help="Carry normalized drift radius and hit type as scalar "
                        "channels. This keeps physical radius visible and avoids "
                        "mixing metadata into geometric multivectors.")
    p.add_argument(
        "--separate_hit_type",
        action="store_true",
        default=False,
        help="Carry only hit type as an invariant scalar instead of adding it "
             "to the geometric token's grade-0 blade. Unlike "
             "--separate_hit_metadata, this does not also add drift radius.",
    )
    p.add_argument("--legacy_equivariance", action="store_true", default=False,
                   help="Reproduce the pre-2026-07-31 conformal backbone, which "
                        "is rotation- but not translation-equivariant: linear "
                        "basis generated against the origin instead of the point "
                        "at infinity, unweighted attention dot product, and the "
                        "hand-crafted distance features. Conformal arm only.")
    p.add_argument(
        "--equivariance_group",
        choices=["e3", "se3"],
        default="se3",
        help="Symmetry imposed by equivariant linear layers. 'e3' is de Haan "
             "et al.'s published reflection-equivariant basis (20 CGA maps, "
             "9 PGA maps). 'se3' drops reflections, allowing chirality "
             "(40 CGA maps; default and used by corrected existing runs).",
    )
    p.add_argument(
        "--invariant_output_head",
        action="store_true",
        default=False,
        help="Return clustering coordinates and beta through GATr's scalar "
             "output channels. This preserves the backbone's geometric "
             "symmetry; the legacy unconstrained Linear(blades, outputs) "
             "readout is retained only for old checkpoints.",
    )

    p.add_argument("--drop_loopers", action="store_true", default=False,
                   help="Train on GGTF's target set: delete every particle "
                        "whose hits span more than 1600/1600/2800 mm or that "
                        "has fewer than 5 hits, hits included, before the "
                        "network sees them (their remove_loopers). Truth-based, "
                        "so it cannot be applied at inference; it exists to make "
                        "the comparison to their numbers possible. Note it "
                        "removes two thirds of pT > 1 GeV tracks and keeps three "
                        "fifths of genuine multi-turn curlers, so it is a "
                        "track-length cut rather than a curler cut -- see "
                        "paper_adjustment_candidates/curler_question.md.")
    p.add_argument("--seed", type=int, default=42,
                   help="Initialization and shuffling seed. The data split is "
                        "set by --train_seeds/--val_seeds and is unaffected, so "
                        "varying this measures run-to-run scatter at fixed data. "
                        "42 was the hardcoded value every run so far used, so it "
                        "is the default and seed 1 of any multi-seed comparison.")
    p.add_argument("--fix_cga_null", action="store_true", default=False,
                   help="Conformal only. Offset the drift sphere and wire plane "
                        "along the point at infinity, inf = e- - e+, as their "
                        "definitions require, rather than along e+ + e- = 2o, "
                        "the origin. Without it the drift radius enters weighted "
                        "by the hit's squared distance from the origin, the "
                        "point-on-sphere test is not the sphere, and the drift "
                        "embedding is not translation-covariant at all. Off by "
                        "default so runs stay comparable to the existing "
                        "checkpoints, which all trained with the error.")
    p.add_argument("--fix_wire_dir", action="store_true", default=False,
                   help="Build the wire direction as the detector does, "
                        "(sin s sin a, -sin s cos a, cos s), so a stereo wire "
                        "tilts azimuthally. The old slot ordering tilted it "
                        "radially: a median 10.8 degrees off, up to 20.2, and "
                        "not perpendicular to the drift direction the parquet "
                        "carries. Affects every arm that reads the wire "
                        "direction, which is conformal and "
                        "--pga_hit_encoding line, but not 'ggtf', which takes "
                        "the drift axis straight from left/right. Off by "
                        "default so runs stay comparable to the existing "
                        "checkpoints, which all trained with the error.")
    p.add_argument("--two_channel_dc", action="store_true", default=False,
                   help="Conformal only. Give every drift hit a grade-1 sphere "
                        "channel alongside its grade-2 circle, so drift and "
                        "vertex hits share a grade and the conformal inner "
                        "product between them is the point-to-sphere distance "
                        "that motivates the algebra. Doubles the input "
                        "multivector channels and so costs a little capacity in "
                        "the first layer only.")
    p.add_argument("--equi_init", default="default",
                   choices=["default", "identity_algebra"],
                   help="Initialization of the equivariant linear layers. "
                        "'default' is Kaiming-like over all basis maps; "
                        "'identity_algebra' starts every channel pair as a "
                        "multiple of the identity map, so the multivector "
                        "structure is untouched at init. de Haan et al. App. C "
                        "report the latter is best for C-GATr and the former "
                        "best for the projective variants, so running "
                        "Kaiming-like everywhere understates the conformal arm. "
                        "Matched to 'default' layer by layer on random input, so "
                        "the two are not confounded by a per-layer gain.")
    p.add_argument("--merge_daughters", action="store_true", default=False,
                   help="Reassign a daughter's hits to its parent when the "
                        "parent also left hits (their fix_splitted_tracks). "
                        "Runs before --drop_loopers, as in their pipeline.")

    p.add_argument("--qmin", type=float, default=0.1)
    p.add_argument("--attr_weight", type=float, default=1.0)
    p.add_argument("--repul_weight", type=float, default=1.0)
    p.add_argument("--fill_loss_weight", type=float, default=0.0)
    p.add_argument("--use_average_cc_pos", type=float, default=0.0)
    p.add_argument("--beta_suppress_weight", type=float, default=0.1)
    p.add_argument("--var_weight", type=float, default=0.3)
    p.add_argument("--var_warmup_epochs", type=int, default=2)
    p.add_argument(
        "--oc_mode", choices=["paper_hinge", "ggtf"], default="paper_hinge",
        help="Repulsive branch of the hybrid OC formula. Both modes use the "
             "HGCAL/GGTF logarithmic attraction. 'paper_hinge' uses Kieseler's "
             "hinge repulsion and unsoftened charge; it is not the literal "
             "quadratic-attraction paper loss. 'ggtf' uses GGTF's /1.01 charge "
             "stabilization, Gaussian repulsion, and inverse nearest-track "
             "truth-(eta,phi) repulsion weighting.",
    )

    p.add_argument("--tbeta", type=float, default=0.1)
    p.add_argument("--td", type=float, default=0.2)
    p.add_argument("--sweep_tbeta_grid", default="0.35,0.45,0.5,0.6,0.7,0.8")
    p.add_argument("--sweep_td_grid", default="0.15,0.2,0.3,0.4,0.5,0.6")
    p.add_argument("--sweep_min_hits_grid", default="3,4")
    p.add_argument("--sweep_match_metric",
                   choices=("idea", "double_majority", "hungarian"),
                   default="double_majority")
    p.add_argument("--sweep_truth_min_hits", type=int, default=3)
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
        "--legacy_variable_epoch_batches", action="store_true",
        help="Let the epoch length wobble with the reshuffle, as every run before "
             "2026-08-05 did. Only for reproducing those runs: it costs a validation "
             "on any epoch that packs shorter than the first one (see FINDINGS.md M13).",
    )
    p.add_argument(
        "--fix_particle_zero", action="store_true",
        help="Stop treating mc_index 0 as noise. It is not noise: the dataset has no "
             "unassociated hits at all, and index 0 is a real generator-status-1 "
             "particle, charged in a fifth of events, leaving full tracks of a median "
             "122 hits at a median 0.84 GeV. The default trains against them -- beta "
             "suppressed, hits repelled from every object -- and drops 0.67%% of "
             "targets. This moves the noise sentinel to -1, which no hit carries, so "
             "they become targets and the noise class is empty. Off by default only to "
             "keep the in-flight ladder internally comparable (FINDINGS.md M20).",
    )
    p.add_argument(
        "--min_target_hits", type=int, default=0,
        help="Relabel any particle with fewer than this many hits in its event as noise, "
             "for the loss and for the validation metric alike. This is GGTF's "
             "create_garbage_label(minNumHits=3), which they apply during graph "
             "construction so their loss never sees a sub-3-hit target; we had it only in "
             "the scorer, and so trained against 146.6 primaries an event where they train "
             "against 33.2. Use 3 for parity with them. The hits are kept and only their "
             "label changes -- removing them would be the looper filter, which is a "
             "different thing (FINDINGS.md M23, M33). 0 disables, which is the pre-parity "
             "behaviour of every run before 2026-08-08.",
    )
    p.add_argument(
        "--secondaries_as_noise", action="store_true", default=False,
        help="Relabel hits whose particle has gen_status == 0 as noise, matching "
             "GGTF's create_garbage_label, which converts every "
             "isProducedBySecondary hit to noise and drops all-secondary "
             "particles. Our produced_by_secondary column is identically zero "
             "because this production stores secondaries as their own "
             "MCParticles with generatorStatus 0 rather than omitting them, so "
             "the flag has to be derived from gen_status (FINDINGS.md M83). "
             "Turning this on removes 42.5%% of targets and 31.9%% of target "
             "hits from the objective (M81), making the task GGTF's rather than "
             "ours. Off by default; the epoch-24 baseline trained without it.",
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
        "--compile", action="store_true", default=False,
        help="torch.compile the model in-place (dynamic shapes). The xformers "
             "attention is excluded from the graph; the rest fuses. State-dict "
             "keys are unchanged; speed and numerics must be gated per target.",
    )
    p.add_argument(
        "--compile_mode",
        choices=("default", "max-autotune-no-cudagraphs"),
        default="default",
        help="torch.compile mode. CUDAGraph modes are excluded because packed "
             "event shapes are dynamic.",
    )
    p.add_argument(
        "--cpu_threads", type=int, default=4,
        help="Set torch.set_num_threads and OMP/POLARS/MKL thread counts. "
             "Central knob for the whole pipeline.",
    )
    return p.parse_args()


def make_loaders(args):
    """Build (train, val) DataLoaders."""
    shared_files = args.train_files is not None or args.val_files is not None
    if shared_files:
        if args.data_dir is not None or not args.train_files or not args.val_files:
            raise ValueError(
                "Use either --data_dir or both --train_files and --val_files."
            )
    elif args.data_dir is None:
        raise ValueError("Provide --data_dir or both shared file lists")

    tr_a, tr_b = _parse_seed_range(args.train_seeds)
    va_a, va_b = _parse_seed_range(args.val_seeds)

    max_hits = args.max_hits if args.max_hits > 0 else None

    print(("Loading shared training files" if shared_files else
           f"Loading training data: seeds {tr_a}-{tr_b - 1}")
          + f"  max_hits_per_event={max_hits}", flush=True)
    ggtf_targets = dict(drop_loopers=args.drop_loopers,
                        merge_daughters=args.merge_daughters,
                        with_drift_dir=(
                            args.algebra == "projective"
                            and args.pga_hit_encoding in ("ggtf", "ggtf_wire")))
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
        train_ds = IDEAParquetDataset(args.data_dir, seed_range=(tr_a, tr_b),
                                      max_hits_per_event=max_hits,
                                      with_time=args.use_time,
                                      ggtf_loss=args.oc_mode == "ggtf",
                                      min_target_hits=args.min_target_hits,
                                      secondaries_as_noise=args.secondaries_as_noise,
                                      **ggtf_targets)
        print(f"Loading validation data: seeds {va_a}-{va_b - 1}"
              f"  max_hits_per_event={max_hits}", flush=True)
        val_ds = IDEAParquetDataset(args.data_dir, seed_range=(va_a, va_b),
                                    max_hits_per_event=max_hits,
                                    with_time=args.use_time,
                                    ggtf_loss=args.oc_mode == "ggtf",
                                    min_target_hits=args.min_target_hits,
                                    secondaries_as_noise=args.secondaries_as_noise,
                                    **ggtf_targets)
        collate_fn = collate_legacy_idea_events

    _pin = os.environ.get("CGATR_PIN_MEMORY", "1") not in ("0", "false", "False")
    base_kwargs = dict(
        num_workers=args.num_workers,
        collate_fn=collate_fn,
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
            stable_epoch_length=not args.legacy_variable_epoch_batches,
            seed=args.seed,
        )
        # The sampler sorts by event size before packing. With shuffle=False,
        # Lightning's --limit_val_batches takes only the smallest events: for
        # the production gate, 1,564/5,000 events with median 2,202 rather than
        # 3,725 hits. Enable the sampler's deterministic bucket shuffle so a
        # limited validation pass spans the event-size distribution. The val
        # sampler is never advanced by _BatchSamplerEpochCallback, so this
        # subset remains fixed across epochs and across matched runs.
        val_sampler = TokenBudgetBatchSampler(
            val_ds, max_tokens=args.max_tokens,
            shuffle=True, drop_last=False,
            stable_epoch_length=False,
            seed=args.seed,
        )
        train_loader = DataLoader(
            train_ds, batch_sampler=train_sampler, **base_kwargs,
        )
        val_loader = DataLoader(
            val_ds, batch_sampler=val_sampler, **base_kwargs,
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
        train_loader = DataLoader(
            train_ds, batch_sampler=train_sampler, **base_kwargs,
        )
        val_loader = DataLoader(
            val_ds, batch_sampler=val_sampler, **base_kwargs,
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
        # Lightning does not validate every epoch here, and when it does not,
        # `trainer.callback_metrics` still holds the previous epoch's values -- so the row
        # written below was silently a copy of the one above it. Every completed 8-epoch
        # run has exactly that: epochs 4 and 7 duplicate 3 and 6, wall-clock included, in
        # all six arms, deterministically. The mechanism is that Lightning derives its
        # end-of-epoch validation trigger from the *first* epoch's batch count, while the
        # token-budget sampler reshuffles each epoch and some epochs come out a batch
        # short, so the trigger is never reached. See FINDINGS.md.
        #
        # Tracked explicitly rather than inferred from equal values, so the record says
        # "not measured" instead of inventing a measurement.
        self._validated_this_epoch = False

    def _init_file(self):
        # Append, and write the header only into a file that does not yet have one. Opening
        # "w" here truncated the record every time a run resumed from a checkpoint: v2_b_time
        # came back from the crash at epoch 5 and lost epochs 1-4, which then read as a
        # 4-of-8 run to the launcher's completion guard and stopped the whole queue on a
        # healthy result. The rows are the experiment record, and a resume must extend it.
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
            # No validation this epoch, so nothing in callback_metrics is this epoch's.
            # `train_loss` included: its all-reduce lives in the module's
            # on_validation_epoch_end, so it too is last epoch's value. The training
            # statistics themselves are fine -- the accumulators are reset in
            # on_train_epoch_start regardless, so no epoch's mean is contaminated by
            # another; it is only that this epoch's mean is never computed. Recorded as
            # missing rather than duplicated, so readers must tolerate blank columns.
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
    # TF32 MUST stay off: its 10-bit mantissa breaks the geometric-product
    # precision the Pin(4,1)-equivariance relies on. No-op on V100 (Volta has
    # no TF32), but critical on A100/H100 where TF32 is otherwise used.
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
        # In-place compile: preserves module identity + state_dict keys, so
        # checkpoints stay compatible. The xformers attention graph-breaks
        # (decorated with torch._dynamo.disable); everything else fuses.
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

    # Both token and fixed-size paths already divide one canonical global plan
    # between ranks. Lightning must not replace either shared sampler.
    use_distributed_sampler = False

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
