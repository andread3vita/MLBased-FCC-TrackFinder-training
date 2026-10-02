"""Inference-time forward pass for C-GATr FCC eval pipeline.

Runs the model forward pass on all events in a seed range, caches per-hit
outputs (coords, beta, mc_index) to parquet for downstream clustering.

The disk cache is reusable across runs:
  - First run with --cache_path X writes forward_hits.parquet + manifest.json.
  - Subsequent runs with --cache_path X reuse the cache (no GPU needed).

Usage:
  python src/eval/forward_pass.py \
      --data_dir /path/to/v1_zqq_uds \
      --checkpoint checkpoints/cgatr_fcc_prod/last.ckpt \
      --eval_seeds 1001-1196 \
      --embed_dim 4 --num_blocks 10 \
      --cache_path eval_results/my_run/emb_all
"""

from __future__ import annotations

import argparse as _argparse
import hashlib
import json
import os
import sys as _sys
import time
from pathlib import Path

_sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import numpy as np
import polars as pl
import torch

from src.model import CGATrParquetModel
from src.dataset.parquet_dataset import IDEAParquetDataset


# Kept in step with `ggtf_assign.MIN_SIGNAL_MC` through the same environment variable, so a
# cache and the report computed from it can never disagree about whether particle 0 is a
# target. FIX_PARTICLE_ZERO=1 includes it, which is correct; the default excludes it, which is
# what every number reported before 2026-08-05 did. See FINDINGS.md M20/M21.
MIN_SIGNAL_MC = 0 if os.environ.get("FIX_PARTICLE_ZERO") == "1" else 1


def _signal_policy_matches(manifest: dict) -> bool:
    """Fail closed when cache truth rows used another particle-zero policy."""
    return int(manifest.get("min_signal_mc", 1)) == MIN_SIGNAL_MC


def _checkpoint_fingerprint(path: str) -> str:
    """Content identity for cache invalidation.

    `cgatr_best.ckpt` is a mutable path: ModelCheckpoint replaces its contents
    whenever validation improves. Comparing only the path can therefore reuse
    predictions from an older epoch. Hashing once before inference is cheap
    relative to the forward pass and makes cache reuse fail closed.
    """
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _dataset_fingerprint(data_dir: str, seed_range: str) -> dict:
    """Metadata identity for the exact requested seed directories."""
    root = Path(data_dir).resolve()
    seed_start, seed_end = _parse_seed_range(seed_range)
    files = []
    for seed in range(seed_start, seed_end):
        candidates = [
            root / f"seed_{seed}",
            root / str(seed),
            root / f"{seed:04d}",
        ]
        seed_dir = next((path for path in candidates if path.is_dir()), None)
        if seed_dir is None:
            matches = [
                path for path in root.iterdir()
                if path.is_dir() and path.name.rstrip("_").endswith(str(seed))
            ]
            if len(matches) != 1:
                raise RuntimeError(
                    f"cannot uniquely identify data directory for seed {seed}"
                )
            seed_dir = matches[0]
        files.extend(sorted(seed_dir.rglob("*.parquet")))
    if not files:
        raise RuntimeError("dataset fingerprint found no parquet files")
    digest = hashlib.sha256()
    for path in files:
        stat = path.stat()
        digest.update(
            f"{path.relative_to(root)}\0{stat.st_size}\0{stat.st_mtime_ns}\n".encode()
        )
    return {
        "data_dir": str(root),
        "seed_range": seed_range,
        "file_count": len(files),
        "metadata_sha256": digest.hexdigest(),
    }


def _inference_source_fingerprint() -> dict:
    package = Path(__file__).resolve().parents[2]
    files = [
        package / "src/eval/forward_pass.py",
        package / "src/model.py",
        package / "src/dataset/parquet_dataset.py",
        *sorted((package / "src/cgatr").rglob("*.py")),
    ]
    return {
        str(path.relative_to(package)): _checkpoint_fingerprint(str(path))
        for path in files
    }


def _load_state_dict_tolerating_derived_buffers(model, state):
    """Load a checkpoint, allowing only derived-constant buffers to be absent.

    The algebra constants -- the geometric-product and outer-product basis tables and
    the point/line/translation embedding matrices -- are registered as persistent
    buffers, so they live in the checkpoint even though nothing learns them. A
    constant added to the code after a run started is therefore missing from that
    run's checkpoint and `strict=True` refuses to load it: `projective_fixed` began
    training before `translation_matrix` existed and could not be evaluated at all.

    Tolerating that is safe, because the freshly built model has already computed the
    constant correctly in `__init__`. Tolerating anything else is not, so a missing
    *parameter* or an unexpected key still raises, as does a shape mismatch, which
    `load_state_dict` reports even when not strict.

    Deliberately absent: any check that a constant present in both agrees with the one
    this code builds. That looks like the obvious safety net and it is a trap. The
    equivariant linear bases in every checkpoint we have differ from what the current
    code constructs -- `cgatr.linear_in.basis` differs in its nonzero count, not by a
    permutation -- because the basis construction changed after those runs trained.
    Letting the checkpoint's version win is the intended behaviour and the correct one,
    since the weights were fitted against that basis; enforcing agreement would refuse
    every checkpoint we own. See `_make_args` on `legacy_equivariance` for the related
    reason the flag has to match the run.
    """
    missing, unexpected = model.load_state_dict(state, strict=False)
    params = {n for n, _ in model.named_parameters()}
    buffers = {n for n, _ in model.named_buffers()}

    bad = [k for k in missing if k in params or k not in buffers]
    if bad or unexpected:
        raise RuntimeError(
            f"checkpoint does not match the model: missing {bad}, "
            f"unexpected {list(unexpected)}. Only derived-constant buffers may be "
            f"absent; a missing parameter or an unexpected key means the "
            f"architecture differs.")

    if missing:
        print(f"[fwd] recomputed {len(missing)} derived constant(s) absent from the "
              f"checkpoint: {sorted(missing)}")


def _validate_checkpoint_config(checkpoint, args):
    """Fail rather than evaluate a checkpoint with different preprocessing."""
    if not isinstance(checkpoint, dict):
        return
    saved = checkpoint.get("hyper_parameters")
    if not isinstance(saved, dict):
        return

    expected = {
        "num_blocks": args.num_blocks,
        "hidden_mv_channels": args.hidden_mv_channels,
        "hidden_s_channels": args.hidden_s_channels,
        "embed_dim": args.embed_dim,
        "normalize_mv_inputs": args.normalize_mv_inputs,
        "algebra": args.algebra,
        "use_time": args.use_time,
        "pga_hit_encoding": args.pga_hit_encoding,
        "two_channel_dc": args.two_channel_dc,
        "cga_hit_encoding": args.cga_hit_encoding,
        "physical_drift_geometry": args.physical_drift_geometry,
        "separate_hit_metadata": args.separate_hit_metadata,
        "separate_hit_type": getattr(args, "separate_hit_type", False),
        "layernorm_epsilon_mode": getattr(
            args, "layernorm_epsilon_mode", "clamp"
        ),
        "legacy_equivariance": not args.no_legacy_equivariance,
        "equivariance_group": args.equivariance_group,
        "invariant_output_head": args.invariant_output_head,
        "fix_cga_null": args.fix_cga_null,
        "fix_wire_dir": args.fix_wire_dir,
    }
    mismatches = [
        f"{key}: checkpoint={saved[key]!r}, requested={value!r}"
        for key, value in expected.items()
        if key in saved and saved[key] != value
    ]
    if mismatches:
        raise RuntimeError(
            "evaluation flags do not match checkpoint hyperparameters:\n  "
            + "\n  ".join(mismatches)
        )


def _make_args(num_blocks, embed_dim, hidden_mv_channels=16, hidden_s_channels=64,
               legacy_equivariance=True, algebra="conformal", use_time=False,
               pga_hit_encoding="line", two_channel_dc=False,
               cga_hit_encoding="circle", physical_drift_geometry=False,
               separate_hit_metadata=False, separate_hit_type=False,
               layernorm_epsilon_mode="clamp", equi_init="default",
               fix_cga_null=False, fix_wire_dir=False,
               equivariance_group="se3", invariant_output_head=False,
               normalize_mv_inputs=True):
    """Build a minimal Namespace to instantiate CGATrParquetModel.

    `legacy_equivariance` defaults to True because every checkpoint produced
    before 2026-07-31 was trained with the old conformal backbone, and the
    equivariant linear basis is a persistent buffer: loading such a checkpoint
    into a corrected model would overwrite the corrected basis with the old one
    while keeping the corrected attention, giving a hybrid that was never
    trained. Pass False to evaluate a model trained after the fix.
    """
    ns = _argparse.Namespace(
        hidden_mv_channels=hidden_mv_channels,
        hidden_s_channels=hidden_s_channels,
        num_blocks=num_blocks,
        embed_dim=embed_dim,
        beta_mlp=False,
        normalize_mv_inputs=normalize_mv_inputs,
        cosine_norm=False,
        grad_checkpoint=False,
        legacy_equivariance=legacy_equivariance,
        equivariance_group=equivariance_group,
        invariant_output_head=invariant_output_head,
        algebra=algebra,
        use_time=use_time,
        pga_hit_encoding=pga_hit_encoding,
        two_channel_dc=two_channel_dc,
        cga_hit_encoding=cga_hit_encoding,
        physical_drift_geometry=physical_drift_geometry,
        separate_hit_metadata=separate_hit_metadata,
        separate_hit_type=separate_hit_type,
        layernorm_epsilon_mode=layernorm_epsilon_mode,
        equi_init=equi_init,
        fix_cga_null=fix_cga_null,
        fix_wire_dir=fix_wire_dir,
    )
    return ns


def _parse_seed_range(s: str):
    a, b = s.split("-")
    return int(a), int(b) + 1


def cache_to_dataframe(cache, embed_dim) -> pl.DataFrame:
    """Flatten in-memory cache to a single hit-level polars DataFrame."""
    rows = {"event_id": [], "seed": []}
    rows["hit_order"] = []
    for d in range(embed_dim):
        rows[f"coord_{d}"] = []
    rows["beta"] = []
    rows["mc_index"] = []
    rows["n_hits_total"] = []
    # Detector positions, so geometric post-processing (helix-consistency
    # merging) does not need the raw parquet re-joined afterwards.
    has_pos = bool(cache) and "sig_pos" in cache[0]
    if has_pos:
        for c in ("pos_x", "pos_y", "pos_z"):
            rows[c] = []

    for entry in cache:
        n_sig = entry["sig_coords"].shape[0]
        rows["event_id"].extend([entry["event_id"]] * n_sig)
        rows["seed"].extend([entry["seed"]] * n_sig)
        rows["hit_order"].extend(range(n_sig))
        for d in range(embed_dim):
            rows[f"coord_{d}"].extend(entry["sig_coords"][:, d].tolist())
        rows["beta"].extend(entry["sig_beta"].tolist())
        rows["mc_index"].extend(entry["sig_mc"].tolist())
        if has_pos:
            for k, c in enumerate(("pos_x", "pos_y", "pos_z")):
                rows[c].extend(entry["sig_pos"][:, k].tolist())
        nht = entry["n_hits_total_map"]
        rows["n_hits_total"].extend([int(nht.get(int(m), 0))
                                     for m in entry["sig_mc"].tolist()])

    schema = {
        "event_id": pl.Int64,
        "seed": pl.Int64,
        "hit_order": pl.Int64,
    }
    for d in range(embed_dim):
        schema[f"coord_{d}"] = pl.Float32
    schema["beta"] = pl.Float32
    schema["mc_index"] = pl.Int64
    schema["n_hits_total"] = pl.Int64
    if has_pos:
        for c in ("pos_x", "pos_y", "pos_z"):
            schema[c] = pl.Float32
    return pl.DataFrame(rows, schema=schema)


def cache_from_dataframe(df: pl.DataFrame, embed_dim: int) -> list[dict]:
    """Inverse of `cache_to_dataframe`.

    Shared with the clustering path, which is where positions are consumed.
    """
    from src.lowpt_op_sweep import cache_from_dataframe as _impl
    return _impl(df, embed_dim)


@torch.no_grad()
def forward_and_cache(model, dataset, device, embed_dim, log_every=200):
    """Run the model forward pass on every event once and cache per-hit outputs."""
    model.eval()
    cache: list[dict] = []
    n_events = len(dataset)
    n_skipped = 0
    t0 = time.time()

    for idx in range(n_events):
        event = dataset[idx]
        if event is None:
            n_skipped += 1
            continue

        features = event["features"].to(device)
        mc_index_all = event["mc_index"].numpy()
        is_secondary = event["is_secondary"].numpy().astype(bool)
        seq_lens = [event["n_hits"]]

        # Which hits reach the cache at all. Under the default, particle 0's hits are dropped
        # here, which is why no cache written before 2026-08-05 can be re-scored with them:
        # the rows are simply absent and a fresh forward pass is required. The model always
        # saw them -- `features` is the whole event either way -- only the stored truth
        # excluded them. See FINDINGS.md M20/M21.
        sig_mask = (~is_secondary) & (mc_index_all >= MIN_SIGNAL_MC)
        if sig_mask.sum() < 4:
            n_skipped += 1
            continue

        try:
            output = model(features, seq_lens)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            n_skipped += 1
            continue

        coords = output[:, :embed_dim].cpu().numpy().astype(np.float32)
        beta = torch.sigmoid(output[:, embed_dim]).cpu().numpy().astype(np.float32)

        sig_coords = coords[sig_mask]
        sig_beta = beta[sig_mask]
        sig_mc = mc_index_all[sig_mask]
        # features[:, :3] is the hit position in mm for both subdetectors
        sig_pos = event["features"][:, :3].numpy().astype(np.float32)[sig_mask]

        n_hits_total_map = {}
        unique_mc_all = np.unique(mc_index_all)
        unique_mc_all = unique_mc_all[unique_mc_all >= MIN_SIGNAL_MC]
        for mc_idx in unique_mc_all:
            n_hits_total_map[int(mc_idx)] = int((mc_index_all == mc_idx).sum())

        dc_path, _vtx, eid, _ = dataset._index[idx]
        seed = int(Path(dc_path).parent.name.replace("seed_", ""))

        cache.append({
            "event_id": int(eid),
            "seed": int(seed),
            "sig_coords": sig_coords,
            "sig_beta": sig_beta,
            "sig_mc": sig_mc,
            "sig_pos": sig_pos,
            "n_hits_total_map": n_hits_total_map,
        })

        del output, coords, beta
        if (idx + 1) % 200 == 0:
            torch.cuda.empty_cache()

        if (idx + 1) % log_every == 0 or idx + 1 == n_events:
            dt = time.time() - t0
            rate = (idx + 1) / max(dt, 1e-3)
            eta = (n_events - idx - 1) / max(rate, 1e-3)
            print(
                f"  Forward {idx + 1}/{n_events}  "
                f"{rate:.2f} ev/s  ETA {eta / 60.0:.1f} min  skipped={n_skipped}",
                flush=True,
            )

    print(f"Forward done in {(time.time() - t0) / 60.0:.1f} min, "
          f"cached {len(cache)} events ({n_skipped} skipped)")
    return cache


class _EventShard:
    """Strided, read-only view of a dataset for parallel CPU inference."""

    def __init__(self, dataset, shard_index: int, shard_count: int):
        if shard_count < 1:
            raise ValueError("event shard count must be positive")
        if not 0 <= shard_index < shard_count:
            raise ValueError(
                f"event shard index {shard_index} is outside [0, {shard_count})"
            )
        self.dataset = dataset
        self.indices = list(range(shard_index, len(dataset), shard_count))
        # forward_and_cache reads this metadata to recover seed/event identity.
        self._index = [dataset._index[index] for index in self.indices]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        return self.dataset[self.indices[index]]


def main():
    p = _argparse.ArgumentParser(description="C-GATr FCC forward pass + cache")
    p.add_argument("--data_dir", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--eval_seeds", default="1001-1196")
    p.add_argument("--max_hits", type=int, default=0,
                   help="Per-event hit cap. 0 = uncapped.")
    p.add_argument("--embed_dim", type=int, default=4)
    p.add_argument("--num_blocks", type=int, default=10)
    p.add_argument("--hidden_mv_channels", type=int, default=16)
    p.add_argument("--hidden_s_channels", type=int, default=64)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--algebra", choices=["conformal", "projective"],
                   default="conformal")
    p.add_argument("--use_time", action="store_true", default=False)
    p.add_argument("--pga_hit_encoding",
                   choices=["line", "ggtf", "ggtf_wire", "point_line"],
                   default="line")
    p.add_argument("--two_channel_dc", action="store_true", default=False,
                   help="Match a checkpoint trained with --two_channel_dc.")
    p.add_argument("--cga_hit_encoding",
                   choices=[
                       "circle", "sphere_circle", "sphere_plane", "point_line"
                   ],
                   default="circle")
    p.add_argument("--physical_drift_geometry", action="store_true", default=False)
    p.add_argument("--separate_hit_metadata", action="store_true", default=False)
    p.add_argument("--separate_hit_type", action="store_true", default=False)
    p.add_argument(
        "--layernorm_epsilon_mode",
        choices=["clamp", "add"],
        default="clamp",
    )
    p.add_argument(
        "--normalize_mv_inputs",
        action=_argparse.BooleanOptionalAction,
        default=True,
        help="Match the checkpoint's input multivector preprocessing.",
    )
    p.add_argument("--equi_init", default="default",
                   choices=["default", "identity_algebra"],
                   help="Irrelevant for a loaded checkpoint, since the weights "
                        "are overwritten; accepted so an arm's training flags "
                        "can be passed through unchanged.")
    p.add_argument("--fix_cga_null", action="store_true", default=False,
                   help="Match a checkpoint trained with --fix_cga_null. This "
                        "one changes the inputs rather than the weights, so it "
                        "must match how the checkpoint was trained.")
    p.add_argument("--fix_wire_dir", action="store_true", default=False,
                   help="Match a checkpoint trained with --fix_wire_dir. Like "
                        "--fix_cga_null this changes the inputs rather than "
                        "the weights, so it must match how the checkpoint was "
                        "trained.")
    p.add_argument("--no_legacy_equivariance", action="store_true", default=False,
                   help="Evaluate a checkpoint trained after the 2026-07-31 "
                        "equivariance fix. The default assumes the legacy "
                        "conformal backbone, because the equivariant linear "
                        "basis is a persistent buffer and mixing the two gives "
                        "a model that was never trained.")
    p.add_argument(
        "--equivariance_group",
        choices=["e3", "se3"],
        default="se3",
        help="Match the checkpoint's equivariant linear basis. Corrected "
             "checkpoints before this option use se3.",
    )
    p.add_argument(
        "--invariant_output_head",
        action="store_true",
        default=False,
        help="Match a checkpoint trained with GATr scalar output channels "
             "instead of the legacy unconstrained blade readout.",
    )
    p.add_argument(
        "--cache_path", type=str, default=None,
        help="Directory for the on-disk forward-pass cache. "
             "Cache contents: forward_hits.parquet + manifest.json.",
    )
    p.add_argument(
        "--force_refresh_cache", action="store_true",
        help="If set with --cache_path, ignore any existing cache and overwrite.",
    )
    p.add_argument(
        "--drop_loopers", action="store_true",
        help="Reproduce GGTF's remove_loopers before the forward pass: delete "
             "every particle whose hits span >1600/1600/2800 mm or that has "
             "<5 hits. Truth-based, so not deployable; this is for comparing "
             "on their terms. Must be set before the forward pass rather than "
             "at clustering time, because GGTF's network never sees the hits.",
    )
    p.add_argument(
        "--merge_daughters", action="store_true",
        help="Reproduce GGTF's fix_splitted_tracks: reassign a daughter's hits "
             "to its parent when the parent also has hits.",
    )
    p.add_argument(
        "--event_shard_index", type=int, default=0,
        help="Zero-based strided event shard for CPU-parallel forward passes.",
    )
    p.add_argument(
        "--event_shard_count", type=int, default=1,
        help="Number of strided event shards. Default 1 processes every event.",
    )
    args = p.parse_args()
    if args.event_shard_count < 1:
        p.error("--event_shard_count must be positive")
    if not 0 <= args.event_shard_index < args.event_shard_count:
        p.error("--event_shard_index must be in [0, --event_shard_count)")
    checkpoint_sha256 = _checkpoint_fingerprint(args.checkpoint)
    dataset_fingerprint = _dataset_fingerprint(
        args.data_dir, args.eval_seeds
    )
    inference_source_sha256 = _inference_source_fingerprint()

    cache: list[dict] | None = None
    cache_used = False
    if args.cache_path is not None and not args.force_refresh_cache:
        hits_path = os.path.join(args.cache_path, "forward_hits.parquet")
        manifest_path = os.path.join(args.cache_path, "manifest.json")
        if os.path.exists(hits_path) and os.path.exists(manifest_path):
            with open(manifest_path) as f:
                manifest = json.load(f)
            same_ckpt = (
                manifest.get("checkpoint") == args.checkpoint
                and manifest.get("checkpoint_sha256") == checkpoint_sha256
            )
            same_seeds = manifest.get("seeds") == args.eval_seeds
            same_dim = manifest.get("embed_dim") == args.embed_dim
            same_max_hits = manifest.get("max_hits") == args.max_hits
            same_signal_policy = _signal_policy_matches(manifest)
            same_dataset = (
                manifest.get("dataset_fingerprint") == dataset_fingerprint
            )
            same_source = (
                manifest.get("inference_source_sha256")
                == inference_source_sha256
            )
            same_shard = (
                int(manifest.get("event_shard_index", 0))
                == args.event_shard_index
                and int(manifest.get("event_shard_count", 1))
                == args.event_shard_count
            )
            # A looper-filtered cache holds different hits, so it must not be
            # reused for an unfiltered request or vice versa.
            same_regime = (
                bool(manifest.get("drop_loopers", False)) == args.drop_loopers
                and bool(manifest.get("merge_daughters", False)) == args.merge_daughters
                and manifest.get("algebra", "conformal") == args.algebra
                and bool(manifest.get("use_time", False)) == args.use_time
                and manifest.get("pga_hit_encoding", "line")
                == args.pga_hit_encoding
                and bool(manifest.get("two_channel_dc", False))
                == args.two_channel_dc
                and manifest.get("cga_hit_encoding", "circle")
                == args.cga_hit_encoding
                and bool(manifest.get("physical_drift_geometry", False))
                == args.physical_drift_geometry
                and bool(manifest.get("separate_hit_metadata", False))
                == args.separate_hit_metadata
                and bool(manifest.get("separate_hit_type", False))
                == args.separate_hit_type
                and manifest.get("layernorm_epsilon_mode", "clamp")
                == args.layernorm_epsilon_mode
                and bool(manifest.get("normalize_mv_inputs", True))
                == args.normalize_mv_inputs
                and bool(manifest.get("fix_cga_null", False))
                == args.fix_cga_null
                and bool(manifest.get("fix_wire_dir", False))
                == args.fix_wire_dir
                and bool(manifest.get("no_legacy_equivariance", False))
                == args.no_legacy_equivariance
                and manifest.get("equivariance_group", "se3")
                == args.equivariance_group
                and bool(manifest.get("invariant_output_head", False))
                == args.invariant_output_head
            )
            if (
                same_ckpt and same_seeds and same_dim and same_max_hits
                and same_signal_policy and same_dataset and same_source
                and same_shard and same_regime
            ):
                print(f"\n=== Reusing forward cache from {args.cache_path} ===")
                t0 = time.time()
                hits_df = pl.read_parquet(hits_path)
                cache = cache_from_dataframe(hits_df, args.embed_dim)
                print(
                    f"  loaded {len(cache)} events ({hits_df.height} hits) "
                    f"in {time.time() - t0:.1f}s"
                )
                cache_used = True
            else:
                print(
                    "\n=== Cache exists but manifest mismatch; regenerating "
                    f"(ckpt={same_ckpt} seeds={same_seeds} dim={same_dim} "
                    f"max_hits={same_max_hits} signal_policy={same_signal_policy} "
                    f"dataset={same_dataset} source={same_source} "
                    f"shard={same_shard} "
                    f"regime={same_regime}) ==="
                )

    if not cache_used:
        device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
        print(f"Device: {device}")
        if device.type == "cuda":
            frac = float(os.environ.get("CGATR_EVAL_MEM_FRACTION", "0.0"))
            if frac > 0:
                torch.cuda.set_per_process_memory_fraction(frac, device)
                print(f"[cgatr_eval] Capped CUDA memory fraction to {frac:.2f} ({frac*32:.1f} GiB)", flush=True)

        print(f"Loading checkpoint: {args.checkpoint}")
        checkpoint = torch.load(
            args.checkpoint, map_location="cpu", weights_only=False
        )
        _validate_checkpoint_config(checkpoint, args)
        model_args = _make_args(
            num_blocks=args.num_blocks,
            embed_dim=args.embed_dim,
            hidden_mv_channels=args.hidden_mv_channels,
            hidden_s_channels=args.hidden_s_channels,
            legacy_equivariance=not args.no_legacy_equivariance,
            equivariance_group=args.equivariance_group,
            invariant_output_head=args.invariant_output_head,
            algebra=args.algebra,
            use_time=args.use_time,
            pga_hit_encoding=args.pga_hit_encoding,
            two_channel_dc=args.two_channel_dc,
            cga_hit_encoding=args.cga_hit_encoding,
            physical_drift_geometry=args.physical_drift_geometry,
            separate_hit_metadata=args.separate_hit_metadata,
            separate_hit_type=args.separate_hit_type,
            layernorm_epsilon_mode=args.layernorm_epsilon_mode,
            equi_init=args.equi_init,
            fix_cga_null=args.fix_cga_null,
            fix_wire_dir=args.fix_wire_dir,
            normalize_mv_inputs=args.normalize_mv_inputs,
        )
        model = CGATrParquetModel(model_args)

        state = checkpoint
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        elif isinstance(state, dict) and "state_dict" in state:
            if "ema_state_dict" in state and state["ema_state_dict"] is not None:
                state = state["ema_state_dict"]
            else:
                sd = state["state_dict"]
                state = {k[len("model."):] if k.startswith("model.") else k: v
                         for k, v in sd.items()}
        _load_state_dict_tolerating_derived_buffers(model, state)
        model = model.to(device)
        print(f"Model loaded ({sum(p.numel() for p in model.parameters()):,} params)")

        seed_start, seed_end = _parse_seed_range(args.eval_seeds)
        max_hits = args.max_hits if args.max_hits > 0 else None
        print(f"Loading eval data: seeds {seed_start}-{seed_end - 1}  max_hits={max_hits}")
        dataset = IDEAParquetDataset(
            args.data_dir,
            seed_range=(seed_start, seed_end),
            max_hits_per_event=max_hits,
            with_time=args.use_time,
            drop_loopers=args.drop_loopers,
            merge_daughters=args.merge_daughters,
            with_drift_dir=model.needs_drift_dir,
        )
        if args.event_shard_count > 1:
            dataset = _EventShard(
                dataset, args.event_shard_index, args.event_shard_count
            )
            print(
                f"Event shard: {args.event_shard_index}/{args.event_shard_count} "
                f"({len(dataset)} events)"
            )
        if args.drop_loopers or args.merge_daughters:
            print(f"  GGTF target regime: drop_loopers={args.drop_loopers} "
                  f"merge_daughters={args.merge_daughters}")
        print(f"Dataset: {len(dataset)} events")

        print("\n=== Forward pass + cache ===")
        cache = forward_and_cache(model, dataset, device, args.embed_dim)

        del model
        torch.cuda.empty_cache()

        if args.cache_path is not None:
            os.makedirs(args.cache_path, exist_ok=True)
            print(f"\nSaving forward cache to {args.cache_path}...")
            t0 = time.time()
            hits_df = cache_to_dataframe(cache, args.embed_dim)
            hits_df.write_parquet(
                os.path.join(args.cache_path, "forward_hits.parquet"),
                compression="zstd",
            )
            with open(os.path.join(args.cache_path, "manifest.json"), "w") as f:
                json.dump({
                    "checkpoint": args.checkpoint,
                    "checkpoint_sha256": checkpoint_sha256,
                    "dataset_fingerprint": dataset_fingerprint,
                    "inference_source_sha256": inference_source_sha256,
                    "seeds": args.eval_seeds,
                    "embed_dim": args.embed_dim,
                    "max_hits": args.max_hits,
                    "min_signal_mc": MIN_SIGNAL_MC,
                    "fix_particle_zero": MIN_SIGNAL_MC == 0,
                    "event_shard_index": args.event_shard_index,
                    "event_shard_count": args.event_shard_count,
                    "drop_loopers": args.drop_loopers,
                    "merge_daughters": args.merge_daughters,
                    "algebra": args.algebra,
                    "use_time": args.use_time,
                    "pga_hit_encoding": args.pga_hit_encoding,
                    "two_channel_dc": args.two_channel_dc,
                    "cga_hit_encoding": args.cga_hit_encoding,
                    "physical_drift_geometry": args.physical_drift_geometry,
                    "separate_hit_metadata": args.separate_hit_metadata,
                    "separate_hit_type": args.separate_hit_type,
                    "layernorm_epsilon_mode": args.layernorm_epsilon_mode,
                    "normalize_mv_inputs": args.normalize_mv_inputs,
                    "fix_cga_null": args.fix_cga_null,
                    "fix_wire_dir": args.fix_wire_dir,
                    "no_legacy_equivariance": args.no_legacy_equivariance,
                    "equivariance_group": args.equivariance_group,
                    "invariant_output_head": args.invariant_output_head,
                    "n_events": len(cache),
                    "n_hits": int(hits_df.height),
                }, f, indent=2)
            print(
                f"  cache written ({hits_df.height} hits, "
                f"{os.path.getsize(os.path.join(args.cache_path, 'forward_hits.parquet')) / 1e6:.1f} MB) "
                f"in {time.time() - t0:.1f}s"
            )
    else:
        print(f"Cache loaded: {len(cache)} events (no GPU forward pass needed)")


if __name__ == "__main__":
    main()
