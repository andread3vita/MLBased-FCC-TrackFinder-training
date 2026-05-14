"""FCC-matching tracking metrics for v36-EF (per-particle + per-cluster cache).

Computes the same metrics the FCC group uses on slides 16, 24, 26:

  - Tracking efficiency vs (pT, theta, eta, vertex-R, n_hits)
  - Fake rate (fraction of reconstructed clusters not matched to a
    reconstructable particle at >=75% purity)
  - N_hits-per-particle distribution
  - Reconstructable-particle masks: IDEA (slide 24), CLD (slide 16),
    displaced (slide 26).

Operating point is fixed at v36 phase-2 sweep optimum
(`tbeta=0.025`, `td=0.10`).

Outputs a per-particle and per-cluster Parquet cache so the plotting
script can be re-run without re-running inference.

Usage:
    cd model_training
    PYTHONPATH=. python src/eval_fcc_metrics_v36.py \
        --data_dir /home/marko.cechovic/cgatr/data_parquet_train \
        --checkpoint checkpoints/cgatr_v36_EF/cgatr_best.pt \
        --embed_dim 5 --eval_seeds 181-190 --max_events 5000 \
        --tbeta 0.025 --td 0.10 --gpu 0 \
        --output_dir eval_results/v36ef_fcc
"""

import os
import sys
import argparse
import json
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import polars as pl
import torch

from src.eval_sweep_v33 import (
    CGATrParquetModel,
    get_clustering_greedy,
    parse_seed_range,
)
from src.dataset.parquet_dataset import IDEAParquetDataset


def per_track_records(pred_labels, mc_index_signal, n_hits_total_map):
    """Per primary-track records computed on the signal subset.

    `mc_index_signal` is the `mc_index` array on the signal subset that
    was actually clustered. `n_hits_total_map` maps mc_idx -> total
    detector hit count (including secondaries) for the reconstructable cut.
    """
    unique_true = np.unique(mc_index_signal)
    unique_true = unique_true[unique_true > 0]
    rows = []
    for tid in unique_true:
        tmask = mc_index_signal == tid
        n_true_signal = int(tmask.sum())
        pred_for_track = pred_labels[tmask]
        pred_for_track = pred_for_track[pred_for_track >= 0]
        if len(pred_for_track) == 0:
            best_label = -1
            best_match = 0
            cluster_size = 0
        else:
            best_label, best_match = Counter(pred_for_track).most_common(1)[0]
            cluster_size = int((pred_labels == best_label).sum())
        eff = best_match / max(n_true_signal, 1)
        purity = best_match / cluster_size if cluster_size > 0 else 0.0
        rows.append({
            "mc_idx": int(tid),
            "n_hits_signal": n_true_signal,
            "n_hits_total": int(n_hits_total_map.get(int(tid), n_true_signal)),
            "best_label": int(best_label),
            "best_match": int(best_match),
            "cluster_size": int(cluster_size),
            "efficiency_per_hit": float(eff),
            "purity_of_match": float(purity),
            "matched": bool(purity >= 0.75 and best_match > 0),
        })
    return rows


def per_cluster_records(pred_labels, mc_index_signal):
    """Per-cluster records. Used for fake rate.

    Each cluster gets:
      - cluster_id (the seed-hit index used as label by greedy)
      - cluster_size
      - matched_mc_idx (best-match true mc_index, -1 if cluster has no signal)
      - best_match (count of hits from matched_mc_idx)
      - purity (best_match / cluster_size)
    """
    unique_pred = np.unique(pred_labels[pred_labels >= 0])
    rows = []
    for pid in unique_pred:
        cmask = pred_labels == pid
        cluster_mc = mc_index_signal[cmask]
        cluster_size = int(cmask.sum())
        if cluster_size == 0:
            continue
        # all entries of cluster_mc are guaranteed > 0 since mc_index_signal
        # comes from sig_mask (mc_index != 0), but bincount needs non-negative
        counts = np.bincount(cluster_mc.astype(np.int64))
        matched_mc = int(counts.argmax())
        best_match = int(counts.max())
        purity = best_match / cluster_size
        rows.append({
            "cluster_id": int(pid),
            "cluster_size": cluster_size,
            "matched_mc_idx": matched_mc,
            "best_match": best_match,
            "purity": float(purity),
        })
    return rows


@torch.no_grad()
def run_inference(model, dataset, device, max_events, embed_dim, tbeta, td,
                  log_every=50):
    """Inference + clustering loop; emits per-track + per-cluster records."""
    model.eval()
    track_records = []
    cluster_records = []

    n_events = min(max_events, len(dataset))
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

        sig_mask = (~is_secondary) & (mc_index_all != 0)
        if sig_mask.sum() < 4:
            n_skipped += 1
            continue

        try:
            output = model(features, seq_lens)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            n_skipped += 1
            continue
        coords = output[:, :embed_dim].cpu().numpy()
        beta = torch.sigmoid(output[:, embed_dim]).cpu().numpy()

        sig_coords = coords[sig_mask]
        sig_beta = beta[sig_mask]
        sig_mc = mc_index_all[sig_mask]

        n_hits_total_map = {}
        unique_mc_all = np.unique(mc_index_all)
        unique_mc_all = unique_mc_all[unique_mc_all > 0]
        for mc_idx in unique_mc_all:
            n_hits_total_map[int(mc_idx)] = int((mc_index_all == mc_idx).sum())

        labels = get_clustering_greedy(sig_beta, sig_coords, tbeta=tbeta, td=td)

        track_rows = per_track_records(labels, sig_mc, n_hits_total_map)
        cluster_rows = per_cluster_records(labels, sig_mc)

        dc_path, vtx_path, eid, _ = dataset._index[idx]
        seed = int(Path(dc_path).parent.name.replace("seed_", ""))

        for tr in track_rows:
            tr["event_id"] = int(eid)
            tr["seed"] = int(seed)
            track_records.append(tr)
        for cr in cluster_rows:
            cr["event_id"] = int(eid)
            cr["seed"] = int(seed)
            cluster_records.append(cr)

        if (idx + 1) % log_every == 0 or idx + 1 == n_events:
            dt = time.time() - t0
            rate = (idx + 1) / max(dt, 1e-3)
            eta = (n_events - idx - 1) / max(rate, 1e-3)
            print(
                f"  Event {idx + 1}/{n_events}  "
                f"({len(track_records)} tracks, {len(cluster_records)} clusters)  "
                f"{rate:.2f} ev/s  ETA {eta / 60.0:.1f} min  skipped={n_skipped}",
                flush=True,
            )

        del output, coords, beta
        if (idx + 1) % 200 == 0:
            torch.cuda.empty_cache()

    print(f"Inference done in {(time.time() - t0) / 60.0:.1f} min "
          f"({len(track_records)} tracks, {len(cluster_records)} clusters, "
          f"{n_skipped} events skipped)")

    return track_records, cluster_records


def load_mc_particles(data_dir, seed_start, seed_end):
    """Load per-(event,seed,mc_index) truth columns we need for cuts."""
    dfs = []
    cols = [
        "mc_index", "pt", "theta", "phi", "vx", "vy", "vz",
        "gen_status", "decayed_in_tracker", "charge", "pdg",
        "event_id", "seed",
    ]
    for seed in range(seed_start, seed_end):
        mc_path = Path(data_dir) / f"seed_{seed}" / "mc_particles_train.parquet"
        if not mc_path.exists():
            continue
        df = pl.read_parquet(str(mc_path), columns=cols)
        dfs.append(df)
    if not dfs:
        return None
    return pl.concat(dfs)


def add_reconstructable_masks(joined: pl.DataFrame) -> pl.DataFrame:
    """Add IDEA / CLD / displaced reconstructable masks + derived columns."""
    eps = 1e-6
    joined = joined.with_columns([
        ((pl.col("vx") ** 2 + pl.col("vy") ** 2).sqrt()).alias("vertex_r"),
        (pl.col("theta") * (180.0 / np.pi)).alias("theta_deg"),
        (-((pl.col("theta") / 2.0).tan() + eps).log()).alias("eta"),
    ])
    joined = joined.with_columns(pl.col("theta").cos().alias("cos_theta"))

    joined = joined.with_columns([
        (
            (pl.col("n_hits_total") > 10)
            & (pl.col("theta_deg") > 15.0)
            & (pl.col("theta_deg") < 165.0)
            & pl.col("gen_status").is_in([0, 1])
            & (pl.col("charge").abs() > 0.0)
        ).alias("is_reconstructable_idea"),
        (
            (pl.col("pt") > 0.1)
            & (pl.col("cos_theta").abs() < 0.99)
            & (pl.col("n_hits_total") >= 4)
            & (pl.col("decayed_in_tracker") == 0)
            & (pl.col("charge").abs() > 0.0)
        ).alias("is_reconstructable_cld"),
        (
            (pl.col("pt") > 1.0)
            & (pl.col("theta_deg") > 10.0)
            & (pl.col("theta_deg") < 170.0)
            & (pl.col("charge").abs() > 0.0)
        ).alias("is_reconstructable_displaced"),
    ])
    return joined


def main():
    parser = argparse.ArgumentParser(description="FCC-style metrics for v36-EF")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--eval_seeds", type=str, default="181-190")
    parser.add_argument("--max_hits", type=int, default=3000,
                        help="Subsample DC hits per event to this max")
    parser.add_argument("--max_events", type=int, default=5000)
    parser.add_argument("--embed_dim", type=int, default=5)
    parser.add_argument("--num_blocks", type=int, default=10)
    parser.add_argument("--tbeta", type=float, default=0.025,
                        help="v36 phase-2 sweep optimum")
    parser.add_argument("--td", type=float, default=0.10,
                        help="v36 phase-2 sweep optimum")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--output_dir", type=str,
                        default="eval_results/v36ef_fcc")
    parser.add_argument("--tag", type=str, default="v36ef")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print(f"Loading checkpoint: {args.checkpoint}")
    model = CGATrParquetModel(num_blocks=args.num_blocks, embed_dim=args.embed_dim)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)

    # Three checkpoint formats are supported:
    #   1. v34/v35-style flat state_dict (keys are "bn_pos.weight" etc.)
    #   2. v37-style {"model_state_dict": flat_state_dict, ...}
    #   3. Lightning ckpt: {"state_dict": prefixed_state_dict, "ema_state_dict": ...}
    #      where keys carry a leading "model." prefix from the LightningModule
    #      attribute name and EMA shadow (if any) is stored alongside.
    if isinstance(state, dict) and "model_state_dict" in state:
        # v37 per-epoch ckpt
        state = state["model_state_dict"]
    elif isinstance(state, dict) and "state_dict" in state:
        # Lightning ckpt — prefer EMA shadow if present (eval-time semantics
        # match v37, which always swaps in EMA weights for validation).
        if "ema_state_dict" in state and state["ema_state_dict"] is not None:
            print("  Lightning ckpt: using EMA shadow for evaluation")
            ema_sd = state["ema_state_dict"]
            # ema_state_dict from our LightningModule was built from
            # CGATrParquetModel.state_dict(), so its keys have NO "model."
            # prefix — they match the eval model directly.
            state = ema_sd
        else:
            print("  Lightning ckpt: using raw state_dict (no EMA found)")
            sd = state["state_dict"]
            # Strip the "model." prefix that LightningModule auto-adds.
            state = {k[len("model."):] if k.startswith("model.") else k: v
                     for k, v in sd.items()}
    model.load_state_dict(state, strict=True)
    model = model.to(device)
    print(f"Model loaded ({sum(p.numel() for p in model.parameters()):,} params)")

    seed_start, seed_end = parse_seed_range(args.eval_seeds)
    print(f"Loading eval data: seeds {seed_start}-{seed_end - 1}")
    dataset = IDEAParquetDataset(
        args.data_dir,
        seed_range=(seed_start, seed_end),
        max_hits_per_event=args.max_hits,
    )
    print(f"Dataset: {len(dataset)} events (max_hits={args.max_hits})")

    print(f"\n=== Inference + greedy clustering "
          f"(tbeta={args.tbeta}, td={args.td}) ===")
    tracks, clusters = run_inference(
        model, dataset, device,
        max_events=args.max_events,
        embed_dim=args.embed_dim,
        tbeta=args.tbeta,
        td=args.td,
    )
    if not tracks:
        print("No tracks produced. Exiting.")
        return

    print(f"\nLoading mc_particles for seeds {seed_start}-{seed_end - 1}...")
    mc_df = load_mc_particles(args.data_dir, seed_start, seed_end)
    if mc_df is None:
        print("ERROR: No mc_particles found")
        return
    print(f"  loaded {len(mc_df)} particle rows")

    track_pl = pl.DataFrame(tracks)
    joined = track_pl.join(
        mc_df,
        left_on=["mc_idx", "event_id", "seed"],
        right_on=["mc_index", "event_id", "seed"],
        how="left",
    )
    n_matched = joined.filter(pl.col("pt").is_not_null()).height
    n_total = joined.height
    print(f"Joined {n_matched}/{n_total} tracks with mc_particles")
    joined = joined.filter(pl.col("pt").is_not_null())
    if len(joined) == 0:
        print("ERROR: No tracks could be joined")
        return

    joined = add_reconstructable_masks(joined)

    # Same join for cluster records: matched_mc_idx -> mc_particles (so we can
    # check whether the matched particle is reconstructable)
    cluster_pl = pl.DataFrame(clusters)
    cluster_joined = cluster_pl.join(
        mc_df.select(["mc_index", "event_id", "seed", "pt", "theta", "phi",
                      "gen_status", "charge", "decayed_in_tracker"]),
        left_on=["matched_mc_idx", "event_id", "seed"],
        right_on=["mc_index", "event_id", "seed"],
        how="left",
    )

    # Build set of reconstructable mc_indices for fake-rate denominator
    # IDEA cut is the headline number on slide 24
    reco_idea_set = set(
        zip(
            joined.filter(pl.col("is_reconstructable_idea"))["mc_idx"].to_list(),
            joined.filter(pl.col("is_reconstructable_idea"))["event_id"].to_list(),
            joined.filter(pl.col("is_reconstructable_idea"))["seed"].to_list(),
        )
    )
    reco_cld_set = set(
        zip(
            joined.filter(pl.col("is_reconstructable_cld"))["mc_idx"].to_list(),
            joined.filter(pl.col("is_reconstructable_cld"))["event_id"].to_list(),
            joined.filter(pl.col("is_reconstructable_cld"))["seed"].to_list(),
        )
    )

    purs = cluster_joined["purity"].to_numpy()
    cl_mc = cluster_joined["matched_mc_idx"].to_numpy()
    cl_ev = cluster_joined["event_id"].to_numpy()
    cl_sd = cluster_joined["seed"].to_numpy()

    def fake_array(reco_set):
        out = np.empty(len(purs), dtype=bool)
        for i in range(len(purs)):
            if purs[i] < 0.75:
                out[i] = True
            else:
                out[i] = (int(cl_mc[i]), int(cl_ev[i]), int(cl_sd[i])) not in reco_set
        return out

    is_fake_idea = fake_array(reco_idea_set)
    is_fake_cld = fake_array(reco_cld_set)
    cluster_joined = cluster_joined.with_columns([
        pl.Series("is_fake_idea", is_fake_idea),
        pl.Series("is_fake_cld", is_fake_cld),
    ])

    # Save caches
    out_tracks = os.path.join(args.output_dir, "cache.parquet")
    out_clusters = os.path.join(args.output_dir, "cache_clusters.parquet")
    joined.write_parquet(out_tracks)
    cluster_joined.write_parquet(out_clusters)
    print(f"Saved {out_tracks}  ({len(joined)} rows)")
    print(f"Saved {out_clusters}  ({len(cluster_joined)} rows)")

    # Print headline numbers
    def overall(df: pl.DataFrame, mask_col: str | None):
        if mask_col is not None:
            df = df.filter(pl.col(mask_col))
        n = len(df)
        if n == 0:
            return {"n": 0, "efficiency": 0.0, "match_rate": 0.0}
        return {
            "n": n,
            "efficiency": float(df["efficiency_per_hit"].mean()),
            "match_rate": float(df["matched"].cast(pl.Float64).mean()),
        }

    summary = {
        "checkpoint": args.checkpoint,
        "eval_seeds": args.eval_seeds,
        "max_events": args.max_events,
        "tbeta": args.tbeta,
        "td": args.td,
        "n_tracks_total": len(joined),
        "n_clusters_total": len(cluster_joined),
        "no_cuts": overall(joined, None),
        "idea": overall(joined, "is_reconstructable_idea"),
        "cld": overall(joined, "is_reconstructable_cld"),
        "displaced": overall(joined, "is_reconstructable_displaced"),
    }

    n_fake_idea = int(cluster_joined["is_fake_idea"].sum())
    n_fake_cld = int(cluster_joined["is_fake_cld"].sum())
    summary["fake_rate_idea"] = n_fake_idea / max(len(cluster_joined), 1)
    summary["fake_rate_cld"] = n_fake_cld / max(len(cluster_joined), 1)
    summary["overall_purity"] = float(cluster_joined["purity"].mean())

    print("\n" + "=" * 70)
    print(f"FCC-style summary  ({args.tag})")
    print("=" * 70)
    print(f"tracks={summary['n_tracks_total']}  clusters={summary['n_clusters_total']}")
    print(f"  overall purity (cluster mean): {summary['overall_purity']:.3f}")
    print(f"  fake rate (IDEA reco set):     {summary['fake_rate_idea']:.3f}")
    print(f"  fake rate (CLD reco set):      {summary['fake_rate_cld']:.3f}")
    for tag in ["no_cuts", "idea", "cld", "displaced"]:
        s = summary[tag]
        print(f"  [{tag:>9}] n={s['n']:>6}  eff_hit={s['efficiency']:.3f}  "
              f"match_rate={s['match_rate']:.3f}")

    out_summary = os.path.join(args.output_dir, "summary.json")
    with open(out_summary, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved {out_summary}")


if __name__ == "__main__":
    main()
