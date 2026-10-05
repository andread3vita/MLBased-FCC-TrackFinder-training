"""Parallel version of fcc_cache_at_op.py.

Two improvements over the original:
  1. Event-loop parallelised across --workers processes (embarrassingly parallel).
  2. fake-flag computation vectorised with np.isin instead of a Python for-loop.

Extra flag:
  --beta_mode sigmoid   use stored sig_beta as-is  (default, same as original)
  --beta_mode raw       apply logit transform first: beta_raw = log(p/(1-p))
                        Threshold --tbeta is then in logit-space.
                        Equivalent sigmoid threshold = sigmoid(tbeta).
                        e.g. --tbeta -3.664 with raw == --tbeta 0.025 with sigmoid.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from multiprocessing import Pool

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import polars as pl

from src.eval_fcc_metrics_v36 import (
    add_reconstructable_masks,
    per_cluster_records,
    per_track_records,
)
from src.eval.ggtf_assign import IOU_THRESHOLD, ggtf_assignment
from src.eval.helix_merge import merge_helix
from src.eval_sweep_v33 import get_clustering_greedy
from src.lowpt_op_sweep import cache_from_dataframe


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def get_clustering_self_seed(betas, X, tbeta=0.5, td=0.5):
    """Greedy OC clustering that suppresses seeds already claimed by a seed."""
    indices = np.nonzero(betas > tbeta)[0]
    indices = indices[np.argsort(-betas[indices])]
    unassigned = np.arange(len(betas))
    clustering = -np.ones(len(betas), dtype=np.int32)
    for idx in indices:
        if clustering[idx] >= 0:
            continue
        d = np.linalg.norm(X[unassigned] - X[idx], axis=-1)
        take = d < td
        clustering[unassigned[take]] = idx
        unassigned = unassigned[~take]
    return clustering


def cluster_event(clusterer, betas, X, tbeta, td, min_hits):
    """Apply one truth-free clusterer behind a common scoring interface."""
    if clusterer == "greedy":
        return get_clustering_greedy(betas, X, tbeta=tbeta, td=td)
    if clusterer == "self_seed_greedy":
        return get_clustering_self_seed(betas, X, tbeta=tbeta, td=td)
    if clusterer == "dbscan":
        from sklearn.cluster import DBSCAN
        return DBSCAN(
            eps=td, min_samples=max(min_hits, 2), n_jobs=1
        ).fit_predict(X).astype(np.int32)
    if clusterer == "hdbscan":
        try:
            from hdbscan import HDBSCAN
        except ImportError as exc:
            raise RuntimeError(
                "clusterer=hdbscan requires the optional hdbscan package"
            ) from exc
        return HDBSCAN(
            min_cluster_size=max(min_hits, 2),
            min_samples=max(min_hits, 2),
            cluster_selection_epsilon=td,
        ).fit_predict(X).astype(np.int32)
    raise ValueError(f"unknown clusterer {clusterer!r}")


# ---------------------------------------------------------------------------
# Fragment merging (truth-free post-pass)
# ---------------------------------------------------------------------------

def merge_fragments(labels, X, betas, t_merge):
    """Star-merge clusters whose condensation points lie near a root seed.

    Greedy NMS labels each cluster by its seed hit index; fragments of one
    particle appear as several clusters with nearby seeds (the attractive
    potential pulls all of a particle's hits toward one condensation point).
    Seeds are processed in descending beta: each unclaimed seed becomes a
    root, and any remaining seed within t_merge OF THE ROOT joins it (no
    transitive chaining, so the merged diameter stays <= 2*t_merge and
    nearby particles are not daisy-chained together). Hits keep their
    tight-td assignment; only whole clusters are relabelled to the root.
    """
    seeds = np.unique(labels[labels >= 0])
    n = len(seeds)
    if n < 2:
        return labels
    order = np.argsort(-betas[seeds])          # descending beta
    P = X[seeds][order]
    seeds_o = seeds[order]
    root_of = np.full(n, -1, dtype=np.int64)   # index into seeds_o
    for i in range(n):
        if root_of[i] >= 0:
            continue
        root_of[i] = i
        free = np.nonzero(root_of < 0)[0]
        if len(free) == 0:
            continue
        d = np.linalg.norm(P[free] - P[i], axis=-1)
        root_of[free[d < t_merge]] = i
    lut = dict(zip(seeds_o.tolist(), seeds_o[root_of].tolist()))
    out = labels.copy()
    m = labels >= 0
    out[m] = np.fromiter((lut[s] for s in labels[m].tolist()), dtype=labels.dtype,
                         count=int(m.sum()))
    return out


def attach_debris(labels, X, t_attach, min_core=5):
    """Adopt debris (hits in clusters smaller than min_core, plus unassigned
    hits) into the nearest core cluster if within t_attach of any core hit.

    Tight-td NMS shreds the outskirts of a track into singletons and tiny
    fragments; those debris clusters both hide hit efficiency and dominate the
    fake count. Cores (>= min_core hits) are left untouched, so the match
    decision is unchanged; debris only ever joins, single-linkage, within
    t_attach of a core hit.
    """
    vals, cnts = np.unique(labels[labels >= 0], return_counts=True)
    core_lbls = vals[cnts >= min_core]
    if len(core_lbls) == 0:
        return labels
    core_mask = np.isin(labels, core_lbls)
    debris_idx = np.nonzero(~core_mask)[0]          # tiny clusters + unassigned
    if len(debris_idx) == 0:
        return labels
    Xc = X[core_mask]
    core_lab = labels[core_mask]
    out = labels.copy()
    step = 256
    for a0 in range(0, len(debris_idx), step):
        idx = debris_idx[a0:a0 + step]
        d = np.linalg.norm(X[idx][:, None, :] - Xc[None, :, :], axis=-1)
        j = np.argmin(d, axis=1)
        ok = d[np.arange(len(idx)), j] < t_attach
        out[idx[ok]] = core_lab[j[ok]]
    return out


# ---------------------------------------------------------------------------
# Worker function (must be top-level for multiprocessing pickling)
# ---------------------------------------------------------------------------

def _process_chunk(args):
    """Cluster one chunk; return min-3 tracks, clusters, and all targets."""
    (chunk, tbeta, td, beta_mode, merge_td, attach_td, min_hits,
     helix_tol, helix_min_hits, min_target_hits, clusterer) = args
    tracks, clusters, all_target_tracks = [], [], []
    for e in chunk:
        betas = e["sig_beta"]
        if beta_mode == "raw":
            betas = np.clip(betas, 1e-7, 1.0 - 1e-7)
            betas = np.log(betas / (1.0 - betas))
        labels = cluster_event(
            clusterer, betas, e["sig_coords"], tbeta, td, min_hits
        )
        if merge_td > 0:
            labels = merge_fragments(labels, e["sig_coords"], betas, merge_td)
        # Geometry before debris attachment: the circle fits describe arcs, and
        # adopting debris first would blur them.
        if helix_tol > 0:
            labels = merge_helix(labels, e["sig_pos"], betas,
                                 min_hits=helix_min_hits, tol=helix_tol)
        if attach_td > 0:
            labels = attach_debris(labels, e["sig_coords"], attach_td)
        if min_hits > 1:
            # track-candidate requirement: clusters below min_hits are not
            # promoted to tracks (their hits become unassigned)
            vals, cnts = np.unique(labels[labels >= 0], return_counts=True)
            small = set(vals[cnts < min_hits].tolist())
            if small:
                labels = np.array([-1 if l in small else l for l in labels],
                                  dtype=labels.dtype)
        # GGTF's `create_garbage_label(minNumHits=3)`, applied to the truth and not to the
        # input. A particle with too few hits stops being a target while its hits stay in the
        # event, which is what their loader does -- it relabels, it does not delete (G3). The
        # relabel happens after clustering for exactly that reason: the hits must still be
        # available to be clustered, and only their truth changes.
        #
        # `n_hits_total_map` counts the particle's hits in the whole event rather than in the
        # clustered subset, which is the right denominator for their cut.
        sig_mc_all_targets = e["sig_mc"]
        for t in per_track_records(
            labels, sig_mc_all_targets, e["n_hits_total_map"]
        ):
            t["event_id"] = e["event_id"]
            t["seed"] = e["seed"]
            all_target_tracks.append(t)

        sig_mc = sig_mc_all_targets
        if min_target_hits > 1:
            nmap = e["n_hits_total_map"]
            sig_mc = sig_mc.copy()
            garbage = np.fromiter(
                (nmap.get(int(m), 0) < min_target_hits for m in sig_mc.tolist()),
                dtype=bool, count=len(sig_mc))
            sig_mc[garbage] = -1

        for t in per_track_records(labels, sig_mc, e["n_hits_total_map"]):
            t["event_id"] = e["event_id"]
            t["seed"] = e["seed"]
            tracks.append(t)
        # GGTF's own assignment, computed here because it needs the per-hit
        # labels; reconstructing it later from the aggregates only supports a
        # greedy rule that overstates their fake rate (validate_hungarian.py).
        cl_lbl, cl_assigned, cl_iou = ggtf_assignment(labels, sig_mc)
        assign_map = {int(l): (bool(a), float(i))
                      for l, a, i in zip(cl_lbl, cl_assigned, cl_iou)}
        for c in per_cluster_records(labels, sig_mc):
            c["event_id"] = e["event_id"]
            c["seed"] = e["seed"]
            a, i = assign_map.get(int(c["cluster_id"]), (False, 0.0))
            c["ggtf_assigned"] = a
            c["ggtf_iou"] = i
            clusters.append(c)
    return tracks, clusters, all_target_tracks


# ---------------------------------------------------------------------------
# Vectorised fake-flag helper
# ---------------------------------------------------------------------------

# Bit layout for the (mc_idx, event_id, seed) → int64 key.
# seed   occupies bits  0-12  (up to 8 191)
# event  occupies bits 13-29  (up to 131 071)
# mc_idx occupies bits 30+    (up to ~8 billion before int64 overflow)
_SEED_BITS  = 13
_EVENT_BITS = 17


def _encode_keys(mc_arr, ev_arr, sd_arr):
    return (
        mc_arr.astype(np.int64) << (_SEED_BITS + _EVENT_BITS)
        | ev_arr.astype(np.int64) << _SEED_BITS
        | sd_arr.astype(np.int64)
    )


def _reco_to_keys(reco_set):
    if not reco_set:
        return np.empty(0, dtype=np.int64)
    mc_arr = np.array([m for m, _, _ in reco_set], dtype=np.int64)
    ev_arr = np.array([e for _, e, _ in reco_set], dtype=np.int64)
    sd_arr = np.array([s for _, _, s in reco_set], dtype=np.int64)
    return _encode_keys(mc_arr, ev_arr, sd_arr)


def fake_flags_vec(purs, cl_mc, cl_ev, cl_sd, reco_set):
    """Vectorised replacement for the original fake_array loop.

    A cluster is fake when purity <= 0.75 OR its best-match particle is not
    in the reconstructable set.
    """
    low_purity = purs <= 0.75
    cluster_keys = _encode_keys(cl_mc, cl_ev, cl_sd)
    reco_keys    = _reco_to_keys(reco_set)
    in_reco      = np.isin(cluster_keys, reco_keys)
    return low_purity | ~in_reco


# ---------------------------------------------------------------------------
# Summary helper (identical to original)
# ---------------------------------------------------------------------------

def overall(df: pl.DataFrame, mask_col):
    """Per-particle efficiency under both conventions.

    match_rate is GGTF efficiency definition 1 (purity of the matched cluster
    > 75%); def2 is their definition 2 (purity > 50% AND hit efficiency > 50%),
    the one their IDEA plots use. The two rank configurations differently
    because def1 accepts a pure fragment of a track while def2 does not, so
    both are reported. split/multiple/bad partition the def2 failures:
    split    = pure but partial   (fragmented track)
    multiple = complete but mixed (merged with another particle)
    bad      = neither
    """
    if mask_col is not None:
        df = df.filter(pl.col(mask_col))
    n = len(df)
    keys = ["n", "efficiency", "match_rate", "def2", "split", "multiple", "bad"]
    if n == 0:
        return dict.fromkeys(keys, 0)
    p = df["purity_of_match"].to_numpy()
    e = df["efficiency_per_hit"].to_numpy()
    return {
        "n": n,
        "efficiency": float(df["efficiency_per_hit"].mean()),
        "match_rate": float((p > 0.75).mean()),
        "def2":     float(((p > 0.5) & (e > 0.5)).mean()),
        "split":    float(((p > 0.5) & (e <= 0.5)).mean()),
        "multiple": float(((p <= 0.5) & (e > 0.5)).mean()),
        "bad":      float(((p <= 0.5) & (e <= 0.5)).mean()),
    }


def ggtf_stats(cl: pl.DataFrame, cuts=(0, 3, 10)):
    """GGTF's own fake rate, from the assignment computed during clustering.

    Their formula is unassigned / assigned over candidates above a hit cut, so
    it is unbounded above rather than capped at 100%, and it charges for clones
    because their assignment is one-to-one. `> 3` is the cut behind the CLD
    numbers in their paper, `> 10` the one on the IDEA slide quoting 8%.

    Two properties of this number are easy to misread, so each cut also reports
    what it is made of.

    It can exceed 100%, and routinely does. Split one particle into six
    candidates and the assignment keeps one, leaving five unassigned over one
    assigned: 500%. So a rate above 100% is a statement about fragmentation, not
    a sign of a broken calculation. To keep that readable the fakes are split by
    whether the candidate had a plausible partner at all -- `clone` overlaps some
    particle at IoU >= 0.02 but lost the one-to-one contest to a better candidate
    on it, `spurious` never reached the threshold against anything. At our
    operating point the > 10 bucket is 76% clones, so the number is mostly
    charging us for splitting tracks rather than for inventing them.

    The cut is on the *candidate's* hit count. Their notebook's `> 3 unique hits`
    is a cut on the *particle*, and which side the published figures cut on is
    the open question in the mail to Andrea and Dolores. Until that is answered
    these are our reading of their formula, not their number.

    Note also that `> 3` cannot bind once min_cluster_hits >= 4, because every
    surviving candidate then has at least 4 hits: at our operating point
    `ggtf_fake_rate_gt3` is identically `ggtf_fake_rate_gt0` and is not a looser
    variant of it. `n_cand` is reported per cut so that collapse is visible
    rather than inferred.
    """
    out = {}
    for cut in cuts:
        k = cl.filter(pl.col("cluster_size") > cut) if cut else cl
        n_match = int(k["ggtf_assigned"].sum())
        n_fake = len(k) - n_match
        unmatched = k.filter(~pl.col("ggtf_assigned"))
        n_clone = len(unmatched.filter(pl.col("ggtf_iou") >= IOU_THRESHOLD))
        out[f"ggtf_fake_rate_gt{cut}"] = (
            n_fake / n_match if n_match else math.inf
        )
        out[f"ggtf_candidate_fake_fraction_gt{cut}"] = (
            n_fake / max(len(k), 1)
        )
        out[f"ggtf_n_cand_gt{cut}"] = len(k)
        out[f"ggtf_n_matched_gt{cut}"] = n_match
        out[f"ggtf_n_fake_clone_gt{cut}"] = n_clone
        out[f"ggtf_n_fake_spurious_gt{cut}"] = n_fake - n_clone
    out["ggtf_median_iou"] = float(cl["ggtf_iou"].median() or 0.0)
    return out


def candidate_stats(cl: pl.DataFrame, n_events: int):
    """Per-candidate fake and clone rates under the standard conventions.

    ghost_rate is the fraction of candidates no particle contributes >= 75% of
    (LHCb ghost rate, Belle II fake rate); it is the number to quote against
    other trackers, whereas fake_rate_idea additionally counts clean
    reconstructions of particles outside the IDEA selection. Clones are extra
    candidates for a particle another candidate already matches, which no fake
    definition charges for -- they inflate the denominator of every rate above,
    so candidates_per_event is reported alongside.
    """
    if len(cl) == 0:
        return {
            "candidates_per_event": 0.0,
            "ghost_rate": 0.0,
            "clone_rate": 0.0,
            "mc_clone_rate": 0.0,
            "clones_per_event": 0.0,
            "ghosts_per_event": 0.0,
            "fakes_per_event_idea": 0.0,
        }
    n = len(cl)
    matched = cl.filter(pl.col("purity") > 0.75)
    n_uniq = len(matched.group_by(["matched_mc_idx", "event_id", "seed"]).len())
    n_clone = len(matched) - n_uniq
    return {
        "candidates_per_event": len(cl) / max(n_events, 1),
        "ghost_rate":     float((cl["purity"] <= 0.75).mean()),
        "clone_rate":     n_clone / n,
        "mc_clone_rate":  n_clone / max(len(matched), 1),
        "clones_per_event": n_clone / max(n_events, 1),
        "ghosts_per_event": int((cl["purity"] <= 0.75).sum()) / max(n_events, 1),
        "fakes_per_event_idea": int(cl["is_fake_idea"].sum()) / max(n_events, 1),
    }


def _slice_summary(
    tracks: pl.DataFrame,
    clusters: pl.DataFrame,
    n_events: int,
    all_target_tracks: pl.DataFrame | None = None,
) -> dict:
    """Metrics used by the leakage/stability report for one event slice."""
    low_pt = {}
    for low, high in ((0.1, 0.2), (0.2, 0.5)):
        selected = tracks.filter(
            pl.col("is_reconstructable_idea")
            & (pl.col("pt") >= low)
            & (pl.col("pt") < high)
        )
        low_pt[f"{low:g}-{high:g}"] = overall(selected, None)
    all_targets = tracks if all_target_tracks is None else all_target_tracks
    return {
        "n_events": n_events,
        "no_cuts": overall(all_targets, None),
        "all_targets_no_reconstruction_cuts": overall(all_targets, None),
        "min3_targets_no_reconstruction_cuts": overall(tracks, None),
        "idea": overall(tracks, "is_reconstructable_idea"),
        "low_pt_idea": low_pt,
        **candidate_stats(clusters, n_events),
        **ggtf_stats(clusters),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache_path", required=True,  help="dir with forward_hits.parquet")
    ap.add_argument("--mc_signal",  required=True,  help="compact mc_signal.parquet")
    ap.add_argument("--embed_dim",  type=int, required=True)
    ap.add_argument("--tbeta",      type=float, required=True)
    ap.add_argument("--td",         type=float, required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--tag",        default="op")
    ap.add_argument(
        "--clusterer",
        choices=["greedy", "self_seed_greedy", "dbscan", "hdbscan"],
        default="greedy",
    )
    ap.add_argument("--beta_mode",  choices=["sigmoid", "raw"], default="sigmoid",
                    help="sigmoid=use cached sig_beta as-is; raw=apply logit transform first")
    ap.add_argument("--workers",    type=int, default=32,
                    help="number of parallel worker processes")
    ap.add_argument("--merge_td",   type=float, default=0.0,
                    help="if >0, merge clusters whose condensation points are "
                         "within this radius (truth-free fragment merging)")
    ap.add_argument("--attach_td",  type=float, default=0.0,
                    help="if >0, adopt debris (clusters <5 hits + unassigned) "
                         "into the nearest core cluster within this radius")
    ap.add_argument("--min_cluster_hits", type=int, default=0,
                    help="track-candidate requirement: clusters with fewer "
                         "hits are not promoted to tracks")
    ap.add_argument("--helix_tol", type=float, default=0.0,
                    help="if >0, merge candidates that fit the same circle in "
                         "the transverse plane, to this fractional tolerance on "
                         "centre and radius. Targets curler fragmentation, "
                         "which embedding-distance merging cannot reach. "
                         "Requires positions in the cache (see "
                         "src/eval/attach_positions.py)")
    ap.add_argument("--helix_min_hits", type=int, default=10,
                    help="candidates below this many hits are not circle-fitted")
    ap.add_argument("--max_events", type=int, default=0,
                    help="cap the number of events, for fast operating-point "
                         "sweeps. 0 = all. Rates are ratios so a subset is "
                         "unbiased, but confirm the chosen point on the full "
                         "sample before quoting it.")
    ap.add_argument("--min_target_hits", type=int, default=0,
                    help="GGTF's create_garbage_label(minNumHits=3): a particle with fewer "
                         "hits than this stops being a target, while its hits stay in the "
                         "event to be clustered. Set 3 to reproduce their target definition; "
                         "0 or 1 leaves every particle a target, which is what every number "
                         "reported before 2026-08-05 did. Note this is a cut on the "
                         "*particle*, unlike --min_cluster_hits which is on the candidate.")
    ap.add_argument(
        "--summary_only", action="store_true",
        help="write summary.json only; useful for large operating-point grids",
    )
    ap.add_argument(
        "--sweep_provenance_sha256",
        default=None,
        help="hash of the fail-closed sweep campaign manifest",
    )
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    t0 = time.time()

    # Print threshold info so the user can judge the operating point
    if args.beta_mode == "raw":
        sig_equiv = 1.0 / (1.0 + math.exp(-args.tbeta))
        print(f"[beta_mode=raw]  tbeta={args.tbeta:.4f} in logit-space  "
              f"(equivalent sigmoid threshold = {sig_equiv:.4f})")
    else:
        raw_equiv = math.log(args.tbeta / (1.0 - args.tbeta))
        print(f"[beta_mode=sigmoid]  tbeta={args.tbeta:.4f}  "
              f"(equivalent raw/logit threshold = {raw_equiv:.4f})")

    print(f"Loading cache from {args.cache_path} ...")
    hits_df = pl.read_parquet(os.path.join(args.cache_path, "forward_hits.parquet"))
    pos_path = os.path.join(args.cache_path, "forward_positions.parquet")
    if args.helix_tol > 0 and "pos_x" not in hits_df.columns:
        if not os.path.exists(pos_path):
            raise SystemExit(
                f"--helix_tol needs hit positions; neither forward_hits.parquet "
                f"nor {pos_path} has them. Run src/eval/attach_positions.py "
                f"--cache_path {args.cache_path} first.")
        position_manifest_path = os.path.join(
            args.cache_path, "forward_positions.manifest.json"
        )
        if not os.path.exists(position_manifest_path):
            raise RuntimeError("position sidecar has no verification manifest")
        with open(position_manifest_path) as handle:
            position_manifest = json.load(handle)
        if (
            position_manifest.get("source_forward_hits_sha256")
            != _sha256(os.path.join(args.cache_path, "forward_hits.parquet"))
            or position_manifest.get("positions_sha256") != _sha256(pos_path)
            or position_manifest.get("rows") != len(hits_df)
        ):
            raise RuntimeError(
                "position sidecar provenance does not match forward cache"
            )
        positions = pl.read_parquet(pos_path)
        if len(positions) != len(hits_df):
            raise RuntimeError("position sidecar row count mismatch")
        hits_df = pl.concat([hits_df, positions], how="horizontal")
        print(f"  positions attached from {os.path.basename(pos_path)}")
    cache = cache_from_dataframe(hits_df, args.embed_dim)
    del hits_df  # free before forking so workers don't copy-on-write it
    if args.max_events > 0:
        cache = cache[: args.max_events]
    n_events = len(cache)
    print(f"Loaded {n_events} events | workers={args.workers} | "
          f"~{n_events // args.workers} events/worker")

    # Split cache into chunks (one per worker)
    n_workers   = args.workers
    chunk_size  = max(1, (len(cache) + n_workers - 1) // n_workers)
    chunks      = [cache[i:i + chunk_size] for i in range(0, len(cache), chunk_size)]
    task_args   = [(chunk, args.tbeta, args.td, args.beta_mode, args.merge_td,
                    args.attach_td, args.min_cluster_hits, args.helix_tol,
                    args.helix_min_hits, args.min_target_hits, args.clusterer)
                   for chunk in chunks]
    del cache  # workers have their own copy via fork

    print(f"Clustering {len(chunks)} chunks in parallel ...")
    tracks, clusters, all_target_tracks = [], [], []
    with Pool(processes=n_workers) as pool:
        for i, (t, c, a) in enumerate(
            pool.imap_unordered(_process_chunk, task_args)
        ):
            tracks.extend(t)
            clusters.extend(c)
            all_target_tracks.extend(a)
            elapsed = time.time() - t0
            print(f"  chunk {i+1:>3}/{len(chunks)}  "
                  f"tracks={len(tracks):>9,}  clusters={len(clusters):>9,}  "
                  f"elapsed={elapsed:.0f}s",
                  flush=True)

    t_cluster = time.time() - t0
    print(f"Clustering done in {t_cluster:.0f}s  "
          f"({len(tracks):,} tracks, {len(clusters):,} clusters)")

    # ------------------------------------------------------------------
    # Join with mc_signal
    # ------------------------------------------------------------------
    mc_df = pl.read_parquet(args.mc_signal)
    print(f"Loaded {mc_df.height:,} signal mc rows")

    track_pl = pl.DataFrame(tracks)
    all_target_track_pl = pl.DataFrame(all_target_tracks)
    duplicate_mc = (
        mc_df.group_by(["mc_index", "event_id", "seed"])
        .len()
        .filter(pl.col("len") != 1)
    )
    if len(duplicate_mc):
        raise RuntimeError(
            f"mc_signal has {len(duplicate_mc)} duplicate truth keys"
        )
    joined = track_pl.join(
        mc_df,
        left_on=["mc_idx", "event_id", "seed"],
        right_on=["mc_index", "event_id", "seed"],
        how="left",
        validate="m:1",
    )
    if len(joined) != len(track_pl) or joined["pt"].null_count():
        raise RuntimeError(
            "min-3 track-to-MC join is incomplete or changed row count"
        )
    joined = add_reconstructable_masks(joined)
    all_target_joined = all_target_track_pl.join(
        mc_df,
        left_on=["mc_idx", "event_id", "seed"],
        right_on=["mc_index", "event_id", "seed"],
        how="left",
        validate="m:1",
    )
    if (
        len(all_target_joined) != len(all_target_track_pl)
        or all_target_joined["pt"].null_count()
    ):
        raise RuntimeError(
            "all-target track-to-MC join is incomplete or changed row count"
        )
    all_target_joined = add_reconstructable_masks(all_target_joined)

    cluster_schema = {
        "cluster_id": pl.Int64,
        "cluster_size": pl.Int64,
        "matched_mc_idx": pl.Int64,
        "best_match": pl.Int64,
        "purity": pl.Float64,
        "event_id": pl.Int64,
        "seed": pl.Int64,
        "ggtf_assigned": pl.Boolean,
        "ggtf_iou": pl.Float64,
    }
    cluster_pl = (
        pl.DataFrame(clusters)
        if clusters else pl.DataFrame(schema=cluster_schema)
    )
    cluster_joined = cluster_pl.join(
        mc_df.select(["mc_index", "event_id", "seed", "pt", "theta", "phi",
                      "gen_status", "charge", "decayed_in_tracker"]),
        left_on=["matched_mc_idx", "event_id", "seed"],
        right_on=["mc_index", "event_id", "seed"],
        how="left",
        validate="m:1",
    )

    # ------------------------------------------------------------------
    # Fake flags — vectorised
    # ------------------------------------------------------------------
    def reco_set(mask_col):
        f = joined.filter(pl.col(mask_col))
        return set(zip(f["mc_idx"].to_list(), f["event_id"].to_list(), f["seed"].to_list()))

    reco_idea = reco_set("is_reconstructable_idea")
    reco_cld  = reco_set("is_reconstructable_cld")

    purs  = cluster_joined["purity"].to_numpy()
    cl_mc = cluster_joined["matched_mc_idx"].to_numpy()
    cl_ev = cluster_joined["event_id"].to_numpy()
    cl_sd = cluster_joined["seed"].to_numpy()

    print(f"Computing fake flags (vectorised) over {len(purs):,} clusters ...")
    cluster_joined = cluster_joined.with_columns([
        pl.Series("is_fake_idea", fake_flags_vec(purs, cl_mc, cl_ev, cl_sd, reco_idea)),
        pl.Series("is_fake_cld",  fake_flags_vec(purs, cl_mc, cl_ev, cl_sd, reco_cld)),
    ])

    # ------------------------------------------------------------------
    # Write outputs
    # ------------------------------------------------------------------
    out_tracks = os.path.join(args.output_dir, "cache.parquet")
    out_clusters = os.path.join(args.output_dir, "cache_clusters.parquet")
    if not args.summary_only:
        joined.write_parquet(out_tracks)
        cluster_joined.write_parquet(out_clusters)

    event_splits = {}
    for split_name, remainder in (("even", 0), ("odd", 1)):
        split_tracks = joined.filter(pl.col("event_id") % 2 == remainder)
        split_all_targets = all_target_joined.filter(
            pl.col("event_id") % 2 == remainder
        )
        split_clusters = cluster_joined.filter(pl.col("event_id") % 2 == remainder)
        split_events = split_tracks.select(["seed", "event_id"]).unique().height
        event_splits[split_name] = _slice_summary(
            split_tracks, split_clusters, split_events, split_all_targets
        )

    summary = {
        "tag":             args.tag,
        "sweep_provenance_sha256": args.sweep_provenance_sha256,
        "clusterer":       args.clusterer,
        "clusterer_parameter_semantics": (
            "tbeta and td are seed and assignment thresholds"
            if args.clusterer in ("greedy", "self_seed_greedy")
            else "tbeta is unused; td is the density distance/epsilon"
        ),
        "beta_mode":       args.beta_mode,
        "tbeta":           args.tbeta,
        "td":              args.td,
        "merge_td":        args.merge_td,
        "attach_td":       args.attach_td,
        "min_cluster_hits": args.min_cluster_hits,
        "helix_tol":       args.helix_tol,
        "helix_min_hits":  args.helix_min_hits,
        "oracle_truth_position_helix": args.helix_tol > 0,
        "promotable_operating_point": args.helix_tol <= 0,
        "min_target_hits": args.min_target_hits,
        "fix_particle_zero": os.environ.get("FIX_PARTICLE_ZERO") == "1",
        "min_signal_mc": 0 if os.environ.get("FIX_PARTICLE_ZERO") == "1" else 1,
        "metric_definitions": {
            "def1": "per-target best-cluster purity > 0.75",
            "def2": "purity > 0.5 and hit efficiency > 0.5",
            "ggtf_fake": "unassigned / matched after Hungarian assignment",
            "candidate_fake": "unassigned / all candidates",
            "candidate_cut": "candidate cluster_size > cut",
            "no_cuts": (
                "all cached primary targets before min-3 truth relabelling"
            ),
        },
        "n_events":        n_events,
        "n_tracks_total":  len(joined),
        "n_tracks_all_targets": len(all_target_joined),
        "n_clusters_total": len(cluster_joined),
        "no_cuts":   overall(all_target_joined, None),
        "all_targets_no_reconstruction_cuts": overall(
            all_target_joined, None
        ),
        "min3_targets_no_reconstruction_cuts": overall(joined, None),
        "idea":      overall(joined, "is_reconstructable_idea"),
        "cld":       overall(joined, "is_reconstructable_cld"),
        "displaced": overall(joined, "is_reconstructable_displaced"),
        "fake_rate_idea": int(cluster_joined["is_fake_idea"].sum()) / max(len(cluster_joined), 1),
        "fake_rate_cld":  int(cluster_joined["is_fake_cld"].sum())  / max(len(cluster_joined), 1),
        "overall_purity": float(cluster_joined["purity"].mean() or 0.0),
        **candidate_stats(cluster_joined, n_events),
        **ggtf_stats(cluster_joined),
        "low_pt_idea": _slice_summary(
            joined, cluster_joined, n_events
        )["low_pt_idea"],
        "event_splits": event_splits,
    }
    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 70)
    print(f"FCC summary  ({args.tag}  beta_mode={args.beta_mode}  "
          f"tbeta={args.tbeta} td={args.td} merge_td={args.merge_td} "
          f"attach_td={args.attach_td})")
    print("=" * 70)
    print(f"tracks={summary['n_tracks_total']:,}  clusters={summary['n_clusters_total']:,}"
          f"  events={n_events:,}  candidates/event={summary['candidates_per_event']:.1f}")
    print(f"  overall purity:  {summary['overall_purity']:.3f}")
    print(f"  ghost rate:      {summary['ghost_rate']:.3f}  (purity<=75%, strict complement)")
    print(f"  fake rate IDEA:  {summary['fake_rate_idea']:.3f}   "
          f"CLD: {summary['fake_rate_cld']:.3f}   (ghost OR non-selected particle)")
    print(f"  clone rate:      {summary['clone_rate']:.3f}  "
          f"MC-clone: {summary['mc_clone_rate']:.3f}")
    print(f"  GGTF fake rate:  >3 hits {summary['ggtf_fake_rate_gt3']:.3f}   "
          f">10 hits {summary['ggtf_fake_rate_gt10']:.3f}   "
          f"(unassigned/assigned, their convention; clone-sensitive)")
    print(f"  per event:  ghosts={summary['ghosts_per_event']:.2f}  "
          f"fakes={summary['fakes_per_event_idea']:.2f}  "
          f"clones={summary['clones_per_event']:.2f}")
    for tag in [
        "no_cuts", "min3_targets_no_reconstruction_cuts",
        "idea", "cld", "displaced",
    ]:
        s = summary[tag]
        print(f"  [{tag:>9}] n={s['n']:>8,}  "
              f"eff_hit={s['efficiency']:.3f}  def1={s['match_rate']:.3f}  "
              f"def2={s['def2']:.3f}  (split={s['split']:.3f} "
              f"multi={s['multiple']:.3f} bad={s['bad']:.3f})")
    print(f"\nTotal wall time: {time.time() - t0:.0f}s")
    if args.summary_only:
        print(f"Saved {os.path.join(args.output_dir, 'summary.json')}")
    else:
        print(f"Saved {out_tracks}, {out_clusters}, summary.json")


if __name__ == "__main__":
    main()
