"""Our fake rate next to GGTF's, computed the way GGTF actually computes it.

Their evaluation notebooks (andread3vita/Tracking_DC,
`notebook/2_output_inference.ipynb` and `5_evaluation_CLD_background.ipynb`) do

    more_than_4_hits = sd_hgb["pred_showers_E"] > 3
    percentage_of_fakes = (
        np.sum(np.isnan(sd_hgb["reco_showers_E"][more_than_4_hits].values))
        / np.sum(~np.isnan(sd_hgb["reco_showers_E"][more_than_4_hits].values))
        * 100
    )

Three things about that formula are worth spelling out, because none of them
match what we have been reporting.

1. The denominator is the number of **matched** tracks, not all candidates.
   A rate of u/m rather than u/(u+m) is systematically smaller, and unbounded
   above rather than capped at 100%.
2. `reco_showers_E` is NaN when the assignment in
   `generate_showers_data_frame` left the cluster unpaired. The assignment is
   **one-to-one**, so a second cluster on an already-claimed particle cannot be
   matched and is counted as a fake. Their fake bucket therefore contains our
   clones. Given how many clones we have, this is the dominant term.
3. There is no purity threshold anywhere in it. A cluster is matched or not by
   the assignment, not by passing 75%.

The assignment itself is in `src/layers/inference_oc_tracks.py`: an IoU matrix
between clusters and particles, entries below **0.02 zeroed**, then
`linear_sum_assignment` on the negated matrix, then pairs with IoU still zero
dropped. So a cluster overlapping its best particle by less than 2% of their
union cannot be matched at all, however pure it is.

The hit cut is on the *candidate*, not the particle: `> 3` for the CLD numbers
in the paper, `> 10` for the IDEA slide that carries the 8%.

    PYTHONPATH=. python src/eval/ggtf_fake_rate.py
    PYTHONPATH=. python src/eval/ggtf_fake_rate.py --physics_cuts
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import polars as pl

BASE = "/home/marko.cechovic/cgatr-paper/literature/cgatr_fcc_pkg/eval_results"

VARIANTS = [
    # GGTF clusters at t_beta=0.6, t_d=0.2, so the first row is the closest
    # thing we have to their operating point and the fairest comparison.
    ("CIRCE, t_d=0.2 (GGTF OP)",  "consol_r3_nokeepall/fcc_unmerged"),
    ("CIRCE, t_d=0.05",           "consol_r3_nokeepall/merge_sweep/td0.05_mg0_at0"),
    ("CIRCE, t_d=0.10",           "consol_r3_nokeepall/merge_sweep/td0.10_mg0_at0"),
    ("CIRCE, >=4 hits",           "consol_r3_nokeepall/merge_sweep/td0.10_mg0_at0_mh4"),
    ("CIRCE + merging",           "consol_r3_nokeepall/merge_sweep/td0.10_mg0.15_at0"),
    ("ADOPTED: merge 0.20, >=4",  "consol_r3_nokeepall/merge_sweep/td0.10_mg0.20_at0_mh4"),
]

N_EVENTS = 10000

# Candidate hit cuts. GGTF uses > 3 for the CLD paper numbers and > 10 for the
# IDEA slide that quotes 8%.
CUTS = [0, 3, 10]

# inference_oc_tracks.py: iou_threshold = 0.02, applied before the assignment.
IOU_THRESHOLD = 0.02

# make_cuts_ct in src/evaluation/CLD_eval.py. Their mask_pt is computed and then
# left out of the product, so it is not applied here either. delta_MC (isolation
# from the nearest other particle) we cannot reproduce and note as missing.
PHYS_VERTEX_R_MM = 50.0
PHYS_THETA_DEG = (10.0, 170.0)
PHYS_MIN_UNIQUE_HITS = 3


def add_iou(clusters: pl.DataFrame, tracks: pl.DataFrame) -> pl.DataFrame:
    """IoU of each candidate with its majority particle.

    intersection is `best_match`, the hits the candidate contributes to that
    particle; union is the candidate plus the particle minus the intersection.
    The particle's own hit count comes from the track table's `n_hits_signal`,
    which counts the same signal hits the clustering saw.

    Only the majority particle is available per candidate, so this is the
    diagonal of their IoU matrix rather than the whole thing. For the assignment
    that is almost always enough: a candidate's largest IoU is with the particle
    contributing most of its hits. `validate_hungarian.py` checks that against a
    real `linear_sum_assignment` over the full matrix.
    """
    n_true = tracks.select([
        pl.col("mc_idx").alias("matched_mc_idx"),
        pl.col("event_id"), pl.col("seed"),
        pl.col("n_hits_signal"),
    ]).unique(subset=["matched_mc_idx", "event_id", "seed"])

    out = clusters.join(n_true, on=["matched_mc_idx", "event_id", "seed"], how="left")
    return out.with_columns(
        (
            pl.col("best_match")
            / (pl.col("cluster_size") + pl.col("n_hits_signal")
               - pl.col("best_match"))
        ).fill_null(0.0).alias("iou")
    )


def assign_one_to_one(c: pl.DataFrame, iou_threshold=IOU_THRESHOLD) -> pl.DataFrame:
    """Mark the one candidate per true particle their assignment would keep.

    Their cost is the IoU, so candidates are ranked by IoU rather than by raw
    overlap, and any candidate whose IoU falls below the threshold is ineligible
    before the assignment runs. Every other candidate on the same particle is
    left unassigned, which is what turns a clone into one of their fakes.
    """
    eligible = pl.col("iou") >= iou_threshold
    return c.with_columns(
        (
            eligible
            & (
                pl.when(eligible).then(pl.col("iou")).otherwise(None)
                .rank("ordinal", descending=True)
                .over(["seed", "event_id", "matched_mc_idx"])
                == 1
            )
        ).fill_null(False).alias("assigned")
    )


def eligible_particles(tracks: pl.DataFrame) -> pl.DataFrame:
    """Their notebook's particle selection, as an optional overlay.

    make_cuts_ct keeps gen_status == 1, vertex R < 50 mm, theta in (10, 170)
    degrees, and more than 3 unique hits. delta_MC > 0.02, an isolation cut, has
    no analogue in our tables and is not applied.

    Read the overlay's effect on the fake rate as an upper bound rather than as
    their number. In their notebooks these cuts scope the *efficiency*
    denominator; the fake formula runs over the unrestricted frame. Applying
    them to the particle side here makes every candidate sitting on an excluded
    particle unmatchable, hence a fake, which is the most pessimistic reading.
    It is reported because it bounds how much of the gap could be selection.
    """
    return tracks.filter(
        (pl.col("gen_status") == 1)
        & (pl.col("vertex_r") < PHYS_VERTEX_R_MM)
        & (pl.col("theta_deg") > PHYS_THETA_DEG[0])
        & (pl.col("theta_deg") < PHYS_THETA_DEG[1])
        & (pl.col("n_hits_signal") > PHYS_MIN_UNIQUE_HITS)
    )


def ggtf_rates(clusters: pl.DataFrame, tracks: pl.DataFrame,
               cuts=CUTS, physics_cuts=False, n_events=N_EVENTS) -> dict:
    """Their fake rate, and ours, over one already-clustered variant."""
    if physics_cuts:
        keep = eligible_particles(tracks).select(
            ["mc_idx", "event_id", "seed"]).rename({"mc_idx": "matched_mc_idx"})
        clusters = clusters.join(
            keep.with_columns(pl.lit(True).alias("_ok")),
            on=["matched_mc_idx", "event_id", "seed"], how="left",
        )
        # A candidate whose majority particle is outside the selection can no
        # longer be assigned to it, so it lands in their fake bucket.
        clusters = clusters.with_columns(
            pl.when(pl.col("_ok").fill_null(False))
            .then(pl.col("iou")).otherwise(0.0).alias("iou")
        ).drop("_ok")

    # Prefer the assignment computed during clustering, which runs their real
    # linear_sum_assignment over the full IoU matrix. The emulation below is
    # only a fallback for caches written before that existed, and it is not
    # interchangeable: on the same 300 events it reports 73.1% where the exact
    # assignment reports 41.4%, because a greedy rule hands a particle to a
    # candidate the optimum would have spent elsewhere.
    exact = "ggtf_assigned" in clusters.columns and not physics_cuts
    c = (clusters.with_columns(pl.col("ggtf_assigned").alias("assigned"))
         if exact else assign_one_to_one(clusters))
    out = {
        "exact_assignment": exact,
        "n_cand": len(c),
        "cand_per_ev": len(c) / max(n_events, 1),
        "ours": 100 * c["is_fake_idea"].mean(),
        "ghost": 100 * (c["purity"] < 0.75).mean(),
        "median_iou": float(c["iou"].median() or 0.0),
        "frac_below_iou": 100 * float((c["iou"] < IOU_THRESHOLD).mean()),
    }
    for cut in cuts:
        k = c.filter(pl.col("cluster_size") > cut) if cut else c
        n_match = int(k["assigned"].sum())
        n_fake = len(k) - n_match
        out[f"ggtf>{cut}"] = 100 * n_fake / max(n_match, 1)
        # Same numerator over all candidates, i.e. what the number would be on
        # the conventional denominator.
        out[f"conv>{cut}"] = 100 * n_fake / max(len(k), 1)
        out[f"n>{cut}"] = len(k)
        out[f"matched>{cut}"] = n_match
    return out


def load_variant(path: str):
    cols = ["cluster_size", "matched_mc_idx", "best_match", "purity",
            "event_id", "seed", "is_fake_idea"]
    have = pl.scan_parquet(f"{BASE}/{path}/cache_clusters.parquet").collect_schema().names()
    if "ggtf_assigned" in have:
        cols.append("ggtf_assigned")
    clusters = pl.read_parquet(f"{BASE}/{path}/cache_clusters.parquet", columns=cols)
    tracks = pl.read_parquet(
        f"{BASE}/{path}/cache.parquet",
        columns=["mc_idx", "event_id", "seed", "n_hits_signal", "gen_status",
                 "vertex_r", "theta_deg"],
    )
    return add_iou(clusters, tracks), tracks


def rates(path: str, physics_cuts=False) -> dict:
    clusters, tracks = load_variant(path)
    return ggtf_rates(clusters, tracks, physics_cuts=physics_cuts)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--physics_cuts", action="store_true",
                    help="restrict the particle side to their notebook's "
                         "make_cuts_ct selection")
    args = ap.parse_args()

    print(__doc__)
    if args.physics_cuts:
        print(">>> particle side restricted to make_cuts_ct "
              "(gen_status==1, R<50mm, 10<theta<170 deg, >3 hits).")
        print(">>> Upper bound, not their number: their notebooks apply these "
              "cuts to the efficiency\n>>> denominator, while the fake formula "
              "runs over the unrestricted frame.\n")
    hdr = (f"{'variant':<26}{'cand/ev':>9}{'>10/ev':>8}{'ours':>8}{'ghost':>8}"
           f"{'<IoU':>7}{'GGTF>3':>9}{'GGTF>10':>9}{'conv>10':>9}")
    print("=" * len(hdr))
    print(hdr)
    print("-" * len(hdr))
    for name, path in VARIANTS:
        try:
            r = rates(path, physics_cuts=args.physics_cuts)
        except FileNotFoundError:
            print(f"{name:<26}{'(missing)':>9}")
            continue
        mark = "" if r["exact_assignment"] else " *"
        print(f"{name:<26}{r['cand_per_ev']:9.1f}{r['n>10'] / N_EVENTS:8.1f}"
              f"{r['ours']:7.1f}%{r['ghost']:7.1f}%{r['frac_below_iou']:6.1f}%"
              f"{r['ggtf>3']:8.1f}%{r['ggtf>10']:8.1f}%{r['conv>10']:8.1f}%{mark}")
    print()
    print("*  greedy emulation, no ggtf_assigned column in the cache. Overstates")
    print("   their rate badly (73% vs 41% on a 300-event check); re-run "
          "fcc_cache_parallel.py")
    print("   to get the exact assignment.")
    print("ours     our strict rate: ghost, or the majority particle fails the "
          "IDEA selection")
    print("ghost    no true particle contributes >= 75% of the candidate's hits")
    print(f"<IoU     candidates whose IoU with their own majority particle is "
          f"below {IOU_THRESHOLD}, so")
    print("         their assignment cannot match them however pure they are")
    print("GGTF>n   their formula: unassigned / assigned, candidates with > n hits")
    print("conv>n   same numerator over all candidates, the conventional denominator")
    print()
    print("Andrea's IDEA figure is 8% and the slide carries an N_hits > 10 cut, "
          "so GGTF>10 is the column to compare against.")


if __name__ == "__main__":
    main()
