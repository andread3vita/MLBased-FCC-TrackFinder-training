"""Why is a cluster counted as fake? Decomposition of the fake population.

The IDEA fake flag is `purity < 0.75  OR  majority particle not
IDEA-reconstructable`, and the second clause fires for clusters that are clean
reconstructions of particles which merely fail the fiducial selection (theta
outside 15-165 deg, <= 10 hits, secondary, neutral). This script separates the
two, so the fake rate can be quoted with and without the fiducial clause.

  PYTHONPATH=. python src/eval/fake_decompose.py
"""
import sys

sys.path.insert(0, "/home/marko.cechovic/cgatr-paper/literature/cgatr_fcc_pkg")
import numpy as np
import polars as pl

BASE = "/home/marko.cechovic/cgatr-paper/literature/cgatr_fcc_pkg/eval_results"
EDGES = np.logspace(np.log10(0.1), np.log10(10), 13)

CASES = [
    ("Loopers", "r3_loopers", 1000, [
        ("CIRCE",                      "merge_sweep/td0.10_mg0_at0"),
        ("CIRCE, >=4 hits",            "merge_sweep/td0.10_mg0_at0_mh4"),
        ("CIRCE + merging",            "merge_sweep/td0.10_mg0.15_at0"),
        ("CIRCE + merging, >=4 hits",  "merge_sweep/td0.10_mg0.15_at0_mh4"),
    ]),
    ("Zqq seeds 181-200", "consol_r3_nokeepall", 10000, [
        ("CIRCE (t_d=0.05, paper OP)", "merge_sweep/td0.05_mg0_at0"),
        ("CIRCE",                      "merge_sweep/td0.10_mg0_at0"),
        ("CIRCE, >=4 hits",            "merge_sweep/td0.10_mg0_at0_mh4"),
        ("CIRCE + merging, >=4 hits",  "merge_sweep/td0.10_mg0.15_at0_mh4"),
    ]),
]


def load(cfg, sub):
    """Cluster table + the fiducial properties of each cluster's majority
    particle (n_hits_total lives on the track table, so join it back in)."""
    c = pl.read_parquet(f"{BASE}/{cfg}/{sub}/cache_clusters.parquet")
    t = (pl.read_parquet(f"{BASE}/{cfg}/{sub}/cache.parquet",
                         columns=["mc_idx", "event_id", "seed", "n_hits_total",
                                  "theta_deg", "gen_status", "charge",
                                  "is_reconstructable_idea"])
         .rename({"mc_idx": "matched_mc_idx"}))
    return c.select(["matched_mc_idx", "event_id", "seed", "cluster_size",
                     "purity", "pt", "is_fake_idea"]).join(
        t, on=["matched_mc_idx", "event_id", "seed"], how="left")


def report(name, cfg, n_events, sub):
    d = load(cfg, sub)
    n = len(d)
    impure = d["purity"].to_numpy() < 0.75
    fake = d["is_fake_idea"].to_numpy()
    # clusters whose majority particle exists in the track table but fails the
    # fiducial selection; `reco` is null when the majority particle has no
    # track row at all (no pt in mc_signal -> treated as non-reconstructable)
    reco = d["is_reconstructable_idea"].fill_null(False).to_numpy()
    theta = d["theta_deg"].to_numpy()
    nh = d["n_hits_total"].fill_null(0).to_numpy()
    gs = d["gen_status"].fill_null(-1).to_numpy()
    q = d["charge"].fill_null(0.0).to_numpy()
    size = d["cluster_size"].to_numpy()

    clean_nonreco = fake & ~impure & ~reco
    out_theta = clean_nonreco & ((theta < 15.0) | (theta > 165.0))
    few_hits = clean_nonreco & ~out_theta & (nh <= 10)
    secondary = clean_nonreco & ~out_theta & ~few_hits & ~np.isin(gs, [0, 1])
    neutral = clean_nonreco & ~out_theta & ~few_hits & (np.abs(q) == 0)
    other = clean_nonreco & ~(out_theta | few_hits | secondary | neutral)

    print(f"\n{name}  [{sub}]")
    print(f"  candidates/event {n / n_events:8.1f}     fake rate "
          f"{100 * fake.mean():5.2f}%   fakes/event {fake.sum() / n_events:6.2f}")
    for lbl, m in [("impure (purity < 75%)", impure & fake),
                   ("clean, majority outside 15-165 deg", out_theta),
                   ("clean, majority has <= 10 hits", few_hits),
                   ("clean, majority is a secondary", secondary),
                   ("clean, majority is neutral", neutral),
                   ("clean, other/unmatched", other)]:
        if m.sum() == 0:
            continue
        print(f"    {lbl:<38} {100 * m.sum() / max(fake.sum(), 1):5.1f}% of fakes"
              f"  = {100 * m.mean():5.2f}% of candidates"
              f"  (median size {int(np.median(size[m])) if m.sum() else 0})")
    # alternative definitions
    purity_only = impure
    physics = impure | (clean_nonreco & (~np.isin(gs, [0, 1]) | (np.abs(q) == 0)))
    print(f"    fake rate, current definition            {100 * fake.mean():5.2f}%")
    print(f"    fake rate, drop the fiducial clause      {100 * physics.mean():5.2f}%"
          "   (still fake: impure, secondary, neutral)")
    print(f"    fake rate, purity-only (< 75%)           {100 * purity_only.mean():5.2f}%")
    return d, fake, impure, physics


def profile(name, cfg, n_events, sub):
    d, fake, impure, physics = report(name, cfg, n_events, sub)
    pt = d["pt"].to_numpy()
    ok = ~np.isnan(pt)
    print("    per-pT fake rate [%]:  bin      current   no-fiducial   purity-only")
    for a, b in zip(EDGES[:-1], EDGES[1:]):
        sel = ok & (pt >= a) & (pt < b)
        if sel.sum() < 40:
            continue
        print(f"      {np.sqrt(a * b):20.2f} {100 * fake[sel].mean():9.1f}"
              f" {100 * physics[sel].mean():13.1f} {100 * impure[sel].mean():13.1f}")


for label, cfg, n_events, subs in CASES:
    print("=" * 78)
    print(label)
    print("=" * 78)
    for name, sub in subs:
        if "Loopers" in label and "merging, >=4" in name:
            profile(name, cfg, n_events, sub)
        else:
            report(name, cfg, n_events, sub)
