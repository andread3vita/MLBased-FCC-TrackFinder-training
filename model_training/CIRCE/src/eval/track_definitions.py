"""One table per sample with every efficiency and fake convention side by side.

Efficiency conventions (on IDEA-reconstructable particles):
  def1  purity of the matched cluster > 75%                  <- our "match rate",
        GGTF efficiency definition 1, the CLD convention
  def2  purity > 50% AND track hit efficiency > 50%          <- GGTF efficiency
        definition 2, the one their IDEA plots use; penalises split tracks
        categories: good / split (pure but partial) / multiple (merged) / bad

Fake and clone conventions (on emitted track candidates):
  ghost      no true particle contributes >= 75% of the candidate's hits.
             This is the standard convention: LHCb ghost rate = fraction of
             reconstructed tracks not matched to an MC particle, matching at
             >= 70% of hits from one particle (LHCb-PUB-2021-005 eq. 3);
             Belle II fake rate = ghost+background candidates / all candidates,
             ghost = best MC candidate below 2/3 purity (basf2 track matching).
             Neither counts a clean track of a real-but-non-fiducial particle.
  ours       ghost OR the majority particle fails the IDEA selection
  clone      extra candidate for a particle another candidate already matches
             (LHCb: clone rate = N_clones / N_reconstructed;
              MC-clone rate = N_clones / N_matched)

  PYTHONPATH=. python src/eval/track_definitions.py
"""
import sys

sys.path.insert(0, "/home/marko.cechovic/cgatr-paper/literature/cgatr_fcc_pkg")
import polars as pl

BASE = "/home/marko.cechovic/cgatr-paper/literature/cgatr_fcc_pkg/eval_results"

CASES = [
    ("Loopers, 1000 seeds, keep-all truth", "r3_loopers", 1000, [
        ("CIRCE",                     "merge_sweep/td0.10_mg0_at0"),
        ("CIRCE, >=4 hits",           "merge_sweep/td0.10_mg0_at0_mh4"),
        ("CIRCE + merging",           "merge_sweep/td0.10_mg0.15_at0"),
        ("CIRCE + merging, >=4 hits", "merge_sweep/td0.10_mg0.15_at0_mh4"),
        ("oracle merge (ceiling)",    "merge_sweep/td0.10_mg0_at0_oracle_T0.75"),
    ]),
    ("Zqq seeds 181-200, 10000 events", "consol_r3_nokeepall", 10000, [
        ("CIRCE, t_d=0.2 draft OP",   "fcc_unmerged"),
        ("CIRCE, t_d=0.05",           "merge_sweep/td0.05_mg0_at0"),
        ("CIRCE",                     "merge_sweep/td0.10_mg0_at0"),
        ("CIRCE, >=4 hits",           "merge_sweep/td0.10_mg0_at0_mh4"),
        ("CIRCE + merging",           "merge_sweep/td0.10_mg0.15_at0"),
        ("CIRCE + merging, >=4 hits", "merge_sweep/td0.10_mg0.15_at0_mh4"),
        ("ADOPTED: merge 0.20, >=4",  "merge_sweep/td0.10_mg0.20_at0_mh4"),
    ]),
]

HDR = (f"{'variant':<26}{'def1':>7}{'def2':>7}{'split':>7}{'multi':>7}{'bad':>6}"
       f"{'cand/ev':>9}{'ghost':>7}{'ours':>7}{'clone':>7}{'MCclone':>9}")


def row(name, cfg, n_events, sub):
    t = (pl.read_parquet(f"{BASE}/{cfg}/{sub}/cache.parquet",
                         columns=["matched", "purity_of_match", "efficiency_per_hit",
                                  "is_reconstructable_idea"])
         .filter("is_reconstructable_idea"))
    p = t["purity_of_match"].to_numpy()
    e = t["efficiency_per_hit"].to_numpy()
    d1 = 100 * t["matched"].mean()
    good = 100 * ((p > 0.5) & (e > 0.5)).mean()
    split = 100 * ((p > 0.5) & (e <= 0.5)).mean()
    multi = 100 * ((p <= 0.5) & (e > 0.5)).mean()
    bad = 100 * ((p <= 0.5) & (e <= 0.5)).mean()

    c = pl.read_parquet(f"{BASE}/{cfg}/{sub}/cache_clusters.parquet",
                        columns=["matched_mc_idx", "event_id", "seed", "purity",
                                 "is_fake_idea"])
    cand = len(c) / n_events
    if "oracle" in sub:                    # truth-derived candidates: no fakes
        return (f"{name:<26}{d1:6.1f}%{good:6.1f}%{split:6.1f}%{multi:6.1f}%"
                f"{bad:5.1f}%{cand:9.1f}{'n/a':>7}{'n/a':>7}{'n/a':>7}{'n/a':>9}")
    ghost = 100 * (c["purity"] < 0.75).mean()
    ours = 100 * c["is_fake_idea"].mean()
    matched = c.filter(pl.col("purity") >= 0.75)
    n_uniq = len(matched.group_by(["matched_mc_idx", "event_id", "seed"]).len())
    n_clone = len(matched) - n_uniq
    clone = 100 * n_clone / len(c)
    mcclone = 100 * n_clone / max(len(matched), 1)
    return (f"{name:<26}{d1:6.1f}%{good:6.1f}%{split:6.1f}%{multi:6.1f}%{bad:5.1f}%"
            f"{cand:9.1f}{ghost:6.1f}%{ours:6.1f}%{clone:6.1f}%{mcclone:8.1f}%")


print(__doc__)
for label, cfg, n_events, subs in CASES:
    print("=" * len(HDR))
    print(label)
    print("=" * len(HDR))
    print(HDR)
    print("-" * len(HDR))
    for name, sub in subs:
        print(row(name, cfg, n_events, sub))
    print()
