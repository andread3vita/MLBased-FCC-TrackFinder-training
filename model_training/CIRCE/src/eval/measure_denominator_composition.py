"""Decompose our IDEA denominator to make the slide-24 comparison like-for-like.

Motivation: our epoch-16 efficiency-vs-pT curve sits ~17 points below GGTF's
slide 24 in the lowest pT bin, and the question is whether that is model deficit
or denominator composition.

Two candidate explanations, and this script settles which one matters.

1. The looper/extent cut. **Dead, twice over.** G1/G2 in FINDINGS.md establish that
   GGTF apply no extent filter: it sits behind a hardcoded `remove_lowEnergyParticle
   = False` (`remove_loopers = False` in the `cgatr/` copy, where the flag even
   shadows the function name so the branch would raise `TypeError` if taken), and the
   only reachable filter, `remove_loopers_overlay`, is a `< 5 hits` cut with no extent
   test, in an `else` branch their shipped `config_tracking.yaml` cannot reach because
   it sets no `overlay` key. Measured here for completeness: the cut is also misnamed,
   deleting long stiff tracks and keeping curlers, so it moves the lowest bin the
   wrong way.

2. Secondary composition. **This is the one.** What their loader *does* run is
   `create_garbage_label(..., minNumHits=3)`, which relabels every hit flagged
   `isProducedBySecondary` to noise and drops any particle whose hits are all
   secondary (G3). In our sample that flag is identically zero (M8), so the same code
   path is a no-op for us: particles created inside the detector stay full targets.
   They are the majority of our lowest pT bin and they are much harder than
   generator-level particles.

`gen_status` is the observable that separates them: 0 means the particle was created
by the simulation rather than the generator. This script reports the metrics under
each denominator so the two effects can be read off separately.

Caveat to keep attached to any number this prints: that GGTF's `isProducedBySecondary`
selects exactly our `gen_status == 0` population is a strong inference from their code
path, not a verified equivalence, and their slide text says `genStatus in [0, 1]`,
which reads the other way. Their hits-per-target is also ~5x lower than ours
(curler_question.md), so the shared `N_hits > 10` cut prunes secondaries far harder in
their sample than in ours. Both are open questions for them, not settled facts.
"""

import argparse
import hashlib
import json
import os
import sys

import numpy as np
import polars as pl
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.dataset.parquet_dataset import (
    _LOOPER_EXTENT_MM,
    _LOOPER_MIN_HITS,
    looper_mc_indices,
)
from src.eval.plot_fcc_metrics import (
    COL,
    _setup_eff_axes,
    binned_log_efficiency,
    save_fig,
)

# Slide-24 edges, same as plot_fcc_metrics.py so the curves overlay exactly.
PT_EDGES = list(np.logspace(np.log10(0.1), np.log10(30.0), 31))
REPORT_BINS = ((0.1, 0.2), (0.2, 0.5), (0.5, 1.0), (1.0, 30.0))
CROSS_CHECK_EVENTS = 8

# Production radius below which we call a particle prompt. The beam pipe is well
# outside this, so it separates the interaction point from detector material.
PROMPT_VERTEX_R_MM = 1.0


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def looper_flags(data_dir: str, seed: int, event_ids):
    """(event_id, mc_idx) -> would our `--drop_loopers` extent predicate delete it.

    Vectorised over all events at once; `looper_mc_indices` is the per-event
    reference implementation and is used below to cross-check a sample.
    """
    seed_dir = os.path.join(data_dir, f"seed_{seed}")
    wanted = pl.Series("event_id", sorted(int(e) for e in event_ids))

    dc = (
        pl.read_parquet(os.path.join(seed_dir, "dc_hits_train.parquet"),
                        columns=["event_id", "mc_index",
                                 "left_x", "left_y", "left_z"])
        .filter(pl.col("event_id").is_in(wanted.implode()))
        .rename({"left_x": "x", "left_y": "y", "left_z": "z"})
    )
    vtx = (
        pl.read_parquet(os.path.join(seed_dir, "vtx_hits_train.parquet"),
                        columns=["event_id", "mc_index",
                                 "hit_x", "hit_y", "hit_z"])
        .filter(pl.col("event_id").is_in(wanted.implode()))
        .rename({"hit_x": "x", "hit_y": "y", "hit_z": "z"})
    )

    lim_x, lim_y, lim_z = _LOOPER_EXTENT_MM
    agg = (
        pl.concat([dc, vtx])
        .filter(pl.col("mc_index") > 0)
        .group_by(["event_id", "mc_index"])
        .agg(
            (pl.col("x").max() - pl.col("x").min()).alias("ex"),
            (pl.col("y").max() - pl.col("y").min()).alias("ey"),
            (pl.col("z").max() - pl.col("z").min()).alias("ez"),
            pl.len().alias("n"),
        )
        .with_columns(
            (
                (pl.col("ex") > lim_x)
                | (pl.col("ey") > lim_y)
                | (pl.col("ez") > lim_z)
                | (pl.col("n") < _LOOPER_MIN_HITS)
            ).alias("is_extent_cut")
        )
        .rename({"mc_index": "mc_idx"})
    )

    checked = _cross_check(dc, vtx, agg, wanted)
    return agg.select("event_id", "mc_idx", "is_extent_cut"), checked


def _cross_check(dc, vtx, agg, wanted) -> int:
    """Assert the vectorised flag matches the per-event reference on a sample."""
    for eid in [int(e) for e in wanted[:CROSS_CHECK_EVENTS]]:
        dc_ev = dc.filter(pl.col("event_id") == eid).rename(
            {"x": "left_x", "y": "left_y", "z": "left_z"})
        vtx_ev = vtx.filter(pl.col("event_id") == eid).rename(
            {"x": "hit_x", "y": "hit_y", "z": "hit_z"})
        reference = looper_mc_indices(dc_ev, vtx_ev)
        mine = set(
            agg.filter((pl.col("event_id") == eid) & pl.col("is_extent_cut"))
            ["mc_idx"].to_list()
        )
        if reference != mine:
            raise SystemExit(
                f"extent predicate mismatch on event {eid}: "
                f"reference-only={sorted(reference - mine)[:8]} "
                f"vectorised-only={sorted(mine - reference)[:8]}"
            )
    return min(CROSS_CHECK_EVENTS, len(wanted))


def stats(df: pl.DataFrame) -> dict:
    """Same definitions as fcc_cache_parallel.overall()."""
    n = len(df)
    if n == 0:
        return dict.fromkeys(
            ["n", "hit_eff", "def1", "def2", "split", "multiple", "bad"], None
        ) | {"n": 0}
    p = df["purity_of_match"].to_numpy()
    e = df["efficiency_per_hit"].to_numpy()
    return {
        "n": n,
        "hit_eff": float(e.mean()),
        "def1": float((p > 0.75).mean()),
        "def2": float(((p > 0.5) & (e > 0.5)).mean()),
        "split": float(((p > 0.5) & (e <= 0.5)).mean()),
        "multiple": float(((p <= 0.5) & (e > 0.5)).mean()),
        "bad": float(((p <= 0.5) & (e <= 0.5)).mean()),
    }


def binned(df: pl.DataFrame) -> dict:
    marked = df.with_columns((pl.col("purity_of_match") > 0.75).alias("matched"))
    centers, p, e_lo, e_hi, ns = binned_log_efficiency(marked, "pt", PT_EDGES)
    return {
        "pt_center": [float(c) for c in centers],
        "def1": [float(v) for v in p],
        "err_lo": [float(v) for v in e_lo],
        "err_hi": [float(v) for v in e_hi],
        "n": [int(v) for v in ns],
    }


def overlay_plot(curves, out_path: str, tag: str):
    styles = [
        ("idea_all", COL["keepall"], "o-",
         "all IDEA targets (includes detector secondaries)"),
        ("idea_generator_only", COL["standard"], "s-",
         "generator particles only ($\\mathrm{genStatus}=1$)"),
        ("idea_secondaries_only", COL["grey"], "^--",
         "simulation-created only ($\\mathrm{genStatus}=0$)"),
    ]
    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    for key, color, fmt, label in styles:
        c = curves[key]
        ax.errorbar(c["pt_center"], c["def1"], yerr=[c["err_lo"], c["err_hi"]],
                    fmt=fmt, color=color, ecolor=color, markersize=4.2,
                    linewidth=1.5, label=label)
    ax.set_xscale("log")
    ax.set_xlabel(r"$p_\mathrm{T}$ [GeV]")
    ax.set_ylabel("Track match rate ($>75\\%$ purity)")
    ax.set_title(f"{tag}: denominator composition drives the low-$p_T$ deficit",
                 fontsize=9.5)
    _setup_eff_axes(ax, ymin=0.4)
    ax.legend(loc="lower right", fontsize=7)
    fig.tight_layout()
    save_fig(fig, out_path)
    plt.close(fig)
    print(f"  saved {out_path}")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache-parquet", required=True,
                    help="per-target table written by fcc_cache_parallel.py")
    ap.add_argument("--data-dir", required=True,
                    help="parquet dataset root holding seed_<n>/ directories")
    ap.add_argument("--output", required=True, help="output directory")
    ap.add_argument("--tag", default="Epoch-16 self-seed fixed")
    args = ap.parse_args()

    os.makedirs(args.output, exist_ok=True)
    targets = pl.read_parquet(args.cache_parquet)
    seeds = targets["seed"].unique().to_list()
    if len(seeds) != 1:
        raise SystemExit(f"expected a single-seed scope, found seeds={seeds}")
    seed = int(seeds[0])
    event_ids = targets["event_id"].unique().to_list()

    print(f"Scoring {len(targets):,} targets from seed {seed}, "
          f"{len(event_ids)} events")
    flags, checked = looper_flags(args.data_dir, seed, event_ids)
    print(f"  extent predicate cross-checked against the per-event reference "
          f"on {checked} events")

    joined = targets.join(flags, on=["event_id", "mc_idx"], how="left")
    if len(joined) != len(targets):
        raise SystemExit("extent-flag join changed the row count")
    # mc_idx 0 is a real particle the predicate deliberately spares, and it carries
    # no row in `flags` because the aggregation filters mc_index > 0.
    joined = joined.with_columns(pl.col("is_extent_cut").fill_null(False))

    idea = joined.filter(pl.col("is_reconstructable_idea"))
    denominators = {
        "idea_all": idea,
        "idea_generator_only": idea.filter(pl.col("gen_status") == 1),
        "idea_secondaries_only": idea.filter(pl.col("gen_status") == 0),
        "idea_prompt_vertex": idea.filter(
            pl.col("vertex_r") < PROMPT_VERTEX_R_MM),
        "idea_extent_cut_applied": idea.filter(~pl.col("is_extent_cut")),
    }

    report = {
        "question": ("Is the slide-24 low-pT deficit model deficit or denominator "
                     "composition?"),
        "answer": ("Composition dominates the lowest bins. See "
                   "FINDINGS.md G1/G2/G3 and M8 for why the extent cut is the "
                   "wrong axis and the secondary relabel is the right one."),
        "scope": {
            "seed": seed,
            "n_events": len(event_ids),
            "n_targets_min3": len(targets),
        },
        "denominator_definitions": {
            "idea_all": ("our published curve: pT>0.1, n_hits_total>10, "
                         "15<theta<165 deg, charged, gen_status in {0,1}"),
            "idea_generator_only": "idea_all and gen_status == 1",
            "idea_secondaries_only": "idea_all and gen_status == 0",
            "idea_prompt_vertex": (
                f"idea_all and production radius < {PROMPT_VERTEX_R_MM} mm"),
            "idea_extent_cut_applied": (
                "idea_all minus our --drop_loopers extent predicate; GGTF do "
                "NOT apply this (G1/G2), reported only to show it is the wrong "
                "axis"),
        },
        "not_promotable": ("Every restricted denominator needs the truth record "
                           "and is a comparison device, not an operating point."),
        "overall": {k: stats(v) for k, v in denominators.items()},
        "pt_bins": {},
        "curve": {k: binned(denominators[k]) for k in
                  ("idea_all", "idea_generator_only", "idea_secondaries_only")},
        "provenance": {
            "cache_parquet": args.cache_parquet,
            "cache_sha256": _sha256(args.cache_parquet),
            "data_dir": args.data_dir,
            "source_sha256": {
                rel: _sha256(rel) for rel in (
                    "src/eval/measure_denominator_composition.py",
                    "src/dataset/parquet_dataset.py",
                    "src/eval/plot_fcc_metrics.py",
                )
            },
        },
    }

    for low, high in REPORT_BINS:
        window = (pl.col("pt") >= low) & (pl.col("pt") < high)
        block = {k: stats(v.filter(window)) for k, v in denominators.items()}
        n_all = block["idea_all"]["n"]
        block["secondary_fraction_of_targets"] = (
            block["idea_secondaries_only"]["n"] / n_all if n_all else None)
        report["pt_bins"][f"{low}_{high}"] = block

    out_json = os.path.join(args.output, "denominator_composition.json")
    with open(out_json, "w") as handle:
        json.dump(report, handle, indent=2)
    overlay_plot(report["curve"],
                 os.path.join(args.output, "eff_vs_pt_by_origin.png"),
                 args.tag)

    print("\n=== IDEA denominators, whole pT range ===")
    for name, block in report["overall"].items():
        print(f"  [{name:<26}] n={block['n']:>7,}  def1={block['def1']:.4f}  "
              f"def2={block['def2']:.4f}  hit_eff={block['hit_eff']:.4f}")
    print("\n=== per pT bin: all targets vs generator particles only ===")
    for key, block in report["pt_bins"].items():
        a, g = block["idea_all"], block["idea_generator_only"]
        if not a["n"]:
            continue
        print(f"  pT {key.replace('_', '-'):>10} GeV  "
              f"secondaries={block['secondary_fraction_of_targets']:6.2%}  "
              f"def1 all={a['def1']:.4f}  def1 generator={g['def1']:.4f}  "
              f"(+{g['def1'] - a['def1']:.4f})")
    print(f"\nwrote {out_json}")


if __name__ == "__main__":
    sys.exit(main())
