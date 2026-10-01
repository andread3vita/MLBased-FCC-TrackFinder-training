"""Select operating points from a clustering sweep and build the full plot suite.

The sweep itself runs with --summary_only, which keeps metrics but discards the
per-hit assignment table that every plot needs. This script re-runs only the
selected points with the full cache written, then produces the complete FCC and
GGTF plot suite for each, plus cross-model overlays.

Selection is by explicit named policy rather than a single hidden "best", because
efficiency and fake rate trade against each other and the honest deliverable is
the trade-off. Every selected point is reported against the reference point that
the epoch-16/20/24 milestones used.

All selection happens on the same 500 validation events that the sweep scanned,
so these points are chosen, not confirmed; promotion still requires the frozen
independent-data evaluation.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

STAGES = ("density", "density_refinement", "full_grid", "refinement")


def _slug(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def _metric(summary: dict, *path, default=None):
    node = summary
    for key in path:
        if not isinstance(node, dict) or key not in node:
            return default
        node = node[key]
    return node


def _row(summary: dict) -> dict:
    return {
        "clusterer": summary["clusterer"],
        "tbeta": summary["tbeta"],
        "td": summary["td"],
        "min_cluster_hits": summary["min_cluster_hits"],
        "merge_td": summary["merge_td"],
        "attach_td": summary["attach_td"],
        "helix_tol": summary.get("helix_tol", 0.0),
        "def1": _metric(summary, "idea", "match_rate"),
        "def2": _metric(summary, "idea", "def2"),
        "hit_eff": _metric(summary, "idea", "efficiency"),
        "low_pt_def1": _metric(summary, "low_pt_idea", "0.1-0.2", "match_rate"),
        "low_pt_def2": _metric(summary, "low_pt_idea", "0.1-0.2", "def2"),
        "ggtf10": _metric(summary, "ggtf_fake_rate_gt10"),
        "candidates_per_event": summary.get("candidates_per_event"),
        "n_matched_gt10": summary.get("ggtf_n_matched_gt10", 0),
        "promotable": summary.get("promotable_operating_point", True),
    }


def _config_key(row: dict) -> tuple:
    return (
        row["clusterer"], row["tbeta"], row["td"], row["min_cluster_hits"],
        row["merge_td"], row["attach_td"], row["helix_tol"],
    )


def _load_points(sweep_root: Path) -> list[dict]:
    """Load every sweep point, de-duplicating configs run in multiple stages."""
    seen: dict[tuple, dict] = {}
    for stage in STAGES:
        for path in sorted((sweep_root / stage).glob("*/*/summary.json")):
            summary = json.loads(path.read_text())
            row = _row(summary)
            row["stage"] = stage
            row["path"] = str(path.parent)
            key = _config_key(row)
            seen.setdefault(key, row)
    return list(seen.values())


def _eligible(rows: list[dict]) -> list[dict]:
    return [
        r for r in rows
        if r["n_matched_gt10"] > 0 and r["promotable"]
        and r["def1"] is not None and r["ggtf10"] is not None
    ]


def _pareto(rows: list[dict], gain: str) -> list[dict]:
    """Non-dominated points maximising `gain` while minimising the fake rate."""
    frontier = []
    for cand in rows:
        if cand[gain] is None:
            continue
        dominated = any(
            other[gain] is not None
            and other[gain] >= cand[gain] and other["ggtf10"] <= cand["ggtf10"]
            and (other[gain] > cand[gain] or other["ggtf10"] < cand["ggtf10"])
            for other in rows
        )
        if not dominated:
            frontier.append(cand)
    return sorted(frontier, key=lambda r: r["ggtf10"])


def _select(rows: list[dict], reference: dict) -> dict[str, dict]:
    """Named operating points spanning the efficiency/fake-rate trade-off."""
    ref_fakes = reference["ggtf10"]
    ref_def1 = reference["def1"]
    selected: dict[str, dict] = {}

    def best(candidates, key):
        candidates = [c for c in candidates if c[key] is not None]
        return max(candidates, key=lambda r: r[key]) if candidates else None

    selected["max_def2"] = best(rows, "def2")
    selected["max_def1"] = best(rows, "def1")
    selected["max_low_pt_def1"] = best(rows, "low_pt_def1")
    # Constrained picks: improve efficiency without paying more fakes than the
    # reference, and cut fakes without losing reference-level efficiency.
    selected["best_def2_at_reference_fakes"] = best(
        [r for r in rows if r["ggtf10"] <= ref_fakes], "def2"
    )
    at_ref_def1 = [r for r in rows if r["def1"] >= ref_def1]
    selected["min_fakes_at_reference_def1"] = (
        min(at_ref_def1, key=lambda r: r["ggtf10"]) if at_ref_def1 else None
    )
    return {name: row for name, row in selected.items() if row is not None}


def _run(command: list[str], cwd: Path, log: Path, env: dict) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w") as handle:
        result = subprocess.run(
            command, cwd=cwd, env=env, stdout=handle,
            stderr=subprocess.STDOUT, check=False,
        )
    if result.returncode:
        tail = log.read_text(errors="replace").splitlines()[-15:]
        print("\n".join(tail), file=sys.stderr)
        raise RuntimeError(f"command failed: {' '.join(command[:4])}... see {log}")


def _cluster_full(args, name: str, row: dict, env: dict) -> Path:
    """Re-run one point writing the full per-hit cache the plots require."""
    out = args.output / name
    cache = out / "cache.parquet"
    if cache.is_file():
        print(f"[{name}] full cache already present", flush=True)
        return out
    out.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable, "-u", str(args.package / "src/eval/fcc_cache_parallel.py"),
        "--cache_path", str(args.cache),
        "--mc_signal", str(args.mc_signal),
        "--embed_dim", str(args.embed_dim),
        "--clusterer", row["clusterer"],
        "--tbeta", str(row["tbeta"]),
        "--td", str(row["td"]),
        "--beta_mode", "sigmoid",
        "--min_cluster_hits", str(row["min_cluster_hits"]),
        "--min_target_hits", "3",
        "--workers", str(args.workers),
        "--merge_td", str(row["merge_td"]),
        "--attach_td", str(row["attach_td"]),
        "--helix_tol", str(row["helix_tol"]),
        "--helix_min_hits", "10",
        "--output_dir", str(out),
        "--tag", f"{args.label} {name}",
    ]
    started = time.time()
    _run(command, args.package, out / "cluster.log", env)
    print(f"[{name}] clustered in {time.time() - started:.1f}s", flush=True)
    return out


def _plot_suite(args, name: str, point_dir: Path, env: dict) -> None:
    """Full FCC suite, GGTF principal panels, and denominator composition."""
    _run([
        sys.executable, "-u", str(args.package / "src/eval/plot_fcc_metrics.py"),
        "--cache_dir", str(point_dir),
        "--output_dir", str(point_dir / "plots"),
        "--tag", f"{args.label} {name}",
    ], args.package, point_dir / "plot_fcc.log", env)
    _run([
        sys.executable, "-u", str(args.package / "src/eval/plot_ggtf_preliminary.py"),
        "--cache-dir", str(point_dir),
        "--output-dir", str(point_dir / "ggtf_plots"),
        "--tag", f"{args.label} {name}",
    ], args.package, point_dir / "plot_ggtf.log", env)
    if args.data_dir is not None:
        _run([
            sys.executable, "-u",
            str(args.package / "src/eval/measure_denominator_composition.py"),
            "--cache-parquet", str(point_dir / "cache.parquet"),
            "--data-dir", str(args.data_dir),
            "--output", str(point_dir / "denominator_composition"),
            "--tag", f"{args.label} {name}",
        ], args.package, point_dir / "denominator.log", env)
    print(f"[{name}] plot suite complete", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--sweep-root", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--mc-signal", type=Path, required=True)
    parser.add_argument("--reference-summary", type=Path, required=True)
    parser.add_argument("--reference-cache", type=Path, default=None,
                        help="cache.parquet of the reference point, for overlays")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--label", default="CGA epoch 24")
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--embed-dim", type=int, default=4)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument(
        "--overlay-cache", action="append", default=[], metavar="LABEL=PATH",
        help="extra cache.parquet to include in the overlay, e.g. the PGA arm",
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env.update({
        "PYTHONPATH": str(args.package),
        "CUDA_VISIBLE_DEVICES": "",
        "FIX_PARTICLE_ZERO": "1",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "POLARS_MAX_THREADS": "4",
    })

    reference = _row(json.loads(args.reference_summary.read_text()))
    rows = _eligible(_load_points(args.sweep_root))
    if not rows:
        raise RuntimeError(f"no eligible sweep points under {args.sweep_root}")
    print(f"loaded {len(rows)} eligible unique sweep points", flush=True)

    selected = _select(rows, reference)
    # Deduplicate: several policies often land on the same configuration.
    unique: dict[tuple, str] = {}
    for name, row in selected.items():
        unique.setdefault(_config_key(row), name)
    aliases: dict[str, list[str]] = {}
    for name, row in selected.items():
        aliases.setdefault(unique[_config_key(row)], []).append(name)

    selection = {
        "scope": (
            "Operating points chosen on the same 500 seed-181 validation events "
            "the sweep scanned. Chosen, not confirmed."
        ),
        "reference": reference,
        "n_eligible_points": len(rows),
        "selected": selected,
        "aliases": aliases,
        "pareto": {
            gain: _pareto(rows, gain)
            for gain in ("def1", "def2", "low_pt_def1", "low_pt_def2")
        },
    }
    (args.output / "selection.json").write_text(
        json.dumps(selection, indent=2) + "\n"
    )

    overlays: list[str] = list(args.overlay_cache)
    if args.reference_cache is not None and args.reference_cache.is_file():
        overlays.insert(0, f"reference (self-seed td0.2)={args.reference_cache}")

    for key, name in unique.items():
        row = selected[name]
        point_dir = _cluster_full(args, name, row, env)
        _plot_suite(args, name, point_dir, env)
        overlays.append(f"{name}={point_dir / 'cache.parquet'}")

    overlay_command = [
        sys.executable, "-u",
        str(args.package / "src/eval/plot_checkpoint_milestones.py"),
    ]
    for spec in overlays:
        overlay_command += ["--cache", spec]
    overlay_command += [
        "--output-dir", str(args.output / "overlay"),
        "--filename", "eff_vs_pt_operating_points",
        "--title", f"{args.label}: operating-point trade-off and PGA baseline",
        # Each curve has its own operating point here, so the caption must not
        # claim a single shared one the way the milestone plot does. The sample
        # itself is derived from the caches by the milestone script (M86), so it
        # must not be restated here either.
        "--subtitle", (
            "each curve at its own operating point, see selection.json · "
            "PGA arm shown at the reference point"
        ),
    ]
    _run(overlay_command, args.package, args.output / "overlay.log", env)

    print("\n=== selected operating points ===", flush=True)
    header = (
        f"{'policy':<34}{'clusterer':<18}{'td':>7}{'mh':>4}{'atch':>7}"
        f"{'def1':>8}{'def2':>8}{'lowpT1':>8}{'ggtf10':>8}"
    )
    print(header)
    print(
        f"{'reference':<34}{reference['clusterer']:<18}{reference['td']:>7g}"
        f"{reference['min_cluster_hits']:>4d}{reference['attach_td']:>7g}"
        f"{reference['def1'] * 100:>8.2f}{reference['def2'] * 100:>8.2f}"
        f"{reference['low_pt_def1'] * 100:>8.2f}{reference['ggtf10'] * 100:>8.2f}"
    )
    for key, name in unique.items():
        row = selected[name]
        label = "+".join(aliases[name])
        print(
            f"{label[:33]:<34}{row['clusterer']:<18}{row['td']:>7g}"
            f"{row['min_cluster_hits']:>4d}{row['attach_td']:>7g}"
            f"{row['def1'] * 100:>8.2f}{row['def2'] * 100:>8.2f}"
            f"{row['low_pt_def1'] * 100:>8.2f}{row['ggtf10'] * 100:>8.2f}"
        )
    print(f"\nartifacts under {args.output}", flush=True)


if __name__ == "__main__":
    main()
