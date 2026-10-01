"""Write the locked final-run physics slices as JSON and Markdown.

The report consumes finished ``fcc_cache_parallel`` directories. It keeps the
operating point fixed and evaluates efficiency versus low pT, truth-track
proximity, production radius, and transverse impact parameter. Candidate-level
fake and clone metrics are copied from each run's summary.
"""

from __future__ import annotations

import argparse
import json
import math
import os

import numpy as np
import polars as pl

from src.eval.report_displacement import add_d0


PT_EDGES = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, math.inf]
PROXIMITY_EDGES = [0.0, 0.02, 0.05, 0.1, 0.2, 0.5, math.inf]
VERTEX_R_EDGES = [0.0, 0.1, 1.0, 10.0, 50.0, 350.0, 1000.0, math.inf]
D0_EDGES = [0.0, 0.05, 0.2, 1.0, 5.0, 20.0, 100.0, math.inf]


def _edge_label(low: float, high: float) -> str:
    if math.isinf(high):
        return f">={low:g}"
    return f"{low:g}-{high:g}"


def nearest_truth_distance(df: pl.DataFrame) -> np.ndarray:
    """Nearest other reconstructable truth particle in wrapped (eta, phi)."""
    try:
        from scipy.spatial import cKDTree
    except ImportError as exc:  # pragma: no cover - production env has scipy
        raise RuntimeError("scipy is required for the proximity report") from exc

    out = np.full(len(df), np.nan, dtype=np.float64)
    indexed = df.with_row_index("_row")
    for event in indexed.partition_by(["seed", "event_id"], maintain_order=False):
        n = len(event)
        if n < 2:
            continue
        eta = event["eta"].to_numpy()
        phi = event["phi"].to_numpy()
        points = np.column_stack([eta, phi])
        tiled = np.concatenate([
            np.column_stack([eta, phi - 2 * np.pi]),
            points,
            np.column_stack([eta, phi + 2 * np.pi]),
        ])
        source = np.tile(np.arange(n), 3)
        tree = cKDTree(tiled)
        distances, indices = tree.query(points, k=min(6, len(tiled)))
        if distances.ndim == 1:
            distances = distances[:, None]
            indices = indices[:, None]
        nearest = np.full(n, np.nan)
        for i in range(n):
            valid = source[indices[i]] != i
            if valid.any():
                nearest[i] = distances[i][np.flatnonzero(valid)[0]]
        out[event["_row"].to_numpy()] = nearest
    return out


def _rates(frame: pl.DataFrame) -> dict:
    n = len(frame)
    if n == 0:
        return {"n": 0}
    purity = frame["purity_of_match"].to_numpy()
    efficiency = frame["efficiency_per_hit"].to_numpy()
    return {
        "n": n,
        "match_rate": float((purity > 0.75).mean()),
        "strict50": float(((purity > 0.75) & (efficiency > 0.5)).mean()),
        "def2": float(((purity > 0.5) & (efficiency > 0.5)).mean()),
        "mean_hit_efficiency": float(efficiency.mean()),
    }


def _binned(frame: pl.DataFrame, column: str, edges: list[float]) -> list[dict]:
    rows = []
    for low, high in zip(edges[:-1], edges[1:]):
        selected = frame.filter(
            (pl.col(column) >= low) & (pl.col(column) < high)
        )
        rows.append({
            "bin": _edge_label(low, high),
            "low": low,
            "high": None if math.isinf(high) else high,
            **_rates(selected),
        })
    return rows


def analyze(label: str, directory: str) -> dict:
    summary_path = os.path.join(directory, "summary.json")
    cache_path = os.path.join(directory, "cache.parquet")
    if not os.path.isfile(summary_path) or not os.path.isfile(cache_path):
        raise FileNotFoundError(
            f"{label}: expected summary.json and cache.parquet under {directory}"
        )
    with open(summary_path) as handle:
        summary = json.load(handle)
    tracks = pl.read_parquet(cache_path)
    required = {
        "seed", "event_id", "pt", "eta", "phi", "vx", "vy", "vertex_r", "charge",
        "n_hits_signal", "purity_of_match", "efficiency_per_hit",
        "is_reconstructable_idea",
    }
    missing = required - set(tracks.columns)
    if missing:
        raise ValueError(f"{label}: cache is missing columns {sorted(missing)}")

    reconstructable = tracks.filter(pl.col("is_reconstructable_idea"))
    reconstructable = add_d0(reconstructable).with_columns(
        pl.Series(
            "nearest_truth_delta_eta_phi",
            nearest_truth_distance(reconstructable),
        )
    )
    displacement = reconstructable.filter(pl.col("n_hits_signal") >= 20)
    matched = max(int(summary.get("ggtf_n_matched_gt10", 0)), 1)
    candidate = {
        "candidates_per_event": summary.get("candidates_per_event"),
        "ggtf_fake_rate_gt10": summary.get("ggtf_fake_rate_gt10"),
        "clone_per_matched_gt10": (
            int(summary.get("ggtf_n_fake_clone_gt10", 0)) / matched
        ),
        "spurious_per_matched_gt10": (
            int(summary.get("ggtf_n_fake_spurious_gt10", 0)) / matched
        ),
    }
    return {
        "label": label,
        "directory": os.path.abspath(directory),
        "n_events": int(summary.get("n_events", 0)),
        "overall": _rates(reconstructable),
        "candidate": candidate,
        "low_pt": _binned(reconstructable, "pt", PT_EDGES),
        "truth_proximity_delta_eta_phi": _binned(
            reconstructable.drop_nulls("nearest_truth_delta_eta_phi"),
            "nearest_truth_delta_eta_phi",
            PROXIMITY_EDGES,
        ),
        "production_radius_mm_min20_hits": _binned(
            displacement, "vertex_r", VERTEX_R_EDGES
        ),
        "impact_parameter_abs_d0_mm_min20_hits": _binned(
            displacement, "d0", D0_EDGES
        ),
    }


def _fmt_rate(value) -> str:
    return "--" if value is None else f"{100 * float(value):.2f}%"


def markdown(payload: dict) -> str:
    lines = [
        "# Final locked-CGA evaluation slices",
        "",
        "The checkpoint and operating point were selected on no-keep-all "
        "validation before these holdouts were opened.",
        "",
    ]
    for run in payload["runs"]:
        lines.extend([
            f"## {run['label']}",
            "",
            f"Events: {run['n_events']:,}; IDEA-reconstructable targets: "
            f"{run['overall']['n']:,}.",
            "",
            "| overall | match | strict50 | def2 | hit efficiency | GGTF>10 | clone | spurious |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
            "| locked raw | "
            + " | ".join([
                _fmt_rate(run["overall"].get("match_rate")),
                _fmt_rate(run["overall"].get("strict50")),
                _fmt_rate(run["overall"].get("def2")),
                _fmt_rate(run["overall"].get("mean_hit_efficiency")),
                _fmt_rate(run["candidate"].get("ggtf_fake_rate_gt10")),
                _fmt_rate(run["candidate"].get("clone_per_matched_gt10")),
                _fmt_rate(run["candidate"].get("spurious_per_matched_gt10")),
            ])
            + " |",
            "",
        ])
        for title, key in (
            ("Low-pT performance", "low_pt"),
            ("Truth-track proximity in wrapped eta-phi", "truth_proximity_delta_eta_phi"),
            (
                "Production radius (length-controlled: at least 20 signal hits)",
                "production_radius_mm_min20_hits",
            ),
            (
                "Transverse impact parameter (length-controlled: at least 20 signal hits)",
                "impact_parameter_abs_d0_mm_min20_hits",
            ),
        ):
            lines.extend([
                f"### {title}",
                "",
                "| bin | targets | match | strict50 | def2 | hit efficiency |",
                "|---|---:|---:|---:|---:|---:|",
            ])
            for row in run[key]:
                lines.append(
                    f"| {row['bin']} | {row['n']:,} | "
                    f"{_fmt_rate(row.get('match_rate'))} | "
                    f"{_fmt_rate(row.get('strict50'))} | "
                    f"{_fmt_rate(row.get('def2'))} | "
                    f"{_fmt_rate(row.get('mean_hit_efficiency'))} |"
                )
            lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run", action="append", required=True, metavar="LABEL=DIR",
        help="Finished locked evaluation directory; repeat for multiple samples",
    )
    parser.add_argument("--out", required=True, help="Output prefix without suffix")
    args = parser.parse_args()

    runs = []
    for item in args.run:
        if "=" not in item:
            raise SystemExit(f"--run must be LABEL=DIR, got {item!r}")
        label, directory = item.split("=", 1)
        runs.append(analyze(label, directory))

    payload = {
        "definitions": {
            "match_rate": "best-cluster purity > 75%",
            "strict50": "purity > 75% and hit efficiency > 50%",
            "def2": "purity > 50% and hit efficiency > 50%",
            "proximity": "nearest other reconstructable truth particle in wrapped (eta,phi)",
        },
        "runs": runs,
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out + ".json", "w") as handle:
        json.dump(payload, handle, indent=2)
    with open(args.out + ".md", "w") as handle:
        handle.write(markdown(payload) + "\n")
    print(f"wrote {args.out}.json and {args.out}.md")


if __name__ == "__main__":
    main()
