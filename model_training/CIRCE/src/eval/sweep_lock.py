"""Sweep a small (tbeta, td) grid via `ggtf_full_eval.py` and lock one operating point per arm.

Purpose-built for the two-arm (CGA vs PGA) Sunday comparison rather than reusing the heavier
multi-clusterer `ggtf_sweep_parallel.py` infrastructure: this only needs greedy clustering, a
handful of grid points, and the same selection rule already established in this project
(`select_milestone_checkpoint.py`) -- minimize the GGTF-exact fake rate among grid points within
`TOLERANCE` points of the arm's own best matched-target rate. Applied on validation only; the
winning point is then reused, unswept, to score the held-out sets.

    python -m src.eval.sweep_lock \
        --forward val/cga/emb/forward_hits.parquet --mc-signal val/mc_signal.parquet \
        --embed-dim 4 --workers 40 --outdir val/cga/op_sweep --lock-out val/cga/locked_op.json
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

TOLERANCE = 2.0  # points of matched-target rate given up to buy a better fake rate; see M-log.

TBETA_GRID = (0.3, 0.6, 0.8)
TD_GRID = (0.1, 0.2, 0.3)


def run_point(py: str, forward: Path, mc_signal: Path, embed_dim: int,
              tbeta: float, td: float, workers: int, outdir: Path) -> dict:
    point_dir = outdir / f"tb{tbeta:g}_td{td:g}"
    marker = point_dir / f"fake_rate_{tbeta:g}_{td:g}_secondaries_in.json"
    if not marker.exists():
        subprocess.run(
            [py, "-u", "-m", "src.eval.ggtf_full_eval",
             "--forward", str(forward), "--mc-signal", str(mc_signal),
             "--embed-dim", str(embed_dim), "--tbeta", str(tbeta), "--td", str(td),
             "--workers", str(workers), "--outdir", str(point_dir)],
            check=True,
        )
    return json.loads(marker.read_text())


def lock(rows: list[dict], tolerance: float = TOLERANCE) -> dict:
    best_eff = max(r["ggtf_efficiency"] for r in rows)
    eligible = [r for r in rows if r["ggtf_efficiency"] >= best_eff - tolerance / 100.0]
    winner = min(eligible, key=lambda r: r["ggtf_fake_rate"])
    return winner


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--forward", type=Path, required=True)
    ap.add_argument("--mc-signal", type=Path, required=True)
    ap.add_argument("--embed-dim", type=int, required=True)
    ap.add_argument("--workers", type=int, default=40)
    ap.add_argument("--outdir", type=Path, required=True)
    ap.add_argument("--lock-out", type=Path, required=True)
    ap.add_argument("--python", default=sys.executable)
    a = ap.parse_args()
    a.outdir.mkdir(parents=True, exist_ok=True)

    rows = []
    for tbeta in TBETA_GRID:
        for td in TD_GRID:
            res = run_point(a.python, a.forward, a.mc_signal, a.embed_dim,
                             tbeta, td, a.workers, a.outdir)
            rows.append(res)
            print(f"  tbeta={tbeta:g} td={td:g}  "
                  f"matched={100*res['ggtf_efficiency']:.2f}%  "
                  f"fake={100*res['ggtf_fake_rate']:.2f}%", flush=True)

    winner = lock(rows, TOLERANCE)
    a.lock_out.write_text(json.dumps({
        "tbeta": winner["tbeta"], "td": winner["td"],
        "ggtf_efficiency": winner["ggtf_efficiency"],
        "ggtf_fake_rate": winner["ggtf_fake_rate"],
        "tolerance_points": TOLERANCE,
        "grid": {"tbeta": list(TBETA_GRID), "td": list(TD_GRID)},
        "all_points": rows,
    }, indent=2) + "\n")
    print(f"LOCKED tbeta={winner['tbeta']:g} td={winner['td']:g} "
          f"matched={100*winner['ggtf_efficiency']:.2f}% "
          f"fake={100*winner['ggtf_fake_rate']:.2f}% -> {a.lock_out}", flush=True)


if __name__ == "__main__":
    main()
