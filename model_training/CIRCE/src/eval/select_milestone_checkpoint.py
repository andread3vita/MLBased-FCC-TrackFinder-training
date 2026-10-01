"""Select one checkpoint and operating point using validation data only.

All operating points from all requested milestone checkpoints are pooled before
applying the locked rule: minimize GGTF>10 among points no more than two
percentage points below the global best def2. This avoids a two-stage selection
artifact where each checkpoint first sacrifices a different amount of def2.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re

from src.eval.report_op_sweep import DEF2_TOLERANCE, collect


EPOCH_RE = re.compile(r"_epoch(\d+)$")


def checkpoint_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def select_milestone(
    sweep_root: str,
    arms: list[str],
    run_dir: str,
    n_tag: str = "0",
    tolerance: float = DEF2_TOLERANCE,
) -> dict:
    rows: list[dict] = []
    milestone_best: dict[str, dict] = {}
    for arm in arms:
        match = EPOCH_RE.search(arm)
        if match is None:
            raise ValueError(
                f"arm {arm!r} must end in _epochNN so its checkpoint is unambiguous"
            )
        epoch = int(match.group(1))
        arm_rows = collect(sweep_root, arm, n_tag)
        if not arm_rows:
            raise ValueError(f"no complete sweep rows found for {arm!r}")
        for row in arm_rows:
            if not all(math.isfinite(float(row[key])) for key in ("def2", "ggtf10")):
                raise ValueError(f"non-finite validation metric in {arm}: {row}")
            rows.append({**row, "arm": arm, "epoch": epoch})

        local_best_def2 = max(float(row["def2"]) for row in arm_rows)
        local_eligible = [
            row for row in arm_rows
            if float(row["def2"]) >= local_best_def2 - tolerance
        ]
        local_pick = min(
            local_eligible,
            key=lambda row: (
                float(row["ggtf10"]),
                -float(row["def2"]),
                -float(row["eff"]),
                float(row["tbeta"]),
                float(row["td"]),
            ),
        )
        milestone_best[arm] = {
            key: local_pick[key]
            for key in (
                "tbeta", "td", "eff", "def2", "ggtf10", "clone10",
                "spur10", "cand_per_ev", "n_ev", "targets",
            )
        }

    best_def2 = max(float(row["def2"]) for row in rows)
    eligible = [
        row for row in rows if float(row["def2"]) >= best_def2 - tolerance
    ]
    pick = min(
        eligible,
        key=lambda row: (
            float(row["ggtf10"]),
            -float(row["def2"]),
            -float(row["eff"]),
            int(row["epoch"]),
            float(row["tbeta"]),
            float(row["td"]),
        ),
    )

    checkpoint = os.path.join(
        run_dir, f"cgatr_epoch{int(pick['epoch']) - 1:02d}.ckpt"
    )
    if not os.path.isfile(checkpoint):
        raise FileNotFoundError(
            f"selected epoch {pick['epoch']} has no checkpoint at {checkpoint}"
        )

    metrics = {
        key: pick[key]
        for key in (
            "eff", "def2", "ggtf10", "clone10", "spur10",
            "cand_per_ev", "n_ev", "targets",
        )
    }
    return {
        "selection_sample": n_tag,
        "criterion": (
            "minimum GGTF>10 fake rate among every milestone/grid point "
            f"within {tolerance:g}pt of the global best def2"
        ),
        "validation_sweep_root": os.path.abspath(sweep_root),
        "selected_arm": pick["arm"],
        "epoch": int(pick["epoch"]),
        "checkpoint": os.path.abspath(checkpoint),
        "checkpoint_sha256": checkpoint_sha256(checkpoint),
        "operating_point": {
            "tbeta": float(pick["tbeta"]),
            "td": float(pick["td"]),
        },
        "validation_metrics_percent": metrics,
        "milestone_local_choices": milestone_best,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, help="Operating-point sweep root")
    parser.add_argument("--arms", required=True, help="Space-delimited milestone arms")
    parser.add_argument("--run-dir", required=True, help="Directory holding checkpoints")
    parser.add_argument("--n", default="0", help="Event-cap tag in sweep directories")
    parser.add_argument("--def2-tolerance", type=float, default=DEF2_TOLERANCE)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    selected = select_milestone(
        args.root,
        args.arms.split(),
        args.run_dir,
        n_tag=args.n,
        tolerance=args.def2_tolerance,
    )
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as handle:
        json.dump(selected, handle, indent=2)

    op = selected["operating_point"]
    metrics = selected["validation_metrics_percent"]
    print(
        f"Selected epoch {selected['epoch']} at tbeta={op['tbeta']:.3g}, "
        f"td={op['td']:.3g}: def2={metrics['def2']:.2f}%, "
        f"GGTF>10={metrics['ggtf10']:.2f}%"
    )
    print(f"Locked selection written to {args.output}")


if __name__ == "__main__":
    main()
