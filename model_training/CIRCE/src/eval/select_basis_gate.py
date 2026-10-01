"""Select the CGA basis from validation-locked clustering metrics.

The in-training ``strict50`` metric is an efficiency measure and does not
penalize the clone/fragmentation failure that dominates this tracking task.
Basis promotion therefore uses each arm's own validation-selected operating
point:

1. retain arms within ``--def2-tolerance`` points of the best def2;
2. among retained arms, choose the lower GGTF>10 fake rate;
3. prefer the published, smaller E(3) basis when fake rates are within
   ``--fake-tie-tolerance`` points.

No test-set metric enters this decision.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def select_basis(
    metrics: dict[str, dict],
    *,
    e3_arm: str = "e3_gate",
    se3_arm: str = "se3_gate",
    def2_tolerance: float = 2.0,
    fake_tie_tolerance: float = 1.0,
) -> tuple[str, str]:
    """Return ``(basis, explanation)`` from two validation metric records."""
    rows = {"e3": metrics[e3_arm], "se3": metrics[se3_arm]}
    for basis, row in rows.items():
        for key in ("def2", "ggtf10"):
            value = float(row[key])
            if not math.isfinite(value):
                raise ValueError(f"{basis} {key} is not finite: {value}")

    best_def2 = max(float(row["def2"]) for row in rows.values())
    eligible = {
        basis: row
        for basis, row in rows.items()
        if float(row["def2"]) >= best_def2 - def2_tolerance
    }
    if len(eligible) == 1:
        selected = next(iter(eligible))
        reason = (
            f"{selected} is the only basis within {def2_tolerance:g}pt "
            f"of best validation def2={best_def2:.3f}"
        )
        return selected, reason

    fake_gap = abs(
        float(eligible["e3"]["ggtf10"]) - float(eligible["se3"]["ggtf10"])
    )
    if fake_gap <= fake_tie_tolerance:
        return (
            "e3",
            f"both bases are def2-eligible and GGTF>10 differs by "
            f"{fake_gap:.3f}pt <= {fake_tie_tolerance:g}pt; prefer published E(3)",
        )

    selected = min(
        eligible, key=lambda basis: float(eligible[basis]["ggtf10"])
    )
    return (
        selected,
        f"both bases are def2-eligible; {selected} has lower validation "
        f"GGTF>10 fake rate",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--e3-arm", default="e3_gate")
    parser.add_argument("--se3-arm", default="se3_gate")
    parser.add_argument("--def2-tolerance", type=float, default=2.0)
    parser.add_argument("--fake-tie-tolerance", type=float, default=1.0)
    args = parser.parse_args()

    payload = json.loads(Path(args.lock).read_text())
    selected, reason = select_basis(
        payload["arms"],
        e3_arm=args.e3_arm,
        se3_arm=args.se3_arm,
        def2_tolerance=args.def2_tolerance,
        fake_tie_tolerance=args.fake_tie_tolerance,
    )
    Path(args.output).write_text(selected + "\n")

    print("Validation-only basis gate")
    for basis, arm in (("E(3)", args.e3_arm), ("SE(3)", args.se3_arm)):
        row = payload["arms"][arm]
        print(
            f"  {basis}: def2={float(row['def2']):.3f}% "
            f"GGTF>10={float(row['ggtf10']):.3f}% "
            f"(tbeta={row['tbeta']}, td={row['td']})"
        )
    print(f"Selected {selected}: {reason}")


if __name__ == "__main__":
    main()
