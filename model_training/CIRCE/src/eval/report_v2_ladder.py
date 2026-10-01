"""Report the conformal v2 ladder as deltas against the Phase 1 baseline.

Each row adds one ingredient to the row above, and every arm trained on the same
recipe and target set, so the per-row delta is attributable to that ingredient
alone. Absolute numbers are printed too, since a ladder of deltas hides whether
the starting point was any good.

    PYTHONPATH=. python src/eval/report_v2_ladder.py

Reads eval_results/v2_ladder/<arm>/op_hx<tol>/summary.json, written by
run_v2_eval.sh. Arms with no checkpoint yet are listed as pending rather than
omitted, so a partial ladder cannot be misread as a complete one.

Ends by naming the winning prefix of the ladder and printing the command for the
long final run, so that choice is made against a stated metric and margin rather
than by eyeballing the table.
"""

import argparse
import glob
import json
import os

BASE = "/home/marko.cechovic/cgatr-paper/literature/cgatr_fcc_pkg/eval_results"

# The ladder, in order, with what each row adds relative to the row above, and the
# train.py flag that row adds. The flags are cumulative, so the flag set for any arm
# is the concatenation of its own and every row above it -- which is what makes
# "the winning prefix" a well-defined thing to recommend.
LADDER = [
    # Which arm this is depends on the regime run_v2_eval.sh was run in, and the two
    # land in different roots so they cannot be mixed: unfiltered (the default) reads
    # eval_results/v2_ladder and the baseline is ablation-v1/conformal, the arm at
    # val_loss 0.566 that the five-arm table is built on; filtered reads
    # eval_results/v2_ladder_filtered and the baseline is Phase 1's conformal_noloop.
    # The unfiltered pairing is the one that matters, because the ladder's purpose is
    # to be set against `projective_ggtf` at 0.429 and that arm is unfiltered.
    ("a0_baseline", "conformal baseline, no v2 ingredients", ""),
    ("a1_nullfix", "+ drift sphere/plane offset along inf, not the origin",
     "--fix_cga_null"),
    ("a2_time", "+ hit time", "--use_time"),
    ("a3_twochan", "+ sphere channel, so drift and vertex hits share a grade",
     "--two_channel_dc"),
    ("a4_init", "+ identity-on-algebra initialization",
     "--equi_init identity_algebra"),
]

METRICS = [
    ("def1", lambda s: 100 * s["idea"]["match_rate"], "%6.1f"),
    ("def2", lambda s: 100 * s["idea"]["def2"], "%6.1f"),
    ("ghost", lambda s: 100 * s["ghost_rate"], "%6.1f"),
    ("clone", lambda s: 100 * s["clone_rate"], "%6.1f"),
    ("GGTF>10", lambda s: 100 * s.get("ggtf_fake_rate_gt10", float("nan")), "%7.1f"),
    ("cand/ev", lambda s: s["candidates_per_event"], "%7.1f"),
]

# Lower is better for these, so a negative delta is an improvement.
LOWER_BETTER = {"ghost", "clone", "GGTF>10", "cand/ev"}


def load(root, op=None):
    """cells[(arm, helix tol)] = summary dict, for one operating point.

    Cell directories are `<operating point>_hx<tol>`. The operating point is part of
    the path because it has already changed once, and a ladder assembled from two of
    them would credit postprocessing to whichever ingredient happened to be measured
    at the better point.
    """
    found = sorted(glob.glob(f"{root}/*/*_hx*/summary.json"))
    ops = sorted({os.path.basename(os.path.dirname(p)).rsplit("_hx", 1)[0]
                  for p in found})
    op = op or (ops[-1] if ops else None)

    cells = {}
    for path in found:
        parts = path.split(os.sep)
        arm, leaf = parts[-3], parts[-2]
        cell_op, helix = leaf.rsplit("_hx", 1)
        if cell_op == op:
            cells[(arm, helix)] = json.load(open(path))
    return cells, op, ops


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=f"{BASE}/v2_ladder")
    ap.add_argument("--op", default=None,
                    help="operating point to report, as it appears in the cell "
                         "directory names. Defaults to the newest present.")
    ap.add_argument("--decide_on", default="def2",
                    choices=[n for n, _, _ in METRICS],
                    help="metric the final-run recommendation is based on. def2 by "
                         "default: it is the efficiency definition GGTF quote, and "
                         "unlike GGTF>10 it is not sensitive to the candidate count "
                         "that postprocessing is still being tuned against.")
    ap.add_argument("--margin", type=float, default=0.5,
                    help="points of --decide_on an ingredient must win by to be "
                         "kept. Default 0.5, below which single-seed runs are not "
                         "separable.")
    args = ap.parse_args()

    cells, op_used, ops = load(args.root, args.op)
    if not cells:
        raise SystemExit(
            f"no summaries under {args.root}"
            + (f" at operating point {args.op}" if args.op else "")
            + "; run run_v2_eval.sh first (and launch_v2_ablations.sh before that)")

    helices = sorted({k[1] for k in cells})
    print(__doc__)
    print(f"operating point: {op_used}")
    others = [o for o in ops if o != op_used]
    if others:
        print(f"also on disk, not shown: {', '.join(others)}  (--op to select)")

    for hx in helices:
        label = "no helix merge" if hx in ("0", "0.0") else f"helix merge tol {hx}"
        width = 34 + sum(8 for _ in METRICS) + 8
        print(f"\n{'=' * width}\n{label}\n{'=' * width}")
        header = f"{'arm':<34}"
        for name, _, _ in METRICS:
            header += f"{name:>8}"
        print(header)
        print("-" * width)

        prev = None
        for arm, desc, _ in LADDER:
            s = cells.get((arm, hx))
            if s is None:
                print(f"{arm:<34}{'pending':>8}")
                continue
            line = f"{arm:<34}"
            for name, fn, fmt in METRICS:
                line += f"{fn(s):>8.1f}"
            print(line)
            if prev is not None:
                dline = f"{'  vs above: ' + desc[:20]:<34}"
                for name, fn, fmt in METRICS:
                    d = fn(s) - fn(prev)
                    mark = ""
                    if abs(d) >= 0.05:
                        better = (d < 0) if name in LOWER_BETTER else (d > 0)
                        mark = "+" if better else "-"
                    dline += f"{d:>+7.1f}{mark}"
                print(dline)
            prev = s

        print()
        for arm, desc, _ in LADDER[1:]:
            print(f"  {arm}: {desc}")

    have = {k[0] for k in cells}
    missing = [a for a, _, _ in LADDER if a not in have]
    if missing:
        print(f"\nstill pending: {', '.join(missing)}")

    recommend(cells, helices, args.decide_on, args.margin)


def recommend(cells, helices, decide_on, margin):
    """Name the winning prefix of the ladder and print the command to run it long.

    Only a prefix is a legitimate answer: the arms are cumulative, so there is no
    checkpoint for, say, time-without-nullfix, and picking a non-prefix subset would
    mean claiming a result for a configuration nobody trained. Ties inside `margin`
    resolve to the shorter prefix, on the grounds that an ingredient has to pay for
    itself to be carried into the headline run.
    """
    hx = helices[-1]  # the strongest postprocessing, which the final run will use
    metric = dict((n, f) for n, f, _ in METRICS)[decide_on]
    sign = -1 if decide_on in LOWER_BETTER else 1

    scored = [(arm, metric(cells[(arm, hx)]))
              for arm, _, _ in LADDER if (arm, hx) in cells]
    if len(scored) < 2:
        print("\nnot enough of the ladder is evaluated to recommend a final run")
        return

    # The queue evaluates after each stage, so this runs first with only the decisive
    # rungs present. A recommendation from a prefix of the ladder is still worth printing
    # -- it is the running answer -- but it must not read as final, because the command
    # below launches a 22-epoch run and the untested rungs are cumulative: one of them
    # winning would change the flag set that command should carry.
    pending = [arm for arm, _, _ in LADDER if (arm, hx) not in cells]

    best_i, best_v = 0, scored[0][1]
    for i, (_, v) in enumerate(scored):
        if sign * v > sign * best_v + margin:
            best_i, best_v = i, v

    label = "no helix merge" if hx in ("0", "0.0") else f"helix merge tol {hx}"
    heading = ("recommended final run" if not pending
               else "PROVISIONAL, the ladder is incomplete")
    print(f"\n{'=' * 78}\n{heading}\n{'=' * 78}")
    if pending:
        print(f"Not yet evaluated: {', '.join(pending)}. Those rungs are cumulative, so "
              f"one of them winning would change the flags below. Do not launch the final "
              f"run on this output; treat it as the running answer only.")
    print(f"deciding on {decide_on} at {label}, margin {margin:g} points; "
          f"an ingredient has to beat the prefix above it by more than the margin "
          f"to be kept")
    for i, (arm, v) in enumerate(scored):
        print(f"  {'>' if i == best_i else ' '} {arm:<14}{decide_on} {v:7.1f}")

    if best_i == 0:
        print("\nNo ingredient paid for itself. Do not launch a v2 final run; the "
              "conformal baseline in the a0 row stays the headline.")
        return

    flags = " ".join(f for _, _, f in LADDER[1:best_i + 1] if f)
    if len(scored) > best_i + 1:
        dropped = ", ".join(a for a, _ in scored[best_i + 1:])
        print(f"\nDropping {dropped}: past the winning prefix, so not carried in.")
    print("\n  cd /home/marko.cechovic/cgatr-runs/ablation-v1")
    print(f"  EPOCHS=22 V2_FINAL=1 V2_FINAL_FLAGS='{flags}' \\")
    print("    bash launch_v2_ablations.sh")


if __name__ == "__main__":
    main()
