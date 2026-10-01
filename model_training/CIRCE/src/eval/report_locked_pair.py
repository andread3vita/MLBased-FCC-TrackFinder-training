"""Report test metrics at operating points selected only on validation data."""

import argparse
import json
import os


def pct(x):
    return 100.0 * float(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--lock", required=True)
    ap.add_argument("--arms", required=True)
    args = ap.parse_args()

    locked = json.load(open(args.lock))["arms"]
    print("Locked test-set report")
    print("Operating points selected on validation seeds; test seeds evaluated once.")
    print("Raw greedy clustering only: helix_tol=0 (no SimTrackerHit-position merge).")
    print()
    print(
        f"{'arm':<28}{'tbeta':>7}{'td':>7}{'eff':>9}{'def2':>9}"
        f"{'GGTF>10':>11}{'clone':>9}{'spurious':>11}"
    )
    print("-" * 91)
    for arm in args.arms.split():
        summary_path = os.path.join(
            args.root, arm, "locked_raw", "summary.json")
        s = json.load(open(summary_path))
        op = locked[arm]
        nc = s["no_cuts"]
        matched = max(int(s.get("ggtf_n_matched_gt10", 0)), 1)
        clone = int(s.get("ggtf_n_fake_clone_gt10", 0)) / matched
        spurious = int(s.get("ggtf_n_fake_spurious_gt10", 0)) / matched
        print(
            f"{arm:<28}{op['tbeta']:7.2f}{op['td']:7.2f}"
            f"{pct(nc['efficiency']):8.2f}%{pct(nc['def2']):8.2f}%"
            f"{pct(s['ggtf_fake_rate_gt10']):10.2f}%"
            f"{pct(clone):8.2f}%{pct(spurious):10.2f}%"
        )


if __name__ == "__main__":
    main()
