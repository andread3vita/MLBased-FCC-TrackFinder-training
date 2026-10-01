"""Consolidated operating-point report over all td / merge_td eval caches.

Walks eval_results/<config>/{fcc_unmerged,td_sweep/*,merge_sweep/*} dirs,
computes IDEA-cut metrics + per-pT match-rate bins from each cache.parquet,
and writes one markdown table per config, flagging rows that pass the GGTF
bars (every pT bin >= 90% match AND fake rate < 8%).

  PYTHONPATH=. python src/eval/op_report.py --out eval_results/OVERNIGHT_REPORT.md
"""
from __future__ import annotations
import argparse
import glob
import json
import os

import numpy as np
import polars as pl

PT_BINS = [(0.1, 0.15), (0.15, 0.2), (0.2, 0.3), (0.3, 0.5),
           (0.5, 1.0), (1.0, 3.0), (3.0, 30.0)]
CONFIGS = ["final_keepall", "final_nokeepall", "final_loopers", "r3_loopers",
           "consol_r3_keepall", "consol_r3_nokeepall",
           "consol_keepall", "consol_nokeepall",
           "ft_ep6_keepall", "ft_ep6_nokeepall",
           "ft_ep4_keepall", "ft_ep4_nokeepall",
           "ep19_keepall", "ep19_nokeepall"]


def one_point(d):
    sj = os.path.join(d, "summary.json")
    cp = os.path.join(d, "cache.parquet")
    if not (os.path.isfile(sj) and os.path.isfile(cp)):
        return None
    with open(sj) as f:
        s = json.load(f)
    df = (pl.read_parquet(cp, columns=["pt", "matched", "purity_of_match",
                                       "efficiency_per_hit", "is_reconstructable_idea"])
            .filter(pl.col("is_reconstructable_idea")))
    bins = []
    for a, b in PT_BINS:
        sub = df.filter((pl.col("pt") >= a) & (pl.col("pt") < b))
        bins.append(float(sub["matched"].cast(pl.Float64).mean()) if sub.height else float("nan"))
    strict = float(((df["purity_of_match"] >= 0.5) &
                    (df["efficiency_per_hit"] >= 0.5)).mean())
    return {
        "td": s.get("td"), "merge": s.get("merge_td", 0.0),
        "attach": s.get("attach_td", 0.0),
        "match": s["idea"]["match_rate"], "strict50": strict,
        "hiteff": s["idea"]["efficiency"], "purity": s["overall_purity"],
        "fake": s["fake_rate_idea"], "bins": bins,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="eval_results")
    ap.add_argument("--out", default="eval_results/OVERNIGHT_REPORT.md")
    args = ap.parse_args()

    lines = ["# Operating-point report (IDEA cuts)",
             "",
             "PASS = every pT bin >= 0.90 match AND fake < 0.08.",
             "pT bins [GeV]: " + ", ".join(f"{a}-{b}" for a, b in PT_BINS), ""]
    for cfg in CONFIGS:
        root = os.path.join(args.base, cfg)
        if not os.path.isdir(root):
            continue
        dirs = sorted(glob.glob(os.path.join(root, "td_sweep", "td_*")) +
                      glob.glob(os.path.join(root, "merge_sweep", "*")))
        dirs.insert(0, os.path.join(root, "fcc_unmerged"))
        rows = [r for r in (one_point(d) for d in dirs) if r]
        if not rows:
            continue
        rows.sort(key=lambda r: (r["td"], r["merge"], r["attach"]))
        lines += [f"## {cfg}", "",
                  "| td | merge | attach | match | strict50 | hit-eff | purity | fake | "
                  + " | ".join(f"{a}-{b}" for a, b in PT_BINS) + " | min bin | PASS |",
                  "|" + "---|" * (15 + len(PT_BINS) - 7)]
        for r in rows:
            mn = np.nanmin(r["bins"])
            ok = mn >= 0.90 and r["fake"] < 0.08
            lines.append(
                f"| {r['td']:.2f} | {r['merge']:.2f} | {r['attach']:.2f} | "
                f"{r['match']:.3f} | "
                f"{r['strict50']:.3f} | {r['hiteff']:.3f} | {r['purity']:.3f} | "
                f"{r['fake']:.3f} | "
                + " | ".join(f"{v:.3f}" for v in r["bins"])
                + f" | {mn:.3f} | {'**PASS**' if ok else ''} |")
        lines.append("")
    with open(args.out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {args.out} ({sum(1 for l in lines if l.startswith('| 0'))} rows)")


if __name__ == "__main__":
    main()
