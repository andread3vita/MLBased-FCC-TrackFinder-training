"""Noise-hit statistics for the two IDEA productions. Raw measurements only.

**The premise of this script is wrong; read the numbers as particle-0 statistics.**
It calls `mc_index == 0` noise, meaning "no associated MC particle". There is no such
population: every one of 2.0M drift hits checked in seed 1 maps to a particle present
in its own event. Index 0 is a real generator-status-1 particle, charged in about a
fifth of events, where it leaves a full track of a median 122 hits. See FINDINGS.md
M20. So what this script plots as a noise fraction is the fraction of hits belonging
to one specific real particle, which is why it is small and bimodal rather than a
detector-noise distribution.

Left computing the same quantity on purpose, since the measurements themselves are
correct and are the input to M20. Only the interpretation changes.

Outputs (eval_results/share/):
  noise_report.md            tables (per-seed totals, per-event distribution)
  noise_per_event_hist.png   per-event noise-fraction histograms, both samples
  noise_vs_event_index.png   per-event noise fraction vs event_id, one seed each
"""
import os
import sys

sys.path.insert(0, "/home/marko.cechovic/cgatr-paper/literature/cgatr_fcc_pkg")
import numpy as np
import polars as pl
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src.eval.plotstyle import COL

OUT = "/home/marko.cechovic/cgatr-paper/literature/cgatr_fcc_pkg/eval_results/share"
os.makedirs(OUT, exist_ok=True)

SAMPLES = {
    "key4hep_2026_06_16 IDEA_v4_o1 Zqq_uds_minKineticEnergy0":
        ("/home/marko.cechovic/cgatr-data/data-final/parquet", [1, 50, 150, 181]),
    "v1 production Zqq_uds":
        ("/home/marko.cechovic/cgatr/data_parquet_train/v1_zqq_uds", [1, 50, 150, 1001]),
}


def per_event(root, seed):
    dc = pl.read_parquet(f"{root}/seed_{seed}/dc_hits_train.parquet",
                         columns=["event_id", "mc_index"])
    vt = pl.read_parquet(f"{root}/seed_{seed}/vtx_hits_train.parquet",
                         columns=["event_id", "mc_index"])
    hits = pl.concat([dc, vt])
    g = (hits.group_by("event_id")
             .agg([(pl.col("mc_index") == 0).mean().alias("noise_frac"),
                   pl.len().alias("n_hits")])
             .sort("event_id"))
    dc_f = float((dc["mc_index"] == 0).mean())
    vt_f = float((vt["mc_index"] == 0).mean())
    return g, dc_f, vt_f


lines = ["# Noise-hit statistics (mc_index == 0)",
         "",
         "Per-seed totals:",
         "",
         "| sample | seed | events | DC noise | VTX noise | median event | q90 event | events < 0.5% | events > 2% |",
         "|---|---|---|---|---|---|---|---|---|"]
hists, traces = {}, {}
for name, (root, seeds) in SAMPLES.items():
    fr_all = []
    for s in seeds:
        g, dc_f, vt_f = per_event(root, s)
        nf = g["noise_frac"].to_numpy()
        fr_all.append(nf)
        lines.append(
            f"| {name} | {s} | {len(nf)} | {100*dc_f:.2f}% | {100*vt_f:.2f}% | "
            f"{100*np.median(nf):.2f}% | {100*np.quantile(nf, .9):.2f}% | "
            f"{100*np.mean(nf < 0.005):.0f}% | {100*np.mean(nf > 0.02):.0f}% |")
        if s == seeds[0]:
            traces[name] = g
    hists[name] = np.concatenate(fr_all)

lines += ["",
          "Aggregate over the seeds above:",
          "",
          "| sample | events | mean noise/event | median | frac events < 0.5% | frac events > 2% |",
          "|---|---|---|---|---|---|"]
for name, nf in hists.items():
    lines.append(f"| {name} | {len(nf)} | {100*nf.mean():.2f}% | {100*np.median(nf):.2f}% | "
                 f"{100*np.mean(nf < 0.005):.1f}% | {100*np.mean(nf > 0.02):.1f}% |")
with open(f"{OUT}/noise_report.md", "w") as f:
    f.write("\n".join(lines) + "\n")
print("wrote", f"{OUT}/noise_report.md")

# histogram
fig, ax = plt.subplots(figsize=(6.4, 4.2))
bins = np.linspace(0, 0.08, 81)
for (name, nf), color in zip(hists.items(), (COL["standard"], COL["keepall"])):
    ax.hist(np.clip(nf, 0, 0.08), bins=bins, histtype="step", lw=1.8,
            color=color, label=name.split(" Zqq")[0], density=True)
ax.set_xlabel("noise-hit fraction per event (mc_index == 0)")
ax.set_ylabel("event density")
ax.legend(fontsize=8.5)
ax.set_title("Per-event noise fraction, 4 seeds per sample (2000 events each)")
fig.savefig(f"{OUT}/noise_per_event_hist.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print("wrote", f"{OUT}/noise_per_event_hist.png")

# per-event trace (structure check)
fig, axes = plt.subplots(2, 1, figsize=(8.6, 5.2), sharex=False)
for ax, (name, g), color in zip(axes, traces.items(), (COL["standard"], COL["keepall"])):
    ax.plot(g["event_id"].to_numpy(), 100 * g["noise_frac"].to_numpy(),
            ".", ms=2.5, color=color)
    ax.set_ylabel("noise %")
    ax.set_title(f"{name}, seed {'1' if 'key4hep' in name else '1'}", fontsize=9.5)
axes[-1].set_xlabel("event_id within seed")
fig.tight_layout()
fig.savefig(f"{OUT}/noise_vs_event_index.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print("wrote", f"{OUT}/noise_vs_event_index.png")
