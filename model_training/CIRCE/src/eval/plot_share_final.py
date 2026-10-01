"""Share-ready best-results plots for the key4hep_2026_06_16 IDEA_v4_o1 eval set.

Plot 1 (headline): match rate vs pT on the new-production eval seeds (181-200,
never trained on), best checkpoint (consolidation r3, epoch 6), at the two
bar-clearing operating points (td=0.05, 0.07) plus the paper base OP (td=0.2)
and the oracle-merge ceiling, with the 90% target line and fake rates.

Plot 2: two-config overview (new-production no-keep-all + old-production
keep-all) at the base OP with oracle ceilings, paper style.

Outputs to eval_results/share/ (persistent, NOT /tmp).
"""
import json
import os
import sys

sys.path.insert(0, "/home/marko.cechovic/cgatr-paper/literature/cgatr_fcc_pkg")
import numpy as np
import polars as pl
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src.eval.plotstyle import COL, yticks

BASE = "/home/marko.cechovic/cgatr-paper/literature/cgatr_fcc_pkg/eval_results"
OUT = f"{BASE}/share"
os.makedirs(OUT, exist_ok=True)
EDGES = np.logspace(np.log10(0.1), np.log10(30), 26)


def binned(df, min_n=20):
    d = df.filter(pl.col("is_reconstructable_idea"))
    pt = d["pt"].to_numpy(); m = d["matched"].to_numpy().astype(float)
    cen, rate, lo, hi = [], [], [], []
    for a, b in zip(EDGES[:-1], EDGES[1:]):
        sel = (pt >= a) & (pt < b); n = sel.sum()
        if n < min_n:
            continue
        k = m[sel].sum(); p = k / n
        z = 1.0; z2 = z * z
        denom = 1 + z2 / n
        c = (p + z2 / (2 * n)) / denom
        half = z * np.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / denom
        cen.append(np.sqrt(a * b)); rate.append(p)
        lo.append(max(0, p - (c - half))); hi.append((c + half) - p)
    return np.array(cen), np.array(rate), np.array(lo), np.array(hi)


def fake_of(path):
    with open(path) as f:
        return json.load(f)["fake_rate_idea"]


# ---------------- Plot 1: headline, new-production eval set ----------------
fig, ax = plt.subplots(figsize=(7.0, 4.8))
CFG = "consol_r3_nokeepall"
series = [
    (f"{CFG}/merge_sweep/td0.05_mg0_at0", "$t_d=0.05$", COL["standard"], "-",  "s"),
    (f"{CFG}/merge_sweep/td0.07_mg0_at0", "$t_d=0.07$", COL["keepall"],  "-",  "o"),
    (f"{CFG}/fcc_unmerged",               "$t_d=0.20$ (baseline OP)", COL["ref"], "--", "^"),
]
for sub, label, color, ls, mk in series:
    c, r, lo, hi = binned(pl.read_parquet(f"{BASE}/{sub}/cache.parquet"))
    fk = fake_of(f"{BASE}/{sub}/summary.json")
    ax.errorbar(c, r, yerr=[lo, hi], fmt=f"{mk}{ls}", color=color, ms=4.5,
                lw=1.8, label=f"CIRCE, {label}  (fake {100*fk:.1f}%)", zorder=3)
co, ro, _, _ = binned(pl.read_parquet(f"{BASE}/{CFG}/fcc_oracle_T0.75/cache.parquet"))
ax.plot(co, ro, ":", color=COL["ink"], lw=1.5, alpha=0.75,
        label="oracle-merge ceiling", zorder=2)
ax.axhline(0.90, color=COL["fake"], lw=1.3, ls=":", zorder=1)
ax.text(0.95, 0.9035, "target: $\\geq$90%", color=COL["fake"], fontsize=9.5,
        va="bottom", ha="left")
ax.set_xscale("log"); ax.set_xlim(0.095, 33); ax.set_ylim(0.78, 1.005)
yticks(ax, 0.05)
ax.set_xlabel("$p_\\mathrm{T}$ [GeV]")
ax.set_ylabel("Track match rate ($>75\\%$ purity)")
ax.set_title("CIRCE on key4hep_2026_06_16 IDEA_v4_o1 $Z\\to q\\bar q$ (uds), eval seeds 181-200")
ax.text(0.035, 0.05, "$\\sqrt{s}=91.2$ GeV\n$15^\\circ<\\theta<165^\\circ$,  $N_\\mathrm{hits}>10$",
        transform=ax.transAxes, fontsize=8.5, va="bottom", ha="left",
        bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#C9D0DE", alpha=0.9))
ax.legend(loc="lower right", fontsize=8.5, framealpha=0.9)
for e in ("png", "pdf"):
    fig.savefig(f"{OUT}/circe_best_eff_vs_pt_key4hep.{e}", dpi=200, bbox_inches="tight")
plt.close(fig)
print("wrote", f"{OUT}/circe_best_eff_vs_pt_key4hep.png")

# ---------------- Plot 3: clean plot, reconstruction vs oracle -------------
fig, ax = plt.subplots(figsize=(6.4, 4.4))
CFG = "consol_r3_nokeepall"
c, r, lo, hi = binned(pl.read_parquet(
    f"{BASE}/{CFG}/merge_sweep/td0.05_mg0_at0/cache.parquet"))
ax.errorbar(c, r, yerr=[lo, hi], fmt="s-", color=COL["standard"], ms=4.5,
            lw=1.8, label="CIRCE", zorder=3)
co, ro, olo, ohi = binned(pl.read_parquet(
    f"{BASE}/{CFG}/merge_sweep/td0.05_mg0_at0_oracle_T0.75/cache.parquet"))
ax.errorbar(co, ro, yerr=[olo, ohi], fmt="o--", color=COL["keepall"], ms=4.0,
            lw=1.6, alpha=0.9, label="CIRCE, oracle merge", zorder=2)
ax.set_xscale("log"); ax.set_xlim(0.095, 33); ax.set_ylim(0.85, 1.005)
yticks(ax, 0.05)
ax.set_xlabel("$p_\\mathrm{T}$ [GeV]")
ax.set_ylabel("Track match rate ($>75\\%$ purity)")
ax.set_title("IDEA_v4_o1 $Z\\to q\\bar q$ (uds)")
ax.text(0.035, 0.05, "$\\sqrt{s}=91.2$ GeV\n$15^\\circ<\\theta<165^\\circ$,  $N_\\mathrm{hits}>10$",
        transform=ax.transAxes, fontsize=8.5, va="bottom", ha="left",
        bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#C9D0DE", alpha=0.9))
ax.legend(loc="lower right", fontsize=9, framealpha=0.9)
for e in ("png", "pdf"):
    fig.savefig(f"{OUT}/circe_clean_eff_vs_pt.{e}", dpi=200, bbox_inches="tight")
plt.close(fig)
print("wrote", f"{OUT}/circe_clean_eff_vs_pt.png")

# ---------------- Plot 2: two-config overview at base OP -------------------
fig, ax = plt.subplots(figsize=(5.8, 4.1))
for label, cfg, color, mk in [
        ("no-keep-all (key4hep 2026_06_16)", "consol_r3_nokeepall", COL["standard"], "s"),
        ("keep-all (v1 production)",         "consol_r3_keepall",   COL["keepall"],  "o")]:
    c, r, lo, hi = binned(pl.read_parquet(f"{BASE}/{cfg}/fcc_unmerged/cache.parquet"))
    ax.errorbar(c, r, yerr=[lo, hi], fmt=f"{mk}-", color=color, ms=4.5, lw=1.6,
                label=f"CIRCE, {label}", zorder=3)
    co, ro, _, _ = binned(pl.read_parquet(f"{BASE}/{cfg}/fcc_oracle_T0.75/cache.parquet"))
    ax.plot(co, ro, "--", color=color, lw=1.4, alpha=0.8,
            label=f"{label.split(' (')[0]}, oracle merge", zorder=2)
ax.axhline(1.0, color=COL["ref"], lw=0.8, ls=":", zorder=0)
ax.axhline(0.90, color=COL["fake"], lw=1.1, ls=":", zorder=0, alpha=0.7)
ax.set_xscale("log"); ax.set_xlabel(r"$p_\mathrm{T}$ [GeV]")
ax.set_ylabel("Track match rate ($>75\\%$ purity)")
ax.set_ylim(0.68, 1.015); yticks(ax, 0.05)
ax.legend(loc="lower right", fontsize=7.5, framealpha=0.9)
ax.set_title("Best model, baseline OP $(t_\\beta,t_d)=(0.1,0.2)$")
for e in ("png", "pdf"):
    fig.savefig(f"{OUT}/circe_two_configs_baseOP.{e}", dpi=200, bbox_inches="tight")
plt.close(fig)
print("wrote", f"{OUT}/circe_two_configs_baseOP.png")
