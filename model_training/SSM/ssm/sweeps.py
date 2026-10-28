"""Validation physics sweep + plot production for SSM (independent).

Runs the working-point sweep over a subset of validation events, selects the
pareto-F1 and max-efficiency operating points, renders the same figure +
CSV payload shared_training produces for GATR_CIRCE_LOSS and CIRCE (pT,
displacement, theta; double-majority vs Hungarian at both working points),
and returns exact-contract scalar keys plus media for wandb offline logging.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch

from .metrics import (DEFAULT_TBETA_GRID, DEFAULT_TD_GRID, GEN_STATUS_OK,
                      MATCHING_LABELS, MIN_THETA_DEG, MAX_THETA_DEG,
                      PT_BINS, DISP_BINS_MM, THETA_BINS_DEG, SweepAccumulator)

MATCHING_METRICS = ("double_majority", "hungarian")


@torch.no_grad()
def run_sweep(model, loader, device, max_events: int,
              tbeta_grid=DEFAULT_TBETA_GRID, td_grid=DEFAULT_TD_GRID,
              token_budget_is_padded=True):
    model.eval()
    acc = SweepAccumulator()
    n_done = 0
    for batch in loader:
        feats = batch["features"].to(device)
        emb, beta_logit = model(feats)
        beta = torch.sigmoid(beta_logit).float().cpu().squeeze(-1)
        emb = emb.float().cpu()
        lens = batch["lengths"]
        for i in range(emb.shape[0]):
            L = int(lens[i])
            acc.update_event(
                emb[i, :L], beta[i, :L], batch["mc_index"][i, :L].cpu(),
                tbeta_grid, td_grid,
                pt_map=batch["pt_maps"][i], disp_map=batch["disp_maps"][i],
                theta_map=batch["theta_maps"][i],
                genstatus_map=batch["genstatus_maps"][i])
            n_done += 1
            if n_done >= max_events:
                break
        if n_done >= max_events:
            break
    return acc


def working_point_scalars(wp):
    """Exact GATR_CIRCE_LOSS scalar keys for the two working points."""
    pareto, best = wp["pareto_f1"], wp["max_efficiency"]
    out = {}
    if pareto is not None:
        out.update({
            "validation/pareto_f1": pareto["f1"],
            "validation/fake_rate_at_pareto_f1": pareto["fake_rate"],
            "validation/fake_rate_double_majority_at_pareto_f1": pareto["fake_rate"],
            "validation/tracking_efficiency_at_pareto_f1": pareto["efficiency"],
            "validation/pareto_f1_tbeta": pareto["tbeta"],
            "validation/pareto_f1_td": pareto["td"],
        })
    if best is not None:
        out.update({
            "validation/max_tracking_efficiency": best["efficiency"],
            "validation/fake_rate_at_max_efficiency": best["fake_rate"],
            "validation/fake_rate_double_majority_at_max_efficiency": best["fake_rate"],
            "validation/max_efficiency_tbeta": best["tbeta"],
            "validation/max_efficiency_td": best["td"],
        })
    return out


def _mpl():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        print("matplotlib not installed — skipping sweep plots "
              "(install from wheelhouse to enable)", flush=True)
        return None


def _wp_series(row):
    """Plot records for both matching criteria at one working point."""
    series = []
    for metric in MATCHING_METRICS:
        suffix = "" if metric == "double_majority" else "_hungarian"
        series.append({
            "name": metric,
            "label": MATCHING_LABELS[metric],
            "efficiency": row[f"efficiency{suffix}"],
            "fake_rate": row[f"fake_rate{suffix}"],
            "total_pt": row["sel_pt"], "matched_pt": row[f"mat_pt{suffix}"],
            "total_dp": row["sel_dp"], "matched_dp": row[f"mat_dp{suffix}"],
            "total_th": row["sel_th"], "matched_th": row[f"mat_th{suffix}"],
        })
    return series


def _save_binned_curve(series, out_stem, bins, column_prefix, x_label,
                       log_x, min_x, max_x, truth_min_hits,
                       x_major=500.0, x_minor=50.0, text_y=0.40, text_size=15):
    """PNG + CSV identical in construction to the shared_training plotter."""
    bins = np.asarray(bins, dtype=np.float64)
    processed = []
    for item in series:
        total = np.asarray(item["total"], dtype=np.int64)
        matched = np.asarray(item["matched"], dtype=np.int64)
        effs = np.divide(matched, total, out=np.full(total.shape, np.nan),
                         where=total > 0)
        errs = np.zeros_like(effs)
        pop = total > 0
        errs[pop] = np.sqrt(effs[pop] * (1.0 - effs[pop]) / total[pop])
        processed.append({**item, "total": total, "matched": matched,
                          "effs": effs, "errs": errs})

    centers = 0.5 * (bins[:-1] + bins[1:])
    csv_path = out_stem.with_suffix(".csv")
    with csv_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        header = [f"{column_prefix}_low", f"{column_prefix}_high",
                  f"{column_prefix}_center"]
        for item in processed:
            header += [f"n_truth_{item['name']}", f"n_matched_{item['name']}",
                       f"efficiency_{item['name']}", f"error_{item['name']}"]
        writer.writerow(header)
        for b in range(bins.size - 1):
            row = [bins[b], bins[b + 1], centers[b]]
            for item in processed:
                row += [item["total"][b], item["matched"][b],
                        item["effs"][b], item["errs"][b]]
            writer.writerow(row)

    plt = _mpl()
    if plt is None:
        return None
    style = {
        "text.usetex": False, "font.family": "serif", "font.size": 25,
        "axes.labelsize": 25, "xtick.labelsize": 25, "ytick.labelsize": 25,
        "legend.fontsize": 25,
    }
    with plt.rc_context(style):
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111)
        colours = ("#238A8DFF", "#440154FF", "#55C667FF", "#FDE725FF")
        markers = ("s", "o", "^", "D")
        for index, item in enumerate(processed):
            colour, marker = colours[index % 4], markers[index % 4]
            effs, errs = item["effs"], item["errs"]
            valid = np.isfinite(effs)
            summary = (f"eff = {item['efficiency']:.3f}, "
                       f"fake = {item['fake_rate']:.3f}")
            ax.scatter(centers[valid], effs[valid],
                       label=f"{item['label']}\n{summary}", marker=marker,
                       c=[colour] * int(valid.sum()), s=30)
            ax.errorbar(centers[valid], effs[valid],
                        yerr=[errs[valid],
                              np.minimum(effs[valid] + errs[valid], 1.0) - effs[valid]],
                        ecolor=colour, linestyle="none", capsize=4)
        ax.set_xlabel(x_label)
        ax.set_ylabel("Tracking efficiency")
        if log_x:
            ax.set_xscale("log")
        ax.set_xlim([min_x, max_x])
        ax.set_ylim([0.01, 1.01])
        ax.legend(loc="lower right", fontsize=17)
        if log_x:
            ax.xaxis.set_major_locator(plt.LogLocator(base=10.0, numticks=4))
            ax.xaxis.set_minor_locator(
                plt.LogLocator(base=10.0, subs="auto", numticks=10))
        else:
            ax.xaxis.set_major_locator(plt.MultipleLocator(x_major))
            ax.xaxis.set_minor_locator(plt.MultipleLocator(x_minor))
        ax.yaxis.set_major_locator(plt.MultipleLocator(0.1))
        ax.yaxis.set_minor_locator(plt.MultipleLocator(0.1))
        ax.minorticks_on()
        ax.grid(which="major", linestyle=":", linewidth=0.5, color="black")
        ax.grid(which="minor", linestyle=":", linewidth=0.5, color="gray")
        legend = ax.get_legend()
        if legend is not None:
            legend._legend_box.align = "left"
        status_text = ",".join(str(int(v)) for v in GEN_STATUS_OK)
        ax.text(0.45, text_y,
                r"$Z/\gamma^* \rightarrow q\bar{q}\ (q = u, d, s)$" "\n"
                r"$\sqrt{s} = m_Z = 91~\mathrm{GeV}$" "\n"
                rf"${MIN_THETA_DEG:g}^\circ < \theta < {MAX_THETA_DEG:g}^\circ$" "\n"
                rf"$genStatus \in [{status_text}]$" "\n"
                rf"$N_\mathrm{{hits}} \geq {int(truth_min_hits)}$",
                transform=ax.transAxes, fontsize=text_size,
                verticalalignment="center", horizontalalignment="left",
                linespacing=1.4,
                bbox=dict(boxstyle="round,pad=0.35", facecolor="none",
                          edgecolor="none"))
        png_path = out_stem.with_suffix(".png")
        fig.savefig(png_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    return png_path


_CURVES = (
    # (observable, filename stem, bins, csv prefix, x label, log_x, min_x, max_x, locators)
    ("pt", "tracking_efficiency_vs_pt", PT_BINS, "pt", r"$p_T$ [GeV]",
     True, 0.1, 60.0, (500.0, 50.0)),
    ("dp", "tracking_efficiency_vs_displacement", DISP_BINS_MM, "displacement_mm",
     r"$r_\mathrm{vtx}=\sqrt{x_\mathrm{vtx}^2+y_\mathrm{vtx}^2}$ [mm]",
     False, 0.0, 2000.0, (500.0, 50.0)),
    ("th", "tracking_efficiency_vs_theta", THETA_BINS_DEG, "theta_deg",
     r"$\theta$ [deg]", False, 10.0, 170.0, (30.0, 5.0)),
)


def save_sweep_plots(wp, out_dir: str, truth_min_hits: int = 3):
    """Render the same figure payload as the shared plotter; return media dict."""
    out = Path(out_dir) / "plots"
    out.mkdir(parents=True, exist_ok=True)
    media = {}
    plt = _mpl()
    if plt is None:
        return media
    pareto, best = wp["pareto_f1"], wp["max_efficiency"]

    # --- operating-point scatter (F1 landscape over sweep grid) ---
    fig, ax = plt.subplots(figsize=(5, 4))
    for row, lab, c in ((pareto, "pareto_f1", "tab:blue"),
                        (best, "max_efficiency", "tab:red")):
        if row is not None:
            ax.scatter([row["efficiency"]], [row["fake_rate"]], label=lab, c=c)
    ax.set_xlabel("tracking efficiency"); ax.set_ylabel("fake rate")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.legend()
    p = out / "efficiency_fake_pareto.png"
    fig.savefig(p, dpi=120, bbox_inches="tight"); plt.close(fig)
    media["plots/operating_point_sweep/efficiency_fake_pareto"] = p

    # --- binned efficiency curves at both working points, both matchings ---
    for wp_name, row in (("max_efficiency", best), ("pareto_f1", pareto)):
        if row is None:
            continue
        series = [s for s in _wp_series(row) if s["total_pt"] is not None]
        for obs, stem, bins, prefix, xlabel, log_x, min_x, max_x, locs in _CURVES:
            per_obs = [{"name": s["name"], "label": s["label"],
                        "efficiency": s["efficiency"], "fake_rate": s["fake_rate"],
                        "total": s[f"total_{obs}"], "matched": s[f"matched_{obs}"]}
                       for s in series]
            if not any(np.asarray(s["total"]).sum() > 0 for s in per_obs):
                continue
            image = _save_binned_curve(
                per_obs, out / f"{stem}_{wp_name}", bins, prefix, xlabel,
                log_x, min_x, max_x, truth_min_hits,
                x_major=locs[0], x_minor=locs[1])
            if image is not None:
                media[f"plots/{stem}/{wp_name}"] = image
    return media
