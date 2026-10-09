"""Clustering, one-to-one tracking metrics, and validation sweep plots.

The functions in this module deliberately operate on NumPy arrays.  Validation
can therefore cache one network forward pass on CPU and scan many clustering
operating points without using additional GPU memory or repeating inference.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np


MATCHING_COMPARISON_METRICS = ("double_majority", "hungarian")
# Matched/fake counts for every comparison criterion are accumulated alongside
# the selected sweep criterion, so each operating point carries both fake rates.
TRACKING_COUNT_KEYS = (
    "n_truth",
    "n_reco",
    "n_matched",
    "n_fake",
    "shared_hits",
    "matched_truth_hits",
    "matched_reco_hits",
) + tuple(
    f"{prefix}_{metric}"
    for metric in MATCHING_COMPARISON_METRICS
    for prefix in ("n_matched", "n_fake")
)

# Requested logarithmic pT binning for the validation tracking-efficiency plot.
TRACKING_PT_BINS = np.exp(np.arange(np.log(0.1), np.log(60.0), 0.2))
# Explicit uniform 50 mm bins over the tracker-scale displacement range.
TRACKING_DISPLACEMENT_BIN_WIDTH = 50.0
TRACKING_DISPLACEMENT_BINS = np.arange(
    0.0,
    2000.0 + TRACKING_DISPLACEMENT_BIN_WIDTH,
    TRACKING_DISPLACEMENT_BIN_WIDTH,
)
MATCHING_METRICS = ("idea", "double_majority", "hungarian")
MATCHING_LABELS = {
    "idea": "IDEA purity",
    "double_majority": r"Double majority ($\epsilon,p>0.5$)",
    "hungarian": "Hungarian 1-to-1",
}
REJECTED_SEED_POLICIES = ("discard", "keep", "attach-after-accept")


def parse_grid(value: str, cast=float) -> Tuple:
    """Parse a comma-separated CLI grid and reject empty/duplicate values."""
    values = tuple(cast(item.strip()) for item in value.split(",") if item.strip())
    if not values:
        raise ValueError("Sweep grids must contain at least one value")
    return tuple(dict.fromkeys(values))


def greedy_cluster(
    beta: np.ndarray,
    coordinates: np.ndarray,
    tbeta: float,
    td: float,
    min_hits: int = 3,
    rejected_seed_policy: str = "discard",
) -> np.ndarray:
    """Beta-ordered OC clustering with an explicit minimum candidate size.

    Unassigned hits retain label ``-1`` and are never interpreted as a track.
    ``rejected_seed_policy`` controls what happens to the core point when its
    candidate contains fewer than ``min_hits`` hits:

    - ``discard`` removes it permanently, matching the historical behaviour;
    - ``keep`` leaves it available as a member of a later candidate;
    - ``attach-after-accept`` defers it so it cannot make a later candidate
      reach ``min_hits``, but attaches it when a later core independently forms
      a valid cluster within ``td``.

    Non-core members of a rejected candidate always remain available.
    """
    beta = np.asarray(beta, dtype=np.float32).reshape(-1)
    coordinates = np.asarray(coordinates, dtype=np.float32)
    if coordinates.ndim != 2 or coordinates.shape[0] != beta.shape[0]:
        raise ValueError("coordinates must have shape (n_hits, embedding_dim)")
    if min_hits < 1 or td <= 0:
        raise ValueError("min_hits must be positive and td must be > 0")
    rejected_seed_policy = str(rejected_seed_policy).lower()
    if rejected_seed_policy not in REJECTED_SEED_POLICIES:
        raise ValueError(
            "rejected_seed_policy must be one of "
            f"{REJECTED_SEED_POLICIES}; received {rejected_seed_policy!r}"
        )

    labels = np.full(beta.shape[0], -1, dtype=np.int32)
    unassigned = np.ones(beta.shape[0], dtype=bool)
    deferred = np.zeros(beta.shape[0], dtype=bool)
    seeds = np.flatnonzero(beta > tbeta)
    seeds = seeds[np.argsort(-beta[seeds], kind="stable")]
    next_label = 0
    for seed in seeds:
        if not unassigned[seed]:
            continue
        available = np.flatnonzero(unassigned)
        distance = np.linalg.norm(coordinates[available] - coordinates[seed], axis=1)
        members = available[distance < td]
        if members.size < min_hits:
            if rejected_seed_policy == "discard":
                unassigned[seed] = False
            elif rejected_seed_policy == "attach-after-accept":
                unassigned[seed] = False
                deferred[seed] = True
            continue

        cluster_members = members
        attached = np.empty(0, dtype=np.int64)
        if rejected_seed_policy == "attach-after-accept" and np.any(deferred):
            deferred_indices = np.flatnonzero(deferred)
            deferred_distance = np.linalg.norm(
                coordinates[deferred_indices] - coordinates[seed], axis=1
            )
            attached = deferred_indices[deferred_distance < td]
            if attached.size:
                cluster_members = np.concatenate((members, attached))

        labels[cluster_members] = next_label
        unassigned[members] = False
        deferred[attached] = False
        next_label += 1
    return labels


def _overlap_tables(labels: np.ndarray, truth: np.ndarray, truth_min_hits: int):
    reco_ids = np.unique(labels[labels >= 0])
    truth_ids, truth_counts = np.unique(truth[truth > 0], return_counts=True)
    truth_ids = truth_ids[truth_counts >= truth_min_hits]
    truth_counts = truth_counts[truth_counts >= truth_min_hits]
    overlap = np.zeros((truth_ids.size, reco_ids.size), dtype=np.int64)
    reco_counts = np.zeros(reco_ids.size, dtype=np.int64)
    for j, reco_id in enumerate(reco_ids):
        reco_mask = labels == reco_id
        reco_counts[j] = int(reco_mask.sum())
        values, counts = np.unique(truth[reco_mask], return_counts=True)
        count_map = dict(zip(values.tolist(), counts.tolist()))
        overlap[:, j] = [count_map.get(int(tid), 0) for tid in truth_ids]
    return truth_ids, truth_counts, reco_ids, reco_counts, overlap


def _one_to_one_matches(
    overlap: np.ndarray,
    truth_counts: np.ndarray,
    reco_counts: np.ndarray,
    metric: str,
) -> List[Tuple[int, int]]:
    if overlap.size == 0:
        return []
    efficiency = overlap / np.maximum(truth_counts[:, None], 1)
    purity = overlap / np.maximum(reco_counts[None, :], 1)
    if metric == "idea":
        valid = (purity >= 0.75) & (overlap > 0)
    elif metric == "double_majority":
        valid = (purity >= 0.50) & (efficiency >= 0.50) & (overlap > 0)
    elif metric == "hungarian":
        # Pure one-to-one assignment: maximize the total number of shared hits
        # without imposing an efficiency or purity threshold.  A zero-overlap
        # assignment is not a reconstructed match.
        valid = overlap > 0
    else:
        raise ValueError(
            f"Unknown matching metric: {metric!r}; expected one of "
            f"{MATCHING_METRICS}"
        )

    # Solve the globally optimal one-to-one assignment over the allowed edges.
    # The existing criteria first apply their threshold mask; ``hungarian``
    # allows every positive-overlap edge.
    score = np.where(valid, overlap, 0)
    try:
        from scipy.optimize import linear_sum_assignment

        rows, cols = linear_sum_assignment(-score)
        return [(int(i), int(j)) for i, j in zip(rows, cols) if valid[i, j]]
    except ImportError as error:
        if metric == "hungarian":
            raise RuntimeError(
                "The hungarian matching criterion requires scipy.optimize."
            ) from error
        candidates = [
            (int(overlap[i, j]), i, j)
            for i, j in zip(*np.nonzero(valid))
        ]
        candidates.sort(reverse=True)
        used_truth, used_reco, matches = set(), set(), []
        for _, i, j in candidates:
            if i not in used_truth and j not in used_reco:
                used_truth.add(i)
                used_reco.add(j)
                matches.append((i, j))
        return matches


def event_metrics(
    labels: np.ndarray,
    truth: np.ndarray,
    metric: str = "double_majority",
    truth_min_hits: int = 3,
) -> Dict[str, float]:
    """Return event-level counts for aggregating efficiency and fake rate."""
    labels = np.asarray(labels).reshape(-1)
    truth = np.asarray(truth).reshape(-1)
    if labels.shape != truth.shape:
        raise ValueError("labels and truth must have identical shapes")
    truth_ids, truth_counts, reco_ids, reco_counts, overlap = _overlap_tables(
        labels, truth, truth_min_hits
    )
    matches = _one_to_one_matches(overlap, truth_counts, reco_counts, metric)
    shared = sum(int(overlap[i, j]) for i, j in matches)
    matched_truth_hits = sum(int(truth_counts[i]) for i, _ in matches)
    matched_reco_hits = sum(int(reco_counts[j]) for _, j in matches)
    counts = {
        "n_truth": int(truth_ids.size),
        "n_reco": int(reco_ids.size),
        "n_matched": int(len(matches)),
        "n_fake": int(reco_ids.size - len(matches)),
        "shared_hits": int(shared),
        "matched_truth_hits": int(matched_truth_hits),
        "matched_reco_hits": int(matched_reco_hits),
    }
    for comparison in MATCHING_COMPARISON_METRICS:
        n_matched = (
            len(matches) if comparison == metric
            else len(_one_to_one_matches(
                overlap, truth_counts, reco_counts, comparison
            ))
        )
        counts[f"n_matched_{comparison}"] = int(n_matched)
        counts[f"n_fake_{comparison}"] = int(reco_ids.size - n_matched)
    return counts


def evaluate_operating_point(
    events: Sequence[Dict[str, np.ndarray]],
    tbeta: float,
    td: float,
    min_hits: int,
    metric: str = "double_majority",
    truth_min_hits: int = 3,
    rejected_seed_policy: str = "discard",
) -> Dict[str, float]:
    totals = {key: 0 for key in TRACKING_COUNT_KEYS}
    for event in events:
        labels = greedy_cluster(
            event["beta"],
            event["coords"],
            tbeta,
            td,
            min_hits,
            rejected_seed_policy=rejected_seed_policy,
        )
        counts = event_metrics(labels, event["truth"], metric, truth_min_hits)
        for key in totals:
            totals[key] += counts[key]

    return tracking_metrics_from_counts(tbeta, td, min_hits, totals)


def tracking_efficiency_binned_counts(
    events: Sequence[Dict[str, np.ndarray]],
    tbeta: float,
    td: float,
    min_hits: int,
    observables: Dict[str, np.ndarray],
    metric: str = "double_majority",
    truth_min_hits: int = 3,
    min_theta: float = 10.0,
    max_theta: float = 170.0,
    gen_status: Sequence[int] = (0, 1),
    rejected_seed_policy: str = "discard",
) -> Tuple[Dict[str, Tuple[np.ndarray, np.ndarray]], Dict[str, int]]:
    """Count total and matched truth tracks for multiple binned observables.

    Each cached event must provide ``particle_info``, keyed by the positive
    event-local truth-cluster ID. Particle records contain ``theta`` (in
    radians), ``gen_status``, and the requested observable values. The track
    matching and selection are evaluated once and shared by all observables.
    """
    if not observables:
        raise ValueError("At least one binned observable is required")
    checked_bins = {}
    counts = {}
    missing_particle_info = {}
    for name, bins in observables.items():
        bins = np.asarray(bins, dtype=np.float64)
        if bins.ndim != 1 or bins.size < 2 or np.any(np.diff(bins) <= 0):
            raise ValueError(
                f"{name} bins must be a strictly increasing one-dimensional array"
            )
        checked_bins[name] = bins
        total = np.zeros(bins.size - 1, dtype=np.int64)
        counts[name] = (total, np.zeros_like(total))
        missing_particle_info[name] = 0
    accepted_status = {int(value) for value in gen_status}

    for event in events:
        labels = greedy_cluster(
            event["beta"],
            event["coords"],
            tbeta,
            td,
            min_hits,
            rejected_seed_policy=rejected_seed_policy,
        )
        truth_ids, truth_counts, _, reco_counts, overlap = _overlap_tables(
            labels, event["truth"], truth_min_hits
        )
        matches = _one_to_one_matches(overlap, truth_counts, reco_counts, metric)
        matched_truth_ids = {int(truth_ids[i]) for i, _ in matches}
        particle_info = event.get("particle_info", {})

        for truth_id in truth_ids:
            info = particle_info.get(int(truth_id))
            if info is None:
                for name in checked_bins:
                    missing_particle_info[name] += 1
                continue

            theta = float(info["theta"])
            status = int(round(float(info["gen_status"])))
            if not np.isfinite(theta):
                continue
            theta_degrees = float(np.degrees(theta))
            if not (min_theta < theta_degrees < max_theta):
                continue
            if status not in accepted_status:
                continue

            for name, bins in checked_bins.items():
                if name not in info:
                    missing_particle_info[name] += 1
                    continue
                value = float(info[name])
                if not np.isfinite(value) or value < 0:
                    continue
                # Include an entry exactly on the upper boundary in the final
                # bin, so the documented 0--2000 mm range is closed at 2000.
                if value == bins[-1]:
                    bin_index = bins.size - 2
                else:
                    bin_index = int(np.searchsorted(bins, value, side="right") - 1)
                total, matched = counts[name]
                if bin_index < 0 or bin_index >= total.size:
                    continue
                total[bin_index] += 1
                if int(truth_id) in matched_truth_ids:
                    matched[bin_index] += 1

    return counts, missing_particle_info


def tracking_efficiency_pt_counts(
    events: Sequence[Dict[str, np.ndarray]],
    tbeta: float,
    td: float,
    min_hits: int,
    bins: np.ndarray = TRACKING_PT_BINS,
    metric: str = "double_majority",
    truth_min_hits: int = 3,
    min_theta: float = 10.0,
    max_theta: float = 170.0,
    gen_status: Sequence[int] = (0, 1),
    rejected_seed_policy: str = "discard",
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Count total and matched selected truth tracks in pT bins."""
    counts, missing = tracking_efficiency_binned_counts(
        events,
        tbeta,
        td,
        min_hits,
        {"pt": bins},
        metric=metric,
        truth_min_hits=truth_min_hits,
        min_theta=min_theta,
        max_theta=max_theta,
        gen_status=gen_status,
        rejected_seed_policy=rejected_seed_policy,
    )
    return counts["pt"][0], counts["pt"][1], missing["pt"]


def matching_comparison_binned_counts(
    events: Sequence[Dict[str, np.ndarray]],
    working_points: Dict[str, Dict[str, float]],
    observables: Dict[str, np.ndarray],
    matching_metrics: Sequence[str] = MATCHING_COMPARISON_METRICS,
    truth_min_hits: int = 3,
    min_theta: float = 10.0,
    max_theta: float = 170.0,
    gen_status: Sequence[int] = (0, 1),
    rejected_seed_policy: str = "discard",
):
    """Build binned counts for each working point and matching criterion.

    The clustering thresholds are held fixed within a working point. Only the
    truth--reconstruction association changes, making the double-majority and
    threshold-free Hungarian curves directly comparable.
    """
    unknown = sorted(set(matching_metrics) - set(MATCHING_METRICS))
    if unknown:
        raise ValueError(f"Unknown matching criteria: {unknown}")
    order = []
    rows = {name: [] for name in observables}
    missing = {name: 0 for name in observables}
    for working_point_name in ("max_efficiency", "pareto_f1"):
        working_point = working_points[working_point_name]
        for metric in matching_metrics:
            counts, local_missing = tracking_efficiency_binned_counts(
                events,
                working_point["tbeta"],
                working_point["td"],
                working_point["min_hits"],
                observables,
                metric=metric,
                truth_min_hits=truth_min_hits,
                min_theta=min_theta,
                max_theta=max_theta,
                gen_status=gen_status,
                rejected_seed_policy=rejected_seed_policy,
            )
            order.append((working_point_name, metric))
            for observable in observables:
                rows[observable].append(counts[observable])
                missing[observable] = max(
                    missing[observable], local_missing[observable]
                )
    return order, rows, missing


def matching_comparison_plot_series(
    order, count_tensor, working_point_name, working_point=None
):
    """Convert reduced comparison counts to the common plotting records.

    When the selected ``working_point`` is given, each record also carries the
    global fake rate of its matching criterion for the plot legend.
    """
    series = []
    for index, (point_name, metric) in enumerate(order):
        if point_name != working_point_name:
            continue
        record = {
            "name": metric,
            "label": MATCHING_LABELS[metric],
            "total": np.asarray(count_tensor[index][0]),
            "matched": np.asarray(count_tensor[index][1]),
        }
        if working_point is not None:
            fake_rate = working_point.get(f"fake_rate_{metric}")
            if fake_rate is not None:
                record["fake_rate"] = float(fake_rate)
        series.append(record)
    return series


def save_tracking_efficiency_pt_plot(
    series: Sequence[Dict],
    output_dir: str,
    filename_stem: str = "tracking_efficiency_vs_pt",
    bins: np.ndarray = TRACKING_PT_BINS,
    min_x: float = 0.1,
    max_x: float = 60.0,
    min_theta: float = 10.0,
    max_theta: float = 170.0,
    gen_status: Sequence[int] = (0, 1),
    truth_min_hits: int = 3,
    _column_prefix: str = "pt",
    _x_label: str = r"$p_T$ [GeV]",
    _log_x: bool = True,
    _text_y: float = 0.5,
    _text_size: float = 17,
):
    """Save binned tracking-efficiency curves for multiple working points.

    Each legend entry reports the total efficiency over the plotted truth
    selection and, when the series provides ``fake_rate``, the global fake rate.
    """
    bins = np.asarray(bins, dtype=np.float64)
    if not series:
        raise ValueError("At least one pT-efficiency series is required")
    processed_series = []
    for item in series:
        total = np.asarray(item["total"], dtype=np.int64)
        matched = np.asarray(item["matched"], dtype=np.int64)
        if total.shape != (bins.size - 1,) or matched.shape != total.shape:
            raise ValueError("pT counts must contain one entry per bin")
        efficiencies = np.divide(
            matched,
            total,
            out=np.full(total.shape, np.nan, dtype=np.float64),
            where=total > 0,
        )
        errors = np.zeros_like(efficiencies)
        populated = total > 0
        errors[populated] = np.sqrt(
            efficiencies[populated]
            * (1.0 - efficiencies[populated])
            / total[populated]
        )
        processed_series.append(
            {
                "name": str(item["name"]),
                "label": str(item["label"]),
                "total": total,
                "matched": matched,
                "efficiencies": efficiencies,
                "errors": errors,
                "fake_rate": item.get("fake_rate"),
            }
        )
    bin_centers = 0.5 * (bins[:-1] + bins[1:])

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    csv_path = output / f"{filename_stem}.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        header = [
            f"{_column_prefix}_low",
            f"{_column_prefix}_high",
            f"{_column_prefix}_center",
        ]
        for item in processed_series:
            header.extend(
                [
                    f"n_truth_{item['name']}",
                    f"n_matched_{item['name']}",
                    f"efficiency_{item['name']}",
                    f"error_{item['name']}",
                ]
            )
        writer.writerow(header)
        for bin_index in range(bins.size - 1):
            row = [bins[bin_index], bins[bin_index + 1], bin_centers[bin_index]]
            for item in processed_series:
                row.extend(
                    [
                        item["total"][bin_index],
                        item["matched"][bin_index],
                        item["efficiencies"][bin_index],
                        item["errors"][bin_index],
                    ]
                )
            writer.writerow(row)

    try:
        import matplotlib
    except ImportError:
        return None

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    style = {
        "text.usetex": False,
        "font.family": "serif",
        "font.size": 25,
        "axes.labelsize": 25,
        "xtick.labelsize": 25,
        "ytick.labelsize": 25,
        "legend.fontsize": 25,
    }
    with plt.rc_context(style):
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111)
        colours = ("#238A8DFF", "#440154FF", "#55C667FF", "#FDE725FF")
        markers = ("s", "o", "^", "D")
        for index, item in enumerate(processed_series):
            colour = colours[index % len(colours)]
            marker = markers[index % len(markers)]
            efficiencies = item["efficiencies"]
            errors = item["errors"]
            valid = np.isfinite(efficiencies)
            summary = (
                f"eff = {item['matched'].sum() / max(item['total'].sum(), 1):.3f}"
            )
            if item["fake_rate"] is not None:
                summary += f", fake = {item['fake_rate']:.3f}"
            ax.scatter(
                bin_centers[valid],
                efficiencies[valid],
                label=f"{item['label']}\n{summary}",
                marker=marker,
                c=[colour for _ in range(int(valid.sum()))],
                s=30,
            )
            yerr_lower = errors[valid]
            yerr_upper = np.minimum(
                efficiencies[valid] + errors[valid], 1.0
            ) - efficiencies[valid]
            ax.errorbar(
                bin_centers[valid],
                efficiencies[valid],
                yerr=[yerr_lower, yerr_upper],
                ecolor=colour,
                linestyle="none",
                capsize=4,
            )

        ax.set_xlabel(_x_label)
        ax.set_ylabel("Tracking efficiency")
        if _log_x:
            ax.set_xscale("log")
        ax.set_xlim([min_x, max_x])
        ax.set_ylim([0.01, 1.01])
        ax.legend(loc="lower right", fontsize=17)
        if _log_x:
            ax.xaxis.set_major_locator(plt.LogLocator(base=10.0, numticks=4))
            ax.xaxis.set_minor_locator(
                plt.LogLocator(base=10.0, subs="auto", numticks=10)
            )
        else:
            ax.xaxis.set_major_locator(plt.MultipleLocator(500.0))
            ax.xaxis.set_minor_locator(plt.MultipleLocator(50.0))
        ax.yaxis.set_major_locator(plt.MultipleLocator(0.1))
        ax.yaxis.set_minor_locator(plt.MultipleLocator(0.1))
        ax.minorticks_on()
        ax.grid(which="major", linestyle=":", linewidth=0.5, color="black")
        ax.grid(which="minor", linestyle=":", linewidth=0.5, color="gray")
        legend = ax.get_legend()
        if legend is not None:
            legend._legend_box.align = "left"

        status_text = ",".join(str(int(value)) for value in gen_status)
        textbox_text = (
            r"$Z/\gamma^* \rightarrow q\bar{q}\ (q = u, d, s)$" "\n"
            r"$\sqrt{s} = m_Z = 91~\mathrm{GeV}$" "\n"
            rf"${min_theta:g}^\circ < \theta < {max_theta:g}^\circ$" "\n"
            rf"$genStatus \in [{status_text}]$" "\n"
            rf"$N_\mathrm{{hits}} \geq {int(truth_min_hits)}$"
        )
        ax.text(
            0.45,
            _text_y,
            textbox_text,
            transform=ax.transAxes,
            fontsize=_text_size,
            verticalalignment="center",
            horizontalalignment="left",
            linespacing=1.4,
            bbox=dict(
                boxstyle="round,pad=0.35", facecolor="none", edgecolor="none"
            ),
        )

        image_path = output / f"{filename_stem}.png"
        fig.savefig(image_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    return image_path


def save_tracking_efficiency_displacement_plot(
    series: Sequence[Dict],
    output_dir: str,
    filename_stem: str = "tracking_efficiency_vs_displacement",
    bins: np.ndarray = TRACKING_DISPLACEMENT_BINS,
    min_x: float = 0.0,
    max_x: float = 2000.0,
    min_theta: float = 10.0,
    max_theta: float = 170.0,
    gen_status: Sequence[int] = (0, 1),
    truth_min_hits: int = 3,
):
    """Save efficiency versus transverse production-vertex displacement."""
    return save_tracking_efficiency_pt_plot(
        series,
        output_dir,
        filename_stem=filename_stem,
        bins=bins,
        min_x=min_x,
        max_x=max_x,
        min_theta=min_theta,
        max_theta=max_theta,
        gen_status=gen_status,
        truth_min_hits=truth_min_hits,
        _column_prefix="displacement_mm",
        _x_label=r"$r_\mathrm{vtx}=\sqrt{x_\mathrm{vtx}^2+y_\mathrm{vtx}^2}$ [mm]",
        _log_x=False,
        # Displacement efficiencies reach down to ~0.5, so keep the physics
        # text compact and between the data and the legend.
        _text_y=0.40,
        _text_size=15,
    )


def tracking_metrics_from_counts(
    tbeta: float,
    td: float,
    min_hits: int,
    totals: Dict[str, int],
) -> Dict[str, float]:
    """Build globally weighted tracking metrics from additive raw counts."""
    totals = {key: int(totals[key]) for key in TRACKING_COUNT_KEYS}
    efficiency = totals["n_matched"] / max(totals["n_truth"], 1)
    fake_rate = totals["n_fake"] / max(totals["n_reco"], 1)
    precision = 1.0 - fake_rate
    f1_denominator = precision + efficiency
    f1 = (
        2.0 * precision * efficiency / f1_denominator
        if f1_denominator > 0.0
        else 0.0
    )
    hit_efficiency = totals["shared_hits"] / max(totals["matched_truth_hits"], 1)
    hit_purity = totals["shared_hits"] / max(totals["matched_reco_hits"], 1)
    comparison_fake_rates = {
        f"fake_rate_{metric}": float(
            totals[f"n_fake_{metric}"] / max(totals["n_reco"], 1)
        )
        for metric in MATCHING_COMPARISON_METRICS
    }
    return {
        "tbeta": float(tbeta),
        "td": float(td),
        "min_hits": int(min_hits),
        "efficiency": float(efficiency),
        "fake_rate": float(fake_rate),
        "precision": float(precision),
        "f1": float(f1),
        "hit_efficiency": float(hit_efficiency),
        "hit_purity": float(hit_purity),
        "physics_score": float(efficiency - fake_rate),
        **comparison_fake_rates,
        **totals,
    }


def run_operating_point_sweep(
    events: Sequence[Dict[str, np.ndarray]],
    tbeta_grid: Iterable[float],
    td_grid: Iterable[float],
    min_hits_grid: Iterable[int],
    metric: str = "double_majority",
    truth_min_hits: int = 3,
    rejected_seed_policy: str = "discard",
) -> List[Dict[str, float]]:
    return [
        evaluate_operating_point(
            events,
            tbeta,
            td,
            min_hits,
            metric=metric,
            truth_min_hits=truth_min_hits,
            rejected_seed_policy=rejected_seed_policy,
        )
        for min_hits in min_hits_grid
        for tbeta in tbeta_grid
        for td in td_grid
    ]


def pareto_front(rows: Sequence[Dict[str, float]]) -> List[Dict[str, float]]:
    """Return points not dominated in efficiency/fake-rate space."""
    return [
        row
        for row in rows
        if not any(
            (
                other["efficiency"] >= row["efficiency"]
                and other["fake_rate"] <= row["fake_rate"]
                and (
                    other["efficiency"] > row["efficiency"]
                    or other["fake_rate"] < row["fake_rate"]
                )
            )
            for other in rows
        )
    ]


def select_tracking_working_points(
    rows: Sequence[Dict[str, float]], fixed_min_hits: int = 3
) -> Dict[str, Dict[str, float]]:
    """Select maximum-efficiency and maximum-F1 Pareto working points."""
    candidates = [row for row in rows if int(row["min_hits"]) == fixed_min_hits]
    if not candidates:
        raise ValueError(f"No sweep rows have min_hits={fixed_min_hits}")

    max_efficiency = max(
        candidates,
        key=lambda row: (row["efficiency"], -row["fake_rate"], row["f1"]),
    )
    frontier = pareto_front(candidates)
    pareto_f1 = max(
        frontier,
        key=lambda row: (row["f1"], row["efficiency"], -row["fake_rate"]),
    )
    return {
        "max_efficiency": dict(max_efficiency),
        "pareto_f1": dict(pareto_f1),
    }


def save_sweep_results(
    rows: Sequence[Dict[str, float]],
    output_dir: str,
    metadata: Dict,
    fixed_min_hits: int = 3,
) -> Dict[str, Dict[str, float]]:
    """Write sweep outputs and return both requested working points."""
    if not rows:
        raise ValueError("Cannot save an empty operating-point sweep")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    working_points = select_tracking_working_points(rows, fixed_min_hits)
    best = working_points["pareto_f1"]

    with (output / "sweep_results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    with (output / "summary.json").open("w") as handle:
        json.dump(
            {"metadata": metadata, "best": best, "working_points": working_points},
            handle,
            indent=2,
            sort_keys=True,
        )

    try:
        import matplotlib
    except ImportError:
        return working_points

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tbetas = sorted({row["tbeta"] for row in rows})
    tds = sorted({row["td"] for row in rows})
    min_hits_values = sorted({row["min_hits"] for row in rows})
    lookup = {(r["min_hits"], r["tbeta"], r["td"]): r for r in rows}
    for min_hits in min_hits_values:
        fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True)
        for axis, key, title, vmin, vmax in (
            (axes[0], "efficiency", "Track efficiency", 0.0, 1.0),
            (axes[1], "fake_rate", "Fake rate", 0.0, 1.0),
            (axes[2], "f1", "Track-level F1", 0.0, 1.0),
        ):
            values = np.array([
                [lookup[(min_hits, tbeta, td)][key] for td in tds]
                for tbeta in tbetas
            ])
            image = axis.imshow(values, origin="lower", aspect="auto", vmin=vmin, vmax=vmax)
            axis.set_xticks(range(len(tds)), [f"{v:g}" for v in tds], rotation=45)
            axis.set_yticks(range(len(tbetas)), [f"{v:g}" for v in tbetas])
            axis.set_xlabel("distance threshold td")
            axis.set_ylabel("beta threshold tbeta")
            axis.set_title(title)
            fig.colorbar(image, ax=axis, shrink=0.85)
        fig.suptitle(f"Validation operating-point sweep, min_hits={min_hits}")
        fig.savefig(output / f"sweep_min_hits_{min_hits}.png", dpi=150)
        plt.close(fig)

    fig, axis = plt.subplots(figsize=(7, 5), constrained_layout=True)
    for min_hits in min_hits_values:
        selected = [row for row in rows if row["min_hits"] == min_hits]
        axis.scatter(
            [row["efficiency"] for row in selected],
            [row["fake_rate"] for row in selected],
            s=28,
            alpha=0.75,
            label=f"min hits {min_hits}",
        )
    fixed_rows = [row for row in rows if int(row["min_hits"]) == fixed_min_hits]
    frontier = sorted(pareto_front(fixed_rows), key=lambda row: row["efficiency"])
    axis.plot(
        [row["efficiency"] for row in frontier],
        [row["fake_rate"] for row in frontier],
        color="black",
        linewidth=1.2,
        alpha=0.7,
        label=f"Pareto front (min hits {fixed_min_hits})",
    )
    max_efficiency = working_points["max_efficiency"]
    axis.scatter(
        [max_efficiency["efficiency"]],
        [max_efficiency["fake_rate"]],
        marker="*",
        s=220,
        color="#238A8DFF",
        label="Maximum efficiency WP",
    )
    axis.scatter(
        [best["efficiency"]],
        [best["fake_rate"]],
        marker="X",
        s=150,
        color="#440154FF",
        label="Pareto maximum-F1 WP",
    )
    axis.set(xlabel="track efficiency", ylabel="fake rate", xlim=(0, 1), ylim=(0, 1))
    axis.grid(alpha=0.25)
    axis.legend()
    fig.savefig(output / "efficiency_fake_pareto.png", dpi=150)
    plt.close(fig)
    return working_points
