"""Physics metrics for SSM — independent implementation, GATR_CIRCE_LOSS-compatible keys.

Validation-time pipeline (per val epoch):
  greedy clustering at each (tbeta, td) working point
  -> one-to-one assignment vs truth particles (double-majority and Hungarian)
  -> tracking efficiency / fake rate; F1 over a sweep grid
  -> selected working points: pareto_f1 and max_efficiency
  -> pT-, displacement- and theta-binned efficiency at both working points
     for both matching criteria.

Conventions mirror shared_training/tracking_metrics (used by GATR_CIRCE_LOSS
and CIRCE), so the three trainings produce the same plots:
  - all truth particles enter the matching; the scalar efficiency/fake-rate
    working-point metrics use the full sample with no truth cuts (the fake
    rate cannot take truth-side cuts, so its paired efficiency takes none);
  - the curve selections (N_hits >= min_hits, 10 deg < theta < 170 deg,
    genStatus in {0, 1}) are post-reconstruction cuts applied only when
    filling the binned counts;
  - clustering: beta-ordered greedy OC clustering; a candidate with fewer
    than ``cluster_min_hits`` hits rejects only the seed ("discard" policy);
  - double majority: cluster and particle share >= purity=50% each way,
    one-to-one via global assignment; hungarian: threshold-free one-to-one
    assignment maximizing shared hits over positive-overlap edges;
  - fake rate: clusters w/o matched truth partner / total clusters;
  - binning: value == last edge falls into the last bin, out-of-range values
    are skipped (no clamping).
"""

from __future__ import annotations

import numpy as np
import torch

# GATR sweep defaults (slurmTrain.sh SWEEP_TBETA_GRID / SWEEP_TD_GRID)
DEFAULT_TBETA_GRID = [0.05, 0.1, 0.15, 0.2, 0.35, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
DEFAULT_TD_GRID = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.55, 0.6]
# Identical to shared_training/tracking_metrics.py constants.
PT_BINS = list(np.exp(np.arange(np.log(0.1), np.log(60.0), 0.2)))
DISP_BINS_MM = list(np.arange(0.0, 2000.0 + 20.0, 20.0))
THETA_BINS_DEG = list(np.arange(10.0, 170.0 + 5.0, 5.0))
MIN_THETA_DEG = 10.0
MAX_THETA_DEG = 170.0
GEN_STATUS_OK = (0, 1)

MATCHING_LABELS = {
    "double_majority": r"Double majority ($\epsilon,p>0.5$)",
    "hungarian": "Hungarian 1-to-1",
}


def greedy_cluster(emb: torch.Tensor, beta: torch.Tensor, tbeta: float, td: float,
                   min_hits: int = 3):
    """Cluster labels per hit (>=1), 0 = unclustered. Single event, torch both dirs.

    Seeds in descending beta order; a candidate with fewer than ``min_hits``
    hits rejects only the seed ("discard" policy), matching the
    shared_training clustering used by GATR_CIRCE_LOSS and CIRCE.
    """
    cand = torch.nonzero(beta > tbeta).squeeze(-1)
    labels = torch.zeros(emb.shape[0], dtype=torch.long)
    if cand.numel() == 0:
        return labels
    e = emb.float().cpu().numpy()
    cand_list = cand[torch.argsort(beta[cand], descending=True)].cpu().numpy().tolist()
    td2 = td * td
    assigned = np.zeros(len(e), dtype=bool)
    cid = 0
    for seed in cand_list:
        if assigned[seed]:
            continue
        d2 = ((e - e[seed]) ** 2).sum(axis=1)
        take = (d2 < td2) & ~assigned
        take[seed] = True
        if take.sum() < min_hits:
            assigned[seed] = True
            continue
        cid += 1
        labels[torch.from_numpy(take)] = cid
        assigned |= take
    return labels


def _dm_table(labels: torch.Tensor, mc: torch.Tensor):
    """One event: particle/cluster shared-hit table.

    Returns dict with per-particle {mc_id, n_hits, best_cluster, frac_own, frac_cluster},
    and per-cluster totals.
    """
    sig = mc > 0
    parts, p_inv, p_counts = mc[sig].unique(return_inverse=True, return_counts=True)
    lc = labels[sig]
    n_c = int(labels.max().item())
    if n_c == 0 or len(parts) == 0:
        return {"parts": parts, "p_counts": p_counts, "M": np.zeros((len(parts), 1), dtype=np.int64),
                "cl_counts": np.zeros(1, dtype=np.int64), "n_c": 0}
    m = torch.zeros(len(parts), n_c + 1, dtype=torch.long)
    m.index_put_((p_inv, lc), torch.ones(len(p_inv), dtype=torch.long), accumulate=True)
    cl_counts = torch.bincount(labels.clamp(min=0), minlength=n_c + 1)
    return {"parts": parts, "p_counts": p_counts,
            "M": m.numpy(), "cl_counts": cl_counts.numpy(), "n_c": n_c}


def _one_to_one_matches(overlap, truth_counts, cl_counts, metric):
    """Global one-to-one assignment over allowed edges (scipy Hungarian).

    double_majority: >=50% shared hits both ways; hungarian: threshold-free,
    any positive overlap is an allowed edge. Mirrors shared_training's
    _one_to_one_matches; assignment returns (particle_idx, cluster_idx).
    """
    eff = overlap / np.clip(truth_counts[:, None], 1, None)
    pur = overlap / np.clip(cl_counts[None, :], 1, None)
    if metric == "double_majority":
        valid = (pur >= 0.50) & (eff >= 0.50) & (overlap > 0)
    elif metric == "hungarian":
        valid = overlap > 0
    else:
        raise ValueError(f"unknown matching metric: {metric!r}")
    if not valid.any():
        return []
    try:
        from scipy.optimize import linear_sum_assignment
    except ImportError as error:
        raise RuntimeError(
            "matching requires scipy.optimize (install scipy)"
        ) from error
    score = np.where(valid, overlap, 0)
    rows, cols = linear_sum_assignment(-score)
    return [(int(i), int(j)) for i, j in zip(rows, cols) if valid[i, j]]


def _bin_index(bins, value):
    """Shared binning rule: exact upper edge -> last bin; out-of-range -> None."""
    n = len(bins)
    if value == bins[-1]:
        return n - 2
    b = int(np.searchsorted(bins, value, side="right")) - 1
    return b if 0 <= b < n - 1 else None


class SweepAccumulator:
    """Accumulate metrics across events for a working-point sweep."""

    def __init__(self, min_hits: int = 3, purity: float = 0.5):
        self.min_hits = min_hits
        self.purity = purity
        self._acc = {}

    def update_event(self, emb, beta, mc, tbeta_grid, td_grid,
                     pt_map=None, disp_map=None, theta_map=None, genstatus_map=None):
        have_maps = pt_map is not None and disp_map is not None
        for tb in tbeta_grid:
            for td in td_grid:
                labels = greedy_cluster(emb, beta, tb, td, min_hits=self.min_hits)
                tab = _dm_table(labels, mc)
                M, cl, parts, pc = tab["M"], tab["cl_counts"], tab["parts"], tab["p_counts"]
                n_c = tab["n_c"]
                sel = (pc >= self.min_hits).numpy()
                matches = {}
                if n_c:
                    for metric in ("double_majority", "hungarian"):
                        matches[metric] = set(_one_to_one_matches(
                            M[:, 1:], pc.numpy(), cl[1:], metric))
                else:
                    matches = {"double_majority": set(), "hungarian": set()}
                used_dm = {j for _, j in matches["double_majority"]}
                used_h = {j for _, j in matches["hungarian"]}
                # Scalar efficiency/fake rate use the full truth/cluster
                # sample with no cuts: fake rate cannot take truth-side
                # cuts, so its paired efficiency takes none either.
                row = {
                    "tbeta": tb, "td": td,
                    "n_sel": int(len(parts)),
                    "n_matched": len(matches["double_majority"]),
                    "n_clusters": n_c, "n_fake": n_c - len(used_dm),
                    "n_matched_hungarian": len(matches["hungarian"]),
                    "n_fake_hungarian": n_c - len(used_h),
                    "sel_pt": np.zeros(len(PT_BINS) - 1, dtype=np.int64),
                    "mat_pt": np.zeros(len(PT_BINS) - 1, dtype=np.int64),
                    "mat_pt_hungarian": np.zeros(len(PT_BINS) - 1, dtype=np.int64),
                    "sel_dp": np.zeros(len(DISP_BINS_MM) - 1, dtype=np.int64),
                    "mat_dp": np.zeros(len(DISP_BINS_MM) - 1, dtype=np.int64),
                    "mat_dp_hungarian": np.zeros(len(DISP_BINS_MM) - 1, dtype=np.int64),
                    "sel_th": np.zeros(len(THETA_BINS_DEG) - 1, dtype=np.int64),
                    "mat_th": np.zeros(len(THETA_BINS_DEG) - 1, dtype=np.int64),
                    "mat_th_hungarian": np.zeros(len(THETA_BINS_DEG) - 1, dtype=np.int64),
                }
                if have_maps:
                    matched_dm = {i for i, _ in matches["double_majority"]}
                    matched_h = {i for i, _ in matches["hungarian"]}
                    truth_ids = parts.numpy()
                    for pi in np.where(sel)[0]:
                        mc_id = int(truth_ids[pi])
                        # Curve selection cuts (shared_training convention):
                        # theta window and genStatus, applied at counting time.
                        if theta_map is not None:
                            th_rad = theta_map.get(mc_id)
                            if th_rad is None or not np.isfinite(th_rad):
                                continue
                            th_deg = float(np.degrees(th_rad))
                            if not (MIN_THETA_DEG < th_deg < MAX_THETA_DEG):
                                continue
                        else:
                            th_deg = None
                        if genstatus_map is not None:
                            gs = genstatus_map.get(mc_id)
                            if gs is None or gs not in GEN_STATUS_OK:
                                continue
                        for value, bins, sel_key in (
                                (pt_map.get(mc_id), PT_BINS, "sel_pt"),
                                (disp_map.get(mc_id), DISP_BINS_MM, "sel_dp"),
                                (th_deg, THETA_BINS_DEG, "sel_th")):
                            if value is None or not np.isfinite(value) or value < 0:
                                continue
                            b = _bin_index(bins, float(value))
                            if b is None:
                                continue
                            mat_key = "mat" + sel_key[3:]
                            row[sel_key][b] += 1
                            if pi in matched_dm:
                                row[mat_key][b] += 1
                            if pi in matched_h:
                                row[mat_key + "_hungarian"][b] += 1
                key = (tb, td)
                acc = self._acc
                if key not in acc:
                    acc[key] = {k: (v.copy() if isinstance(v, np.ndarray) else v)
                                for k, v in row.items() if k not in ("tbeta", "td")}
                else:
                    for k, v in row.items():
                        if k in ("tbeta", "td"):
                            continue
                        acc[key][k] = acc[key][k] + v

    def working_points(self):
        """Return {pareto_f1: row, max_efficiency: row} over accumulated values."""
        best_f1, best_eff = None, None
        for (tb, td), a in self._acc.items():
            eff = a["n_matched"] / a["n_sel"] if a["n_sel"] else 0.0
            fake = a["n_fake"] / a["n_clusters"] if a["n_clusters"] else 0.0
            purity_ = 1.0 - fake
            f1 = (2 * eff * purity_ / (eff + purity_)) if (eff + purity_) > 0 else 0.0
            fake_h = (a["n_fake_hungarian"] / a["n_clusters"]
                      if a["n_clusters"] else 0.0)
            eff_h = a["n_matched_hungarian"] / a["n_sel"] if a["n_sel"] else 0.0
            row = {**a, "tbeta": tb, "td": td, "efficiency": eff, "fake_rate": fake,
                   "efficiency_hungarian": eff_h, "fake_rate_hungarian": fake_h, "f1": f1}
            if best_f1 is None or f1 > best_f1["f1"]:
                best_f1 = row
            if best_eff is None or eff > best_eff["efficiency"]:
                best_eff = row
        return {"pareto_f1": best_f1, "max_efficiency": best_eff}
