"""Lightweight training-time clustering metrics shared by the algebra arms."""

from collections import Counter

import numpy as np
import torch

from src.layers.object_cond import get_clustering_np


def compute_batch_metrics_greedy(
    coords: torch.Tensor,
    beta_logits: torch.Tensor,
    mc_index: torch.Tensor,
    is_secondary: torch.Tensor,
    seq_lens,
    *,
    tbeta: float = 0.1,
    td: float = 0.2,
    noise_index: int = 0,
):
    """Mirror CIRCE's online strict50 metric at one fixed operating point.

    ``strict50`` is the mean per-event fraction of truth tracks whose
    best-overlap predicted cluster has purity > 0.75 and hit efficiency >= 0.5.
    Noise and secondary hits are excluded from this lightweight online metric,
    matching CIRCE's training validator. Full FCC evaluation remains the
    checkpoint-selection metric.
    """
    coords_np = coords.detach().float().cpu().numpy()
    beta = torch.sigmoid(beta_logits.squeeze(-1)).detach().float().cpu().numpy()
    mc_np = mc_index.detach().cpu().numpy()
    secondary_np = is_secondary.detach().bool().cpu().numpy()

    event_loose = []
    event_strict50 = []
    offset = 0
    for n_hits in seq_lens:
        n_hits = int(n_hits)
        sl = slice(offset, offset + n_hits)
        offset += n_hits

        signal = (~secondary_np[sl]) & (mc_np[sl] != noise_index)
        event_coords = coords_np[sl][signal]
        event_beta = beta[sl][signal]
        event_truth = mc_np[sl][signal]
        if len(event_coords) == 0:
            continue

        pred = get_clustering_np(
            event_beta, event_coords, tbeta=tbeta, td=td
        )
        truth_ids = np.unique(event_truth[event_truth != noise_index])
        n_matchable = 0
        loose = 0
        strict50 = 0
        for truth_id in truth_ids:
            truth_mask = event_truth == truth_id
            n_true = int(truth_mask.sum())
            if n_true < 2:
                continue
            n_matchable += 1

            assigned = pred[truth_mask]
            assigned = assigned[assigned >= 0]
            if len(assigned) == 0:
                continue
            best_label, best_match = Counter(assigned.tolist()).most_common(1)[0]
            efficiency = best_match / n_true
            purity = best_match / int((pred == best_label).sum())
            if purity > 0.75:
                loose += 1
                if efficiency >= 0.5:
                    strict50 += 1

        if n_matchable:
            event_loose.append(loose / n_matchable)
            event_strict50.append(strict50 / n_matchable)

    return {
        "match_rate": float(np.mean(event_loose)) if event_loose else 0.0,
        "match_rate_strict50": (
            float(np.mean(event_strict50)) if event_strict50 else 0.0
        ),
    }
