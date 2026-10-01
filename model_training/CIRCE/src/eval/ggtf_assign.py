"""GGTF's candidate-to-particle assignment, run exactly rather than emulated.

From `Tracking_DC/src/layers/inference_oc_tracks.py`:

    iou_threshold = 0.02
    iou_matrix_num[iou_matrix_num < iou_threshold] = 0
    row_ind, col_ind = linear_sum_assignment(-iou_matrix_num)
    mask_matching_matrix = iou_matrix_num[row_ind, col_ind] > 0

So: an IoU matrix between candidates and true particles, entries under 0.02
zeroed, an optimal one-to-one assignment maximising total IoU, and pairs that
survive with non-zero IoU are the matches. Everything else is, in their
accounting, a fake -- including a perfectly pure second candidate on a particle
some other candidate already claimed, which is how their fake bucket comes to
contain our clones.

This has to run where the per-hit labels still exist. Reconstructing it later
from the cached per-candidate aggregates forces a greedy rule, and
`validate_hungarian.py` measures that greedy overstates the fake rate by about
12 points at their > 10 hit cut, so the emulation is not good enough to quote.
"""

import os

import numpy as np
from scipy.optimize import linear_sum_assignment

IOU_THRESHOLD = 0.02

# The smallest mc_index that counts as a real particle. 1 excludes index 0, which is what
# every number reported before 2026-08-05 did; 0 includes it, which is correct, because
# index 0 is a real particle and not noise (FINDINGS.md M20/M21). Switched by
# FIX_PARTICLE_ZERO=1 in the environment rather than a parameter, because this function is
# called from several scripts and the two conventions must never be mixed within one report.
# No hit carries a negative index, so 0 admits exactly the particle-0 tracks and nothing else.
MIN_SIGNAL_MC = 0 if os.environ.get("FIX_PARTICLE_ZERO") == "1" else 1


def contingency(labels, mc):
    """Hit-overlap table between candidates and true particles for one event.

    Rows are candidate labels in ascending order, columns are true particle ids; ids below
    `MIN_SIGNAL_MC` are excluded. Also returns each side's own hit count.

    Which particles count as targets is decided by `MIN_SIGNAL_MC` above. Under the default
    the exclusion is wrong in one specific way, and this is where it bites hardest: dropping
    particle 0 means failing to find that track is free, while finding it yields a candidate
    with no target to match, counted as a fake.
    """
    cl = np.unique(labels[labels >= 0])
    pr = np.unique(mc[mc >= MIN_SIGNAL_MC])
    if len(cl) == 0 or len(pr) == 0:
        return (cl, pr, np.zeros((len(cl), len(pr)), dtype=np.int64),
                np.zeros(len(cl), dtype=np.int64), np.zeros(len(pr), dtype=np.int64))

    cl_pos = np.searchsorted(cl, labels)
    pr_pos = np.searchsorted(pr, mc)
    sel = (labels >= 0) & (mc >= MIN_SIGNAL_MC)
    inter = np.zeros((len(cl), len(pr)), dtype=np.int64)
    np.add.at(inter, (cl_pos[sel], pr_pos[sel]), 1)

    n_cl = np.bincount(cl_pos[labels >= 0], minlength=len(cl))
    n_pr = np.bincount(pr_pos[mc >= MIN_SIGNAL_MC], minlength=len(pr))
    return cl, pr, inter, n_cl, n_pr


def iou_matrix(labels, mc):
    """Candidate-by-particle IoU matrix, as in their construction."""
    cl, pr, inter, n_cl, n_pr = contingency(labels, mc)
    if not inter.size:
        return cl, pr, np.zeros((len(cl), len(pr)), dtype=np.float64)
    union = n_cl[:, None] + n_pr[None, :] - inter
    return cl, pr, inter / np.maximum(union, 1)


def ggtf_assignment(labels, mc, iou_threshold=IOU_THRESHOLD):
    """Per-candidate assignment flag and the IoU it was assigned at.

    Returns (cluster_labels, assigned, iou_of_pair). `iou_of_pair` is the IoU
    with the particle the assignment gave the candidate, or its best available
    IoU when it went unassigned, so a candidate that lost a contested particle
    is distinguishable from one that never had a plausible partner.
    """
    cl, pr, iou = iou_matrix(labels, mc)
    assigned = np.zeros(len(cl), dtype=bool)
    pair_iou = iou.max(axis=1) if iou.size else np.zeros(len(cl))
    if not iou.size:
        return cl, assigned, pair_iou

    gated = np.where(iou < iou_threshold, 0.0, iou)
    row, col = linear_sum_assignment(-gated)
    keep = gated[row, col] > 0
    assigned[row[keep]] = True
    pair_iou[row[keep]] = gated[row[keep], col[keep]]
    return cl, assigned, pair_iou
