"""User-provided greedy clusterer (self-seed variant).

This is the algorithm from the colleague's ``get_clustering`` function
(see the user's notes for the torch version). The differences from our
existing ``eval_sweep_v33.get_clustering_greedy`` are:

  1. **Self-seeding**: a cond-point can only seed a *new* cluster if it
     itself is still unassigned. In the v33 variant, a condpoint that
     was absorbed by an earlier cluster still iterates and can pull
     other unassigned points into a "ghost" cluster centred at its
     absorbed position.

  2. **Re-ranking each step**: after each seed assigns its members, the
     remaining cond-points are re-filtered (only those still
     unassigned) and re-sorted by beta. In the v33 variant the initial
     beta-descending order is fixed for the whole loop.

Both implementations are dimension-agnostic: ``np.linalg.norm(..., axis=-1)``
works for any embedding dimensionality, so the same code that the user
ran on 3D embeddings works on our 5D ``coord_0..coord_4``. The user's
note about needing to adjust for 5D is unnecessary at the algorithm
level - we simply pass the full 5D coords through.

The numpy port mirrors the user's torch version exactly. We use the
``unassigned_mask`` boolean rather than a shrinking index array, but
the resulting cluster labels are identical (modulo dtype).
"""

from __future__ import annotations

import numpy as np


def get_clustering_user(
    betas: np.ndarray,
    X: np.ndarray,
    tbeta: float = 0.7,
    td: float = 0.05,
) -> np.ndarray:
    """Self-seed greedy clustering on any-D embedding.

    Parameters
    ----------
    betas : (N,) float array of sigmoid-squashed beta values per hit.
    X : (N, D) float array of embedding coordinates. ``D`` is arbitrary;
        we use ``D = 5`` for v35.
    tbeta : float, scalar threshold a hit must exceed to be a seed.
    td : float, distance radius around a seed that absorbs unassigned
         hits into the seed's cluster.

    Returns
    -------
    labels : (N,) int32 array. ``-1`` for noise. Otherwise the integer
             label is the index of the seed that created the cluster.
    """
    n_points = betas.shape[0]
    unassigned_mask = np.ones(n_points, dtype=bool)
    labels = -1 * np.ones(n_points, dtype=np.int32)

    while True:
        cand = unassigned_mask & (betas > tbeta)
        if not cand.any():
            break
        cand_idx = np.flatnonzero(cand)
        seed = int(cand_idx[np.argmax(betas[cand_idx])])

        un_idx = np.flatnonzero(unassigned_mask)
        d = np.linalg.norm(X[un_idx] - X[seed], axis=-1)
        absorbed = un_idx[d < td]
        labels[absorbed] = seed
        unassigned_mask[absorbed] = False

    return labels
