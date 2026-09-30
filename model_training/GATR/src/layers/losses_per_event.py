"""Block-sparse, event-local implementation of GGTF's tracking OC loss.

This module is intentionally not wired into training yet.  It provides a
numerically equivalent candidate for the exact ``hgcalimplementation`` recipe
without materialising hit-by-object entries for pairs from different events.
"""

from __future__ import annotations

from typing import Tuple

import torch
from torch_scatter import scatter_add, scatter_max

from src.layers.object_cond import calculate_delta_MC


def object_condensation_loss_tracking_per_event(
    batch,
    pred: torch.Tensor,
    y: torch.Tensor,
    return_resolution: bool = False,
    clust_loss_only: bool = True,
    add_energy_loss: bool = False,
    calc_e_frac_loss: bool = False,
    q_min: float = 0.1,
    frac_clustering_loss: float = 0.1,
    attr_weight: float = 1.0,
    repul_weight: float = 1.0,
    fill_loss_weight: float = 1.0,
    use_average_cc_pos: float = 0.0,
    loss_type: str = "hgcalimplementation",
    output_dim: int = 4,
    clust_space_norm: str = "none",
    tracking: bool = False,
) -> Tuple[torch.Tensor, tuple]:
    """Evaluate the active GGTF loss using only within-event dense blocks.

    Arguments intentionally mirror ``object_condensation_loss_tracking``.
    Options unused by the active GGTF tracking recipe are accepted for API
    parity but unsupported alternatives fail closed.
    """
    del (
        add_energy_loss,
        calc_e_frac_loss,
        frac_clustering_loss,
        fill_loss_weight,
        clust_space_norm,
        tracking,
    )
    if return_resolution:
        raise NotImplementedError("regression resolution is outside the tracking recipe")
    if not clust_loss_only or output_dim != 4:
        raise NotImplementedError("only the active four-output clustering recipe is supported")
    if loss_type != "hgcalimplementation":
        raise NotImplementedError("only GGTF's active hgcalimplementation loss is supported")
    if use_average_cc_pos != 0.0:
        raise NotImplementedError("average condensation coordinates are not active")

    coords = pred[:, :3]
    beta = torch.sigmoid(pred[:, 3])
    cluster_ids = batch.ndata["particle_number"].view(-1).long()
    event_sizes = [int(size) for size in batch.batch_num_nodes().tolist()]
    if sum(event_sizes) != pred.shape[0]:
        raise ValueError("graph node counts do not match predictions")

    attractive_per_object = []
    repulsive_per_object = []
    beta_signal_per_object = []
    beta_noise_per_event = []
    object_counts = []
    node_offset = 0

    for event_size in event_sizes:
        event_slice = slice(node_offset, node_offset + event_size)
        event_coords = coords[event_slice]
        event_beta = beta[event_slice]
        event_ids = cluster_ids[event_slice]
        node_offset += event_size

        n_objects = int(event_ids.max().item())
        if n_objects < 1:
            raise ValueError("the active loss requires at least one signal object per event")
        object_counts.append(n_objects)
        is_signal = event_ids != 0
        object_index = event_ids[is_signal] - 1
        expected = torch.arange(n_objects, device=pred.device)
        if not torch.equal(torch.unique(object_index), expected):
            raise ValueError("signal cluster IDs must be dense and one-indexed")

        q = (event_beta.clip(0.0, 1 - 1e-4).arctanh() / 1.01) ** 2 + q_min
        q_signal = q[is_signal]
        q_alpha, index_alpha = scatter_max(q_signal, object_index, dim=0)
        x_alpha = event_coords[is_signal][index_alpha]
        beta_alpha = event_beta[is_signal][index_alpha]
        n_hits_per_object = scatter_add(
            torch.ones_like(object_index), object_index, dim=0, dim_size=n_objects
        )

        # Attraction only uses each signal hit's own object.  The original
        # computes the full matrix and zeros all other columns afterward.
        own_sq_distance = torch.square(
            event_coords[is_signal] - x_alpha[object_index]
        ).sum(dim=-1)
        own_potential = torch.log(
            torch.exp(pred.new_tensor(1.0)) * own_sq_distance / 2 + 1
        )
        attractive = scatter_add(
            q_signal * q_alpha[object_index] * own_potential,
            object_index,
            dim=0,
            dim_size=n_objects,
        )
        attractive_per_object.append(
            attractive / (n_hits_per_object.to(attractive.dtype) + 1e-3)
        )

        # Repulsion genuinely needs all hit/object pairs, but only within this
        # event.  This is the block that replaces the global dense matrix.
        sq_distance = torch.square(
            event_coords.unsqueeze(1) - x_alpha.unsqueeze(0)
        ).sum(dim=-1)
        anti_connectivity = torch.ones_like(sq_distance)
        anti_connectivity[is_signal, object_index] = 0
        repulsive = (
            q.unsqueeze(1)
            * q_alpha.unsqueeze(0)
            * torch.exp(-sq_distance / 2)
            * anti_connectivity
        ).sum(dim=0)
        n_repulsive_terms = anti_connectivity.sum(dim=0).clamp(min=1)
        repulsive_per_object.append(repulsive / n_repulsive_terms)

        beta_sum = scatter_add(
            event_beta[is_signal],
            object_index,
            dim=0,
            dim_size=n_objects,
        )
        beta_signal_per_object.append(
            1 - beta_alpha + 1 - torch.clip(beta_sum, 0, 1)
        )
        noise_beta = event_beta[~is_signal]
        beta_noise_per_event.append(
            noise_beta.mean() if noise_beta.numel() else beta.new_zeros(())
        )

    attractive_all = torch.cat(attractive_per_object)
    repulsive_all = torch.cat(repulsive_per_object)
    beta_signal_all = torch.cat(beta_signal_per_object)
    if attractive_all.numel() != sum(object_counts):
        raise RuntimeError("internal object count mismatch")

    # GGTF weights every object by inverse nearest truth-track separation.
    track_weight = 1 / (calculate_delta_MC(y, batch) + 0.001)
    if track_weight.numel() != repulsive_all.numel():
        raise ValueError("truth rows and signal objects do not align")

    loss_attractive = attractive_all.mean()
    loss_repulsive = (repulsive_all * track_weight).sum() / track_weight.sum()
    loss_beta_signal = beta_signal_all.mean()
    loss_beta_noise = torch.stack(beta_noise_per_event).mean()
    loss_v = attr_weight * loss_attractive + repul_weight * loss_repulsive
    loss_beta = loss_beta_noise + loss_beta_signal
    components = (
        loss_v,
        loss_beta,
        loss_attractive,
        loss_repulsive,
        loss_beta_signal,
        loss_beta_noise,
    )
    return loss_v + loss_beta, components
