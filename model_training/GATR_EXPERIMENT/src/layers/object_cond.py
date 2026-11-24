from typing import Optional, Tuple, Union
import numpy as np
import torch
from torch_scatter import scatter_max, scatter_add, scatter_mean
import dgl
import sys

from src.utils.track_weighting import pt_track_weights, weighted_mean

def safe_index(arr, index):
    # One-hot index (or zero if it's not in the array)
    if index not in arr:
        return 0
    else:
        return arr.index(index) + 1


def assert_no_nans(x):
    """
    Raises AssertionError if there is a nan in the tensor
    """
    if torch.isnan(x).any():
        print(x)
    assert not torch.isnan(x).any()


def second_highest_beta_loss(
    signal_beta: torch.Tensor,
    object_index: torch.Tensor,
    index_alpha: torch.Tensor,
    n_hits_per_object: torch.Tensor,
    n_objects: int,
    object_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Mean second-highest signal beta per truth object.

    A single-hit object has no competing condensation point and therefore
    contributes zero.  This reduction is called only when its loss weight is
    non-zero.
    """
    second_candidates = signal_beta.clone()
    second_candidates[index_alpha] = -1.0
    second_beta = scatter_max(
        second_candidates,
        object_index,
        dim=0,
        dim_size=n_objects,
    )[0]
    second_beta = torch.where(
        n_hits_per_object > 1,
        second_beta,
        torch.zeros_like(second_beta),
    )
    return weighted_mean(second_beta, object_weights)


# FIXME: Use a logger instead of this
DEBUG = False


def calculate_weights_for_class_hit_type_batch(g, object_index, is_sig):
    vx_hits = torch.sum(
    1 * (g.ndata["hit_type"] == 1)[is_sig]
    )
    dc_hits = torch.sum(
        1 * (g.ndata["hit_type"] ==0)[is_sig]
    ) 
    
    
    # muon_hits = torch.sum(
    #     1 * (g.ndata["hit_type"] == 4)[is_sig]
    # )
    number_classes = 1*(vx_hits>0)+1*(dc_hits>0)
    weights = 1.0 * torch.ones_like(g.ndata["hit_type"][is_sig])
    weight_vx_per_object = 1.0 * torch.ones_like(vx_hits)
    weight_dc_per_object = 1.0 *  torch.ones_like(vx_hits)
    weight_vx_per_object = (vx_hits + dc_hits) / (
        number_classes * vx_hits
    )
    weight_dc_per_object = (vx_hits + dc_hits) / (
        number_classes* dc_hits
    )

    # weight_muon_per_object = (ecal_hits + hcal_hits+track_hits) / (
    #     number_classes* muon_hits
    # )
    weights[g.ndata["hit_type"][is_sig] == 1] = weight_vx_per_object.view(-1)
    weights[g.ndata["hit_type"][is_sig] == 0] = weight_dc_per_object.view(-1)
    weights = weights

    return weights


def debug(*args, **kwargs):
    if DEBUG:
        print(*args, **kwargs)


def calc_LV_Lbeta(
    original_coords,
    g,
    y,
    distance_threshold,
    energy_correction,
    beta: torch.Tensor,
    cluster_space_coords: torch.Tensor,  # Predicted by model
    cluster_index_per_event: torch.Tensor,  # Truth hit->cluster index
    batch: torch.Tensor,
    predicted_pid=None,  # predicted PID embeddings - will be aggregated by summing up the clusters and applying the post_pid_pool_module MLP afterwards
    post_pid_pool_module=None,  # MLP to apply to the pooled embeddings to get the PID predictions torch.nn.Module
    # From here on just parameters
    qmin: float = 0.1,
    s_B: float = 1.0,
    noise_cluster_index: int = 0,  # cluster_index entries with this value are noise/noise
    beta_stabilizing="soft_q_scaling",
    huberize_norm_for_V_attractive=False,
    beta_term_option="paper",
    return_components=False,
    return_regression_resolution=False,
    clust_space_dim=3,
    frac_combinations=0,  # fraction of the all possible pairs to be used for the clustering loss
    attr_weight=1.0,
    repul_weight=1.0,
    fill_loss_weight=0.0,
    use_average_cc_pos=0.0,
    beta_suppress_weight=0.0,
    beta_second_weight=0.0,
    var_weight=0.0,
    hard_negative_weight=1.0,
    hard_negative_max_weight=100.0,
    pt_track_weighting=False,
    pt_track_weight_bin_edges=(0.4, 0.9, 5.0),
    pt_track_weight_bin_weights=(1.5, 1.2, 0.75, 2.0),
    loss_type="hgcalimplementation",
    hit_energies=None,
    tracking=False,
    dis=False,
    CLD=False,
) -> Union[Tuple[torch.Tensor, torch.Tensor], dict]:
    """
    Calculates the L_V and L_beta object condensation losses.
    Concepts:
    - A hit belongs to exactly one cluster (cluster_index_per_event is (n_hits,)),
      and to exactly one event (batch is (n_hits,))
    - A cluster index of `noise_cluster_index` means the cluster is a noise cluster.
      There is typically one noise cluster per event. Any hit in a noise cluster
      is a 'noise hit'. A hit in an object is called a 'signal hit' for lack of a
      better term.
    - An 'object' is a cluster that is *not* a noise cluster.
    beta_stabilizing: Choices are ['paper', 'clip', 'soft_q_scaling']:
        paper: beta is sigmoid(model_output), q = beta.arctanh()**2 + qmin
        clip:  beta is clipped to 1-1e-4, q = beta.arctanh()**2 + qmin
        soft_q_scaling: beta is sigmoid(model_output), q = (clip(beta)/1.002).arctanh()**2 + qmin
    huberize_norm_for_V_attractive: Huberizes the norms when used in the attractive potential
    beta_term_option: Choices are ['paper', 'short-range-potential']:
        Choosing 'short-range-potential' introduces a short range potential around high
        beta points, acting like V_attractive.
    Note this function has modifications w.r.t. the implementation in 2002.03605:
    - The norms for V_repulsive are now Gaussian (instead of linear hinge)
    """
    # remove dummy rows added for dataloader #TODO think of better way to do this
    device = beta.device
    # Most loss tensors follow Lightning's precision policy.  The charge
    # transformation is handled in float32 below because its 1e-4 clamp is not
    # representable near one in bf16/float16.
    somethingIsNaN = False
    if torch.isnan(beta).any():
        print("There are nans in beta! L198", len(beta[torch.isnan(beta)]))
        somethingIsNaN = True
    beta = torch.nan_to_num(beta, nan=0.0, posinf=1.0, neginf=0.0)
    assert_no_nans(beta)
    # ________________________________

    # Calculate a bunch of needed counts and indices locally

    # cluster_index: unique index over events
    # E.g. cluster_index_per_event=[ 0, 0, 1, 2, 0, 0, 1], batch=[0, 0, 0, 0, 1, 1, 1]
    #      -> cluster_index=[ 0, 0, 1, 2, 3, 3, 4 ]
    
    # print("[DEBUG] cluster_index_per_event:", cluster_index_per_event)
    cluster_index, n_clusters_per_event = batch_cluster_indices(cluster_index_per_event, batch)
    # print("[DEBUG] cluster_index:", cluster_index)
    # print("[DEBUG] n_clusters_per_event:", n_clusters_per_event)
    # Cluster space coordinates
    n_hits, cluster_space_dim = cluster_space_coords.size()
    # print("[DEBUG] cluster_space_coords shape:", cluster_space_coords.shape)
    batch_size = n_clusters_per_event.numel()
    # print("[DEBUG] batch size:", batch_size)

    n_hits_per_event = scatter_count(batch)
    # print("[DEBUG] n_hits_per_event:", n_hits_per_event)

    # Index mapping clusters to events
    batch_cluster = scatter_counts_to_indices(n_clusters_per_event)
    n_clusters = batch_cluster.numel()
    # print("[DEBUG] total number of clusters:", n_clusters)
    # print("[DEBUG] batch_cluster shape:", batch_cluster.shape)
    # print("[DEBUG] batch_cluster sample:", batch_cluster)
    


    # Signal / noise masks
    is_noise = cluster_index_per_event == noise_cluster_index
    is_sig = ~is_noise
    # print("[DEBUG] number of noise hits:", is_noise.sum())
    # print("[DEBUG] number of signal hits:", is_sig.sum())

    n_hits_sig = is_sig.sum()
    n_sig_hits_per_event = scatter_add(
        torch.ones_like(batch[is_sig], dtype=torch.long),
        batch[is_sig],
        dim=0,
        dim_size=batch_size,
    )
    # print("[DEBUG] n_hits_sig:", n_hits_sig)
    # print("[DEBUG] n_sig_hits_per_event:", n_sig_hits_per_event)

    # Object / noise cluster masks
    is_object = scatter_max(is_sig.long(), cluster_index)[0].bool()
    is_noise_cluster = ~is_object
    # print("[DEBUG] number of objects:", is_object.sum())
    # print("[DEBUG] number of noise clusters:", is_noise_cluster.sum())

    n_objects = int(is_object.sum().item())

    # The noise-beta term is well-defined even when the batch has no truth
    # objects.  Compute it before constructing object indices so an all-noise
    # batch can return cleanly without reductions such as max() on empty
    # tensors.
    n_noise_hits_per_event = scatter_add(
        torch.ones_like(batch[is_noise], dtype=torch.long),
        batch[is_noise],
        dim=0,
        dim_size=batch_size,
    ).clamp(min=1)
    beta_noise_per_event = scatter_add(
        beta[is_noise],
        batch[is_noise],
        dim=0,
        dim_size=batch_size,
    )
    L_beta_noise = (
        s_B * (beta_noise_per_event / n_noise_hits_per_event).sum() / batch_size
    )

    zero_embedding_loss = cluster_space_coords.sum() * 0.0
    zero_beta_loss = beta.sum() * 0.0
    if n_objects == 0:
        L_V_attractive = zero_embedding_loss
        L_V_repulsive = zero_embedding_loss
        L_V = attr_weight * L_V_attractive + repul_weight * L_V_repulsive
        L_beta_sig = zero_beta_loss
        L_beta_suppress = zero_beta_loss
        L_beta_second = zero_beta_loss
        L_beta_second_contribution = zero_beta_loss
        L_var = zero_embedding_loss
        L_beta = (
            L_beta_noise
            + L_beta_sig
            + L_beta_suppress
            + L_beta_second_contribution
            + var_weight * L_var
        )
        return (
            L_V,
            L_beta,
            L_V_attractive,
            L_V_repulsive,
            L_beta_sig,
            L_beta_noise,
            L_beta_suppress,
            L_var,
            L_beta_second,
            L_beta_second_contribution,
        )

    if pt_track_weighting and loss_type != "hgcalimplementation":
        raise ValueError(
            "pT track weighting is currently supported only for "
            "loss_type=hgcalimplementation"
        )

    # Object indices for hits.  Build them from the complete cluster list,
    # rather than calling batch_cluster_indices() on signal hits only.  The
    # latter cannot represent an event that contains no signal objects.
    if noise_cluster_index != 0:
        raise NotImplementedError
    cluster_to_object = torch.full(
        (n_clusters,), -1, dtype=torch.long, device=device
    )
    cluster_to_object[is_object] = torch.arange(n_objects, device=device)
    object_index = cluster_to_object[cluster_index[is_sig]]
    n_objects_per_event = scatter_add(
        is_object.long(), batch_cluster, dim=0, dim_size=batch_size
    )
    n_hits_per_object = scatter_add(
        torch.ones_like(object_index, dtype=torch.long),
        object_index,
        dim=0,
        dim_size=n_objects,
    )
    batch_object = batch_cluster[is_object]

    object_pt_weights = None
    if pt_track_weighting:
        if y is None or y.ndim != 2 or y.shape[1] <= 6:
            raise ValueError(
                "pT track weighting requires truth rows with pT in column 6"
            )
        if y.shape[0] != n_objects:
            raise ValueError(
                "pT track weighting requires one aligned truth row per object "
                f"(got {y.shape[0]} rows for {n_objects} objects)"
            )
        if y.shape[1] > 7 and not torch.equal(
            y[:, -1].long().to(device=batch_object.device), batch_object.long()
        ):
            raise ValueError(
                "pT track weighting found truth rows that are not aligned with "
                "the batched object ordering"
            )
        object_pt_weights = pt_track_weights(
            y[:, 6].to(device=device),
            pt_track_weight_bin_edges,
            pt_track_weight_bin_weights,
            dtype=cluster_space_coords.dtype,
        )

    # Assertions
    # print("[DEBUG] object_index shape:", object_index.shape)
    # print("[DEBUG] n_hits_per_object:", n_hits_per_object)
    # print("[DEBUG] batch_object shape:", batch_object.shape)
    # print("[DEBUG] n_objects:", n_objects)

    assert object_index.size() == (n_hits_sig,)
    assert is_object.size() == (n_clusters,)
    assert torch.all(n_hits_per_object > 0)
    assert object_index.max() + 1 == n_objects
    
    # ________________________________
    # L_V term

    # Calculate q
    beta_for_q = (
        beta.float()
        if beta.dtype in (torch.bfloat16, torch.float16)
        else beta
    )
    if loss_type == "hgcalimplementation" or loss_type == "vrepweighted":
        q = (beta_for_q.clip(0.0, 1 - 1e-4).arctanh() / 1.01) ** 2 + qmin
    elif beta_stabilizing == "paper":
        q = beta_for_q.arctanh() ** 2 + qmin
    elif beta_stabilizing == "clip":
        beta_for_q = beta_for_q.clip(0.0, 1 - 1e-4)
        q = beta_for_q.arctanh() ** 2 + qmin
    elif beta_stabilizing == "soft_q_scaling":
        q = (
            (beta_for_q.clip(0.0, 1 - 1e-4) / 1.002).arctanh() ** 2
            + qmin
        )
    else:
        raise ValueError(f"beta_stablizing mode {beta_stabilizing} is not known")
    assert_no_nans(q)
    assert q.device == device
    assert q.size() == (n_hits,)

    # Calculate q_alpha, the max q per object, and the indices of said maxima
    # assert hit_energies.shape == q.shape
    # q_alpha, index_alpha = scatter_max(hit_energies[is_sig], object_index)
    q_alpha, index_alpha = scatter_max(q[is_sig], object_index)
    assert q_alpha.size() == (n_objects,)

    # Get the cluster space coordinates and betas for these maxima hits too
    x_alpha = cluster_space_coords[is_sig][index_alpha]
    x_alpha_original = original_coords[is_sig][index_alpha]
    if use_average_cc_pos > 0:
        #! this is a func of beta and q so maybe we could also do it with only q
        x_alpha_sum = scatter_add(
            q[is_sig].view(-1, 1).repeat(1, cluster_space_dim)
            * cluster_space_coords[is_sig],
            object_index,
            dim=0,
        )  # * beta[is_sig].view(-1, 1).repeat(1, 3)
        qbeta_alpha_sum = scatter_add(q[is_sig], object_index) + 1e-9  # * beta[is_sig]
        div_fac = 1 / qbeta_alpha_sum
        div_fac = torch.nan_to_num(div_fac, nan=0)
        x_alpha_mean = torch.mul(
            x_alpha_sum, div_fac.view(-1, 1).repeat(1, cluster_space_dim)
        )
        x_alpha = use_average_cc_pos * x_alpha_mean + (1 - use_average_cc_pos) * x_alpha
    if dis:
        phi_sum = scatter_add(
            beta[is_sig].view(-1) * distance_threshold[is_sig].view(-1),
            object_index,
            dim=0,
        )
        phi_alpha_sum = scatter_add(beta[is_sig].view(-1), object_index) + 1e-9
        phi_alpha = phi_sum / phi_alpha_sum

    beta_alpha = beta[is_sig][index_alpha]
    assert x_alpha.size() == (n_objects, cluster_space_dim)
    assert beta_alpha.size() == (n_objects,)

    # The training configuration uses the HGCAL-style potential.  Its dense
    # implementation below materializes every hit/object pair in the complete
    # mini-batch and masks cross-event pairs only afterwards.  Keep the legacy
    # matrices for the other loss variants, but avoid constructing them for the
    # HGCAL path.
    eventwise_hgcal = loss_type == "hgcalimplementation"
    compute_attractive = attr_weight != 0.0
    compute_repulsive = repul_weight != 0.0
    # Connectivity matrix from hit (row) -> cluster (column)
    # Index to matrix, e.g.:
    # [1, 3, 1, 0] --> [
    #     [0, 1, 0, 0],
    #     [0, 0, 0, 1],
    #     [0, 1, 0, 0],
    #     [1, 0, 0, 0]
    #     ]
    if not eventwise_hgcal:
        M = torch.nn.functional.one_hot(cluster_index).long()

        # Anti-connectivity matrix; be sure not to connect hits to clusters in different events!
        M_inv = get_inter_event_norms_mask(batch, n_clusters_per_event) - M

        # Throw away noise cluster columns; we never need them
        M = M[:, is_object]
        M_inv = M_inv[:, is_object]
        assert M.size() == (n_hits, n_objects)
        assert M_inv.size() == (n_hits, n_objects)

    # -------
    # Attractive potential term
    # First get all the relevant norms: We only want norms of signal hits
    # w.r.t. the object they belong to, i.e. no noise hits and no noise clusters.
    # First select all norms of all signal hits w.r.t. all objects, mask out later

    if eventwise_hgcal and compute_attractive:
        # Only the distance from each signal hit to its own object contributes
        # to the attractive potential.  Computing all N_signal x K distances
        # and multiplying the unwanted entries by zero is unnecessary.
        signal_coords = cluster_space_coords[is_sig]
        norms_att = torch.sum(
            torch.square(signal_coords - x_alpha[object_index]), dim=-1
        )
        if dis:
            norms_att = norms_att / (
                2 * phi_alpha[object_index].square() + 1e-6
            )
            norms_att = torch.log(
                norms_att.new_tensor(np.e) * norms_att + 1
            )
        else:
            norms_att = torch.log(
                norms_att.new_tensor(np.e) * norms_att / 2 + 1
            )

        attractive_per_hit = (
            q[is_sig] * q_alpha[object_index] * norms_att
        )
        attractive_per_object = scatter_add(
            attractive_per_hit, object_index, dim=0
        )
        attractive_per_object = attractive_per_object / (
            n_hits_per_object.to(attractive_per_object.dtype) + 1e-3
        )
        L_V_attractive = weighted_mean(
            attractive_per_object, object_pt_weights
        )
    elif eventwise_hgcal:
        L_V_attractive = zero_embedding_loss
    elif loss_type == "hgcalimplementation" or loss_type == "vrepweighted":
        if dis:
            N_k = torch.sum(M, dim=0)  # number of hits per object
            norms = torch.sum(
                torch.square(cluster_space_coords.unsqueeze(1) - x_alpha.unsqueeze(0)),
                dim=-1,
            )
            norms_att = norms[is_sig]
            norms_att = norms_att / (2 * phi_alpha.unsqueeze(0) ** 2 + 1e-6)
            #! att func as in line 159 of object condensation
            norms_att = torch.log(
                norms_att.new_tensor(np.e) * norms_att + 1
            )
        else:
            N_k = torch.sum(M, dim=0)  # number of hits per object
            norms = torch.sum(
                torch.square(cluster_space_coords.unsqueeze(1) - x_alpha.unsqueeze(0)),
                dim=-1,
            )
            norms_att = norms[is_sig]
            #! att func as in line 159 of object condensation

            norms_att = torch.log(
                norms_att.new_tensor(np.e) * norms_att / 2 + 1
            )
    elif huberize_norm_for_V_attractive:
        norms_att = norms[is_sig]
        # Huberized version (linear but times 4)
        # Be sure to not move 'off-diagonal' away from zero
        # (i.e. norms of hits w.r.t. clusters they do _not_ belong to)
        norms_att = huber(norms_att + 1e-5, 4.0)
    else:
        norms_att = norms[is_sig]
        # Paper version is simply norms squared (no need for mask)
        norms_att = norms_att**2
    if not eventwise_hgcal:
        assert norms_att.size() == (n_hits_sig, n_objects)

        # Now apply the mask to keep only norms of signal hits w.r.t. to the object
        # they belong to
        norms_att *= M[is_sig]

    # Sum over hits, then sum per event, then divide by n_hits_per_event, then sum over events
    if eventwise_hgcal:
        # L_V_attractive was reduced directly per object above.
        pass
    elif loss_type == "hgcalimplementation":
        # Final potential term
        # (n_sig_hits, 1) * (1, n_objects) * (n_sig_hits, n_objects)
        V_attractive = q[is_sig].unsqueeze(-1) * q_alpha.unsqueeze(0) * norms_att
        assert V_attractive.size() == (n_hits_sig, n_objects)
        #! each shower is account for separately
        V_attractive = V_attractive.sum(dim=0)  # K objects
        #! divide by the number of accounted points
        V_attractive = V_attractive.view(-1) / (N_k.view(-1) + 1e-3)
        L_V_attractive = torch.mean(V_attractive)
    elif loss_type =="weighted":
        weights = calculate_weights_for_class_hit_type_batch(g, torch.zeros_like(batch), is_sig)
        # (n_sig_hits, 1) * (1, n_objects) * (n_sig_hits, n_objects)
        V_attractive = weights.unsqueeze(-1)*q[is_sig].unsqueeze(-1) * q_alpha.unsqueeze(0) * norms_att
        
        assert V_attractive.size() == (n_hits_sig, n_objects)
        #! each shower is account for separately
        V_attractive = V_attractive.sum(dim=0)  # K objects
        weight_per_object = scatter_add(weights, object_index)# weight per object 
        V_attractive= V_attractive/weight_per_object
        L_V_attractive = torch.mean(V_attractive)
    
    elif loss_type == "vrepweighted":
        
        # weight the vtx hits inside the shower
        V_attractive = (
                g.ndata["weights"][is_sig].unsqueeze(-1)
                * q[is_sig].unsqueeze(-1)
                * q_alpha.unsqueeze(0)
                * norms_att
            )
        assert V_attractive.size() == (n_hits_sig, n_objects)
        V_attractive = V_attractive.sum(dim=0)  # K objects

        L_V_attractive = torch.mean(V_attractive.view(-1))
            
    else:
        # Final potential term
        # (n_sig_hits, 1) * (1, n_objects) * (n_sig_hits, n_objects)
        V_attractive = q[is_sig].unsqueeze(-1) * q_alpha.unsqueeze(0) * norms_att
        assert V_attractive.size() == (n_hits_sig, n_objects)
        #! in comparison this works per hit
        V_attractive = scatter_add(
            V_attractive.sum(dim=0), batch_object, dim=0, dim_size=batch_size
        ) / n_hits_per_event
        assert V_attractive.size() == (batch_size,)
        L_V_attractive = V_attractive.sum()

    # -------
    # Repulsive potential term

    # Get all the relevant norms: We want norms of any hit w.r.t. to
    # objects they do *not* belong to, i.e. no noise clusters.
    # We do however want to keep norms of noise hits w.r.t. objects
    # Power-scale the norms: Gaussian scaling term instead of a cone
    # Mask out the norms of hits w.r.t. the cluster they belong to
    if eventwise_hgcal and compute_repulsive:
        # Repulsion is defined only between hits and other truth objects in the
        # same event.  Build one event block at a time instead of first forming
        # an N_total x K_total matrix and masking its cross-event blocks.
        repulsive_per_object_blocks = []
        for event_index in range(len(n_hits_per_event)):
            event_hit_mask = batch == event_index
            event_object_mask = batch_object == event_index
            event_centers = x_alpha[event_object_mask]
            if event_centers.shape[0] == 0:
                continue

            event_coords = cluster_space_coords[event_hit_mask]
            event_norms = torch.sum(
                torch.square(
                    event_coords.unsqueeze(1) - event_centers.unsqueeze(0)
                ),
                dim=-1,
            )
            if dis:
                event_norms = event_norms / (
                    2 * phi_alpha[event_object_mask].unsqueeze(0).square() + 1e-6
                )

            # Particle labels are local to each event: zero denotes noise and
            # positive labels map consecutively onto that event's objects.
            event_truth_object = (
                cluster_index_per_event[event_hit_mask].long() - 1
            )
            local_object_ids = torch.arange(
                event_centers.shape[0], device=device
            )
            repulsion_mask = (
                event_truth_object.unsqueeze(1)
                != local_object_ids.unsqueeze(0)
            )
            event_norms_rep = torch.exp(
                -event_norms if dis else -event_norms / 2
            ) * repulsion_mask
            event_v_repulsive = (
                q[event_hit_mask].unsqueeze(1)
                * q_alpha[event_object_mask].unsqueeze(0)
                * event_norms_rep
            )
            repulsive_counts = repulsion_mask.sum(dim=0).clamp(min=1)
            repulsive_per_object_blocks.append(
                event_v_repulsive.sum(dim=0)
                / repulsive_counts.to(event_v_repulsive.dtype)
            )

        L_V_repulsive = torch.cat(repulsive_per_object_blocks)
        delta_MC = calculate_delta_MC(
            y, g, dtype=cluster_space_coords.dtype
        )
        weight_track = torch.pow(
            delta_MC.clamp(min=0.001), -hard_negative_weight
        ).clamp(max=hard_negative_max_weight)
        if object_pt_weights is not None:
            weight_track = weight_track * object_pt_weights
        L_V_repulsive = torch.sum(
            L_V_repulsive * weight_track.view(-1)
        ) / torch.sum(weight_track)
    elif eventwise_hgcal:
        L_V_repulsive = zero_embedding_loss
    elif loss_type == "hgcalimplementation" or loss_type == "vrepweighted" or loss_type == "weighted":
        if dis:
            norms = norms / (2 * phi_alpha.unsqueeze(0) ** 2 + 1e-6)
            norms_rep = torch.exp(-(norms)) * M_inv
        else:
            norms_rep = torch.exp(-(norms) / 2) * M_inv
    else:
        norms_rep = torch.exp(-4.0 * norms**2) * M_inv

    # (n_sig_hits, 1) * (1, n_objects) * (n_sig_hits, n_objects)
    if not eventwise_hgcal:
        V_repulsive = q.unsqueeze(1) * q_alpha.unsqueeze(0) * norms_rep

        # No need to apply a V = max(0, V); by construction V>=0
        assert V_repulsive.size() == (n_hits, n_objects)

    # Sum over hits, then sum per event, then divide by n_hits_per_event, then sum up events
    nope = (n_objects_per_event - 1).clamp(min=1)
    if eventwise_hgcal:
        # L_V_repulsive was globally normalized above after concatenating the
        # event-local per-object contributions in their original order.
        pass
    elif loss_type == "hgcalimplementation" or loss_type == "vrepweighted" or loss_type == "weighted":
        #! sum each object repulsive terms
        L_V_repulsive = V_repulsive.sum(dim=0)  # size number of objects
        number_of_repulsive_terms_per_object = torch.sum(M_inv, dim=0)
        # print (
        #     "number_of_repulsive_terms_per_object", number_of_repulsive_terms_per_object
        # )
        number_of_repulsive_terms_per_object[
            number_of_repulsive_terms_per_object < 1
        ] = 1
        # print(
        #     "number_of_repulsive_terms_per_object", number_of_repulsive_terms_per_object
        # )
        L_V_repulsive = L_V_repulsive.view(
            -1
        ) / number_of_repulsive_terms_per_object.view(-1)

     
        if loss_type == "vrepweighted":
            L_V_repulsive = torch.sum(
                modified_showers.view(-1) * L_V_repulsive.view(-1)
            ) / len(modified_showers)
        
        else:
            # this needs to be done per graph
            delta_MC = calculate_delta_MC(
                y, g, dtype=cluster_space_coords.dtype
            )
            # Nearby truth tracks are the difficult negatives. Periodic delta-R
            # is used by calculate_delta_MC; the exponent controls how strongly
            # these cases are emphasized and the cap prevents unstable spikes.
            weight_track = torch.pow(
                delta_MC.clamp(min=0.001), -hard_negative_weight
            ).clamp(max=hard_negative_max_weight)
            #L_V_repulsive = torch.mean(L_V_repulsive)
            
            L_V_repulsive = torch.sum(L_V_repulsive * weight_track.view(-1))/torch.sum(weight_track)
            # sys.exit()
            
    else:
        L_V_repulsive = (
            scatter_add(
                V_repulsive.sum(dim=0), batch_object, dim=0, dim_size=batch_size
            )
            / (n_hits_per_event * nope)
        ).sum()

    L_V = attr_weight * L_V_attractive + repul_weight * L_V_repulsive

    
    # -------
    # L_beta signal term
    if loss_type == "hgcalimplementation" or loss_type == "weighted":
        beta_per_object_c = scatter_add(beta[is_sig], object_index)

        beta_alpha = beta[is_sig][index_alpha]

        beta_signal_per_object = (
            1 - beta_alpha + 1 - torch.clip(beta_per_object_c, 0, 1)
        )
        L_beta_sig = weighted_mean(
            beta_signal_per_object, object_pt_weights
        )

    elif loss_type == "vrepweighted":
        # version one:
        beta_per_object_c = scatter_add(beta[is_sig], object_index)
        beta_alpha = beta[is_sig][index_alpha]
        L_beta_sig = 1 - beta_alpha + 1 - torch.clip(beta_per_object_c, 0, 1)
        L_beta_sig = torch.sum(L_beta_sig.view(-1) * modified_showers.view(-1))
        L_beta_sig = L_beta_sig / len(modified_showers)

        L_beta_noise = L_beta_noise / batch_size
        # ? note: the training that worked quite well was dividing this by the batch size (1/4)

    elif beta_term_option == "paper":
        beta_alpha = beta[is_sig][index_alpha]
        L_beta_sig = torch.sum(  # maybe 0.5 for less aggressive loss
            scatter_add(
                (1 - beta_alpha), batch_object, dim=0, dim_size=batch_size
            )
            / n_objects_per_event.clamp(min=1)
        )
        # print("L_beta_sig", L_beta_sig / batch_size)
        # beta_exp = beta[is_sig]
        # beta_exp[index_alpha] = 0
        # # L_exp = torch.mean(beta_exp)
        # beta_exp = torch.exp(0.5 * beta_exp)
        # L_exp = torch.mean(scatter_add(beta_exp, batch) / n_hits_per_event)

    elif beta_term_option == "short-range-potential":

        # First collect the norms: We only want norms of hits w.r.t. the object they
        # belong to (like in V_attractive)
        # Apply transformation first, and then apply mask to keep only the norms we want,
        # then sum over hits, so the result is (n_objects,)
        norms_beta_sig = (1.0 / (20.0 * norms[is_sig] ** 2 + 1.0) * M[is_sig]).sum(
            dim=0
        )
        assert torch.all(norms_beta_sig >= 1.0) and torch.all(
            norms_beta_sig <= n_hits_per_object
        )
        # Subtract from 1. to remove self interaction, divide by number of hits per object
        norms_beta_sig = (1.0 - norms_beta_sig) / n_hits_per_object
        assert torch.all(norms_beta_sig >= -1.0) and torch.all(norms_beta_sig <= 0.0)
        norms_beta_sig *= beta_alpha
        # Conclusion:
        # lower beta --> higher loss (less negative)
        # higher norms --> higher loss

        # Sum over objects, divide by number of objects per event, then sum over events
        L_beta_norms_term = (
            scatter_add(
                norms_beta_sig, batch_object, dim=0, dim_size=batch_size
            )
            / n_objects_per_event.clamp(min=1)
        ).sum()
        assert L_beta_norms_term >= -batch_size and L_beta_norms_term <= 0.0

        # Logbeta term: Take -.2*torch.log(beta_alpha[is_object]+1e-9), sum it over objects,
        # divide by n_objects_per_event, then sum over events (same pattern as above)
        # lower beta --> higher loss
        L_beta_logbeta_term = (
            scatter_add(
                -0.2 * torch.log(beta_alpha + 1e-9),
                batch_object,
                dim=0,
                dim_size=batch_size,
            )
            / n_objects_per_event.clamp(min=1)
        ).sum()

        # Final L_beta term
        L_beta_sig = L_beta_norms_term + L_beta_logbeta_term

    else:
        valid_options = ["paper", "short-range-potential"]
        raise ValueError(
            f'beta_term_option "{beta_term_option}" is not valid, choose from {valid_options}'
        )

    # Penalize additional high-beta signal hits.  The original signal term can
    # saturate once the beta sum reaches one, which otherwise allows multiple
    # condensation seeds and fragmented reconstructed tracks.
    L_beta_suppress = beta.sum() * 0.0
    if beta_suppress_weight > 0 and n_hits_sig > n_objects:
        is_alpha_sig = torch.zeros(n_hits_sig, dtype=torch.bool, device=device)
        is_alpha_sig[index_alpha] = True
        non_alpha_beta = beta[is_sig][~is_alpha_sig]
        if object_pt_weights is None:
            L_beta_suppress = beta_suppress_weight * non_alpha_beta.mean()
        else:
            non_alpha_object_index = object_index[~is_alpha_sig]
            non_alpha_sum = scatter_add(
                non_alpha_beta,
                non_alpha_object_index,
                dim=0,
                dim_size=n_objects,
            )
            non_alpha_count = scatter_add(
                torch.ones_like(non_alpha_beta),
                non_alpha_object_index,
                dim=0,
                dim_size=n_objects,
            )
            has_competing_hit = non_alpha_count > 0
            suppress_per_object = (
                non_alpha_sum[has_competing_hit]
                / non_alpha_count[has_competing_hit]
            )
            L_beta_suppress = beta_suppress_weight * weighted_mean(
                suppress_per_object,
                object_pt_weights[has_competing_hit],
            )

    # Directly penalize the strongest competing condensation point in every
    # truth object.  Unlike the all-non-alpha average above, this term is not
    # diluted by objects containing many already-small beta values.  Keep the
    # complete top-two reduction out of the graph when its weight is disabled.
    L_beta_second = beta.sum() * 0.0
    L_beta_second_contribution = beta.sum() * 0.0
    if beta_second_weight != 0.0:
        L_beta_second = second_highest_beta_loss(
            beta[is_sig],
            object_index,
            index_alpha,
            n_hits_per_object,
            n_objects,
            object_pt_weights,
        )
        L_beta_second_contribution = beta_second_weight * L_beta_second

    # Compact truth objects around their centroid in the learned embedding.
    # This complements the q-weighted attractive term and reduces long tails
    # that are easily claimed by a neighbouring condensation point.
    L_var = zero_embedding_loss
    if var_weight != 0.0:
        centroid = scatter_mean(cluster_space_coords[is_sig], object_index, dim=0)
        variance_per_hit = torch.sum(
            (cluster_space_coords[is_sig] - centroid[object_index]) ** 2, dim=1
        )
        variance_per_object = scatter_mean(
            variance_per_hit, object_index, dim=0
        )
        L_var = weighted_mean(variance_per_object, object_pt_weights)

    L_beta = (
        L_beta_noise
        + L_beta_sig
        + L_beta_suppress
        + L_beta_second_contribution
        + var_weight * L_var
    )
  
    if (somethingIsNaN):
        print("[DEBUG] Loss values")
        print("[DEBUG] L_V = attr_weight * L_V_attractive + repul_weight * L_V_repulsive ->", "L_V=",attr_weight,"*",L_V_attractive,"+",repul_weight,"*",L_V_repulsive)
        print("[DEBUG]  L_beta = L_beta_noise + L_beta_sig ->"," L_beta =",L_beta_noise,"+",L_beta_sig)

    # ________________________________
    # Returning
    # Also divide by batch size here

    if return_components or DEBUG:
        components = dict(
            L_V=L_V / batch_size,
            L_V_attractive=L_V_attractive / batch_size,
            L_V_repulsive=L_V_repulsive / batch_size,
            L_beta=L_beta / batch_size,
            L_beta_noise=L_beta_noise / batch_size,
            L_beta_sig=L_beta_sig / batch_size,
        )
        if beta_term_option == "short-range-potential":
            components["L_beta_norms_term"] = L_beta_norms_term / batch_size
            components["L_beta_logbeta_term"] = L_beta_logbeta_term / batch_size
    if DEBUG:
        debug(formatted_loss_components_string(components))
    if torch.isnan(L_beta / batch_size):
        print("isnan!!!")
        print(L_beta, batch_size)
        print("L_beta_noise", L_beta_noise)
        print("L_beta_sig", L_beta_sig)
    
    L_exp = L_beta
    if loss_type == "hgcalimplementation" or loss_type == "vrepweighted" or loss_type == "weighted":
        return (
            L_V,  # 0
            L_beta,
            L_V_attractive,
            L_V_repulsive,
            L_beta_sig,
            L_beta_noise,
            L_beta_suppress,
            L_var,
            L_beta_second,
            L_beta_second_contribution,
        )


def formatted_loss_components_string(components: dict) -> str:
    """
    Formats the components returned by calc_LV_Lbeta
    """
    total_loss = components["L_V"] + components["L_beta"]
    fractions = {k: v / total_loss for k, v in components.items()}
    fkey = lambda key: f"{components[key]:+.4f} ({100.*fractions[key]:.1f}%)"
    s = (
        "  L_V                 = {L_V}"
        "\n    L_V_attractive      = {L_V_attractive}"
        "\n    L_V_repulsive       = {L_V_repulsive}"
        "\n  L_beta              = {L_beta}"
        "\n    L_beta_noise        = {L_beta_noise}"
        "\n    L_beta_sig          = {L_beta_sig}".format(
            L=total_loss, **{k: fkey(k) for k in components}
        )
    )
    if "L_beta_norms_term" in components:
        s += (
            "\n      L_beta_norms_term   = {L_beta_norms_term}"
            "\n      L_beta_logbeta_term = {L_beta_logbeta_term}".format(
                **{k: fkey(k) for k in components}
            )
        )
    if "L_noise_filter" in components:
        s += f'\n  L_noise_filter = {fkey("L_noise_filter")}'
    return s


def calc_simple_clus_space_loss(
    cluster_space_coords: torch.Tensor,  # Predicted by model
    cluster_index_per_event: torch.Tensor,  # Truth hit->cluster index
    batch: torch.Tensor,
    # From here on just parameters
    noise_cluster_index: int = 0,  # cluster_index entries with this value are noise/noise
    huberize_norm_for_V_attractive=True,
    pred_edc: torch.Tensor = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Isolating just the V_attractive and V_repulsive parts of object condensation,
    w.r.t. the geometrical mean of truth cluster centers (rather than the highest
    beta point of the truth cluster).
    Most of this code is copied from `calc_LV_Lbeta`, so it's easier to try out
    different scalings for the norms without breaking the main OC function.
    `pred_edc`: Predicted estimated distance-to-center.
    This is an optional column, that should be `n_hits` long. If it is
    passed, a third loss component is calculated based on the truth distance-to-center
    w.r.t. predicted distance-to-center. This quantifies how close a hit is to it's center,
    which provides an ansatz for the clustering.
    See also the 'Concepts' in the doc of `calc_LV_Lbeta`.
    """
    # ________________________________
    # Calculate a bunch of needed counts and indices locally

    # cluster_index: unique index over events
    # E.g. cluster_index_per_event=[ 0, 0, 1, 2, 0, 0, 1], batch=[0, 0, 0, 0, 1, 1, 1]
    #      -> cluster_index=[ 0, 0, 1, 2, 3, 3, 4 ]
    cluster_index, n_clusters_per_event = batch_cluster_indices(
        cluster_index_per_event, batch
    )
    n_hits, cluster_space_dim = cluster_space_coords.size()
    batch_size = batch.max() + 1
    n_hits_per_event = scatter_count(batch)

    # Index of cluster -> event (n_clusters,)
    batch_cluster = scatter_counts_to_indices(n_clusters_per_event)

    # Per-hit boolean, indicating whether hit is sig or noise
    is_noise = cluster_index_per_event == noise_cluster_index
    is_sig = ~is_noise
    n_hits_sig = is_sig.sum()

    # Per-cluster boolean, indicating whether cluster is an object or noise
    is_object = scatter_max(is_sig.long(), cluster_index)[0].bool()

    # # FIXME: This assumes noise_cluster_index == 0!!
    # # Not sure how to do this in a performant way in case noise_cluster_index != 0
    # if noise_cluster_index != 0: raise NotImplementedError
    # object_index_per_event = cluster_index_per_event[is_sig] - 1
    batch_object = batch_cluster[is_object]
    n_objects = is_object.sum()

    # ________________________________
    # Build the masks

    # Connectivity matrix from hit (row) -> cluster (column)
    # Index to matrix, e.g.:
    # [1, 3, 1, 0] --> [
    #     [0, 1, 0, 0],
    #     [0, 0, 0, 1],
    #     [0, 1, 0, 0],
    #     [1, 0, 0, 0]
    #     ]
    M = torch.nn.functional.one_hot(cluster_index).long()

    # Anti-connectivity matrix; be sure not to connect hits to clusters in different events!
    M_inv = get_inter_event_norms_mask(batch, n_clusters_per_event) - M

    # Throw away noise cluster columns; we never need them
    M = M[:, is_object]
    M_inv = M_inv[:, is_object]
    assert M.size() == (n_hits, n_objects)
    assert M_inv.size() == (n_hits, n_objects)

    # ________________________________
    # Loss terms

    # First calculate all cluster centers, then throw out the noise clusters
    cluster_centers = scatter_mean(cluster_space_coords, cluster_index, dim=0)
    object_centers = cluster_centers[is_object]

    # Calculate all norms
    # Warning: Should not be used without a mask!
    # Contains norms between hits and objects from different events
    # (n_hits, 1, cluster_space_dim) - (1, n_objects, cluster_space_dim)
    #   gives (n_hits, n_objects, cluster_space_dim)
    norms = (cluster_space_coords.unsqueeze(1) - object_centers.unsqueeze(0)).norm(
        dim=-1
    )
    assert norms.size() == (n_hits, n_objects)

    # -------
    # Attractive loss

    # First get all the relevant norms: We only want norms of signal hits
    # w.r.t. the object they belong to, i.e. no noise hits and no noise clusters.
    # First select all norms of all signal hits w.r.t. all objects (filtering out
    # the noise), mask out later
    norms_att = norms[is_sig]

    # Power-scale the norms
    if huberize_norm_for_V_attractive:
        # Huberized version (linear but times 4)
        # Be sure to not move 'off-diagonal' away from zero
        # (i.e. norms of hits w.r.t. clusters they do _not_ belong to)
        norms_att = huber(norms_att + 1e-5, 4.0)
    else:
        # Paper version is simply norms squared (no need for mask)
        norms_att = norms_att**2
    assert norms_att.size() == (n_hits_sig, n_objects)

    # Now apply the mask to keep only norms of signal hits w.r.t. to the object
    # they belong to (throw away norms w.r.t. cluster they do *not* belong to)
    norms_att *= M[is_sig]

    # Sum norms_att over hits (dim=0), then sum per event, then divide by n_hits_per_event,
    # then sum over events
    L_attractive = (
        scatter_add(norms_att.sum(dim=0), batch_object) / n_hits_per_event
    ).sum()

    # -------
    # Repulsive loss

    # Get all the relevant norms: We want norms of any hit w.r.t. to
    # objects they do *not* belong to, i.e. no noise clusters.
    # We do however want to keep norms of noise hits w.r.t. objects
    # Power-scale the norms: Gaussian scaling term instead of a cone
    # Mask out the norms of hits w.r.t. the cluster they belong to
    norms_rep = torch.exp(-4.0 * norms**2) * M_inv

    # Sum over hits, then sum per event, then divide by n_hits_per_event, then sum up events
    L_repulsive = (
        scatter_add(norms_rep.sum(dim=0), batch_object) / n_hits_per_event
    ).sum()

    L_attractive /= batch_size
    L_repulsive /= batch_size

    # -------
    # Optional: edc column

    if pred_edc is not None:
        n_hits_per_cluster = scatter_count(cluster_index)
        cluster_centers_expanded = torch.index_select(cluster_centers, 0, cluster_index)
        assert cluster_centers_expanded.size() == (n_hits, cluster_space_dim)
        truth_edc = (cluster_space_coords - cluster_centers_expanded).norm(dim=-1)
        assert pred_edc.size() == (n_hits,)
        d_per_hit = (pred_edc - truth_edc) ** 2
        d_per_object = scatter_add(d_per_hit, cluster_index)[is_object]
        assert d_per_object.size() == (n_objects,)
        L_edc = (scatter_add(d_per_object, batch_object) / n_hits_per_event).sum()
        return L_attractive, L_repulsive, L_edc

    return L_attractive, L_repulsive


def huber(d, delta):
    """
    See: https://en.wikipedia.org/wiki/Huber_loss#Definition
    Multiplied by 2 w.r.t Wikipedia version (aligning with Jan's definition)
    """
    return torch.where(
        torch.abs(d) <= delta, d**2, 2.0 * delta * (torch.abs(d) - delta)
    )


def batch_cluster_indices(
    cluster_id: torch.Tensor, batch: torch.Tensor
) -> Tuple[torch.LongTensor, torch.LongTensor]:
    """
    Turns cluster indices per event to an index in the whole batch
    Example:
    cluster_id = torch.LongTensor([0, 0, 1, 1, 2, 0, 0, 1, 1, 1, 0, 0, 1])
    batch = torch.LongTensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2])
    -->
    offset = torch.LongTensor([0, 0, 0, 0, 0, 3, 3, 3, 3, 3, 5, 5, 5])
    output = torch.LongTensor([0, 0, 1, 1, 2, 3, 3, 4, 4, 4, 5, 5, 6])
    """
    device = cluster_id.device
    assert cluster_id.device == batch.device
    # Count the number of clusters per entry in the batch
    n_clusters_per_event = scatter_max(cluster_id, batch, dim=-1)[0] + 1
    # Offsets are then a cumulative sum
    offset_values_nozero = n_clusters_per_event[:-1].cumsum(dim=-1)
    # Prefix a zero
    offset_values = torch.cat((torch.zeros(1, device=device), offset_values_nozero))
    # Fill it per hit
    offset = torch.gather(offset_values, 0, batch).long()
    return offset + cluster_id, n_clusters_per_event


def get_clustering_np(
    betas: np.array, X: np.array, tbeta: float = 0.1, td: float = 1.0
) -> np.array:
    """
    Returns a clustering of hits -> cluster_index, based on the GravNet model
    output (predicted betas and cluster space coordinates) and the clustering
    parameters tbeta and td.
    Takes numpy arrays as input.
    """
    n_points = betas.shape[0]
    select_condpoints = betas > tbeta
    # Get indices passing the threshold
    indices_condpoints = np.nonzero(select_condpoints)[0]
    # Order them by decreasing beta value
    indices_condpoints = indices_condpoints[np.argsort(-betas[select_condpoints])]
    # Assign points to condensation points
    # Only assign previously unassigned points (no overwriting)
    # Points unassigned at the end are bkg (-1)
    unassigned = np.arange(n_points)
    clustering = -1 * np.ones(n_points, dtype=np.int32)
    for index_condpoint in indices_condpoints:
        d = np.linalg.norm(X[unassigned] - X[index_condpoint], axis=-1)
        assigned_to_this_condpoint = unassigned[d < td]
        clustering[assigned_to_this_condpoint] = index_condpoint
        unassigned = unassigned[~(d < td)]
    return clustering


def get_clustering(betas: torch.Tensor, X: torch.Tensor, tbeta=0.1, td=1.0):
    """
    Returns a clustering of hits -> cluster_index, based on the GravNet model
    output (predicted betas and cluster space coordinates) and the clustering
    parameters tbeta and td.
    Takes torch.Tensors as input.
    """
    n_points = betas.size(0)
    select_condpoints = betas > tbeta
    # Get indices passing the threshold
    indices_condpoints = select_condpoints.nonzero()
    # Order them by decreasing beta value
    indices_condpoints = indices_condpoints[(-betas[select_condpoints]).argsort()]
    # Assign points to condensation points
    # Only assign previously unassigned points (no overwriting)
    # Points unassigned at the end are bkg (-1)
    unassigned = torch.arange(n_points)
    clustering = -1 * torch.ones(n_points, dtype=torch.long)
    for index_condpoint in indices_condpoints:
        d = torch.norm(X[unassigned] - X[index_condpoint][0], dim=-1)
        assigned_to_this_condpoint = unassigned[d < td]
        clustering[assigned_to_this_condpoint] = index_condpoint[0]
        unassigned = unassigned[~(d < td)]
    return clustering


def scatter_count(input: torch.Tensor):
    """
    Returns ordered counts over an index array
    Example:
    >>> scatter_count(torch.Tensor([0, 0, 0, 1, 1, 2, 2])) # input
    >>> [3, 2, 2]
    Index assumptions work like in torch_scatter, so:
    >>> scatter_count(torch.Tensor([1, 1, 1, 2, 2, 4, 4]))
    >>> tensor([0, 3, 2, 0, 2])
    """
    return scatter_add(torch.ones_like(input, dtype=torch.long), input.long())


def scatter_counts_to_indices(input: torch.LongTensor) -> torch.LongTensor:
    """
    Converts counts to indices. This is the inverse operation of scatter_count
    Example:
    input:  [3, 2, 2]
    output: [0, 0, 0, 1, 1, 2, 2]
    """
    return torch.repeat_interleave(
        torch.arange(input.size(0), device=input.device), input
    ).long()


def get_inter_event_norms_mask(
    batch: torch.LongTensor, nclusters_per_event: torch.LongTensor
):
    """
    Creates mask of (nhits x nclusters) that is only 1 if hit i is in the same event as cluster j
    Example:
    cluster_id_per_event = torch.LongTensor([0, 0, 1, 1, 2, 0, 0, 1, 1, 1, 0, 0, 1])
    batch = torch.LongTensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2])
    Should return:
    torch.LongTensor([
        [1, 1, 1, 0, 0, 0, 0],
        [1, 1, 1, 0, 0, 0, 0],
        [1, 1, 1, 0, 0, 0, 0],
        [1, 1, 1, 0, 0, 0, 0],
        [1, 1, 1, 0, 0, 0, 0],
        [0, 0, 0, 1, 1, 0, 0],
        [0, 0, 0, 1, 1, 0, 0],
        [0, 0, 0, 1, 1, 0, 0],
        [0, 0, 0, 1, 1, 0, 0],
        [0, 0, 0, 1, 1, 0, 0],
        [0, 0, 0, 0, 0, 1, 1],
        [0, 0, 0, 0, 0, 1, 1],
        [0, 0, 0, 0, 0, 1, 1],
        ])
    """
    device = batch.device
    # Following the example:
    # Expand batch to the following (nhits x nevents) matrix (little hacky, boolean mask -> long):
    # [[1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #  [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0],
    #  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1]]
    batch_expanded_as_ones = (
        batch
        == torch.arange(batch.max() + 1, dtype=torch.long, device=device).unsqueeze(-1)
    ).long()
    # Then repeat_interleave it to expand it to nclusters rows, and transpose to get (nhits x nclusters)
    return batch_expanded_as_ones.repeat_interleave(nclusters_per_event, dim=0).T


def isin(ar1, ar2):
    """To be replaced by torch.isin for newer releases of torch"""
    return (ar1[..., None] == ar2).any(-1)


def reincrementalize(y: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
    """Re-indexes y so that missing clusters are no longer counted.
    Example:
        >>> y = torch.LongTensor([
            0, 0, 0, 1, 1, 3, 3,
            0, 0, 0, 0, 0, 2, 2, 3, 3,
            0, 0, 1, 1
            ])
        >>> batch = torch.LongTensor([
            0, 0, 0, 0, 0, 0, 0,
            1, 1, 1, 1, 1, 1, 1, 1, 1,
            2, 2, 2, 2,
            ])
        >>> print(reincrementalize(y, batch))
        tensor([0, 0, 0, 1, 1, 2, 2, 0, 0, 0, 0, 0, 1, 1, 2, 2, 0, 0, 1, 1])
    """
    y_offset, n_per_event = batch_cluster_indices(y, batch)
    offset = y_offset - y
    n_clusters = n_per_event.sum()
    holes = (
        (~isin(torch.arange(n_clusters, device=y.device), y_offset))
        .nonzero()
        .squeeze(-1)
    )
    n_per_event_without_holes = n_per_event.clone()
    n_per_event_cumsum = n_per_event.cumsum(0)
    for hole in holes.sort(descending=True).values:
        y_offset[y_offset > hole] -= 1
        i_event = (hole > n_per_event_cumsum).long().argmin()
        n_per_event_without_holes[i_event] -= 1
    offset_per_event = torch.zeros_like(n_per_event_without_holes)
    offset_per_event[1:] = n_per_event_without_holes.cumsum(0)[:-1]
    offset_without_holes = torch.gather(offset_per_event, 0, batch).long()
    reincrementalized = y_offset - offset_without_holes
    return reincrementalized


def L_clusters_calc(batch, cluster_space_coords, cluster_index, frac_combinations, q):
    number_of_pairs = 0
    for batch_id in batch.unique():
        # do all possible pairs...
        bmask = batch == batch_id
        clust_space_filt = cluster_space_coords[bmask]
        pos_pairs_all = []
        neg_pairs_all = []
        if len(cluster_index[bmask].unique()) <= 1:
            continue
        L_clusters = torch.tensor(0.0).to(q.device)
        for cluster in cluster_index[bmask].unique():
            coords_pos = clust_space_filt[cluster_index[bmask] == cluster]
            coords_neg = clust_space_filt[cluster_index[bmask] != cluster]
            if len(coords_neg) == 0:
                continue
            clust_idx = cluster_index[bmask] == cluster
            # all_ones = torch.ones_like((clust_idx, clust_idx))
            # pos_pairs = [[i, j] for i in range(len(coords_pos)) for j in range (len(coords_pos)) if i < j]
            total_num = (len(coords_pos) ** 2) / 2
            num = int(frac_combinations * total_num)
            pos_pairs = []
            for i in range(num):
                pos_pairs.append(
                    [
                        np.random.randint(len(coords_pos)),
                        np.random.randint(len(coords_pos)),
                    ]
                )
            neg_pairs = []
            for i in range(len(pos_pairs)):
                neg_pairs.append(
                    [
                        np.random.randint(len(coords_pos)),
                        np.random.randint(len(coords_neg)),
                    ]
                )
            pos_pairs_all += pos_pairs
            neg_pairs_all += neg_pairs
        pos_pairs = torch.tensor(pos_pairs_all)
        neg_pairs = torch.tensor(neg_pairs_all)
        """# do just a small sample of the pairs. ...
        bmask = batch == batch_id

        #L_clusters = 0   # Loss of randomly sampled distances between points inside and outside clusters

        pos_idx, neg_idx = [], []
        for cluster in cluster_index[bmask].unique():
            clust_idx = (cluster_index == cluster)[bmask]
            perm = torch.randperm(clust_idx.sum())
            perm1 = torch.randperm((~clust_idx).sum())
            perm2 = torch.randperm(clust_idx.sum())
            #cutoff = clust_idx.sum()//2
            pos_lst = clust_idx.nonzero()[perm]
            neg_lst = (~clust_idx).nonzero()[perm1]
            neg_lst_second = clust_idx.nonzero()[perm2]
            if len(pos_lst) % 2:
                pos_lst = pos_lst[:-1]
            if len(neg_lst) % 2:
                neg_lst = neg_lst[:-1]
            len_cap = min(len(pos_lst), len(neg_lst), len(neg_lst_second))
            if len_cap % 2:
                len_cap -= 1
            pos_lst = pos_lst[:len_cap]
            neg_lst = neg_lst[:len_cap]
            neg_lst_second = neg_lst_second[:len_cap]
            pos_pairs = pos_lst.reshape(-1, 2)
            neg_pairs = torch.cat([neg_lst, neg_lst_second], dim=1)
            neg_pairs = neg_pairs[:pos_lst.shape[0]//2, :]
            pos_idx.append(pos_pairs)
            neg_idx.append(neg_pairs)
        pos_idx = torch.cat(pos_idx)
        neg_idx = torch.cat(neg_idx)"""
        assert pos_pairs.shape == neg_pairs.shape
        if len(pos_pairs) == 0:
            continue
        cluster_space_coords_filtered = cluster_space_coords[bmask]
        qs_filtered = q[bmask]
        pos_norms = (
            cluster_space_coords_filtered[pos_pairs[:, 0]]
            - cluster_space_coords_filtered[pos_pairs[:, 1]]
        ).norm(dim=-1)

        neg_norms = (
            cluster_space_coords_filtered[neg_pairs[:, 0]]
            - cluster_space_coords_filtered[neg_pairs[:, 1]]
        ).norm(dim=-1)
        q_pos = qs_filtered[pos_pairs[:, 0]]
        q_neg = qs_filtered[neg_pairs[:, 0]]
        q_s = torch.cat([q_pos, q_neg])
        norms_pos = torch.cat([pos_norms, neg_norms])
        ys = torch.cat([torch.ones_like(pos_norms), -torch.ones_like(neg_norms)])
        L_clusters += torch.sum(
            q_s * torch.nn.HingeEmbeddingLoss(reduce=None)(norms_pos, ys)
        )
        number_of_pairs += norms_pos.shape[0]
    if number_of_pairs > 0:
        L_clusters = L_clusters / number_of_pairs

    return L_clusters


def _calculate_delta_MC_debug(y, batch_g):
    """Verbose diagnostic variant retained for manual loss debugging."""
    graphs = dgl.unbatch(batch_g)
    batch_id = y[:, -1].view(-1)  # event IDs per hit
    df_list = []

    print(f"Total hits: {y.shape[0]}, Total events: {len(graphs)}")

    for i in range(len(graphs)):
        mask = batch_id == i
        y_i = y[mask]

        print(f"\nEvent {i}:")
        print(f"  Hits in event: {mask.sum()}")
        print(f"  y_i shape: {y_i.shape}")

        theta = y_i[:, 0].float()
        phi = y_i[:, 1].float()
        valid_direction = (
            torch.isfinite(theta)
            & torch.isfinite(phi)
            & (theta > 0)
            & (theta < torch.pi)
        )
        theta_safe = torch.nan_to_num(
            theta, nan=torch.pi / 2, posinf=torch.pi / 2, neginf=torch.pi / 2
        ).clamp(1e-6, torch.pi - 1e-6)
        phi_safe = torch.nan_to_num(phi, nan=0.0, posinf=0.0, neginf=0.0)
        pseudorapidity = -torch.log(torch.tan(theta_safe / 2))
        x1 = torch.stack((pseudorapidity, phi), dim=1)

        print(f"  x1 (eta, phi) shape: {x1.shape}")
        print(f"  x1:\n{x1}")

        delta_eta = pseudorapidity[:, None] - pseudorapidity[None, :]
        delta_phi = torch.remainder(
            phi_safe[:, None] - phi_safe[None, :] + torch.pi, 2 * torch.pi
        ) - torch.pi
        distance_matrix = torch.sqrt(delta_eta.square() + delta_phi.square())
        valid_pairs = valid_direction[:, None] & valid_direction[None, :]
        distance_matrix = distance_matrix.masked_fill(~valid_pairs, float("inf"))
        shape_d = distance_matrix.shape[0]

        print(f"  distance_matrix shape: {distance_matrix.shape}")
        if shape_d > 0:
            print(f"  distance_matrix first row: {distance_matrix[0]}")

        values, _ = torch.sort(distance_matrix, dim=1)
        print(f"  Sorted distances shape: {values.shape}")
        if shape_d > 0:
            print(f"  Sorted distances first row: {values[0]}")

        if shape_d > 1:
            delta_MC = values[:, 1]
        else:
            delta_MC = torch.ones((shape_d, 1)).view(-1).to(y_i.device)

        # Invalid directions, or events with no second valid truth track, get
        # neutral hard-negative weight instead of propagating NaN/inf.
        delta_MC = torch.nan_to_num(
            delta_MC, nan=1.0, posinf=1.0, neginf=1.0
        )

        print(f"  delta_MC shape: {delta_MC.shape}")
        print(f"  delta_MC: {delta_MC}")

        df_list.append(delta_MC)

    delta_MC = torch.cat(df_list)
    print(f"\nFinal delta_MC shape: {delta_MC.shape}")
    return delta_MC
def calculate_delta_MC(y, batch_g, dtype=None):
    graphs = dgl.unbatch(batch_g)
    batch_id = y[:, -1].view(-1)  # event IDs per hit
    df_list = []

    # print(f"Total particles: {y.shape[0]}, Total events: {len(graphs)}")

    for i in range(len(graphs)):
        mask = batch_id == i
        y_i = y[mask]

        # print(f"\nEvent {i}:")
        # print(f"  Particles in event: {mask.sum()}")
        # print(f"  y_i shape: {y_i.shape}")

        direction_dtype = y_i.dtype if dtype is None else dtype
        theta = y_i[:, 0].to(dtype=direction_dtype)
        phi = y_i[:, 1].to(dtype=direction_dtype)
        valid_direction = (
            torch.isfinite(theta)
            & torch.isfinite(phi)
            & (theta > 0)
            & (theta < torch.pi)
        )
        theta_safe = torch.nan_to_num(
            theta, nan=torch.pi / 2, posinf=torch.pi / 2, neginf=torch.pi / 2
        ).clamp(1e-6, torch.pi - 1e-6)
        phi_safe = torch.nan_to_num(phi, nan=0.0, posinf=0.0, neginf=0.0)
        pseudorapidity = -torch.log(torch.tan(theta_safe / 2))

        # print(f"  x1 (eta, phi) shape: {x1.shape}")
        # print(f"  x1:\n{x1}")

        delta_eta = pseudorapidity[:, None] - pseudorapidity[None, :]
        delta_phi = torch.remainder(
            phi_safe[:, None] - phi_safe[None, :] + torch.pi, 2 * torch.pi
        ) - torch.pi
        distance_matrix = torch.sqrt(delta_eta.square() + delta_phi.square())
        valid_pairs = valid_direction[:, None] & valid_direction[None, :]
        distance_matrix = distance_matrix.masked_fill(~valid_pairs, float("inf"))
        shape_d = distance_matrix.shape[0]

        values, _ = torch.sort(distance_matrix, dim=1)
    
        if shape_d > 1:
            delta_MC = values[:, 1]
        else:
            delta_MC = torch.ones((shape_d, 1)).view(-1).to(y_i.device)

        # Invalid directions, or events with no second valid truth track, get
        # neutral hard-negative weight instead of propagating NaN/inf.
        delta_MC = torch.nan_to_num(
            delta_MC, nan=1.0, posinf=1.0, neginf=1.0
        )

        # print(f"  delta_MC shape: {delta_MC.shape}")
        # print(f"  delta_MC: {delta_MC}")

        df_list.append(delta_MC)

    delta_MC = torch.cat(df_list)
    # print(f"\nFinal delta_MC shape: {delta_MC.shape}")
    return delta_MC

## deprecated code:
