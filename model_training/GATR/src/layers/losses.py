from typing import Tuple, Union
import numpy as np
import torch
from torch_scatter import scatter_max, scatter_add, scatter_mean
from src.layers.object_cond import calc_LV_Lbeta

def object_condensation_loss_tracking(
    batch,
    pred,
    y,
    return_resolution=False,
    clust_loss_only=True,
    add_energy_loss=False,
    calc_e_frac_loss=False,
    q_min=0.1,
    frac_clustering_loss=0.1,
    attr_weight=1.0,
    repul_weight=1.0,
    fill_loss_weight=1.0,
    use_average_cc_pos=0.0,
    loss_type="hgcalimplementation",
    output_dim=4,
    clust_space_norm="none",
    tracking=False,
    CLD=False,
):

    _, S = pred.shape
    if clust_loss_only:
        clust_space_dim = output_dim - 1
    else:
        clust_space_dim = output_dim - 28

    xj = pred[:, 0:clust_space_dim]  # xj: cluster space coords
    bj = torch.sigmoid(torch.reshape(pred[:, clust_space_dim], [-1, 1]))  # 3: betas
    
    original_coords = batch.ndata["pos_hits_xyz"]  # [:, 0:clust_space_dim]
    

    dev = batch.device
    clustering_index_l = batch.ndata["particle_number"]

    len_batch = len(batch.batch_num_nodes())
    batch_numbers = torch.repeat_interleave(
        torch.arange(len_batch, device=dev),  # [0, 1, 2, ..., len_batch-1]
        batch.batch_num_nodes(),              # number of nodes per graph
    )

    a = calc_LV_Lbeta(
        original_coords,
        batch,
        y,
        None,
        None,
        # momentum=None,
        # predicted_pid=None,
        beta=bj.view(-1),
        cluster_space_coords=xj,  # Predicted by model
        cluster_index_per_event=clustering_index_l.view(-1).long(),  # Truth hit->cluster index
        batch=batch_numbers.long(),
        qmin=q_min,
        return_regression_resolution=return_resolution,
        post_pid_pool_module=None,
        clust_space_dim=clust_space_dim,
        frac_combinations=frac_clustering_loss,
        attr_weight=attr_weight,
        repul_weight=repul_weight,
        fill_loss_weight=fill_loss_weight,
        use_average_cc_pos=use_average_cc_pos,
        loss_type=loss_type,
        tracking=tracking,
        CLD=CLD,
    )
    
    

    loss = a[0] + a[1] 

    return loss, a

# def object_condensation_loss_tracking(
#     batch,
#     pred,
#     y,
#     return_resolution=False,
#     clust_loss_only=True,
#     add_energy_loss=False,
#     calc_e_frac_loss=False,
#     q_min=0.1,
#     frac_clustering_loss=0.1,
#     attr_weight=1.0,
#     repul_weight=1.0,
#     fill_loss_weight=1.0,
#     use_average_cc_pos=0.0,
#     loss_type="hgcalimplementation",
#     output_dim=4,
#     clust_space_norm="none",
#     tracking=False,
#     CLD=False,
#     var_weight=0.0,
#     beta_suppress_weight=0.0,
#     s_B=1.0,
#     noise_index=0,
#     return_components=False,
# ):
#     _, S = pred.shape

#     if clust_loss_only:
#         clust_space_dim = output_dim - 1
#     else:
#         clust_space_dim = output_dim - 28

#     xj = pred[:, 0:clust_space_dim]         # cluster space coords
#     bj = torch.sigmoid(
#         torch.reshape(pred[:, clust_space_dim], [-1, 1])
#     )                                         # betas

#     clustering_index_l = batch.ndata["particle_number"]

#     dev = batch.device
#     len_batch = len(batch.batch_num_nodes())
#     batch_numbers = torch.repeat_interleave(
#         torch.arange(len_batch, device=dev),
#         batch.batch_num_nodes(),
#     )

#     result = object_condensation_loss(
#         coords=xj,
#         beta=bj.view(-1),
#         mc_index=clustering_index_l.view(-1).long(),
#         batch=batch_numbers.long(),
#         noise_index=noise_index,
#         qmin=q_min,
#         attr_weight=attr_weight,
#         repul_weight=repul_weight,
#         fill_loss_weight=fill_loss_weight,
#         use_average_cc_pos=use_average_cc_pos,
#         s_B=s_B,
#         beta_suppress_weight=beta_suppress_weight,
#         var_weight=var_weight,
#         return_components=return_components,
#     )

#     if return_components:
#         total, components = result
#         # Preserve the old (loss, a) return shape; stuff components into a tuple
#         # so callers doing `loss = a[0] + a[1]` can be updated to `loss = a[0]`
#         return total, components

#     # Legacy-compatible: return (total, (total,)) so callers doing
#     # `loss = a[0] + a[1]` break loudly and need updating to `loss = a[0]`
#     return result, (result,)


# def object_condensation_loss(
#     coords, beta, mc_index, batch,
#     noise_index=0, qmin=0.1,
#     attr_weight=1.0, repul_weight=1.0, fill_loss_weight=0.0,
#     use_average_cc_pos=0.0, s_B=1.0,
#     beta_suppress_weight=0.0,
#     var_weight=0.0,
#     return_components=False,
# ):
#     """Object condensation loss (hgcalimplementation-style) using torch_scatter.

#     Matches the original calc_LV_Lbeta semantics:
#     - Attraction: signal hits pulled toward their own condensation point.
#     - Repulsion: ALL hits (incl. noise) pushed away from non-own-cluster objects.
#     - Repulsion computed per-event to control memory.
#     - Beta suppression: penalizes high beta for non-alpha signal hits.

#     Additions on top of the OC loss as used in upstream:
#     - Within-cluster variance regularizer
#           L_var = mean_k mean_i ||x_i - mu_k||^2
#       where mu_k is the mean of signal-hit coords assigned to track k. This
#       directly shrinks the tail of each track cluster (an elongation pattern
#       we observed in embedding analyses, caused by the `log(d^2+1)` attractive
#       term + beta-weighted charge having weak gradients at the cluster edges).
#       Controlled by `var_weight`; if 0, the term is ignored and the loss
#       reduces to the standard OC formulation.
#     - If `return_components=True`, returns (total, dict of components) for
#       logging / diagnostics.
#     """
#     device = coords.device
#     beta = torch.nan_to_num(beta, nan=0.0)

#     is_noise = mc_index == noise_index
#     is_sig = ~is_noise

#     n_hits = coords.shape[0]
#     n_hits_sig = is_sig.sum().item()
#     if n_hits_sig < 4:
#         return torch.tensor(0.0, device=device, requires_grad=True)

#     sig_coords = coords[is_sig]
#     sig_beta = beta[is_sig]
#     sig_mc = mc_index[is_sig]
#     sig_batch = batch[is_sig]

#     # Per-event reincrementalization of signal labels -> contiguous 0..K_e-1
#     object_index = torch.empty_like(sig_mc)
#     n_objects_per_event_list = []
#     unique_events = sig_batch.unique()
#     for evt in unique_events:
#         evt_mask = sig_batch == evt
#         _, inv = sig_mc[evt_mask].unique(return_inverse=True)
#         object_index[evt_mask] = inv
#         n_objects_per_event_list.append(inv.max().item() + 1)

#     n_objects_per_event = torch.tensor(n_objects_per_event_list, device=device, dtype=torch.long)

#     # Make object_index globally unique across events
#     offsets = torch.zeros_like(n_objects_per_event)
#     offsets[1:] = n_objects_per_event[:-1].cumsum(dim=0)
#     _, event_remap = sig_batch.unique(return_inverse=True)
#     object_index = object_index + offsets[event_remap]

#     n_objects = n_objects_per_event.sum().item()
#     if n_objects < 2:
#         return torch.tensor(0.0, device=device, requires_grad=True)

#     # q for ALL hits (repulsion uses noise hits too, matching original)
#     q_all = (beta.clip(0.0, 1 - 1e-4).arctanh() / 1.01) ** 2 + qmin
#     q_sig = q_all[is_sig]

#     # Alpha points (condensation points)
#     q_alpha, index_alpha = scatter_max(q_sig, object_index)
#     x_alpha = sig_coords[index_alpha]
#     beta_alpha = sig_beta[index_alpha]

#     # --- Attractive potential (signal hits only, per-hit) ---
#     e1 = torch.exp(torch.tensor(1.0, device=device))
#     d_sq_own = ((sig_coords - x_alpha[object_index]) ** 2).sum(dim=1)
#     norms_att = torch.log(e1 * d_sq_own / 2 + 1)
#     V_att_per_hit = q_sig * q_alpha[object_index] * norms_att

#     V_att_per_obj = scatter_add(V_att_per_hit, object_index)
#     n_hits_per_obj = scatter_add(torch.ones(n_hits_sig, device=device), object_index)
#     V_att_per_obj = V_att_per_obj / (n_hits_per_obj + 1e-3)
#     L_V_att = V_att_per_obj.mean()

#     # # --- Within-cluster variance regularizer ---
#     # # For each track k, pull every assigned hit toward the track centroid with a
#     # # quadratic penalty. Unlike the attractive term it is (a) unweighted by beta
#     # # so low-beta tail hits still feel the pull, (b) quadratic in distance so
#     # # gradients grow at the tail, (c) anchored at the centroid not the alpha
#     # # point so it is shift-invariant. Controlled by `var_weight`.
#     # x_centroid = scatter_mean(sig_coords, object_index, dim=0)
#     # d_sq_centroid = ((sig_coords - x_centroid[object_index]) ** 2).sum(dim=1)
#     # L_var_per_obj = scatter_mean(d_sq_centroid, object_index)
#     # L_var = L_var_per_obj.mean()

#     d_sq_alpha = ((sig_coords - x_alpha[object_index]) ** 2).sum(dim=1)
#     L_var_alpha = scatter_mean(d_sq_alpha, object_index).mean()


#     # --- Repulsive potential (per-event, ALL hits incl. noise, matching original) ---
#     # Build global object assignment: signal hits get their object_index, noise gets -1
#     all_object_index = torch.full((n_hits,), -1, device=device, dtype=torch.long)
#     all_object_index[is_sig] = object_index

#     rep_sum = torch.tensor(0.0, device=device)
#     rep_n_objects = 0
#     obj_offset = 0

#     for i, evt_val in enumerate(unique_events):
#         n_evt_obj = n_objects_per_event[i].item()
#         if n_evt_obj < 2:
#             obj_offset += n_evt_obj
#             continue

#         evt_mask = batch == evt_val
#         evt_coords = coords[evt_mask]
#         evt_q = q_all[evt_mask]
#         evt_obj = all_object_index[evt_mask]

#         evt_x_alpha = x_alpha[obj_offset:obj_offset + n_evt_obj]
#         evt_q_alpha = q_alpha[obj_offset:obj_offset + n_evt_obj]

#         d_sq = ((evt_coords.unsqueeze(1) - evt_x_alpha.unsqueeze(0)) ** 2).sum(-1)
#         exp_rep = torch.exp(-d_sq / 2)

#         # M_inv: 1 for non-own-cluster pairs. Noise hits (obj=-1) repel from ALL objects.
#         local_obj = evt_obj.clone()
#         has_obj = local_obj >= 0
#         local_obj[has_obj] -= obj_offset
#         own_mask = torch.zeros(evt_coords.shape[0], n_evt_obj, device=device)
#         if has_obj.any():
#             own_mask[has_obj] = torch.nn.functional.one_hot(
#                 local_obj[has_obj], num_classes=n_evt_obj
#             ).float()
#         M_inv = 1.0 - own_mask

#         V_rep = evt_q.unsqueeze(1) * evt_q_alpha.unsqueeze(0) * exp_rep * M_inv
#         V_rep_per_obj = V_rep.sum(dim=0)
#         n_rep_terms = M_inv.sum(dim=0).clamp(min=1.0)
#         V_rep_per_obj = V_rep_per_obj / n_rep_terms

#         rep_sum = rep_sum + V_rep_per_obj.sum()
#         rep_n_objects += n_evt_obj
#         obj_offset += n_evt_obj

#     L_V_rep = rep_sum / max(rep_n_objects, 1)
#     L_V = attr_weight * L_V_att + repul_weight * L_V_rep

#     # --- L_beta signal ---
#     beta_sum_per_obj = scatter_add(sig_beta, object_index)
#     L_beta_sig = torch.mean(1 - beta_alpha + 1 - torch.clip(beta_sum_per_obj, 0, 1))

#     # --- L_beta noise (matches original: .sum() / batch_size) ---
#     batch_size = batch.unique().numel()
#     L_beta_noise = torch.tensor(0.0, device=device)
#     if is_noise.any():
#         noise_beta = beta[is_noise]
#         noise_batch = batch[is_noise]
#         _, noise_evt_remap = noise_batch.unique(return_inverse=True)
#         n_noise_per_evt = scatter_add(
#             torch.ones_like(noise_evt_remap, dtype=torch.float), noise_evt_remap
#         ).clamp(min=1.0)
#         beta_noise_per_evt = scatter_add(noise_beta, noise_evt_remap)
#         L_beta_noise = s_B * (beta_noise_per_evt / n_noise_per_evt).sum() / batch_size

#     # --- Beta suppression: push non-alpha signal betas toward 0 ---
#     L_beta_suppress = torch.tensor(0.0, device=device)
#     if beta_suppress_weight > 0 and n_hits_sig > n_objects:
#         is_alpha = torch.zeros(n_hits_sig, dtype=torch.bool, device=device)
#         is_alpha[index_alpha] = True
#         L_beta_suppress = beta_suppress_weight * sig_beta[~is_alpha].mean()

#     # total = L_V + L_beta_sig + L_beta_noise + L_beta_suppress + var_weight * L_var
#     # if return_components:
#     #     components = {
#     #         "L_V_att": L_V_att.detach(),
#     #         "L_V_rep": L_V_rep.detach(),
#     #         "L_beta_sig": L_beta_sig.detach() if torch.is_tensor(L_beta_sig) else torch.tensor(float(L_beta_sig), device=device),
#     #         "L_beta_noise": L_beta_noise.detach() if torch.is_tensor(L_beta_noise) else torch.tensor(float(L_beta_noise), device=device),
#     #         "L_beta_suppress": L_beta_suppress.detach() if torch.is_tensor(L_beta_suppress) else torch.tensor(float(L_beta_suppress), device=device),
#     #         "L_var": L_var.detach(),
#     #         "var_weight": torch.tensor(float(var_weight), device=device),
#     #     }
#     #     return total, components


#     total = L_V + L_beta_sig + L_beta_noise + L_beta_suppress + var_weight * L_var_alpha
#     if return_components:
#         components = {
#             "L_V_att": L_V_att.detach(),
#             "L_V_rep": L_V_rep.detach(),
#             "L_beta_sig": L_beta_sig.detach() if torch.is_tensor(L_beta_sig) else torch.tensor(float(L_beta_sig), device=device),
#             "L_beta_noise": L_beta_noise.detach() if torch.is_tensor(L_beta_noise) else torch.tensor(float(L_beta_noise), device=device),
#             "L_beta_suppress": L_beta_suppress.detach() if torch.is_tensor(L_beta_suppress) else torch.tensor(float(L_beta_suppress), device=device),
#             "L_var_alpha": L_var_alpha.detach(),
#             "var_weight": torch.tensor(float(var_weight), device=device),
#         }
#         return total, components


#     return total