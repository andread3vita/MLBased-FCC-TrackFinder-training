"""The CIRCE object-condensation objective, shared verbatim by both trainers."""

from __future__ import annotations

import torch
from torch_scatter import scatter_add, scatter_max, scatter_mean


def sequence_lengths_to_batch(sequence_lengths, device):
    """Return the event index of every hit in a packed CIRCE batch."""
    return torch.repeat_interleave(
        torch.arange(len(sequence_lengths), device=device, dtype=torch.long),
        torch.tensor(sequence_lengths, device=device, dtype=torch.long),
    )


def variance_weight(epoch, target_weight, warmup_epochs):
    """CIRCE/GATr linear embedding-variance warmup (one-indexed epoch)."""
    if warmup_epochs <= 0:
        return float(target_weight)
    fraction = min(1.0, max(0.0, (epoch - 1) / max(warmup_epochs, 1)))
    return float(target_weight) * fraction


def object_condensation_loss(
    coords, beta, mc_index, batch,
    noise_index=0, qmin=0.1,
    attr_weight=1.0, repul_weight=1.0, fill_loss_weight=0.0,
    use_average_cc_pos=0.0, s_B=1.0,
    beta_suppress_weight=0.0,
    var_weight=0.0,
    return_components=False,
    detach_components=True,
    oc_mode="paper_hinge",
    track_separation_weight=None,
):
    """CIRCE's selected logarithmic-attraction/hinge-repulsion OC loss.

    ``fill_loss_weight`` and ``use_average_cc_pos`` are retained in the
    signature for checkpoint/CLI compatibility.  They are not terms in the
    selected CIRCE objective.  The fair-comparison launcher always selects
    ``paper_hinge``.
    """
    device = coords.device
    beta = torch.nan_to_num(beta, nan=0.0)
    is_noise = mc_index == noise_index
    is_sig = ~is_noise

    def connected_zero_result():
        zero = coords.sum() * 0.0 + beta.sum() * 0.0
        if not return_components:
            return zero
        component_zero = zero.detach() if detach_components else zero
        return zero, {
            "L_V_att": component_zero,
            "L_V_rep": component_zero,
            "L_beta_sig": component_zero,
            "L_beta_noise": component_zero,
            "L_beta_suppress": component_zero,
            "L_var": component_zero,
            "var_weight": torch.tensor(float(var_weight), device=device),
        }

    n_hits = coords.shape[0]
    n_hits_sig = is_sig.sum().item()
    if n_hits_sig < 4:
        return connected_zero_result()

    sig_coords = coords[is_sig]
    sig_beta = beta[is_sig]
    sig_mc = mc_index[is_sig]
    sig_batch = batch[is_sig]

    object_index = torch.empty_like(sig_mc)
    n_objects_per_event_list = []
    unique_events = sig_batch.unique()
    for evt in unique_events:
        evt_mask = sig_batch == evt
        _, inv = sig_mc[evt_mask].unique(return_inverse=True)
        object_index[evt_mask] = inv
        n_objects_per_event_list.append(inv.max().item() + 1)

    n_objects_per_event = torch.tensor(
        n_objects_per_event_list, device=device, dtype=torch.long
    )
    offsets = torch.zeros_like(n_objects_per_event)
    offsets[1:] = n_objects_per_event[:-1].cumsum(dim=0)
    _, event_remap = sig_batch.unique(return_inverse=True)
    object_index = object_index + offsets[event_remap]
    n_objects = n_objects_per_event.sum().item()
    if n_objects < 2:
        return connected_zero_result()
    if oc_mode not in ("paper_hinge", "ggtf"):
        raise ValueError(f"Unknown object-condensation mode: {oc_mode}")

    q_scale = 1.01 if oc_mode == "ggtf" else 1.0
    q_all = (beta.clip(0.0, 1 - 1e-4).arctanh() / q_scale) ** 2 + qmin
    q_sig = q_all[is_sig]
    q_alpha, index_alpha = scatter_max(q_sig, object_index)
    x_alpha = sig_coords[index_alpha]
    beta_alpha = sig_beta[index_alpha]
    object_repulsion_weight = None
    if track_separation_weight is not None:
        if track_separation_weight.shape != beta.shape:
            raise ValueError("track_separation_weight must have one value per input hit")
        object_repulsion_weight = scatter_mean(
            track_separation_weight[is_sig].float(), object_index
        )

    e1 = torch.exp(torch.tensor(1.0, device=device))
    d_sq_own = ((sig_coords - x_alpha[object_index]) ** 2).sum(dim=1)
    norms_att = torch.log(e1 * d_sq_own / 2 + 1)
    v_att_per_hit = q_sig * q_alpha[object_index] * norms_att
    v_att_per_obj = scatter_add(v_att_per_hit, object_index)
    n_hits_per_obj = scatter_add(torch.ones(n_hits_sig, device=device), object_index)
    l_v_att = (v_att_per_obj / (n_hits_per_obj + 1e-3)).mean()

    x_centroid = scatter_mean(sig_coords, object_index, dim=0)
    d_sq_centroid = ((sig_coords - x_centroid[object_index]) ** 2).sum(dim=1)
    l_var = scatter_mean(d_sq_centroid, object_index).mean()

    all_object_index = torch.full((n_hits,), -1, device=device, dtype=torch.long)
    all_object_index[is_sig] = object_index
    rep_sum = torch.tensor(0.0, device=device)
    rep_normalization = torch.tensor(0.0, device=device)
    obj_offset = 0
    for i, evt_val in enumerate(unique_events):
        n_evt_obj = n_objects_per_event[i].item()
        if n_evt_obj < 2:
            obj_offset += n_evt_obj
            continue
        evt_mask = batch == evt_val
        evt_coords = coords[evt_mask]
        evt_q = q_all[evt_mask]
        evt_obj = all_object_index[evt_mask]
        evt_x_alpha = x_alpha[obj_offset:obj_offset + n_evt_obj]
        evt_q_alpha = q_alpha[obj_offset:obj_offset + n_evt_obj]
        d_sq = ((evt_coords.unsqueeze(1) - evt_x_alpha.unsqueeze(0)) ** 2).sum(-1)
        if oc_mode == "ggtf":
            exp_rep = torch.exp(-d_sq / 2.0)
        else:
            exp_rep = torch.relu(1.0 - torch.sqrt(d_sq.clamp(min=1e-12)))
        local_obj = evt_obj.clone()
        has_obj = local_obj >= 0
        local_obj[has_obj] -= obj_offset
        own_mask = torch.zeros(evt_coords.shape[0], n_evt_obj, device=device)
        if has_obj.any():
            own_mask[has_obj] = torch.nn.functional.one_hot(
                local_obj[has_obj], num_classes=n_evt_obj
            ).float()
        m_inv = 1.0 - own_mask
        v_rep = evt_q.unsqueeze(1) * evt_q_alpha.unsqueeze(0) * exp_rep * m_inv
        v_rep_per_obj = v_rep.sum(dim=0) / m_inv.sum(dim=0).clamp(min=1.0)
        if object_repulsion_weight is None:
            rep_sum = rep_sum + v_rep_per_obj.sum()
            rep_normalization = rep_normalization + n_evt_obj
        else:
            evt_weight = object_repulsion_weight[obj_offset:obj_offset + n_evt_obj]
            rep_sum = rep_sum + (v_rep_per_obj * evt_weight).sum()
            rep_normalization = rep_normalization + evt_weight.sum()
        obj_offset += n_evt_obj

    l_v_rep = rep_sum / rep_normalization.clamp(min=1.0)
    l_v = attr_weight * l_v_att + repul_weight * l_v_rep
    beta_sum_per_obj = scatter_add(sig_beta, object_index)
    l_beta_sig = torch.mean(1 - beta_alpha + 1 - torch.clip(beta_sum_per_obj, 0, 1))
    batch_size = batch.unique().numel()
    l_beta_noise = torch.tensor(0.0, device=device)
    if is_noise.any():
        noise_beta = beta[is_noise]
        noise_batch = batch[is_noise]
        _, noise_evt_remap = noise_batch.unique(return_inverse=True)
        n_noise_per_evt = scatter_add(
            torch.ones_like(noise_evt_remap, dtype=torch.float), noise_evt_remap
        ).clamp(min=1.0)
        beta_noise_per_evt = scatter_add(noise_beta, noise_evt_remap)
        l_beta_noise = s_B * (beta_noise_per_evt / n_noise_per_evt).sum() / batch_size

    l_beta_suppress = torch.tensor(0.0, device=device)
    if beta_suppress_weight > 0 and n_hits_sig > n_objects:
        is_alpha = torch.zeros(n_hits_sig, dtype=torch.bool, device=device)
        is_alpha[index_alpha] = True
        l_beta_suppress = beta_suppress_weight * sig_beta[~is_alpha].mean()

    total = l_v + l_beta_sig + l_beta_noise + l_beta_suppress + var_weight * l_var
    if not return_components:
        return total

    def component(value):
        if not torch.is_tensor(value):
            value = torch.tensor(float(value), device=device)
        return value.detach() if detach_components else value

    return total, {
        "L_V_att": component(l_v_att),
        "L_V_rep": component(l_v_rep),
        "L_beta_sig": component(l_beta_sig),
        "L_beta_noise": component(l_beta_noise),
        "L_beta_suppress": component(l_beta_suppress),
        "L_var": component(l_var),
        "var_weight": torch.tensor(float(var_weight), device=device),
    }
