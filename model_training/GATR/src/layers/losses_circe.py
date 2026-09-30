"""CIRCE's object-condensation loss, portable into the GGTF pipeline.

This is the objective CIRCE trains with: the logarithmic attraction and both
beta terms are identical in form to GGTF's `hgcalimplementation` branch, the
repulsion uses Kieseler's compact hinge max(0, 1 - d) instead of the Gaussian,
and there are two extra regularisers (beta suppression on non-alpha signal
hits, weight 0.1, and a within-cluster variance term, weight 0.3).

Hyperparameter optimization on 50,000 matched events established CIRCE's
champion configuration:
    - qmin = 3.0 (prevents curling-track fragmentation without cluster bloat)
    - attr_weight = 1.0, repul_weight = 2.0 (ensures clean track separation)
    - beta_suppress_weight = 0.1 (essential with compact hinge repulsion to
      prevent multiple condensation seeds along 50-150 hit drift helices)
    - var_weight = 0.3 (tightens intra-track latent spread)

Why ship it: the conformal model trains noticeably better under this
objective than under the Gaussian-repulsion one, so reproducing CIRCE
results inside this pipeline needs it. Select per run: the conformal wrapper
defaults to this loss; set `args.loss_backend = "ggtf"` (as
`train_algebra_ab.py` does) for algebra-only A/B runs where both arms must
share the exact same objective.

`circe_condensation_loss` below is the thin adapter matching the calling
convention of `object_condensation_loss_tracking`.
"""
from __future__ import annotations

import torch
from torch_scatter import scatter_add, scatter_max, scatter_mean

from src.layers.batch_operations import obtain_batch_numbers


def object_condensation_loss(
    coords, beta, mc_index, batch,
    noise_index=0, qmin=3.0,
    attr_weight=1.0, repul_weight=2.0, fill_loss_weight=0.0,
    use_average_cc_pos=0.0, s_B=1.0,
    beta_suppress_weight=0.1,
    var_weight=0.3,
    return_components=False,
    detach_components=True,
    oc_mode="paper_hinge",
    track_separation_weight=None,
):
    """Hybrid object-condensation loss using torch_scatter.

    Both modes use the logarithmic attractive potential and per-object
    normalization from the HGCAL/GGTF implementation. ``paper_hinge`` combines
    that attraction with Kieseler's compact-support hinge repulsion and the
    original arctanh-squared charge. It is the empirically selected tracking
    objective, but it is not the literal Kieseler loss (whose attraction is
    quadratic). ``ggtf`` instead uses GGTF's Gaussian repulsion and softened
    charge, optionally with inverse nearest-track-separation weights.

    Matches the original calc_LV_Lbeta semantics:
    - Attraction: signal hits pulled toward their own condensation point
    - Repulsion: ALL hits (incl. noise) pushed away from non-own-cluster objects
    - Repulsion computed per-event to control memory
    - Beta suppression: penalizes high beta for non-alpha signal hits

    v35 additions:
    - Within-cluster variance regularizer L_var = mean_k mean_i ||x_i - mu_k||^2
      where mu_k is the mean of signal-hit coords assigned to track k.
    - If return_components=True, returns (total, dict of components).
    """
    device = coords.device
    beta = torch.nan_to_num(beta, nan=0.0)

    is_noise = mc_index == noise_index
    is_sig = ~is_noise

    n_hits = coords.shape[0]
    n_hits_sig = is_sig.sum().item()
    if n_hits_sig < 4:
        return torch.tensor(0.0, device=device, requires_grad=True)

    sig_coords = coords[is_sig]
    sig_beta = beta[is_sig]
    sig_mc = mc_index[is_sig]
    sig_batch = batch[is_sig]

    # Per-event reincrementalization of signal labels -> contiguous 0..K_e-1
    object_index = torch.empty_like(sig_mc)
    n_objects_per_event_list = []
    unique_events = sig_batch.unique()
    for evt in unique_events:
        evt_mask = sig_batch == evt
        _, inv = sig_mc[evt_mask].unique(return_inverse=True)
        object_index[evt_mask] = inv
        n_objects_per_event_list.append(inv.max().item() + 1)

    n_objects_per_event = torch.tensor(n_objects_per_event_list, device=device, dtype=torch.long)

    # Make object_index globally unique across events
    offsets = torch.zeros_like(n_objects_per_event)
    offsets[1:] = n_objects_per_event[:-1].cumsum(dim=0)
    _, event_remap = sig_batch.unique(return_inverse=True)
    object_index = object_index + offsets[event_remap]

    n_objects = n_objects_per_event.sum().item()
    if n_objects < 2:
        return torch.tensor(0.0, device=device, requires_grad=True)

    if oc_mode not in ("paper_hinge", "ggtf"):
        raise ValueError(f"Unknown object-condensation mode: {oc_mode}")

    # q for ALL hits (repulsion uses noise hits too, matching original).
    # GGTF's HGCAL-derived implementation divides by 1.01; the paper mode does not.
    q_scale = 1.01 if oc_mode == "ggtf" else 1.0
    q_all = (beta.clip(0.0, 1 - 1e-4).arctanh() / q_scale) ** 2 + qmin
    q_sig = q_all[is_sig]

    # Alpha points (condensation points)
    q_alpha, index_alpha = scatter_max(q_sig, object_index)
    x_alpha = sig_coords[index_alpha]
    beta_alpha = sig_beta[index_alpha]
    object_repulsion_weight = None
    if track_separation_weight is not None:
        if track_separation_weight.shape != beta.shape:
            raise ValueError(
                "track_separation_weight must have one value per input hit"
            )
        object_repulsion_weight = scatter_mean(
            track_separation_weight[is_sig].float(), object_index
        )

    # --- Attractive potential (signal hits only, per-hit) ---
    e1 = torch.exp(torch.tensor(1.0, device=device))
    d_sq_own = ((sig_coords - x_alpha[object_index]) ** 2).sum(dim=1)
    norms_att = torch.log(e1 * d_sq_own / 2 + 1)
    V_att_per_hit = q_sig * q_alpha[object_index] * norms_att

    V_att_per_obj = scatter_add(V_att_per_hit, object_index)
    n_hits_per_obj = scatter_add(torch.ones(n_hits_sig, device=device), object_index)
    V_att_per_obj = V_att_per_obj / (n_hits_per_obj + 1e-3)
    L_V_att = V_att_per_obj.mean()

    # --- Within-cluster variance regularizer (v35) ---
    # older torch_scatter (as in the gatr:v9 image) needs the index expanded
    # to the source's shape for multi-column scatters
    _idx2 = object_index.unsqueeze(1).expand(-1, sig_coords.size(1))
    x_centroid = scatter_mean(sig_coords, _idx2, dim=0)
    d_sq_centroid = ((sig_coords - x_centroid[object_index]) ** 2).sum(dim=1)
    L_var_per_obj = scatter_mean(d_sq_centroid, object_index)
    L_var = L_var_per_obj.mean()

    # --- Repulsive potential (per-event, ALL hits incl. noise, matching original) ---
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
            # GGTF/HGCAL tracking path: Gaussian in squared latent distance.
            exp_rep = torch.exp(-d_sq / 2.0)
        else:
            # Kieseler 2002.03605: compact-support linear hinge.
            exp_rep = torch.relu(1.0 - torch.sqrt(d_sq.clamp(min=1e-12)))

        local_obj = evt_obj.clone()
        has_obj = local_obj >= 0
        local_obj[has_obj] -= obj_offset
        own_mask = torch.zeros(evt_coords.shape[0], n_evt_obj, device=device)
        if has_obj.any():
            own_mask[has_obj] = torch.nn.functional.one_hot(
                local_obj[has_obj], num_classes=n_evt_obj
            ).float()
        M_inv = 1.0 - own_mask

        V_rep = evt_q.unsqueeze(1) * evt_q_alpha.unsqueeze(0) * exp_rep * M_inv
        V_rep_per_obj = V_rep.sum(dim=0)
        n_rep_terms = M_inv.sum(dim=0).clamp(min=1.0)
        V_rep_per_obj = V_rep_per_obj / n_rep_terms

        if object_repulsion_weight is None:
            rep_sum = rep_sum + V_rep_per_obj.sum()
            rep_normalization = rep_normalization + n_evt_obj
        else:
            evt_weight = object_repulsion_weight[
                obj_offset:obj_offset + n_evt_obj
            ]
            rep_sum = rep_sum + (V_rep_per_obj * evt_weight).sum()
            rep_normalization = rep_normalization + evt_weight.sum()
        obj_offset += n_evt_obj

    L_V_rep = rep_sum / rep_normalization.clamp(min=1.0)
    L_V = attr_weight * L_V_att + repul_weight * L_V_rep

    # --- L_beta signal ---
    beta_sum_per_obj = scatter_add(sig_beta, object_index)
    L_beta_sig = torch.mean(1 - beta_alpha + 1 - torch.clip(beta_sum_per_obj, 0, 1))

    # --- L_beta noise (matches original: .sum() / batch_size) ---
    batch_size = batch.unique().numel()
    L_beta_noise = torch.tensor(0.0, device=device)
    if is_noise.any():
        noise_beta = beta[is_noise]
        noise_batch = batch[is_noise]
        _, noise_evt_remap = noise_batch.unique(return_inverse=True)
        n_noise_per_evt = scatter_add(
            torch.ones_like(noise_evt_remap, dtype=torch.float), noise_evt_remap
        ).clamp(min=1.0)
        beta_noise_per_evt = scatter_add(noise_beta, noise_evt_remap)
        L_beta_noise = s_B * (beta_noise_per_evt / n_noise_per_evt).sum() / batch_size

    # --- Beta suppression: push non-alpha signal betas toward 0 ---
    L_beta_suppress = torch.tensor(0.0, device=device)
    if beta_suppress_weight > 0 and n_hits_sig > n_objects:
        is_alpha = torch.zeros(n_hits_sig, dtype=torch.bool, device=device)
        is_alpha[index_alpha] = True
        L_beta_suppress = beta_suppress_weight * sig_beta[~is_alpha].mean()

    total = L_V + L_beta_sig + L_beta_noise + L_beta_suppress + var_weight * L_var
    if return_components:
        def component(value):
            if not torch.is_tensor(value):
                value = torch.tensor(float(value), device=device)
            return value.detach() if detach_components else value

        components = {
            "L_V_att": component(L_V_att),
            "L_V_rep": component(L_V_rep),
            "L_beta_sig": component(L_beta_sig),
            "L_beta_noise": component(L_beta_noise),
            "L_beta_suppress": component(L_beta_suppress),
            "L_var": component(L_var),
            "var_weight": torch.tensor(float(var_weight), device=device),
        }
        return total, components
    return total


def circe_condensation_loss(batch_g, model_output, y, args):
    """Adapter with the same inputs as object_condensation_loss_tracking.

    Reads the clustering coordinates and beta from the model output columns,
    the per-hit particle id from `particle_number` (0 = noise, matching this
    loss's noise_index), and the batch assignment from the graph structure.
    Returns (total, components) like the GGTF loss does.
    """
    coords = model_output[:, :-1]
    beta = torch.sigmoid(model_output[:, -1])
    mc_index = batch_g.ndata["particle_number"].long()
    batch = obtain_batch_numbers(batch_g).long()

    # CIRCE champion defaults from the systematic 5-way ablation:
    # qmin=3.0, repul_weight=2.0, beta_suppress=0.1, var_weight=0.3.
    # If args still carries GGTF parser defaults (qmin=0.1, repul=1.0),
    # use CIRCE's validated defaults instead.
    qmin = getattr(args, "qmin", 3.0)
    if qmin == 0.1:
        qmin = 3.0
    repul_weight = getattr(args, "L_repulsive_weight", 2.0)
    if repul_weight == 1.0:
        repul_weight = 2.0

    total, components = object_condensation_loss(
        coords,
        beta,
        mc_index,
        batch,
        noise_index=0,
        qmin=qmin,
        attr_weight=getattr(args, "L_attractive_weight", 1.0),
        repul_weight=repul_weight,
        s_B=1.0,
        beta_suppress_weight=getattr(args, "beta_suppress_weight", 0.1),
        var_weight=getattr(args, "var_weight", 0.3),
        return_components=True,
        oc_mode="paper_hinge",
    )
    # the wandb logger indexes the components positionally, matching the
    # tuple object_condensation_loss_tracking returns:
    # [lv, lbeta, attractive, repulsive, beta_sig, beta_noise]
    lv = components["L_V_att"] + components["L_V_rep"]
    lbeta = (components["L_beta_sig"] + components["L_beta_noise"]
             + components["L_beta_suppress"])
    losses = [lv, lbeta,
              components["L_V_att"], components["L_V_rep"],
              components["L_beta_sig"], components["L_beta_noise"]]
    return total, losses
