"""C-GATr FCC track-finding model — simplified, M1-M4 baked in.

All ablation flags removed; these improvements are now permanent:
  M1: grade-wise equivariant LayerNorm (normalization.py)
  M2: drift-radius fixed-scale normalization (dc[:,7]/5.0)
  M3: isotropic position normalization (pos/1000.0)
  M4: faithful OC loss (Kieseler 2002.03605 hinge repulsive + arctanh^2)

M5 (invariant-only readout) was tested and discarded — ablation showed
higher loss (~0.95-1.06) vs equivariant readout (~0.78-0.84). The
multivector output is used directly.
"""

import os
import sys
import copy
import math
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch_scatter import scatter_max, scatter_add, scatter_mean

from src.cgatr.nets.cgatr import CGATr
from src.cgatr.layers.attention.config import SelfAttentionConfig
from src.cgatr.layers.mlp.config import MLPConfig
from src.cgatr.interface.point import embed_point
from src.cgatr.interface.scalar import embed_scalar
from src.cgatr.interface.circle import embed_circle_ipns
from src.cgatr.primitives.linear import _compute_se3_equi_linear_basis
from src.cgatr.primitives.attention import _build_dist_basis, block_diagonal_bool_mask
from src.cgatr.primitives.invariants import compute_inner_product_mask
from src.cgatr.primitives.dual import _DualCache
from src.dataset.parquet_dataset import IDEAParquetDataset, collate_idea_events


class CGATrParquetModel(nn.Module):
    """C-GATr model with M1-M5 baked in. Train from scratch (--init_weights none).

    VTX hits  -> CGA null point  (grade-1 vector,  32-dim MV)
    DC  hits  -> IPNS circle     (grade-2 bivector, 32-dim MV)
    Both in ONE multivector channel.
    """

    def __init__(self, args):
        super().__init__()
        self.args = args
        # M3: isotropic norm — single scale, preserves E(3) rotation equivariance.
        self.pos_scale = 1000.0

        gp_sparse = torch.load("cga_utils/cga_geometric_product.pt", weights_only=False)
        self.register_buffer("basis_gp", gp_sparse.to_dense().to(dtype=torch.float32))

        op_sparse = torch.load("cga_utils/cga_outer_product.pt", weights_only=False)
        self.register_buffer("basis_outer", op_sparse.to_dense().to(dtype=torch.float32))

        metadata = torch.load("cga_utils/cga_metadata.pt", weights_only=False)
        _DualCache.init_from_metadata(metadata, device=torch.device("cpu"))

        # Canonical SE(3)-equivariant CGA linear basis (de Haan et al. 2311.04744
        # Sec. 3.3), computed as the null space of the Lie-algebra equivariance
        # constraint. Replaces the old 9-element basis (which carried 3
        # non-equivariant Hodge cross-grade maps). Self-verifies at construction.
        pin_basis = _compute_se3_equi_linear_basis(
            self.basis_gp, device=torch.device("cpu"), dtype=torch.float32
        )
        basis_q, basis_k = _build_dist_basis(device=torch.device("cpu"), dtype=torch.float32)
        basis_ip_weights = compute_inner_product_mask(self.basis_gp, device=torch.device("cpu"))

        self.cgatr = CGATr(
            in_mv_channels=1,
            out_mv_channels=1,
            hidden_mv_channels=args.hidden_mv_channels,
            in_s_channels=None,
            out_s_channels=None,
            hidden_s_channels=args.hidden_s_channels,
            num_blocks=args.num_blocks,
            attention=SelfAttentionConfig(),
            mlp=MLPConfig(),
            basis_gp=self.basis_gp,
            basis_ip_weights=basis_ip_weights,
            basis_outer=self.basis_outer,
            basis_pin=pin_basis,
            basis_q=basis_q,
            basis_k=basis_k,
            checkpoint_blocks=getattr(args, "grad_checkpoint", False),
        )

        self.embed_dim = args.embed_dim
        self.clustering = nn.Linear(32, self.embed_dim, bias=False)
        self.beta = nn.Linear(32, 1)

    def forward(self, features, seq_lens):
        pos = features[:, :3]
        hit_type = features[:, 3:4]
        # M3: isotropic norm
        pos_normed = pos / self.pos_scale

        is_vtx = (hit_type.squeeze(-1) == 0)
        is_dc = (hit_type.squeeze(-1) == 1)

        mv = torch.zeros(features.shape[0], 32, device=features.device, dtype=features.dtype)

        if is_vtx.any():
            mv[is_vtx] = embed_point(pos_normed[is_vtx]).to(mv.dtype)

        if is_dc.any():
            dc = features[is_dc]
            # M3: isotropic norm for wire positions
            wire_normed = dc[:, 4:7] / self.pos_scale
            # M2: fixed positive scale preserves monotonic drift-to-radius mapping
            drift_normed = dc[:, 7] / 5.0
            cos_s = torch.cos(dc[:, 9])
            sin_s = torch.sin(dc[:, 9])
            cos_a = torch.cos(dc[:, 8])
            sin_a = torch.sin(dc[:, 8])
            wire_dir = torch.stack([sin_s * cos_a, sin_s * sin_a, cos_s], dim=-1)
            wire_dir = wire_dir / (torch.norm(wire_dir, dim=-1, keepdim=True) + 1e-8)
            mv[is_dc] = embed_circle_ipns(wire_normed, wire_dir, drift_normed, self.basis_outer).to(mv.dtype)

        mv = mv + embed_scalar(hit_type)

        if self.args.normalize_mv_inputs:
            # Euclidean per-hit normalization. This is ROTATION-equivariant (a
            # rotation acts as a real orthogonal map on the 32 components, so the
            # L2 norm is preserved) — which is the symmetry that matters for
            # tracking. It is deliberately NOT translation-equivariant: the IP and
            # detector sit at fixed positions, so absolute position is meaningful
            # and translation is not a physical symmetry here. The equivariant CGA
            # grade-wise norm cannot be used on the inputs because VTX hits are
            # null vectors (<P,P> = 0), so their CGA norm is identically zero.
            mv_norm = torch.norm(mv, dim=-1, keepdim=True).clamp(min=1e-6)
            mv = mv / mv_norm

        mv = mv.unsqueeze(1)  # (N, 1, 32) — single channel

        # Torch-native block-diagonal mask (no xformers dependency)
        mask = block_diagonal_bool_mask(seq_lens, device=features.device)
        out_mv, _ = self.cgatr(mv, scalars=None, attention_mask=mask)
        out = out_mv[:, 0, :]

        return torch.cat([self.clustering(out), self.beta(out)], dim=1)


# ---------------------------------------------------------------------------
# Object condensation loss — torch_scatter port of hgcalimplementation
# (no DGL dependency)
# ---------------------------------------------------------------------------
def object_condensation_loss(
    coords, beta, mc_index, batch,
    noise_index=0, qmin=0.1,
    attr_weight=1.0, repul_weight=1.0, fill_loss_weight=0.0,
    use_average_cc_pos=0.0, s_B=1.0,
    beta_suppress_weight=0.0,
    var_weight=0.0,
    return_components=False,
):
    """Object condensation loss (hgcalimplementation-style) using torch_scatter.

    M4 baked in: faithful OC (Kieseler 2002.03605 hinge repulsive + arctanh^2
    with /1.0). All ablation flags removed.

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
    # M4: faithful OC (Kieseler 2002.03605) — hardcoded, no env flag.
    _faithful = True
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

    # q for ALL hits (repulsion uses noise hits too, matching original)
    # M4: /1.0 (faithful, no /1.01)
    q_all = (beta.clip(0.0, 1 - 1e-4).arctanh() / 1.0) ** 2 + qmin
    q_sig = q_all[is_sig]

    # Alpha points (condensation points)
    q_alpha, index_alpha = scatter_max(q_sig, object_index)
    x_alpha = sig_coords[index_alpha]
    beta_alpha = sig_beta[index_alpha]

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
    x_centroid = scatter_mean(sig_coords, object_index, dim=0)
    d_sq_centroid = ((sig_coords - x_centroid[object_index]) ** 2).sum(dim=1)
    L_var_per_obj = scatter_mean(d_sq_centroid, object_index)
    L_var = L_var_per_obj.mean()

    # --- Repulsive potential (per-event, ALL hits incl. noise, matching original) ---
    all_object_index = torch.full((n_hits,), -1, device=device, dtype=torch.long)
    all_object_index[is_sig] = object_index

    rep_sum = torch.tensor(0.0, device=device)
    rep_n_objects = 0
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
        # M4: hinge repulsion (faithful Kieseler 2002.03605)
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

        rep_sum = rep_sum + V_rep_per_obj.sum()
        rep_n_objects += n_evt_obj
        obj_offset += n_evt_obj

    L_V_rep = rep_sum / max(rep_n_objects, 1)
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
        components = {
            "L_V_att": L_V_att.detach(),
            "L_V_rep": L_V_rep.detach(),
            "L_beta_sig": L_beta_sig.detach() if torch.is_tensor(L_beta_sig) else torch.tensor(float(L_beta_sig), device=device),
            "L_beta_noise": L_beta_noise.detach() if torch.is_tensor(L_beta_noise) else torch.tensor(float(L_beta_noise), device=device),
            "L_beta_suppress": L_beta_suppress.detach() if torch.is_tensor(L_beta_suppress) else torch.tensor(float(L_beta_suppress), device=device),
            "L_var": L_var.detach(),
            "var_weight": torch.tensor(float(var_weight), device=device),
        }
        return total, components
    return total


# ---------------------------------------------------------------------------
# Beta-greedy evaluation (pure numpy)
# ---------------------------------------------------------------------------
def get_clustering_np(betas, X, tbeta=0.5, td=0.5):
    n_points = betas.shape[0]
    select = betas > tbeta
    indices = np.nonzero(select)[0]
    indices = indices[np.argsort(-betas[select])]
    unassigned = np.arange(n_points)
    clustering = -1 * np.ones(n_points, dtype=np.int32)
    for idx in indices:
        d = np.linalg.norm(X[unassigned] - X[idx], axis=-1)
        assigned = unassigned[d < td]
        clustering[assigned] = idx
        unassigned = unassigned[~(d < td)]
    return clustering


def _seq_lens_to_batch(seq_lens, device):
    return torch.repeat_interleave(
        torch.arange(len(seq_lens), device=device, dtype=torch.long),
        torch.tensor(seq_lens, device=device, dtype=torch.long),
    )


def _compute_var_weight(epoch, args):
    """Linear warmup of var_weight from 0 to args.var_weight over the first
    `args.var_warmup_epochs` epochs (inclusive of the final epoch)."""
    if args.var_warmup_epochs <= 0:
        return float(args.var_weight)
    # epoch is 1-indexed in this codebase
    frac = min(1.0, max(0.0, (epoch - 1) / max(args.var_warmup_epochs, 1)))
    return float(args.var_weight) * frac


def _compute_batch_metrics_greedy(coords, beta_logits, mc_index, is_secondary, seq_lens, tbeta=0.5, td=0.5, cosine_norm=False):
    from collections import Counter

    coords = coords.detach().cpu().numpy()
    beta = torch.sigmoid(beta_logits.squeeze(-1)).detach().cpu().numpy()
    mc_index = mc_index.detach().cpu().numpy()
    is_secondary = is_secondary.detach().cpu().numpy()

    all_purities, all_effs, all_match = [], [], []
    all_match_strict50 = []  # purity > 0.75 AND efficiency_per_hit >= 0.5
    noise_low_beta_counts, noise_total_counts = 0, 0
    offset = 0
    for n_hits in seq_lens:
        sl = slice(offset, offset + n_hits)
        offset += n_hits

        sec_mask = is_secondary[sl] | (mc_index[sl] == 0)
        if sec_mask.any():
            noise_total_counts += sec_mask.sum()
            noise_low_beta_counts += (beta[sl][sec_mask] < 0.1).sum()

        mask = (~is_secondary[sl]) & (mc_index[sl] != 0)
        c = coords[sl][mask]
        if cosine_norm:
            norms = np.linalg.norm(c, axis=1, keepdims=True)
            c = c / np.clip(norms, 1e-6, None)
        b = beta[sl][mask]
        mc = mc_index[sl][mask]
        if len(c) == 0:
            continue

        pred_labels = get_clustering_np(b, c, tbeta=tbeta, td=td)
        unique_pred = np.unique(pred_labels[pred_labels >= 0])
        unique_true = np.unique(mc[mc >= 0])
        if len(unique_pred) == 0:
            all_purities.append(0.0)
            all_effs.append(0.0)
            all_match.append(0.0)
            continue

        purities = []
        for pid in unique_pred:
            cluster_mc = mc[pred_labels == pid]
            if len(cluster_mc) > 0:
                purities.append(np.bincount(cluster_mc).max() / len(cluster_mc))
        if purities:
            all_purities.append(np.mean(purities))

        matched = 0
        matched_strict50 = 0
        n_matchable = 0
        effs = []
        for tid in unique_true:
            tmask = mc == tid
            n_true = tmask.sum()
            if n_true < 2:
                continue
            n_matchable += 1
            pred_for_track = pred_labels[tmask]
            pred_for_track = pred_for_track[pred_for_track >= 0]
            if len(pred_for_track) == 0:
                effs.append(0.0)
                continue
            best_label, best_match = Counter(pred_for_track).most_common(1)[0]
            eff = best_match / n_true
            pur = best_match / (pred_labels == best_label).sum()
            effs.append(eff)
            if pur > 0.75:
                matched += 1
                if eff >= 0.5:
                    matched_strict50 += 1
        if effs:
            all_effs.append(np.mean(effs))
            all_match.append(matched / max(n_matchable, 1))
            all_match_strict50.append(matched_strict50 / max(n_matchable, 1))

    noise_suppression = float(noise_low_beta_counts / max(noise_total_counts, 1))

    return {
        "purity": float(np.mean(all_purities)) if all_purities else 0.0,
        "efficiency": float(np.mean(all_effs)) if all_effs else 0.0,
        "match_rate": float(np.mean(all_match)) if all_match else 0.0,
        "match_rate_strict50": float(np.mean(all_match_strict50)) if all_match_strict50 else 0.0,
        "noise_suppression": noise_suppression,
    }
