"""Geometric attention primitives for CGA Cl(4,1).

Key difference from PGA: CGA distance is computed directly from grade-1 (vector)
components using the CGA inner product: d^2(P1, P2) = -2 <P1, P2>.

Grade-1 indices: [1, 2, 3, 4, 5] (5 components)
Metric on grade-1: diag(+1, +1, +1, +1, -1) for (e1, e2, e3, e+, e-)

SDPA-only version: xformers dependency removed for ONNX export compatibility.
block_diagonal_bool_mask() replaces xformers BlockDiagonalMask.from_seqlens().
"""

import math
from functools import partial
from typing import Callable, Optional, Tuple

import torch
from einops import rearrange
from torch import Tensor, nn
from torch.nn.functional import scaled_dot_product_attention

from src.gatr_v111.utils.tensors import to_nd

# MV size factor for normalization
_MV_SIZE_FACTOR = 16  # Larger than PGA's 8 since we have 32 components

# CGA grade-1 indices (vectors): e1, e2, e3, e+, e-
_GRADE1_IDX = [1, 2, 3, 4, 5]

# Inner product contributing indices: ALL 32 for non-degenerate Cl(4,1)
# But for attention we use a subset that's most informative
# We exclude grade-0 (scalar) and grade-5 (pseudoscalar) for distance
_INNER_PRODUCT_WO_EXTREMES_IDX = list(range(1, 31))  # grades 1-4 (30 components)

# Grade-1 metric: (+1, +1, +1, +1, -1) for (e1, e2, e3, e+, e-)
_GRADE1_METRIC = [1.0, 1.0, 1.0, 1.0, -1.0]


def block_diagonal_bool_mask(seq_lens, device, M=None):
    """Torch-native block-diagonal self-attention mask (True = attend).
    Replaces xformers BlockDiagonalMask.from_seqlens for packed multi-event batches.
    Token i attends to token j iff they belong to the same packed event.
    """
    if M is None:
        M = int(sum(seq_lens))
    lens = torch.as_tensor(seq_lens, device=device, dtype=torch.long)
    event_id = torch.repeat_interleave(
        torch.arange(lens.numel(), device=device), lens
    )
    mask = (event_id[:, None] == event_id[None, :])  # (M, M) bool
    return mask[None, None]  # (1, 1, M, M) broadcasts over batch and heads


def _build_dist_basis(device, dtype) -> Tuple[Tensor, Tensor]:
    """Compute basis features for CGA distance-aware attention.

    For CGA null vectors P = x*e1 + y*e2 + z*e3 + ((1-r^2)/2)*e+ + ((1+r^2)/2)*e-,
    the squared distance is d^2 = -2 * <P1, P2> where <,> uses the CGA metric.

    We construct (5, 5, 6) basis tensors that encode this distance via grade-1 components.

    Returns
    -------
    basis_q : Tensor with shape (5, 5, 6)
    basis_k : Tensor with shape (5, 5, 6)
    """
    r4 = torch.arange(4, device=device)
    basis_q = torch.zeros((5, 5, 6), device=device, dtype=dtype)
    basis_k = torch.zeros((5, 5, 6), device=device, dtype=dtype)

    # For CGA distance: d^2 = -2(P1·P2) where · uses metric (+,+,+,+,-)
    # = -2(q1*k1 + q2*k2 + q3*k3 + q4*k4 - q5*k5)

    # Term 1: -sum_i(q_i^2) * k_5^2 (Minkowski cross terms)
    basis_q[r4, r4, 0] = 1
    basis_k[4, 4, 0] = -1

    # Term 2: -q_5^2 * sum_i(k_i^2)
    basis_q[4, 4, 1] = 1
    basis_k[r4, r4, 1] = -1

    # Term 3: 2*q_i*k_i*q_5*k_5 for i=1..4 (cross terms with e-)
    basis_q[r4, 4, 2 + r4] = 1
    basis_k[r4, 4, 2 + r4] = 2

    return basis_q, basis_k


def _build_dist_vec(tri: Tensor, basis: Tensor, normalizer: Callable[[Tensor], Tensor], device=None) -> Tensor:
    """Build distance feature vector.

    Parameters
    ----------
    tri : Tensor
        Grade-1 components of multivectors, shape (..., channels, 5)
    basis : Tensor with shape (5, 5, 6)
    normalizer : Callable
    """
    # Normalize by the e- component (index 4) for numerical stability
    tri_normed = tri * normalizer(tri[..., [4]])
    vec = torch.einsum("xyz,abcdx->abcdyz", basis, tri_normed)
    vec = torch.einsum("abcdyz,abcdy->abcdz", vec, tri_normed)
    return vec


def lin_square_normalizer(v: Tensor, epsilon=0.001) -> Tensor:
    """Linear square normalization: v / (v^2 + epsilon)."""
    return v / (v.pow(2) + epsilon)


class geometric_attention(nn.Module):
    """CGA geometric attention with distance-aware features.

    Uses grade-1 (vector) components for distance computation instead of
    PGA's trivector-based approach.

    Channel counts (`num_mv_channels_qk`, `num_s_channels_qk`,
    `num_mv_channels_v`, `num_s_channels_v`) can be passed at __init__
    time. When supplied, the forward pass uses them as Python ints
    instead of reading `tensor.shape[-2]` / `tensor.shape[-1]`. This
    avoids `TracerWarning: Converting a tensor to a Python boolean`
    during ONNX export — those shape reads return SymInts under tracing,
    and the subsequent `max(...)` / arithmetic baked them in as
    constants anyway (they're fixed by the model's hyperparameters, not
    by input). We just make the constants explicit.

    Legacy callers (single-event inference paths or older training code)
    pass `None` for the channel counts and fall back to dynamic shape
    reads — those still work, they just emit the warnings.

    SDPA-only: xformers removed. Uses torch.nn.functional.scaled_dot_product_attention
    with a (1, 1, M, M) bool block-diagonal mask for packed multi-event batches.
    """

    def __init__(self, basis_q, basis_k,
                 num_mv_channels_qk=None, num_s_channels_qk=None,
                 num_mv_channels_v=None, num_s_channels_v=None):
        super().__init__()
        self.register_buffer("basis_q", basis_q)
        self.register_buffer("basis_k", basis_k)
        self._GRADE1_IDX = _GRADE1_IDX
        self._INNER_PRODUCT_WO_EXTREMES_IDX = _INNER_PRODUCT_WO_EXTREMES_IDX
        self.num_mv_channels_qk = num_mv_channels_qk
        self.num_s_channels_qk = num_s_channels_qk
        self.num_mv_channels_v = num_mv_channels_v
        self.num_s_channels_v = num_s_channels_v

    def forward(
        self,
        q_mv: Tensor,
        k_mv: Tensor,
        v_mv: Tensor,
        q_s: Tensor,
        k_s: Tensor,
        v_s: Tensor,
        normalizer: Callable[[Tensor], Tensor],
        weights: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """CGA geometric attention.

        Parameters
        ----------
        q_mv : Tensor (..., num_items_out, num_mv_channels_in, 32)
        k_mv : Tensor (..., num_items_in, num_mv_channels_in, 32)
        v_mv : Tensor (..., num_items_in, num_mv_channels_out, 32)
        q_s : Tensor (..., heads, num_items_out, num_s_channels_in)
        k_s : Tensor (..., heads, num_items_in, num_s_channels_in)
        v_s : Tensor (..., heads, num_items_in, num_s_channels_out)
        normalizer : callable
        weights : Optional[Tensor]
        attn_mask : Optional[Tensor]
            Bool tensor of shape (1, 1, M, M) where True = attend.
            None = dense (single-event) attention.

        Returns
        -------
        outputs_mv : Tensor (..., heads, num_items_out, num_channels_out, 32)
        outputs_s : Tensor (..., heads, num_items_out, num_s_channels_out)
        """
        bh_shape = q_mv.shape[:-3]
        q_mv = to_nd(q_mv, 5)
        k_mv = to_nd(k_mv, 5)
        v_mv = to_nd(v_mv, 5)
        q_s = to_nd(q_s, 4)
        k_s = to_nd(k_s, 4)
        v_s = to_nd(v_s, 4)

        # Prefer cached Python ints (set at __init__ from SelfAttentionConfig)
        # so the channel arithmetic below is pure Python and doesn't
        # trip the tracer.  Fall back to `.shape[-]` for legacy callers.
        num_mv_channels_v = (
            self.num_mv_channels_v if self.num_mv_channels_v is not None
            else v_mv.shape[-2]
        )
        num_s_channels_v = (
            self.num_s_channels_v if self.num_s_channels_v is not None
            else v_s.shape[-1]
        )
        num_mv_channels_qk = (
            self.num_mv_channels_qk if self.num_mv_channels_qk is not None
            else q_mv.shape[-2]
        )
        num_s_channels_qk = (
            self.num_s_channels_qk if self.num_s_channels_qk is not None
            else q_s.shape[-1]
        )

        device = q_mv.device
        dtype = q_mv.dtype

        # Extract grade-1 components for distance computation
        q_g1 = q_mv[..., _GRADE1_IDX]  # (..., channels, 5)
        k_g1 = k_mv[..., _GRADE1_IDX]

        q_dist = _build_dist_vec(q_g1, self.basis_q, normalizer, device=device)
        k_dist = _build_dist_vec(k_g1, self.basis_k, normalizer, device=device)

        if weights is not None:
            q_dist = q_dist * weights[..., None].to(q_dist.dtype)

        # Extract grade 1-4 components for inner product attention (30 components)
        q_mv_ip = q_mv[..., _INNER_PRODUCT_WO_EXTREMES_IDX]
        k_mv_ip = k_mv[..., _INNER_PRODUCT_WO_EXTREMES_IDX]

        # Compute channel dimensions
        num_ip_components = len(_INNER_PRODUCT_WO_EXTREMES_IDX)  # 30
        num_dist_features = 6  # from _build_dist_basis
        num_channels_qk = num_mv_channels_qk * (num_ip_components + num_dist_features) + num_s_channels_qk
        num_channels_v = num_mv_channels_v * 32 + num_s_channels_v
        num_channels = max(num_channels_qk, num_channels_v)
        num_channels = 8 * -(-num_channels // 8)  # Ceil to multiple of 8

        # Build queries
        a = rearrange(q_mv_ip, "... c x -> ... (c x)")
        b = rearrange(q_dist, "... c d -> ... (c d)")
        q = torch.cat([
            a, b, q_s,
            torch.zeros(*q_s.shape[:3], num_channels - num_channels_qk, device=device, dtype=dtype),
        ], -1)

        # Build keys
        a_k = rearrange(k_mv_ip, "... c x -> ... (c x)")
        b_k = rearrange(k_dist, "... c d -> ... (c d)")
        k = torch.cat([
            a_k, b_k, k_s,
            torch.zeros(*k_s.shape[:3], num_channels - num_channels_qk, device=device, dtype=dtype),
        ], -1)

        # Build values
        v = torch.cat([
            rearrange(v_mv, "... c x -> ... (c x)"),
            v_s,
            torch.zeros(*v_s.shape[:3], num_channels - num_channels_v, device=device, dtype=dtype),
        ], -1)

        # Scale keys to correct for zero padding
        k = k * math.sqrt(num_channels / num_channels_qk)

        # SDPA-only path: bool mask (True = attend) or None (dense).
        # No xformers dependency — compatible with ONNX export.
        v_out = scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)

        # Split output
        v_out_mv = rearrange(v_out[..., :num_mv_channels_v * 32], "... (c x) -> ... c x", x=32)
        v_out_s = v_out[..., num_mv_channels_v * 32:num_mv_channels_v * 32 + num_s_channels_v]

        v_out_mv = v_out_mv.view(*bh_shape, *v_out_mv.shape[-3:])
        v_out_s = v_out_s.view(*bh_shape, *v_out_s.shape[-2:])

        return v_out_mv, v_out_s
