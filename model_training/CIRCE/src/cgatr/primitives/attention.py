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
import torch._dynamo
from einops import rearrange
from torch import Tensor, nn
from torch.nn.functional import scaled_dot_product_attention


def to_nd(tensor: Tensor, d: int) -> Tensor:
    """Prepend singleton dimensions until `tensor` has dimension `d`."""
    while len(tensor.shape) < d:
        tensor = tensor.unsqueeze(0)
    return tensor

# Optional xformers memory-efficient attention. Used for the packed-batch
# training path: it is block-sparse and O(M) in memory (never materialises the
# M x M score matrix), and it handles this model's large per-head feature dim.
# SDPA — which falls back to the O(M^2) MATH backend for that head dim —
# is kept as the fallback and for single-event eval / ONNX export.
try:
    import xformers.ops as _xops
    from xformers.ops.fmha import BlockDiagonalMask as _BlockDiagonalMask

    _HAS_XFORMERS = True
except Exception:  # pragma: no cover - xformers optional
    _HAS_XFORMERS = False

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
    event_id = torch.repeat_interleave(torch.arange(lens.numel(), device=device), lens)
    mask = event_id[:, None] == event_id[None, :]  # (M, M) bool
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


def pga_distance_features(
    tri: Tensor, normalizer: Callable[[Tensor], Tensor], query: bool
) -> Tensor:
    """GATr's distance-aware query/key features for the projective algebra.

    Brehmer et al. arXiv:2305.18415 App. B. The trivector is first rescaled by
    w(q0), where q0 is its homogeneous weight and w(x) = x / (x^2 + eps); the
    features are then quadratic in the rescaled components qv,

        phi(q) = (q0^2, |qv|^2,  q0 qv)
        psi(k) = (-|kv|^2, -k0^2, 2 k0 kv)

    so that phi(q).psi(k) = -||k0 qv - q0 kv||^2 = -||xq - xk||^2, the negative
    squared Euclidean distance between the points the trivectors represent.
    This is needed because the projective inner product is provably constant in
    point coordinates (de Haan et al. Prop. 3), so without it attention cannot
    see positions at all. The conformal algebra needs no such construction.

    The rescaling must happen before the quadratic, not after: the homogeneous
    weight is gauge, and only w(q0)^2 q0^2 -> 1 divides it back out. Applying
    w once instead leaves the logit proportional to q0 k0, which is the
    distance times an arbitrary learnable factor rather than the distance.
    `src/eval/verify_distance_features.py` pins this against the reference.

    Parameters
    ----------
    tri : Tensor (..., channels, 4)
        Trivector components, homogeneous weight first.
    query : bool
        Selects phi (True) or psi (False).

    Returns
    -------
    features : Tensor (..., channels, 5)
    """
    tri = tri * normalizer(tri[..., :1])
    w0 = tri[..., :1]
    wv = tri[..., 1:]
    sq_v = wv.pow(2).sum(-1, keepdim=True)
    if query:
        parts = [w0.pow(2), sq_v, w0 * wv]
    else:
        parts = [-sq_v, -w0.pow(2), 2.0 * w0 * wv]
    return torch.cat(parts, dim=-1)


def _build_dist_vec(
    tri: Tensor, basis: Tensor, normalizer: Callable[[Tensor], Tensor], device=None
) -> Tensor:
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

    Attention backend: xformers memory_efficient_attention (block-sparse, O(M)
    memory) for packed multi-event training, with SDPA fallbacks. See the
    dispatch in `forward` for details.
    """

    def __init__(
        self,
        basis_q,
        basis_k,
        num_mv_channels_qk=None,
        num_s_channels_qk=None,
        num_mv_channels_v=None,
        num_s_channels_v=None,
        grade1_idx=None,
        ip_idx=None,
        num_blades=32,
        ip_weights=None,
        pga_dist_idx=None,
    ):
        super().__init__()
        # basis_q/basis_k None disables the legacy conformal distance features.
        self.register_buffer("basis_q", basis_q)
        self.register_buffer("basis_k", basis_k)
        self.use_dist = basis_q is not None
        self.num_blades = num_blades
        self._GRADE1_IDX = grade1_idx if grade1_idx is not None else _GRADE1_IDX
        self._INNER_PRODUCT_WO_EXTREMES_IDX = (
            ip_idx if ip_idx is not None else _INNER_PRODUCT_WO_EXTREMES_IDX
        )
        # Metric weights turn the dot product into the invariant inner product.
        # Folded into the query side only, which keeps the whole score a single
        # Euclidean dot product and so keeps the fast attention kernels usable.
        self.register_buffer(
            "ip_weights",
            None if ip_weights is None
            else torch.as_tensor(ip_weights, dtype=torch.float32),
        )
        # Projective distance-aware attention (GATr App. B).
        self._PGA_DIST_IDX = pga_dist_idx
        self.use_pga_dist = pga_dist_idx is not None
        self.num_mv_channels_qk = num_mv_channels_qk
        self.num_s_channels_qk = num_s_channels_qk
        self.num_mv_channels_v = num_mv_channels_v
        self.num_s_channels_v = num_s_channels_v

    # Exclude this method from any torch.compile graph: the xformers
    # memory_efficient_attention backward kernel cannot be traced by Dynamo.
    # `torch._dynamo.disable` is a no-op when the model is not compiled, so the
    # eager path is unaffected; under compile it produces a graph break here
    # and lets the rest of the model (linears, GP, layernorm) fuse and compile.
    @torch._dynamo.disable
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
            self.num_mv_channels_v
            if self.num_mv_channels_v is not None
            else v_mv.shape[-2]
        )
        num_s_channels_v = (
            self.num_s_channels_v
            if self.num_s_channels_v is not None
            else v_s.shape[-1]
        )
        num_mv_channels_qk = (
            self.num_mv_channels_qk
            if self.num_mv_channels_qk is not None
            else q_mv.shape[-2]
        )
        num_s_channels_qk = (
            self.num_s_channels_qk
            if self.num_s_channels_qk is not None
            else q_s.shape[-1]
        )

        device = q_mv.device
        dtype = q_mv.dtype

        if self.use_dist:
            # Extract grade-1 components for distance computation
            q_g1 = q_mv[..., self._GRADE1_IDX]  # (..., channels, 5)
            k_g1 = k_mv[..., self._GRADE1_IDX]

            q_dist = _build_dist_vec(q_g1, self.basis_q, normalizer, device=device)
            k_dist = _build_dist_vec(k_g1, self.basis_k, normalizer, device=device)

            if weights is not None:
                q_dist = q_dist * weights[..., None].to(q_dist.dtype)
        elif self.use_pga_dist:
            q_dist = pga_distance_features(
                q_mv[..., self._PGA_DIST_IDX], normalizer, query=True)
            k_dist = pga_distance_features(
                k_mv[..., self._PGA_DIST_IDX], normalizer, query=False)
            if weights is not None:
                q_dist = q_dist * weights[..., None].to(q_dist.dtype)
        else:
            q_dist = k_dist = None
        use_dist = self.use_dist or self.use_pga_dist

        q_mv_ip = q_mv[..., self._INNER_PRODUCT_WO_EXTREMES_IDX]
        k_mv_ip = k_mv[..., self._INNER_PRODUCT_WO_EXTREMES_IDX]
        if self.ip_weights is not None:
            q_mv_ip = q_mv_ip * self.ip_weights.to(q_mv_ip.dtype)

        # Compute channel dimensions
        num_ip_components = len(self._INNER_PRODUCT_WO_EXTREMES_IDX)
        num_dist_features = (
            self.basis_q.shape[-1] if self.use_dist
            else (5 if self.use_pga_dist else 0)
        )
        num_channels_qk = (
            num_mv_channels_qk * (num_ip_components + num_dist_features)
            + num_s_channels_qk
        )
        num_channels_v = num_mv_channels_v * self.num_blades + num_s_channels_v
        num_channels = max(num_channels_qk, num_channels_v)
        num_channels = 8 * -(-num_channels // 8)  # Ceil to multiple of 8

        # Build queries
        a = rearrange(q_mv_ip, "... c x -> ... (c x)")
        b = (rearrange(q_dist, "... c d -> ... (c d)") if use_dist
             else a[..., :0])
        q = torch.cat(
            [
                a,
                b,
                q_s,
                torch.zeros(
                    *q_s.shape[:3],
                    num_channels - num_channels_qk,
                    device=device,
                    dtype=dtype,
                ),
            ],
            -1,
        )

        # Build keys
        a_k = rearrange(k_mv_ip, "... c x -> ... (c x)")
        b_k = (rearrange(k_dist, "... c d -> ... (c d)") if use_dist
               else a_k[..., :0])
        k = torch.cat(
            [
                a_k,
                b_k,
                k_s,
                torch.zeros(
                    *k_s.shape[:3],
                    num_channels - num_channels_qk,
                    device=device,
                    dtype=dtype,
                ),
            ],
            -1,
        )

        # Build values
        v = torch.cat(
            [
                rearrange(v_mv, "... c x -> ... (c x)"),
                v_s,
                torch.zeros(
                    *v_s.shape[:3],
                    num_channels - num_channels_v,
                    device=device,
                    dtype=dtype,
                ),
            ],
            -1,
        )

        # Scale keys to correct for zero padding
        k = k * math.sqrt(num_channels / num_channels_qk)

        # Attention. Three accepted forms for `attn_mask`:
        #   * None                 -> single dense self-attention (one event).
        #   * bool (..., M, M)      -> legacy dense block-diagonal mask. Kept for
        #                             back-compat; materialises the full M x M
        #                             score matrix (O(M^2) memory).
        #   * 1-D lengths / list    -> per-event hit counts (seq_lens). Block-
        #                             diagonal attention is mathematically just
        #                             independent dense attention per event, so
        #                             we slice the item axis and run SDPA once
        #                             per event. Peak memory becomes O(sum n_i^2)
        #                             instead of O((sum n_i)^2).
        #
        # Why this matters here: the per-head feature dim is ~640 (32-blade MVs x
        # channels), far above the 256-dim cap of the Flash / mem-efficient SDPA
        # kernels, so SDPA always falls back to the MATH backend and materialises
        # the dense score matrix. With packed batches (M up to max_tokens) the
        # cross-event entries — which the mask only zeroes *after* allocation —
        # dominate VRAM. Splitting per event removes them entirely (~8x less on
        # realistic packed batches) and is numerically identical to the mask.
        if attn_mask is None:
            v_out = scaled_dot_product_attention(q, k, v)
        elif torch.is_tensor(attn_mask) and attn_mask.dtype == torch.bool:
            v_out = scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        else:
            seq_lens = [int(n) for n in attn_mask]
            # Leading (batch) dims must collapse to 1: the packed batch is a
            # single sequence-set with all events along the item axis.
            lead_numel = 1
            for d in q.shape[:-3]:
                lead_numel *= int(d)
            Hq, M, C = q.shape[-3], q.shape[-2], q.shape[-1]

            if _HAS_XFORMERS and lead_numel == 1 and len(seq_lens) >= 1:
                # xformers wants (B, M, H, K). q is (.., H, M, C); k/v are
                # (.., 1, M, C) under multi-query -> broadcast to H heads.
                qx = q.reshape(Hq, M, C).permute(1, 0, 2).unsqueeze(0).contiguous()
                kx = (
                    k.reshape(k.shape[-3], M, C)
                    .permute(1, 0, 2)
                    .unsqueeze(0)
                    .expand(1, M, Hq, C)
                )
                vx = (
                    v.reshape(v.shape[-3], M, v.shape[-1])
                    .permute(1, 0, 2)
                    .unsqueeze(0)
                    .expand(1, M, Hq, v.shape[-1])
                )
                bias = _BlockDiagonalMask.from_seqlens(seq_lens)
                ox = _xops.memory_efficient_attention(qx, kx, vx, attn_bias=bias)
                # (1, M, H, Cv) -> (.., H, M, Cv)
                v_out = (
                    ox.squeeze(0).permute(1, 0, 2).reshape(*q.shape[:-1], v.shape[-1])
                )
            elif len(seq_lens) <= 1:
                v_out = scaled_dot_product_attention(q, k, v)
            else:
                # Fallback (no xformers): per-event dense SDPA. Still avoids the
                # cross-event O(M^2) blow-up — O(sum n_i^2) — but unlike xformers
                # each event's score matrix is materialised (MATH backend).
                outs = []
                off = 0
                for n in seq_lens:
                    sl = slice(off, off + n)
                    off += n
                    outs.append(
                        scaled_dot_product_attention(
                            q[..., sl, :], k[..., sl, :], v[..., sl, :]
                        )
                    )
                v_out = torch.cat(outs, dim=-2)

        # Split output
        nb = self.num_blades
        v_out_mv = rearrange(
            v_out[..., : num_mv_channels_v * nb], "... (c x) -> ... c x", x=nb
        )
        v_out_s = v_out[
            ..., num_mv_channels_v * nb : num_mv_channels_v * nb + num_s_channels_v
        ]

        v_out_mv = v_out_mv.view(*bh_shape, *v_out_mv.shape[-3:])
        v_out_s = v_out_s.view(*bh_shape, *v_out_s.shape[-2:])

        return v_out_mv, v_out_s
