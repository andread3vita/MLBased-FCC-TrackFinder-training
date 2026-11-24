"""Inference-only ONNX rewrites for GATR_EXPERIMENT checkpoints (CPU, FP32).

Training never imports this module: it rewrites a disposable copy of the
network after the checkpoint weights are loaded, so checkpoints, the training
graph and the training environment are unchanged. Only torch, numpy and the
vendored ``gatr_v142`` are needed to build the model; ``onnx`` is imported
lazily by the graph rewrites, which run in the conversion environment.

Rewrites (all standard ONNX unless noted):

* Affine hit embedding: ``hits @ E + c`` replaces gatr's in-place embedding
  writes, which the TorchScript exporter cannot handle.
* Folded, fused equivariant linears: the 9 Pin-equivariant basis maps, the
  scalar<->multivector maps and the biases of an ``EquiLinear`` become one
  dense matrix, and linears reading the same input (Q/K/V, and the four
  GP/join inputs of each bilinear) become a single GEMM.
* Geometric product and join as one outer product (x_j y_k, 256 wide) and a
  [256, 16] MatMul over the flattened table, instead of a dense 16x16x16
  einsum. A gathered sparse form over the nonzero terms only (192 of 4096
  entries for the product) does fewer FLOPs but is memory bound in ONNX
  Runtime and ~3x slower per product.
* Masked inner product in the equivariant layer norm and a MatMul form of the
  attention distance features.
* Attention core exported as MatMul/Softmax/MatMul without materialising the
  shared K/V per head, then optionally rewritten on the ONNX graph:
  ``blocked`` (Loop over query blocks against all keys, bounded score memory),
  ``ort-mha`` (ONNX Runtime ``com.microsoft.MultiHeadAttention``; not a
  runtime-neutral file) or ``dense``.
* Optional numerical guards (blocked/dense): zero |x| < 1e-12 in Q/K/V and
  floor logits at row_max - 30 before softmax, which avoids slow subnormal
  arithmetic. This is an approximation; validate every exported checkpoint.
"""

from __future__ import annotations

import contextlib
import math
from pathlib import Path
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from torch import nn
from torch.nn import functional as F

_GATR_V142_ROOT = Path(__file__).resolve().parents[1] / "gatr_v142"
if str(_GATR_V142_ROOT) not in sys.path:
    sys.path.insert(0, str(_GATR_V142_ROOT))

import gatr.primitives.attention as gatr_attention  # noqa: E402
from gatr.interface import embed_point, embed_scalar, embed_translation  # noqa: E402
from gatr.layers.attention.config import SelfAttentionConfig  # noqa: E402
from gatr.layers.attention.qkv import MultiQueryQKVModule  # noqa: E402
from gatr.layers.layer_norm import EquiLayerNorm  # noqa: E402
from gatr.layers.linear import EquiLinear  # noqa: E402
from gatr.layers.mlp.config import MLPConfig  # noqa: E402
from gatr.layers.mlp.geometric_bilinears import GeometricBilinear  # noqa: E402
from gatr.nets.gatr import GATr  # noqa: E402
from gatr.primitives.bilinear import _load_bilinear_basis  # noqa: E402
from gatr.primitives.dual import _compute_efficient_join  # noqa: E402
from gatr.primitives.invariants import INNER_PRODUCT_INDICES  # noqa: E402
from gatr.primitives.linear import _compute_pin_equi_linear_basis  # noqa: E402
from gatr.utils.einsum import enable_cached_einsum  # noqa: E402

# opt_einsum path caching mis-parses some gatr equations; plain torch.einsum
# is numerically equivalent (same choice as conversionModel/convert_to_onnx.py).
enable_cached_einsum(False)

WEIGHT_SOURCES = ("ema", "raw")
ATTENTION_MODES = ("blocked", "ort-mha", "dense")
_BLADES = 16
_JOIN_REFERENCE_BLADE = 14  # e_123 component, as in efficient_equivariant_join


# --------------------------------------------------------------------------
# Single-event model and checkpoint loading
# --------------------------------------------------------------------------
class SingleEventGATr(nn.Module):
    """One-event rebuild of ``ExampleWrapper.forward`` without DGL/Lightning.

    Input ``hits [N, 7] = x, y, z, planar_flag, u, v, w`` (GGTFTrackFinder
    layout); output ``[N, 9]`` = 4 clustering coordinates, raw beta logit and
    4 zero columns. The network has no helix head; the zero columns keep the
    GGTFTrackFinder output contract of the previous exports.
    """

    zero_columns = 4

    def __init__(self, hparams: dict) -> None:
        super().__init__()
        if bool(hparams.get("use_detector_features", False)):
            raise NotImplementedError("only the geometry-only (7-feature) model is supported")
        if int(hparams.get("attention_phi_sectors", 1) or 1) > 1:
            raise NotImplementedError("phi-sector attention is not supported for export")
        self.embedding_dim = int(hparams.get("clustering_space_dim", 4))
        self.output_dim = self.embedding_dim + 1 + self.zero_columns
        self.position_scale = float(hparams.get("position_scale", 1000.0))
        self.hidden_s_channels = int(hparams.get("hidden_s_channels", 64))
        self.gatr = GATr(
            in_mv_channels=1,
            out_mv_channels=1,
            hidden_mv_channels=int(hparams.get("hidden_mv_channels", 16)),
            in_s_channels=None,
            out_s_channels=None,
            hidden_s_channels=self.hidden_s_channels,
            num_blocks=int(hparams.get("gatr_blocks", 10)),
            attention=SelfAttentionConfig(),
            mlp=MLPConfig(),
            checkpoint=None,
        )
        self.clustering = nn.Linear(16, self.embedding_dim, bias=False)
        self.beta = nn.Linear(16, 1)

    def embed(self, hits: torch.Tensor) -> torch.Tensor:
        return (
            embed_point(hits[..., :3] / self.position_scale)
            + embed_scalar(hits[..., 3:4])
            + embed_translation(hits[..., 4:7] / self.position_scale)
        )

    def forward(self, hits: torch.Tensor) -> torch.Tensor:
        hits = hits.to(self.clustering.weight.dtype)
        multivectors = self.embed(hits).unsqueeze(-2)
        # One event is one BlockDiagonalMask block: full attention, and the
        # join reference is the event mean (ExampleWrapper._join_reference).
        join_reference = multivectors.mean(dim=-3, keepdim=True).expand_as(multivectors)
        embedded, _ = self.gatr(
            multivectors, scalars=None, attention_mask=None, join_reference=join_reference
        )
        latent = embedded[..., 0, :]
        # Slice-based zeros keep the output width static in the graph.
        zeros = torch.zeros_like(latent[..., : self.zero_columns])
        return torch.cat((self.clustering(latent), self.beta(latent), zeros), dim=-1)


def select_weights(checkpoint: dict, weights_source: str) -> dict:
    if weights_source == "ema":
        state = checkpoint.get("ema_state_dict")
        if not isinstance(state, dict) or not state:
            raise ValueError("checkpoint has no non-empty ema_state_dict; use --weights-source raw")
        return state
    state = checkpoint.get("state_dict", checkpoint)
    if not isinstance(state, dict) or not state:
        raise ValueError("checkpoint has no non-empty state_dict")
    return state


def load_checkpoint(path, weights_source: str = "ema") -> Tuple[SingleEventGATr, dict]:
    """Rebuild the single-event model from a trusted Lightning checkpoint."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    hparams = dict(checkpoint.get("hyper_parameters") or {})
    model = SingleEventGATr(hparams)
    # Older checkpoints still carry the removed helix head; ignore it.
    state = {k: v for k, v in select_weights(checkpoint, weights_source).items()
             if not k.startswith("helix_proxy.")}
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected:
        raise RuntimeError(f"unexpected checkpoint keys: {sorted(unexpected)[:8]}")
    if missing:
        raise RuntimeError(f"missing checkpoint weights: {sorted(missing)[:8]}")
    return model.eval(), hparams


# --------------------------------------------------------------------------
# Folded and fused equivariant linears
# --------------------------------------------------------------------------
def _dense_equi_linear(linear: EquiLinear) -> Tuple[torch.Tensor, torch.Tensor, int, int, int]:
    """Return (matrix, bias, out_mv, out_s, in_s) of one EquiLinear.

    Columns: ``in_mv * 16`` flattened multivector inputs, then ``in_s``
    scalars. Rows: ``out_mv * 16`` flattened multivector outputs, then
    ``out_s`` scalars.
    """
    weight = linear.weight.detach().double()
    out_mv, in_mv, _ = weight.shape
    basis = _compute_pin_equi_linear_basis(torch.device("cpu"), torch.float64)
    in_s = linear.s2mvs.in_features if linear.s2mvs is not None else (
        linear.s2s.in_features if linear.s2s is not None else 0
    )
    out_s = linear.mvs2s.out_features if linear.mvs2s is not None else 0
    rows, cols = out_mv * _BLADES + out_s, in_mv * _BLADES + in_s
    matrix = torch.zeros(rows, cols, dtype=torch.float64)
    bias = torch.zeros(rows, dtype=torch.float64)
    mv_block = torch.einsum("yxa,aij->yixj", weight, basis).reshape(out_mv * _BLADES, in_mv * _BLADES)
    matrix[: out_mv * _BLADES, : in_mv * _BLADES] = mv_block
    scalar_rows = torch.arange(out_mv) * _BLADES  # blade 0 of each output channel
    if linear.bias is not None:
        bias[scalar_rows] += linear.bias.detach().double()[:, 0]
    if linear.s2mvs is not None:
        matrix[scalar_rows, in_mv * _BLADES :] += linear.s2mvs.weight.detach().double()
        if linear.s2mvs.bias is not None:
            bias[scalar_rows] += linear.s2mvs.bias.detach().double()
    if linear.mvs2s is not None:
        blade0_cols = torch.arange(in_mv) * _BLADES
        matrix[out_mv * _BLADES :, blade0_cols] += linear.mvs2s.weight.detach().double()
        if linear.mvs2s.bias is not None:
            bias[out_mv * _BLADES :] += linear.mvs2s.bias.detach().double()
        if linear.s2s is not None:
            matrix[out_mv * _BLADES :, in_mv * _BLADES :] += linear.s2s.weight.detach().double()
    return matrix, bias, out_mv, out_s, in_s


class FusedEquiLinear(nn.Module):
    """One GEMM evaluating several EquiLinears that read the same input."""

    def __init__(self, linears: Sequence[EquiLinear]) -> None:
        super().__init__()
        parts = [_dense_equi_linear(linear) for linear in linears]
        in_mv = {linear.weight.shape[1] for linear in linears}
        if len(in_mv) != 1:
            raise ValueError("fused linears must share their multivector input")
        self.in_mv = in_mv.pop()
        self.in_s = max(p[4] for p in parts)
        cols = self.in_mv * _BLADES + self.in_s
        matrices = []
        for matrix, *_ in parts:
            padded = torch.zeros(matrix.shape[0], cols, dtype=torch.float64)
            padded[:, : matrix.shape[1]] = matrix
            matrices.append(padded)
        self.register_buffer("matrix", torch.cat(matrices).float().contiguous())
        self.register_buffer("bias", torch.cat([p[1] for p in parts]).float().contiguous())
        self.layout = [(p[2], p[3]) for p in parts]

    def forward(self, multivectors, scalars=None) -> List[Tuple[torch.Tensor, Optional[torch.Tensor]]]:
        if multivectors.dim() != 3:
            raise ValueError("export linears expect [hits, channels, 16] inputs")
        inputs = multivectors.flatten(-2)
        if self.in_s:
            if scalars is None:
                raise RuntimeError("fused linear expects scalar inputs")
            inputs = torch.cat((inputs, scalars), dim=-1)
        outputs = F.linear(inputs, self.matrix.to(inputs.dtype), self.bias.to(inputs.dtype))
        results, start = [], 0
        for out_mv, out_s in self.layout:
            mv = outputs[..., start : start + out_mv * _BLADES]
            # Explicit reshapes (hit axis -1) instead of unflatten: the
            # TorchScript ONNX shape inference crashes on unflatten here.
            mv = mv.reshape(-1, out_mv, _BLADES)
            start += out_mv * _BLADES
            s = outputs[..., start : start + out_s] if out_s else None
            start += out_s
            results.append((mv, s))
        return results


def _blade_terms() -> Tuple[List[Tuple[int, float]], Dict[int, Tuple[int, int, float]]]:
    """Per output blade i: its diagonal basis map (a, coefficient) and, for the
    8 blades reached by multiplication with e0, the extra (a, source j, coefficient).

    The 9 Pin-equivariant maps touch only 24 of the 256 entries of a 16x16
    blade block, so a folded dense matrix does ~10x more arithmetic than needed.
    """
    basis = _compute_pin_equi_linear_basis(torch.device("cpu"), torch.float64)
    diagonal: List[List[Tuple[int, float]]] = [[] for _ in range(_BLADES)]
    extra: Dict[int, List[Tuple[int, int, float]]] = {}
    for a in range(basis.shape[0]):
        for i, j in basis[a].nonzero().tolist():
            if i == j:
                diagonal[i].append((a, float(basis[a, i, j])))
            else:
                extra.setdefault(i, []).append((a, j, float(basis[a, i, j])))
    if any(len(d) != 1 for d in diagonal) or any(len(e) != 1 for e in extra.values()):
        raise ValueError("unexpected equivariant linear basis structure")
    return [d[0] for d in diagonal], {i: e[0] for i, e in extra.items()}


class BladeFusedEquiLinear(nn.Module):
    """Same maps as ``FusedEquiLinear`` with blade-sparse weights.

    Multivectors are evaluated blade-major: output blade i is
    ``x_i @ W_diag(i) + x_src(i) @ W_e0(i)``, one batched [16, hits, 2 in] x
    [16, 2 in, out] MatMul. Scalars, biases and the blade-0 <-> scalar maps go
    through one small GEMM on ``[x_0, scalars]``. Output layout and values
    match ``FusedEquiLinear``.
    """

    def __init__(self, linears: Sequence[EquiLinear]) -> None:
        super().__init__()
        diagonal, extra = _blade_terms()
        in_mv = {linear.weight.shape[1] for linear in linears}
        if len(in_mv) != 1:
            raise ValueError("fused linears must share their multivector input")
        self.in_mv = in_mv.pop()
        self.in_s = max(_dense_equi_linear(linear)[4] for linear in linears)
        out_mv = [linear.weight.shape[0] for linear in linears]
        out_s = [linear.mvs2s.out_features if linear.mvs2s is not None else 0 for linear in linears]
        self.total_mv = sum(out_mv)
        weight = torch.zeros(_BLADES, 2 * self.in_mv, self.total_mv, dtype=torch.float64)
        source = torch.arange(_BLADES)
        zero = torch.zeros(self.total_mv + sum(out_s), self.in_mv + self.in_s, dtype=torch.float64)
        bias = torch.zeros(self.total_mv + sum(out_s), dtype=torch.float64)
        mv_start, s_start = 0, self.total_mv
        for linear, n_mv, n_s in zip(linears, out_mv, out_s):
            w = linear.weight.detach().double()  # [out, in, 9]
            mv = slice(mv_start, mv_start + n_mv)
            for i, (a, c) in enumerate(diagonal):
                weight[i, : self.in_mv, mv] = (c * w[:, :, a]).T
            for i, (a, j, c) in extra.items():
                source[i] = j
                weight[i, self.in_mv :, mv] = (c * w[:, :, a]).T
            if linear.bias is not None:
                bias[mv] += linear.bias.detach().double()[:, 0]
            if linear.s2mvs is not None:
                zero[mv, self.in_mv : self.in_mv + linear.s2mvs.in_features] = linear.s2mvs.weight.detach().double()
                if linear.s2mvs.bias is not None:
                    bias[mv] += linear.s2mvs.bias.detach().double()
            if linear.mvs2s is not None:
                s = slice(s_start, s_start + n_s)
                zero[s, : self.in_mv] = linear.mvs2s.weight.detach().double()
                if linear.mvs2s.bias is not None:
                    bias[s] += linear.mvs2s.bias.detach().double()
                if linear.s2s is not None:
                    zero[s, self.in_mv : self.in_mv + linear.s2s.in_features] = linear.s2s.weight.detach().double()
            mv_start += n_mv
            s_start += n_s
        self.register_buffer("weight", weight.float().contiguous())
        self.register_buffer("source", source)
        self.register_buffer("zero_weight", zero.float().contiguous())
        self.register_buffer("zero_bias", bias.float().contiguous())
        self.layout = list(zip(out_mv, out_s))
        self._check_against_dense(linears)

    def _check_against_dense(self, linears) -> None:
        dense = FusedEquiLinear(linears)
        generator = torch.Generator().manual_seed(0)
        x = torch.randn(5, self.in_mv, _BLADES, generator=generator)
        s = torch.randn(5, self.in_s, generator=generator) if self.in_s else None
        with torch.no_grad():
            for (a_mv, a_s), (b_mv, b_s) in zip(dense(x, s), self(x, s)):
                if not torch.allclose(a_mv, b_mv, atol=1e-4) or (
                    a_s is not None and not torch.allclose(a_s, b_s, atol=1e-4)
                ):
                    raise ValueError("blade-sparse linear does not match the dense fold")

    def forward(self, multivectors, scalars=None) -> List[Tuple[torch.Tensor, Optional[torch.Tensor]]]:
        if multivectors.dim() != 3:
            raise ValueError("export linears expect [hits, channels, 16] inputs")
        blades = multivectors.permute(2, 0, 1)  # [16, hits, in]
        pairs = torch.cat((blades, blades[self.source]), dim=-1)
        outputs = torch.matmul(pairs, self.weight.to(pairs.dtype))  # [16, hits, out]
        zero_inputs = blades[0]
        if self.in_s:
            if scalars is None:
                raise RuntimeError("fused linear expects scalar inputs")
            zero_inputs = torch.cat((zero_inputs, scalars), dim=-1)
        zero = F.linear(zero_inputs, self.zero_weight.to(pairs.dtype), self.zero_bias.to(pairs.dtype))
        outputs = torch.cat(((outputs[0] + zero[:, : self.total_mv]).unsqueeze(0), outputs[1:]))
        results, mv_start, s_start = [], 0, self.total_mv
        for n_mv, n_s in self.layout:
            mv = outputs[..., mv_start : mv_start + n_mv].permute(1, 2, 0)
            s = zero[:, s_start : s_start + n_s] if n_s else None
            results.append((mv, s))
            mv_start += n_mv
            s_start += n_s
        return results


# Measured on CPU (4 threads, 4,000 hits): the blade-sparse form is 1.5-2x
# faster for wide layers, but its transposes dominate for narrow ones.
_BLADE_LINEAR_MIN_CHANNELS = 8
_BLADE_LINEAR_MIN_OUTPUTS = 16


def fused_equi_linear(linears: Sequence[EquiLinear]) -> nn.Module:
    """The faster of the blade-sparse and dense folded forms for these linears."""
    in_mv = linears[0].weight.shape[1]
    out_mv = sum(linear.weight.shape[0] for linear in linears)
    if in_mv >= _BLADE_LINEAR_MIN_CHANNELS and out_mv >= _BLADE_LINEAR_MIN_OUTPUTS:
        return BladeFusedEquiLinear(linears)
    return FusedEquiLinear(linears)


class FoldedEquiLinear(nn.Module):
    """Drop-in replacement for a single EquiLinear."""

    def __init__(self, linear: EquiLinear) -> None:
        super().__init__()
        self.fused = fused_equi_linear([linear])

    def forward(self, multivectors, scalars=None):
        return self.fused(multivectors, scalars)[0]


# --------------------------------------------------------------------------
# Bilinear products
# --------------------------------------------------------------------------
class TableBilinear(nn.Module):
    """``out_i = sum_jk table[i, j, k] x_j y_k`` as an outer product and one GEMM.

    A gathered sparse form (Gather x2, Mul, ReduceSum) is memory bound in ONNX
    Runtime: some output blades have 16 terms, so every multivector expands to
    256 gathered entries twice. One 256-wide outer product followed by a
    [256, 16] MatMul over the flattened table moves the same data once and
    reduces it in MLAS, ~3x faster per product on CPU.
    """

    def __init__(self, table: torch.Tensor) -> None:
        super().__init__()
        table = table.detach().double().to_dense()
        if table.shape != (_BLADES,) * 3:
            raise ValueError("expected a 16x16x16 PGA table")
        # weight[(j, k), i] = table[i, j, k], matching (x_j y_k) flattened over (j, k).
        weight = table.permute(1, 2, 0).reshape(_BLADES * _BLADES, _BLADES)
        self.register_buffer("weight", weight.float())

    def forward(self, x, y):
        pairs = (x.unsqueeze(-1) * y.unsqueeze(-2)).flatten(-2)
        return torch.matmul(pairs, self.weight.to(x.dtype))


def _bilinear_forward(self, multivectors, reference_mv, scalars):
    """GeometricBilinear.forward with one fused input GEMM and sparse products."""
    (left, _), (right, _), (join_left, _), (join_right, _) = self._onnx_inputs(
        multivectors, scalars
    )
    gp_outputs = self._onnx_gp(left, right)
    join_outputs = reference_mv[..., [_JOIN_REFERENCE_BLADE]] * self._onnx_join(join_left, join_right)
    outputs_mv = torch.cat((gp_outputs, join_outputs), dim=-2)
    return self.linear_out(outputs_mv, scalars=scalars)


def _qkv_forward(self, inputs, scalars, additional_qk_features_mv=None, additional_qk_features_s=None):
    """MultiQueryQKVModule.forward with Q, K and V from one GEMM."""
    if additional_qk_features_mv is not None or additional_qk_features_s is not None:
        raise NotImplementedError("reinserted channels are not used by the tracking model")
    (q_mv, q_s), (k_mv, k_s), (v_mv, v_s) = self._onnx_qkv(inputs, scalars)
    heads, hidden_mv = self.config.num_heads, self.config.hidden_mv_channels
    hidden_s = self.config.hidden_s_channels
    # "... items (hidden heads) x -> ... heads items hidden x", as in gatr.
    # Positive permutations: negative movedim axes export an invalid Transpose.
    q_mv = q_mv.reshape(-1, hidden_mv, heads, _BLADES).permute(2, 0, 1, 3)
    q_s = q_s.reshape(-1, hidden_s, heads).permute(2, 0, 1)
    return q_mv, k_mv.unsqueeze(-4), v_mv.unsqueeze(-4), q_s, k_s.unsqueeze(-3), v_s.unsqueeze(-3)


def _layer_norm_forward(self, multivectors, scalars):
    """EquiLayerNorm.forward with a masked inner product and static widths."""
    squared = (multivectors * multivectors * self._onnx_mask.to(multivectors.dtype)).sum(
        -1, keepdim=True
    )
    squared = squared.mean(dim=self.mv_channel_dim, keepdim=True).clamp(min=self.epsilon)
    outputs_mv = multivectors / torch.sqrt(squared)
    outputs_s = F.layer_norm(scalars, (self._onnx_s_channels,))
    return outputs_mv, outputs_s


class AffineEmbedding(nn.Module):
    """The hit embedding is affine in the 7 inputs: ``hits @ E + c`` (no index_put)."""

    def __init__(self, model: SingleEventGATr) -> None:
        super().__init__()
        with torch.no_grad():
            basis = torch.cat((torch.zeros(1, 7), torch.eye(7))).double()
            values = model.embed(basis)
        offset, matrix = values[0], values[1:] - values[0]
        probe = torch.randn(64, 7, dtype=torch.float64) * 1000
        with torch.no_grad():
            if not torch.allclose(probe @ matrix + offset, model.embed(probe), rtol=0, atol=1e-9):
                raise ValueError("hit embedding is not affine; cannot fold it")
        self.register_buffer("matrix", matrix.float())
        self.register_buffer("offset", offset.float())

    def forward(self, hits):
        return torch.matmul(hits, self.matrix.to(hits.dtype)) + self.offset.to(hits.dtype)


def prepare_for_onnx(model: SingleEventGATr) -> Dict[str, int]:
    """Rewrite a disposable eval model in place; return replacement counts."""
    if model.training:
        raise ValueError("call model.eval() before preparing the export model")
    model.embed = AffineEmbedding(model)
    gp = TableBilinear(_load_bilinear_basis("gp", torch.device("cpu"), torch.float64))
    join = TableBilinear(_compute_efficient_join(torch.device("cpu"), torch.float64))
    mask = torch.zeros(_BLADES)
    mask[INNER_PRODUCT_INDICES] = 1.0
    counts = {"bilinear": 0, "qkv": 0, "layer_norm": 0, "linear": 0}

    for module in list(model.modules()):
        if isinstance(module, GeometricBilinear):
            module._onnx_inputs = fused_equi_linear(
                [module.linear_left, module.linear_right,
                 module.linear_join_left, module.linear_join_right]
            )
            module._onnx_gp, module._onnx_join = gp, join
            del module.linear_left, module.linear_right
            del module.linear_join_left, module.linear_join_right
            module.forward = _bilinear_forward.__get__(module)
            counts["bilinear"] += 1
        elif isinstance(module, MultiQueryQKVModule):
            module._onnx_qkv = fused_equi_linear([module.q_linear, module.k_linear, module.v_linear])
            del module.q_linear, module.k_linear, module.v_linear
            module.forward = _qkv_forward.__get__(module)
            counts["qkv"] += 1
        elif isinstance(module, EquiLayerNorm):
            module.register_buffer("_onnx_mask", mask.clone(), persistent=False)
            module._onnx_s_channels = model.hidden_s_channels
            module.forward = _layer_norm_forward.__get__(module)
            counts["layer_norm"] += 1

    def replace(parent):
        for name, child in list(parent.named_children()):
            if isinstance(child, EquiLinear):
                setattr(parent, name, FoldedEquiLinear(child))
                counts["linear"] += 1
            else:
                replace(child)

    replace(model)
    model.requires_grad_(False)
    return counts


# --------------------------------------------------------------------------
# Attention core used during export (patched into gatr only inside a context)
# --------------------------------------------------------------------------
def _export_sdpa(q, k, v):
    """Attention core; K/V keep their singleton head axis (multi-query).

    Exported as Mul -> MatMul(q, K^T) -> Softmax -> MatMul(probs, V), the
    pattern the ONNX attention rewrites below look for.
    """
    scale = 1.0 / math.sqrt(float(q.shape[-1]))
    scores = torch.matmul(q * scale, k.transpose(-1, -2))
    return torch.matmul(torch.softmax(scores, dim=-1), v)


def _matmul_dist_vec(tri, basis, normalizer):
    """``einsum("xyz,...x,...y->...z")`` written as an outer product and MatMul."""
    tri_normed = tri * normalizer(tri[..., [3]])
    outer = (tri_normed.unsqueeze(-1) * tri_normed.unsqueeze(-2)).flatten(-2)
    return torch.matmul(outer, basis.reshape(-1, basis.shape[-1]).to(outer.dtype))


def _make_export_geometric_attention():
    """Single-event ``gatr.primitives.attention.geometric_attention``.

    Same features, padding, key scaling and softmax as gatr; static channel
    widths and Pad instead of runtime-shaped zeros/views, which the TorchScript
    ONNX exporter cannot handle with a dynamic hit axis.
    """
    basis_q, basis_k = (
        b.detach() for b in gatr_attention._build_dist_basis(torch.device("cpu"), torch.float64)
    )
    tri, inner = gatr_attention._TRIVECTOR_IDX, gatr_attention._INNER_PRODUCT_WO_TRI_IDX

    def attention(q_mv, k_mv, v_mv, q_s, k_s, v_s, normalizer, weights=None, attn_mask=None):
        if attn_mask is not None:
            raise NotImplementedError("single-event export expects attention_mask=None")
        q_dist = _matmul_dist_vec(q_mv[..., tri], basis_q, normalizer)
        k_dist = _matmul_dist_vec(k_mv[..., tri], basis_k, normalizer)
        if weights is not None:
            q_dist = q_dist * weights[..., None].to(q_dist.dtype)
        mv_qk, s_qk = int(q_mv.shape[-2]), int(q_s.shape[-1])
        mv_v, s_v = int(v_mv.shape[-2]), int(v_s.shape[-1])
        width_qk = mv_qk * (len(inner) + basis_q.shape[-1]) + s_qk
        width_v = mv_v * _BLADES + s_v
        # gatr pads Q, K and V to one common width (80 for 16 MV channels, while
        # Q/K carry only 64 features) and rescales K so that the logits equal
        # q.k / sqrt(width_qk). Q/K and V are padded separately here, which
        # skips the zero columns in the score MatMul with identical logits.
        padded_qk = 8 * -(-width_qk // 8)
        padded_v = 8 * -(-width_v // 8)

        def features(mv, dist, s, used, width):
            x = torch.cat([part for part in (mv, dist, s) if part is not None], dim=-1)
            return F.pad(x, (0, width - used)) if width > used else x

        q = features(q_mv[..., inner].flatten(-2), q_dist.flatten(-2), q_s, width_qk, padded_qk)
        k = features(k_mv[..., inner].flatten(-2), k_dist.flatten(-2), k_s, width_qk, padded_qk)
        v = features(v_mv.flatten(-2), None, v_s, width_v, padded_v)
        k = k * math.sqrt(padded_qk / width_qk)  # logits = q.k / sqrt(width_qk), as in gatr
        out = _export_sdpa(q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0))[0]
        heads = int(q_mv.shape[0])
        return out[..., : mv_v * _BLADES].reshape(heads, -1, mv_v, _BLADES), out[
            ..., mv_v * _BLADES : mv_v * _BLADES + s_v
        ]

    return attention


@contextlib.contextmanager
def export_attention():
    """Swap gatr's attention primitive for the export form, then restore."""
    import gatr.layers.attention.attention as attention_layer

    saved = attention_layer.geometric_attention
    attention_layer.geometric_attention = _make_export_geometric_attention()
    try:
        yield
    finally:
        attention_layer.geometric_attention = saved


def num_attention_heads() -> int:
    return SelfAttentionConfig().num_heads


# --------------------------------------------------------------------------
# ONNX graph rewrites of the attention core (adapted from CIRCE's exporter)
# --------------------------------------------------------------------------
def _attention_patterns(nodes):
    producers = {out: node for node in nodes for out in node.output}
    for softmax in nodes:
        if softmax.op_type != "Softmax":
            continue
        before = producers.get(softmax.input[0])
        consumers = [n for n in nodes if softmax.output[0] in n.input]
        if (before is None or before.op_type != "MatMul" or len(consumers) != 1
                or consumers[0].op_type != "MatMul" or consumers[0].input[0] != softmax.output[0]):
            raise ValueError(f"unexpected attention pattern around {softmax.name}")
        yield before, softmax, consumers[0]


def _replace_nodes(model, replacements, removed):
    final = []
    for node in model.graph.node:
        if node.name in replacements:
            final.extend(replacements[node.name])
        elif node.name not in removed:
            final.append(node)
    del model.graph.node[:]
    model.graph.node.extend(final)
    del model.graph.value_info[:]  # stale shapes of the replaced dense attention


def blocked_attention(model, block: int = 256) -> int:
    """Each query block of ``block`` rows attends to all keys inside a Loop."""
    import numpy as np
    from onnx import TensorProto as T, helper as h, numpy_helper as nh

    nodes = list(model.graph.node)
    replacements, removed, count = {}, set(), 0
    for before, softmax, after in list(_attention_patterns(nodes)):
        q, kt, v = before.input[0], before.input[1], after.input[1]
        prefix = f"blocked_attention_{count}/"
        new = []

        def const(name, value, dtype=np.int64):
            name = prefix + name
            model.graph.initializer.append(nh.from_array(np.array(value, dtype=dtype), name))
            return name

        def op(kind, inputs, name, **attrs):
            out = prefix + name
            new.append(h.make_node(kind, inputs, [out], name=out, **attrs))
            return out

        axes01, axes0 = const("axes01", [0, 1]), const("axes0", [0])
        zero, one = const("zero", 0), const("one", 1)
        tile, tile_minus_one = const("tile", block), const("tile_minus_one", block - 1)
        condition = const("condition", True, np.bool_)
        steps, starts = const("steps", [1]), const("starts", [0])
        query_shape = op("Shape", [q], "query_shape")
        # Multi-query attention: K/V have singleton batch/head axes, so all
        # (head, query) rows share the same keys and can be flattened together.
        flat_query = op("Flatten", [q], "flat_query", axis=3)
        keys = op("Squeeze", [kt, axes01], "keys_2d")
        values = op("Squeeze", [v, axes01], "values_2d")
        rows = op("Gather", [op("Shape", [flat_query], "flat_shape"), zero], "rows", axis=0)
        trips = op("Div", [op("Add", [rows, tile_minus_one], "rows_plus"), tile], "trips")
        padding = op("Sub", [op("Mul", [trips, tile], "padded_rows"), rows], "padding")
        pads = op(
            "Concat",
            [const("zeros2", [0, 0]), op("Unsqueeze", [padding, axes0], "padding_1d"),
             const("zero1", [0])],
            "pads", axis=0,
        )
        padded_query = op("Pad", [flat_query, pads], "padded_query", mode="constant")
        names = {k: prefix + "body_" + k for k in
                 ("iteration", "condition", "start", "end", "start1", "end1",
                  "query", "scores", "probabilities", "output")}
        body = h.make_graph(
            [
                h.make_node("Mul", [names["iteration"], tile], [names["start"]]),
                h.make_node("Add", [names["start"], tile], [names["end"]]),
                h.make_node("Unsqueeze", [names["start"], axes0], [names["start1"]]),
                h.make_node("Unsqueeze", [names["end"], axes0], [names["end1"]]),
                h.make_node("Slice", [padded_query, names["start1"], names["end1"], axes0, steps],
                            [names["query"]]),
                h.make_node("MatMul", [names["query"], keys], [names["scores"]]),
                h.make_node("Softmax", [names["scores"]], [names["probabilities"]], axis=-1),
                h.make_node("MatMul", [names["probabilities"], values], [names["output"]]),
            ],
            prefix + "body",
            [h.make_tensor_value_info(names["iteration"], T.INT64, []),
             h.make_tensor_value_info(names["condition"], T.BOOL, [])],
            [h.make_tensor_value_info(names["condition"], T.BOOL, []),
             h.make_tensor_value_info(names["output"], T.FLOAT, [block, None])],
        )
        scan = op("Loop", [trips, condition], "loop", body=body)
        width = op("Gather", [op("Shape", [values], "value_shape"), one], "value_width", axis=0)
        scan_shape = op("Concat", [const("minus1", [-1]),
                                   op("Unsqueeze", [width, axes0], "value_width_1d")],
                        "scan_shape", axis=0)
        flat_output = op("Reshape", [scan, scan_shape], "flat_output")
        trimmed = op("Slice", [flat_output, starts, op("Unsqueeze", [rows, axes0], "rows_1d"),
                               axes0, steps], "trimmed")
        # Output shape: the query's leading axes and the value width (Q/K and V
        # may have different widths).
        leading = op("Slice", [query_shape, starts, const("three", [3]), axes0, steps], "leading_shape")
        output_shape = op("Concat", [leading, op("Unsqueeze", [width, axes0], "value_width_out")],
                          "output_shape", axis=0)
        new.append(h.make_node("Reshape", [trimmed, output_shape], list(after.output),
                               name=prefix + "output"))
        replacements[after.name] = new
        removed.update([before.name, softmax.name, after.name])
        count += 1
    if not count:
        raise ValueError("no attention found in the exported graph")
    _replace_nodes(model, replacements, removed)
    return count


def runtime_attention(model, heads: int) -> int:
    """Use ONNX Runtime's built-in MultiHeadAttention (com.microsoft domain)."""
    import numpy as np
    from onnx import helper as h, numpy_helper as nh

    nodes = list(model.graph.node)
    replacements, removed, count = {}, set(), 0
    for before, softmax, after in list(_attention_patterns(nodes)):
        q, kt, v = before.input[0], before.input[1], after.input[1]
        prefix = f"runtime_attention_{count}/"
        new = []

        def op(kind, inputs, name, **attrs):
            name = prefix + name
            new.append(h.make_node(kind, inputs, [name], name=name, **attrs))
            return name

        def const(name, value):
            name = prefix + name
            model.graph.initializer.append(nh.from_array(np.array(value, dtype=np.int64), name))
            return name

        shape = op("Shape", [q], "query_shape")
        k = op("Expand", [op("Transpose", [kt], "key", perm=[0, 1, 3, 2]), shape], "key_heads")
        # V may be wider than Q/K: expand only the batch/head/item axes.
        value_shape = op("Concat", [
            op("Slice", [shape, const("start0", [0]), const("stop3", [3])], "query_leading"),
            op("Slice", [op("Shape", [v], "value_shape"), const("start3", [3]), const("stop4", [4])],
               "value_width"),
        ], "value_heads_shape", axis=0)
        v = op("Expand", [v, value_shape], "value_heads")
        flat = const("flat_shape", [0, 0, -1])

        def to_3d(x, name):
            return op("Reshape", [op("Transpose", [x], name + "_bsnh", perm=[0, 2, 1, 3]), flat],
                      name + "_3d")

        # Q already carries the 1/sqrt(d) scale (see _export_sdpa), hence scale=1.
        attention = op("MultiHeadAttention", [to_3d(q, "query"), to_3d(k, "key"), to_3d(v, "value")],
                       "attention", domain="com.microsoft", num_heads=heads, scale=1.0,
                       unidirectional=0)
        out_shape = op("Gather", [value_shape, const("bsnh", [0, 2, 1, 3])], "output_shape", axis=0)
        out4 = op("Reshape", [attention, out_shape], "output_4d")
        new.append(h.make_node("Transpose", [out4], list(after.output), name=prefix + "output",
                               perm=[0, 2, 1, 3]))
        replacements[after.name] = new
        removed.update([before.name, softmax.name, after.name])
        count += 1
    if not count:
        raise ValueError("no attention found in the exported graph")
    _replace_nodes(model, replacements, removed)
    if not any(o.domain == "com.microsoft" for o in model.opset_import):
        model.opset_import.append(h.make_opsetid("com.microsoft", 1))
    return count


def stabilize_attention(model, tiny: float = 1e-12, logit_range: float = 30.0) -> Dict[str, int]:
    """Numerical approximation: zero |x| < tiny in Q/K/V, floor logits at max - range."""
    import numpy as np
    import onnx
    from onnx import helper as h, numpy_helper as nh

    for value in model.graph.initializer:
        x = nh.to_array(value)
        if x.dtype == np.float32:
            selected = (x != 0) & (np.abs(x) < tiny)
            if selected.any():
                y = x.copy()
                y[selected] = 0
                value.CopyFrom(nh.from_array(y, value.name))
    threshold, zero, negative = "stable_threshold", "stable_zero", "stable_negative_range"
    for name, value in ((threshold, tiny), (zero, 0.0), (negative, -logit_range)):
        model.graph.initializer.append(nh.from_array(np.array(value, dtype=np.float32), name))
    counts = {"softmax": 0, "qkv_guards": 0}

    def guard_targets(graph):
        """Q/K/V tensors to guard in the top-level graph."""
        blocked = [o for n in graph.node for o in n.output
                   if o.endswith(("/flat_query", "/keys_2d", "/values_2d"))]
        if blocked:
            return set(blocked)
        targets = set()
        for before, _, after in _attention_patterns(list(graph.node)):
            targets.update([before.input[0], before.input[1], after.input[1]])
        return targets

    def rewrite(graph, targets):
        final = []
        for n in graph.node:
            for attr in n.attribute:
                if attr.type == onnx.AttributeProto.GRAPH:
                    rewrite(attr.g, set())
            if n.op_type == "Softmax":
                scores = n.input[0]
                final.extend([
                    h.make_node("ReduceMax", [scores], [scores + "/row_max"], axes=[-1], keepdims=1),
                    h.make_node("Add", [scores + "/row_max", negative], [scores + "/floor"]),
                    h.make_node("Max", [scores, scores + "/floor"], [scores + "/floored"]),
                ])
                n.input[0] = scores + "/floored"
                counts["softmax"] += 1
            final.append(n)
            for index, output in enumerate(list(n.output)):
                if output not in targets:
                    continue
                raw = output + "/unguarded"
                n.output[index] = raw
                final.extend([
                    h.make_node("Abs", [raw], [output + "/abs"]),
                    h.make_node("Less", [output + "/abs", threshold], [output + "/tiny"]),
                    h.make_node("Where", [output + "/tiny", zero, raw], [output]),
                ])
                counts["qkv_guards"] += 1
        del graph.node[:]
        graph.node.extend(final)

    rewrite(model.graph, guard_targets(model.graph))
    if not counts["softmax"] or counts["qkv_guards"] != 3 * counts["softmax"]:
        raise ValueError(f"unexpected attention guards: {counts}")
    return counts
