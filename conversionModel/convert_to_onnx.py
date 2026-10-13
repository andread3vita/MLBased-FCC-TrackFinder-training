#!/usr/bin/env python3
"""Convert a GATR_CIRCE_LOSS GATr tracking checkpoint (.ckpt) to ONNX.

The model is rebuilt inference-only from the hyperparameters stored in the
Lightning checkpoint (``hyper_parameters``), weights are taken from the EMA
state (default used for validation/inference in training; ``--weights-source
raw`` selects the raw state). The rebuild mirrors
``GATR_CIRCE_LOSS/src/models/Gatr_withModifications.py::ExampleWrapper.forward``
for a single event (one xformers BlockDiagonalMask block -> no mask):

    hits [N,7] = x,y,z,planar_flag,u,v,w
    mv = embed_point(pos/1000) + embed_scalar(planar_flag)
         + embed_translation(vector/1000)   -> [N,1,16] (PGA)
    GATr(mv, attention_mask=None, join_reference=mean over hits)
    out [N,9] = [clustering coords (4), beta logit (1), helix proxy (4,
                zeros when helix_loss_weight=0)]

The ONNX graph takes float32 hits and returns float32, but computes
internally in float64 (COMPUTE_DTYPE): the geometric attention is
ill-conditioned in float32 (fp32 vs fp64 eager differs by ~1e-1 on a
4k-hit event). ONNX Runtime has no float64 ``Erf`` kernel, so the scalar
GELU (erf form, ``gatr.layers.mlp.nonlinearities``) is re-implemented in
float64 from the fdlibm ``s_erf.c`` polynomials before export (max error
vs libm erf ~1 ulp). The multivector gated GELU uses the tanh
approximation, which ORT supports in float64.

Usage:

    python convert_to_onnx.py -w model.ckpt -o model.onnx

Check the result with validate_onnx.py:

    python validate_onnx.py -w model.ckpt -o model.onnx [-i model_input_event0.csv]
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

import torch
from torch import nn

# The vendored gatr_v142 package of the training repository is the exact
# source the checkpoint was trained with; never import an installed gatr.
_MODEL_TRAINING_DIR = Path(
    os.environ.get(
        "GGTF_MODEL_TRAINING_DIR",
        Path(__file__).resolve().parent.parent / "model_training",
    )
)
_GATR_V142 = _MODEL_TRAINING_DIR / "GATR_CIRCE_LOSS" / "src" / "gatr_v142"
if not (_GATR_V142 / "gatr" / "nets" / "gatr.py").is_file():
    raise RuntimeError(
        "vendored gatr_v142 not found at "
        f"{_GATR_V142}; set GGTF_MODEL_TRAINING_DIR to the model_training directory"
    )
if str(_GATR_V142) not in sys.path:
    sys.path.insert(0, str(_GATR_V142))

from gatr.interface import embed_point, embed_scalar, embed_translation
from gatr.layers.attention.config import SelfAttentionConfig
from gatr.layers.mlp.config import MLPConfig
from gatr.layers.mlp.nonlinearities import ScalarGatedNonlinearity
from gatr.nets.gatr import GATr
from gatr.utils.einsum import enable_cached_einsum

# opt_einsum 3.3.0 mis-parses equations like "xyz,...x,...y->...z" when gatr
# computes (and caches) contraction paths, making the unconverted model raise
# even in plain eager mode. Routing every gatr einsum straight to torch.einsum
# is numerically equivalent and slightly slower only on first calls.
enable_cached_einsum(False)

# Internal compute precision of the exported graph (see module docstring).
COMPUTE_DTYPE = torch.float64

WEIGHT_SOURCES = ("ema", "raw")


class GatrTrackingModel(nn.Module):
    """Inference rebuild of the GATR_CIRCE_LOSS ExampleWrapper for one event.

    Takes the raw ``[N, 7]`` GGTFTrackFinder features instead of a DGL graph
    and returns the same ``[N, 9]`` output tensor as the training model:
    4 clustering coordinates, 1 beta logit (sigmoid applied by consumers),
    4 helix-proxy columns (zeros when the helix loss was disabled).
    """

    helix_proxy_dim = 4

    def __init__(self, hparams: dict) -> None:
        super().__init__()
        if bool(hparams.get("use_detector_features", False)):
            raise NotImplementedError(
                "checkpoint uses detector features; this converter covers the "
                "geometry-only (7-feature) model"
            )
        self.embedding_dim = int(hparams.get("clustering_space_dim", 4))
        self.output_dim = self.embedding_dim + 1 + self.helix_proxy_dim
        self.position_scale = float(hparams.get("position_scale", 1000.0))
        self._use_helix_proxy = float(hparams.get("helix_loss_weight", 0.0)) > 0.0

        self.gatr = GATr(
            in_mv_channels=1,
            out_mv_channels=1,
            hidden_mv_channels=int(hparams.get("hidden_mv_channels", 16)),
            in_s_channels=None,
            out_s_channels=None,
            hidden_s_channels=int(hparams.get("hidden_s_channels", 64)),
            num_blocks=int(hparams.get("gatr_blocks", 10)),
            attention=SelfAttentionConfig(),
            mlp=MLPConfig(),
            checkpoint=None,
        )
        self.clustering = nn.Linear(16, self.embedding_dim, bias=False)
        self.beta = nn.Linear(16, 1)
        self.helix_proxy = nn.Linear(16, self.helix_proxy_dim)

    def forward(self, hits: torch.Tensor) -> torch.Tensor:
        # hits: [N, 7] = (x, y, z, planar_flag, u, v, w), one event.
        dtype = self.clustering.weight.dtype
        hits = hits.to(dtype)
        positions_scaled = hits[..., :3] / self.position_scale
        hit_type = hits[..., 3:4]
        vector_scaled = hits[..., 4:7] / self.position_scale

        multivectors = (
            embed_point(positions_scaled)
            + embed_scalar(hit_type)
            + embed_translation(vector_scaled)
        ).unsqueeze(-2)  # [N, 1, 16]

        # A single event is one BlockDiagonalMask block, i.e. full attention
        # (attention_mask=None), with a data-derived join reference equal to
        # the mean multivector of the event (cf. ExampleWrapper._join_reference).
        join_reference = multivectors.mean(dim=-2, keepdim=True).expand_as(multivectors)

        embedded, _ = self.gatr(
            multivectors,
            scalars=None,
            attention_mask=None,
            join_reference=join_reference,
        )
        latent = embedded[..., 0, :]  # [N, 16]
        if self._use_helix_proxy:
            helix_prediction = self.helix_proxy(latent)
        else:
            helix_prediction = hits.new_zeros(
                (*hits.shape[:-1], self.helix_proxy_dim)
            )
        return torch.cat(
            (self.clustering(latent), self.beta(latent), helix_prediction), dim=-1
        )


# --------------------------------------------------------------------------
# float64 erf from the fdlibm s_erf.c polynomial tables.
#
# ORT implements Erf for float32 only, so the erf-GELU used on GATr's scalar
# channels is swapped for this version before export. Branch cut
# ``_B_CUT`` is the fdlibm boundary between the ra/sa and rb/sb rational
# approximations (high word 0x4006DB6E, i.e. 1/0.35 rounded down).
# fdlibm additionally truncates the exp() argument to pseudo-single
# precision; skipping that changes erf by O(1e-16), far below any tolerance
# relevant here.
# --------------------------------------------------------------------------
_ERX = 8.45062911510467529297e-01  # erf(x) at the 0.84375/1.25 split (approx.)
_PP = (
    1.28379167095512558561e-01,
    -3.25042107247001499370e-01,
    -2.84817495755985104766e-02,
    -5.77027029648944159157e-03,
    -2.37630166566501626084e-05,
)
_QQ = (
    3.97917223959155352819e-01,
    6.50222499887672944485e-02,
    5.08130628187576562776e-03,
    1.32494738004321644526e-04,
    -3.96022827877536812320e-06,
)
_PA = (
    -2.36211856075265944077e-03,
    4.14856118683748331666e-01,
    -3.72207876035701323847e-01,
    3.18346619901161753674e-01,
    -1.10894694282396677476e-01,
    3.54783043256182359371e-02,
    -2.16637559486879084300e-03,
)
_QA = (
    1.06420880400844228286e-01,
    5.40397917702171048937e-01,
    7.18286544141962662868e-02,
    1.26171219808761642112e-01,
    1.36370839120290507362e-02,
    1.19844998467991074170e-02,
)
_RA = (
    -9.86494403484714822705e-03,
    -6.93858572707181708672e-01,
    -1.05586262253232909884e01,
    -6.23753324503260060396e01,
    -1.62396669462573470355e02,
    -1.84605092906711035994e02,
    -8.12874355063065934246e01,
    -9.81432934416914548592e00,
)
_SA = (
    1.96512716674392571292e01,
    1.37657754143519042600e02,
    4.34565877475229228821e02,
    6.45387271733267880336e02,
    4.29008140027567833386e02,
    1.08635005541779435134e02,
    6.57024977031928170135e00,
    -6.04244152148580987438e-02,
)
_RB = (
    -9.86494292470009928597e-03,
    -7.99283237680523006574e-01,
    -1.77579549177547519889e01,
    -1.60636384855821916062e02,
    -6.37566443368389627722e02,
    -1.02509513161107724954e03,
    -4.83519191608651397019e02,
)
_SB = (
    3.03380607434824582924e01,
    3.25792512996573918826e02,
    1.53672958608443695994e03,
    3.19985821950859553908e03,
    2.55305040643316442583e03,
    4.74528541206955367215e02,
    -2.24409524465858183362e01,
)
_B_CUT = 2.857142925262451171875  # fdlibm high-word boundary 0x4006DB6E


def _horner(coeffs, x, one_based=False):
    out = torch.full_like(x, coeffs[-1])
    for c in reversed(coeffs[:-1]):
        out = out * x + c
    if one_based:
        out = out * x + torch.ones_like(x)
    return out


def erf_fdlibm(x: torch.Tensor) -> torch.Tensor:
    """fdlibm erf() for float64 tensors; ~1 ulp against libm everywhere."""
    ax = torch.abs(x)
    sign = torch.sign(x)

    # |x| < 0.84375: erf(x) = x + x * P(x^2)/Q(x^2)
    z = x * x
    r = _horner(_PP, z)
    s = _horner(_QQ, z, one_based=True)
    y_small = x + x * (r / s)

    # 0.84375 <= |x| < 1.25: erf(x) = sign(x) * (erx + Pa(|x|-1)/Qa(|x|-1))
    s1 = ax - 1.0
    p = _horner(_PA, s1)
    q = _horner(_QA, s1, one_based=True)
    y_mid = sign * (_ERX + p / q)

    # |x| >= 1.25: erf(x) = sign(x) * (1 - erfc), erfc via exp + rational in 1/x^2
    sinv = 1.0 / (ax * ax)
    ra = _horner(_RA, sinv)
    sa = _horner(_SA, sinv, one_based=True)
    rb = _horner(_RB, sinv)
    sb = _horner(_SB, sinv, one_based=True)
    ratio = torch.where(ax < _B_CUT, ra / sa, rb / sb)
    erfc = torch.exp(-ax * ax - 0.5625) * torch.exp(ratio) / torch.clamp(ax, min=1.25)
    y_large = sign * (1.0 - erfc)
    y_large = torch.where(ax >= 6.0, sign, y_large)

    return torch.where(
        ax < 0.84375, y_small, torch.where(ax < 1.25, y_mid, y_large)
    )


def _gelu_erf_fdlibm(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * x * (1.0 + erf_fdlibm(x / math.sqrt(2.0)))


def _math_sdpa(query, key, value, attn_mask=None):
    """Mask-free scaled_dot_product_attention decomposition for ONNX export.

    Only used during export: single-event inference passes attention_mask=None,
    and torch's fp64 SDPA has no ONNX symbolic; the math below is numerically
    identical (softmax with max subtraction in float64).
    """
    if attn_mask is not None:
        raise NotImplementedError("export path expects attention_mask=None")
    # einsum, not matmul(key.transpose(-2, -1)): ORT fuses MatMul+Transpose
    # into com.microsoft.FusedMatMul, which has no float64 kernel and crashes
    # CPU session creation.
    scores = torch.einsum("... i d, ... j d -> ... i j", query, key)
    scores = scores / math.sqrt(float(query.shape[-1]))
    return torch.einsum(
        "... i j, ... j d -> ... i d", torch.softmax(scores, dim=-1), value
    )


def _export_sdpa_wrapper(q, k, v, attn_mask):
    """Un-wrapped replacement for gatr's ``_sdpa_graph_breaking``.

    The original is decorated with ``@torch.compiler.disable``, which hard
    stops the dynamo-based ONNX exporter. The content is identical.
    """
    from gatr.utils.tensors import expand_pairwise

    q, k, v = expand_pairwise(q, k, v, exclude_dims=(-2,))
    return q, k, _math_sdpa(q, k, v, attn_mask=attn_mask)


def _replace_erf_gelu(model: nn.Module) -> int:
    """Swap the erf-form scalar GELU of GATr MLPs for the fdlibm version."""
    count = 0
    for module in model.modules():
        if isinstance(module, ScalarGatedNonlinearity):
            if module.scalar_nonlinearity is nn.functional.gelu:
                module.scalar_nonlinearity = _gelu_erf_fdlibm
                count += 1
    return count


def _add_to_scalar_slot(mv: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    """``mv[..., 0] += values`` written as a cat+add (ONNX-safe, no index_put)."""
    padded = torch.cat(
        (values.unsqueeze(-1), values.new_zeros(*values.shape, 15)), dim=-1
    )
    return mv + padded


def _equilayernorm_export_forward(self, multivectors, scalars):
    """EquiLayerNorm.forward with a static-shape manual scalar LayerNorm.

    The original passes ``scalars.shape[-1:]`` to F.layer_norm, which the
    legacy tracer cannot keep static. The decomposition below is bit-for-bit
    equivalent in float64 (unbiased=False, eps=1e-5, no affine parameters).
    """
    from gatr.primitives import equi_layer_norm

    outputs_mv = equi_layer_norm(
        multivectors, channel_dim=self.mv_channel_dim, epsilon=self.epsilon
    )
    mean = scalars.mean(dim=-1, keepdim=True)
    var = scalars.var(dim=-1, unbiased=False, keepdim=True)
    outputs_s = (scalars - mean) / torch.sqrt(var + 1e-5)
    return outputs_mv, outputs_s


def _equilinear_export_forward(self, multivectors, scalars=None):
    """EquiLinear.forward without the in-place ``mv[..., 0] +=`` index_put.

    ``outputs_mv[..., 0] += self.s2mvs(scalars)`` is replaced by a cat+add
    equivalent, which exports to plain ONNX ops instead of an unsupported
    index_put (``embed_scalar`` cannot be used: it expects [..., 1]).
    """
    from gatr.interface import embed_scalar
    from gatr.primitives.linear import equi_linear

    outputs_mv = equi_linear(multivectors, self.weight)
    if self.bias is not None:
        outputs_mv = outputs_mv + embed_scalar(self.bias)
    if self.s2mvs is not None and scalars is not None:
        outputs_mv = _add_to_scalar_slot(outputs_mv, self.s2mvs(scalars))
    if self.mvs2s is not None:
        outputs_s = self.mvs2s(multivectors[..., 0])
        if self.s2s is not None and scalars is not None:
            outputs_s = outputs_s + self.s2s(scalars)
    else:
        outputs_s = None
    return outputs_mv, outputs_s


def _patch_for_export(model: nn.Module) -> None:
    from gatr.layers.layer_norm import EquiLayerNorm
    from gatr.layers.linear import EquiLinear

    for module in model.modules():
        if isinstance(module, EquiLayerNorm):
            module.forward = _equilayernorm_export_forward.__get__(module)
        elif isinstance(module, EquiLinear):
            module.forward = _equilinear_export_forward.__get__(module)


class _OnnxExportWrapper(nn.Module):
    """float32 I/O around the float64-computing model."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, hits: torch.Tensor) -> torch.Tensor:
        return self.model(hits.to(COMPUTE_DTYPE)).float()


def select_weights(checkpoint: dict, weights_source: str) -> dict:
    if weights_source == "ema":
        state = checkpoint.get("ema_state_dict")
        if not isinstance(state, dict) or not state:
            raise ValueError(
                "checkpoint has no non-empty ema_state_dict; "
                "use --weights-source raw"
            )
        return state
    state = checkpoint.get("state_dict", checkpoint)
    if not isinstance(state, dict) or not state:
        raise ValueError("checkpoint has no non-empty state_dict")
    return state


def build_model(weight_path: str, weights_source: str = "ema"):
    """Rebuild the tracking model from a checkpoint and load its weights.

    Returns ``(model, description)``; the model maps a [N, 7] float tensor to
    the [N, 9] network output and can be run in float32 (``model(hits)``) or
    float64 (``model.double()(hits.double())``).
    """
    checkpoint = torch.load(weight_path, map_location="cpu", weights_only=False)
    hparams = dict(checkpoint.get("hyper_parameters") or {})
    model = GatrTrackingModel(hparams)
    state = select_weights(checkpoint, weights_source)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected:
        raise RuntimeError(f"unexpected checkpoint keys: {sorted(unexpected)[:8]}")
    pinned = [k for k in missing if "helix_proxy" not in k]
    if pinned:
        raise RuntimeError(f"missing checkpoint weights: {pinned[:8]}")
    model.eval()
    description = f"{Path(weight_path).name} [{weights_source}_state_dict]"
    return model, description


def convert(weight_path: str, onnx_path: str, weights_source: str = "ema") -> None:
    model, description = build_model(weight_path, weights_source)
    print(f"checkpoint weights: {description}")

    # Internal float64 compute; fp32 tensors at the graph boundary.
    model = model.to(COMPUTE_DTYPE)
    n_erf = _replace_erf_gelu(model)
    _patch_for_export(model)
    print(f"replaced erf-GELU with fdlibm float64 erf in {n_erf} MLP blocks")

    exportable = _OnnxExportWrapper(model).eval()
    dummy = torch.zeros(8, 7, dtype=torch.float32)
    dynamic_axes = {"hits": {0: "N"}, "output": {0: "N"}}

    # gatr lazily torch.load()s its geometric-product/join basis tables on
    # first use of a given dtype; running one eager pass first fills those
    # caches with real tensors so tracing never executes the load.
    with torch.no_grad():
        exportable(dummy)

    export_errors = []
    for use_dynamo in (True, False):
        try:
            # Replace torch's SDPA by its math decomposition: the fp64 variant
            # has no reliable ONNX symbolic. ``_sdpa_graph_breaking`` (wrapped
            # in @torch.compiler.disable) is replaced wholesale so the dynamo
            # exporter does not stop at it.
            import gatr.primitives.attention as _gatr_attn

            _gatr_attn.scaled_dot_product_attention = _math_sdpa
            _gatr_attn._sdpa_graph_breaking = _export_sdpa_wrapper
            with torch.no_grad():
                torch.onnx.export(
                    exportable,
                    (dummy,),
                    onnx_path,
                    input_names=["hits"],
                    output_names=["output"],
                    dynamic_axes=dynamic_axes,
                    opset_version=18,
                    dynamo=use_dynamo,
                )
            print(f"exported with torch.onnx dynamo={use_dynamo}")
            break
        except Exception as exc:  # noqa: BLE001 - retry with the other exporter
            export_errors.append(f"dynamo={use_dynamo}: {type(exc).__name__}: {exc}")
    else:
        raise RuntimeError("onnx export failed:\n" + "\n".join(export_errors))

    import onnx

    model_proto = onnx.load(onnx_path)
    onnx.checker.check_model(model_proto)
    size_mb = os.path.getsize(onnx_path) / 1e6
    print(f"wrote {onnx_path} ({size_mb:.2f} MB), opset "
          + ",".join(str(o.version) for o in model_proto.opset_import))

    # Smoke check: ORT vs float64 eager reference on a seeded random event.
    import onnxruntime as ort

    torch.manual_seed(42)
    hits = torch.randn(30, 7, dtype=torch.float32)
    with torch.no_grad():
        reference = model(hits.to(COMPUTE_DTYPE)).float()
    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    (output,) = session.run(None, {"hits": hits.numpy()})
    diff = (torch.from_numpy(output) - reference).abs().max().item()
    print(f"smoke check 30x7 random hits: max |ORT - fp64 eager| = {diff:.3e}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a GATR_CIRCE_LOSS GATr checkpoint to ONNX."
    )
    parser.add_argument("-w", "--weightPath", required=True, help="input .ckpt file")
    parser.add_argument("-o", "--onnxPath", required=True, help="output .onnx file")
    parser.add_argument(
        "--weights-source",
        choices=WEIGHT_SOURCES,
        default="ema",
        help="which checkpoint weights to convert (default: ema, the "
        "validation/inference weights used in training)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    convert(args.weightPath, args.onnxPath, args.weights_source)
    return 0


if __name__ == "__main__":
    sys.exit(main())
