"""Optional CPU/FP32 inference preparation; call after loading checkpoint weights."""

from pathlib import Path

import torch
from torch import nn

from src.cgatr.layers.layer_norm import EquiLayerNorm
from src.cgatr.primitives.attention import geometric_attention
from src.cgatr.primitives.bilinear import geometric_product
from src.cgatr.primitives.normalization import _GRADE_RANGES_BY_BLADES


class CPUGeometricProduct(nn.Module):
    """Sparse Cayley-table contraction without a (..., 32, 32) temporary."""

    def __init__(self, gp):
        super().__init__()
        gp = gp.detach().cpu().to_dense().float()
        if gp.shape != (32, 32, 32) or not torch.isfinite(gp).all():
            raise ValueError("Expected a finite 32-blade CGA Cayley table")
        count = (gp != 0).sum(dim=1).long()
        width = max(1, int(count.max()))
        left = torch.zeros(32, 32, width, dtype=torch.long)
        coefficient = torch.zeros(32, 32, width, dtype=torch.float32)
        for i in range(32):
            for k in range(32):
                j = torch.nonzero(gp[i, :, k], as_tuple=False).flatten()
                left[i, k, :len(j)] = j
                coefficient[i, k, :len(j)] = gp[i, j, k]
        self.register_buffer("left", left, persistent=False)
        self.register_buffer("coefficient", coefficient, persistent=False)
        self.register_buffer("count", count, persistent=False)

    def forward(self, x, y):
        if torch.is_grad_enabled():
            raise RuntimeError("CPU geometric product requires torch.inference_mode() or no_grad()")
        return torch.ops.circe_cpu.geometric_product(
            x, y, self.left, self.coefficient, self.count
        )


class _TraceableLayerNorm(nn.Module):
    """Resolve the fixed algebra's grade ranges before TorchScript tracing."""

    def __init__(self, source):
        super().__init__()
        self.register_buffer("ip_weights", source.ip_weights.detach().clone())
        self.grade_ranges = _GRADE_RANGES_BY_BLADES[source.ip_weights.numel()]
        self.channel_dim = source.mv_channel_dim
        self.epsilon = source.epsilon
        self.epsilon_mode = source.epsilon_mode
        self.scalar_channels = source.hidden_s_channels

    def forward(self, multivectors, scalars):
        wsq = self.ip_weights.to(multivectors.dtype) * multivectors * multivectors
        squared = wsq[..., :1].sum(-1, keepdim=True).abs()
        for a, b in self.grade_ranges[1:]:
            squared = squared + wsq[..., a:b].sum(-1, keepdim=True).abs()
        squared = squared.mean(dim=self.channel_dim, keepdim=True)
        squared = (squared.clamp_min(self.epsilon) if self.epsilon_mode == "clamp"
                   else squared + self.epsilon)
        return (multivectors / squared.sqrt(),
                torch.nn.functional.layer_norm(scalars, [self.scalar_channels]))


def prepare_cpu_inference(model: nn.Module, library: str | Path) -> nn.Module:
    """Modify a disposable CPU/FP32 eval model in place, after loading weights.

    Replaces CGA products and trace-sensitive normalization, and expands shared
    attention heads as views. The result is inference-only; keep the original
    checkpoint for training. Load the same operator library before jit.load().
    """
    if any(layer.training for layer in model.modules()):
        raise ValueError("Call model.eval() before preparing CPU inference")
    for tensor in (*model.parameters(), *model.buffers()):
        if tensor.device.type != "cpu" or (tensor.is_floating_point() and tensor.dtype != torch.float32):
            raise ValueError("CPU inference requires a CPU/FP32 model")
    products = [m for m in model.modules() if isinstance(m, geometric_product)]
    if not products or any(tuple(m.gp.shape) != (32, 32, 32) for m in products):
        raise ValueError("CPU inference supports 32-blade CGA models only")
    torch.ops.load_library(str(Path(library).resolve()))

    def replace(parent):
        for name, child in list(parent.named_children()):
            if isinstance(child, geometric_product):
                setattr(parent, name, CPUGeometricProduct(child.gp).eval())
            elif isinstance(child, EquiLayerNorm):
                setattr(parent, name, _TraceableLayerNorm(child).eval())
            else:
                if isinstance(child, geometric_attention):
                    child._expand_cpu_heads = True
                replace(child)

    replace(model)
    return model.requires_grad_(False)
