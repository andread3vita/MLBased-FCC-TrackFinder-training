"""Optional CPU/FP32 inference preparation; call after loading checkpoint weights."""

from pathlib import Path
import argparse
import json

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


class SingleEvent(nn.Module):
    """Measured [N,10] features to clustering coordinates and raw beta logit."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, features):
        mv, scalars = self.model.embed(features)
        _, output = self.model.cgatr(mv, scalars=scalars, attention_mask=None)
        return output


def export_cpu_checkpoint(checkpoint_path, features, library, output):
    """Save a LibTorch graph and packed algebra constants; prefer EMA weights.

    Builds a separate CPU model so export never modifies the training model.
    The same function handles old checkpoints and automatic training exports.
    """
    from src.model import CGATrParquetModel
    from src.eval.forward_pass import _load_state_dict_tolerating_derived_buffers

    features = features.detach().cpu().float().contiguous()
    if features.ndim != 2 or features.shape[1] != 10 or not torch.isfinite(features).all():
        raise ValueError("Expected finite measured float32 [N,10] features")
    vertex, drift = features[features[:, 3] == 0], features[features[:, 3] == 1]
    if not len(vertex) or not len(drift):
        raise ValueError("Export requires an example with both vertex and drift hits")
    output = Path(output)
    if output.resolve() == Path(checkpoint_path).resolve():
        raise ValueError("Inference output must differ from the training checkpoint")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = checkpoint.get("ema_state_dict")
    weights_source = "ema" if state else "model"
    if not state:
        state = {k[6:]: v for k, v in checkpoint["state_dict"].items() if k.startswith("model.")}
        if not state:
            raise ValueError("Checkpoint has neither EMA nor Lightning model weights")
    # Model construction initializes weights before loading them. Preserve the
    # training process's CPU RNG sequence when exporting during a run.
    with torch.random.fork_rng(devices=[]), torch.inference_mode():
        config = dict(checkpoint["hyper_parameters"], grad_checkpoint=False, equi_init="default")
        model = CGATrParquetModel(argparse.Namespace(**config))
        if not model.invariant_output_head or model.num_blades != 32 or model.needs_drift_dir:
            raise ValueError("CPU export requires the 32-blade CGA invariant head with 10 input features")
        _load_state_dict_tolerating_derived_buffers(model, state)
        runner = SingleEvent(prepare_cpu_inference(model.cpu().float().eval(), library)).eval()
        traced = torch.jit.trace(runner, (features,), check_trace=False, strict=False)
        traced = torch.jit.freeze(traced.eval())
        for example in (features, features[:128], vertex[:17], drift[:17]):
            torch.testing.assert_close(traced(example), runner(example), atol=1e-4, rtol=1e-4)
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_name(output.name + ".tmp")
        try:
            traced.save(str(temporary), _extra_files={"circe_cpu.json": json.dumps({"weights_source": weights_source})})
            temporary.replace(output)
        finally:
            temporary.unlink(missing_ok=True)
    return weights_source
