"""Run with CIRCE_CPU_LIBRARY=/path/to/libcirce_cpu.so pytest -q <this file>."""

import os
from pathlib import Path

import pytest
import torch
from torch import nn

from src.cgatr.cpu_inference import CPUGeometricProduct, prepare_cpu_inference
from src.cgatr.layers.layer_norm import EquiLayerNorm
from src.cgatr.primitives.attention import geometric_attention
from src.cgatr.primitives.bilinear import geometric_product


@pytest.fixture(scope="module")
def library():
    path = os.environ.get("CIRCE_CPU_LIBRARY")
    if not path:
        pytest.skip("Build the optional CPU operator and set CIRCE_CPU_LIBRARY")
    torch.ops.load_library(path)
    return path


@pytest.fixture(scope="module")
def gp():
    path = Path(__file__).resolve().parents[3] / "cga_utils/cga_geometric_product.pt"
    return torch.load(path, map_location="cpu", weights_only=False).to_dense().float()


def dense_product(gp, x, y):
    return torch.einsum("...ik,...k->...i", torch.einsum("ijk,...j->...ik", gp, x), y)


@pytest.mark.parametrize("shape", [(32,), (0, 32), (17, 6, 32), (2, 13, 7, 32)])
def test_product_matches_dense(library, gp, shape):
    rng = torch.Generator().manual_seed(42)
    x, y = [torch.randn(shape, generator=rng) for _ in range(2)]
    with torch.inference_mode():
        torch.testing.assert_close(CPUGeometricProduct(gp)(x, y), dense_product(gp, x, y), atol=2e-5, rtol=2e-5)


def test_noncontiguous_and_dynamic_trace(library, gp, tmp_path):
    module = CPUGeometricProduct(gp).eval()
    x = torch.randn(7, 13, 32).transpose(0, 1)
    y = torch.randn(7, 13, 32).transpose(0, 1)
    with torch.inference_mode():
        torch.testing.assert_close(module(x, y), dense_product(gp, x, y), atol=2e-5, rtol=2e-5)
        traced = torch.jit.trace(module, (x, y))
        path = tmp_path / "product.pt"
        torch.jit.freeze(traced).save(str(path))
        reloaded = torch.jit.load(str(path))
        a, b = torch.randn(31, 5, 32), torch.randn(31, 5, 32)
        torch.testing.assert_close(reloaded(a, b), dense_product(gp, a, b), atol=2e-5, rtol=2e-5)


def test_rejects_invalid_inputs_and_lookup_tables(library, gp):
    module = CPUGeometricProduct(gp)
    x = torch.randn(3, 32)
    with pytest.raises(RuntimeError, match="inference_mode"):
        module(x, x)
    with torch.inference_mode():
        for a, b in [(x.double(), x.double()), (x, x[:1]), (x[:, :31], x[:, :31])]:
            with pytest.raises(RuntimeError):
                module(a, b)
        module.count[0, 0] = module.left.shape[-1] + 1
        with pytest.raises(RuntimeError, match="term count"):
            module(x, x)
        module.count[0, 0] = 1
        module.left[0, 0, 0] = 32
        with pytest.raises(RuntimeError, match="blade index"):
            module(x, x)


def test_preparation_and_original_training_backward(library, gp):
    original = geometric_product(gp)
    x = torch.randn(3, 32, requires_grad=True)
    y = torch.randn(3, 32, requires_grad=True)
    original(x, y).square().sum().backward()
    assert torch.isfinite(x.grad).all() and torch.isfinite(y.grad).all()

    model = nn.ModuleDict({
        "product": original,
        "norm": EquiLayerNorm(torch.ones(32), hidden_s_channels=4),
        "attention": geometric_attention(None, None),
    })
    assert not model["attention"]._expand_cpu_heads
    with pytest.raises(ValueError, match="eval"):
        prepare_cpu_inference(model, library)
    model.eval()
    mv, scalar = torch.randn(11, 3, 32), torch.randn(11, 4)
    with torch.inference_mode():
        expected_mv, expected_s = model["norm"](mv, scalar)
        prepare_cpu_inference(model, library)
        actual_mv, actual_s = model["norm"](mv, scalar)
        torch.testing.assert_close(actual_mv, expected_mv, atol=0, rtol=0)
        torch.testing.assert_close(actual_s, expected_s, atol=0, rtol=0)
    assert isinstance(model["product"], CPUGeometricProduct)
    assert model["attention"]._expand_cpu_heads
