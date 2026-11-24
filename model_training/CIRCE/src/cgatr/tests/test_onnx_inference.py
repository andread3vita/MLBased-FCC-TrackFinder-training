"""ONNX rewrites: algebra identity, scalar paths and dynamic full attention.

Run from model_training/CIRCE with onnx and onnxruntime installed.
"""

import numpy as np
import pytest
import torch

pytest.importorskip("onnx")
ort = pytest.importorskip("onnxruntime")
import onnx
from onnx import helper as h, TensorProto as T
from tools.export_onnx import (
    MatrixGP,
    PackedLinear,
    blocked_attention,
    runtime_attention,
    stabilize_attention,
)
from src.cgatr.layers.linear import EquiLinear


def session(path):
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    return ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])


def test_geometric_product_all_blades_and_broadcasting():
    table = torch.load("cga_utils/cga_geometric_product.pt", weights_only=False).to_dense().float()
    product = MatrixGP(table)
    blades = torch.eye(32)
    torch.testing.assert_close(
        product(blades[:, None], blades[None]), table.permute(1, 2, 0), rtol=0, atol=0
    )
    torch.manual_seed(42)
    left, right = torch.randn(2, 7, 3, 32), torch.randn(1, 7, 3, 32)
    expected = torch.einsum("ijk,...j,...k->...i", table, left, right)
    torch.testing.assert_close(product(left, right), expected, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("scalars,bias", [(False, False), (False, True), (True, True)])
def test_folded_linear_includes_bias_and_scalar_paths(scalars, bias):
    torch.manual_seed(42)
    # Deliberately exercise arbitrary basis maps as well as scalar passthrough.
    layer = EquiLinear(
        torch.randn(9, 32, 32),
        2,
        3,
        in_s_channels=4 if scalars else None,
        out_s_channels=5 if scalars else None,
        bias=bias,
    ).eval()
    x = torch.randn(2, 7, 2, 32)
    s = torch.randn(2, 7, 4) if scalars else None
    with torch.inference_mode():
        expected, expected_s = layer(x, s)
        actual, actual_s = PackedLinear(layer)(x, s)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
    if scalars:
        torch.testing.assert_close(actual_s, expected_s)
    else:
        assert actual_s is None


def attention_graph(path):
    # Shape used by the exporter: shared K/V, Q and K already scaled.
    nodes = [
        h.make_node("Transpose", ["key"], ["kt"], name="transpose", perm=[0, 1, 3, 2]),
        h.make_node("MatMul", ["query", "kt"], ["scores"], name="qk"),
        h.make_node("Softmax", ["scores"], ["probabilities"], name="softmax", axis=-1),
        h.make_node("MatMul", ["probabilities", "value"], ["output"], name="pv"),
    ]
    inputs = [
        h.make_tensor_value_info(name, T.FLOAT, [1, heads, "hits", 8])
        for name, heads in [("query", 2), ("key", 1), ("value", 1)]
    ]
    graph = h.make_graph(
        nodes,
        "shared_attention",
        inputs,
        [h.make_tensor_value_info("output", T.FLOAT, [1, 2, "hits", 8])],
    )
    model = h.make_model(graph, opset_imports=[h.make_opsetid("", 17)], ir_version=10)
    onnx.save(model, path)


@pytest.mark.parametrize("hits", [1, 7, 8, 9, 33])
def test_attention_rewrites_keep_all_keys_and_dynamic_partial_blocks(tmp_path, hits):
    raw, blocked, stable, native = [
        tmp_path / (name + ".onnx") for name in ("raw", "blocked", "stable", "native")
    ]
    attention_graph(raw)
    blocked_attention(raw, blocked, block=8)
    stabilize_attention(blocked, stable)
    runtime_attention(raw, native, heads=2)
    rng = np.random.default_rng(42)
    inputs = {
        name: rng.normal(size=(1, heads, hits, 8)).astype(np.float32) * 0.25
        for name, heads in [("query", 2), ("key", 1), ("value", 1)]
    }
    expected = session(raw).run(None, inputs)[0]
    for path in (blocked, stable, native):
        actual = session(path).run(None, inputs)[0]
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)

    def domains(graph):
        for node in graph.node:
            yield node.domain
            for attr in node.attribute:
                if attr.type == onnx.AttributeProto.GRAPH:
                    yield from domains(attr.g)

    assert set(domains(onnx.load(stable).graph)) <= {"", "ai.onnx"}


def test_stabilization_avoids_subnormal_attention_tails(tmp_path):
    raw, blocked, stable = [tmp_path / (name + ".onnx") for name in ("raw", "blocked", "stable")]
    attention_graph(raw)
    blocked_attention(raw, blocked, block=8)
    stabilize_attention(blocked, stable)
    inputs = {
        "query": np.full((1, 2, 9, 8), 10, np.float32),
        "key": np.full((1, 1, 9, 8), -10, np.float32),
        "value": np.full((1, 1, 9, 8), 1e-30, np.float32),
    }
    inputs["key"][:, :, 0] = 10
    actual = session(stable).run(None, inputs)[0]
    assert np.isfinite(actual).all()
    # The declared guard zeroes V components smaller than 1e-12.
    np.testing.assert_array_equal(actual, 0)
