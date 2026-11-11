#!/usr/bin/env python3
"""Export existing sphere/circle CIRCE checkpoints for optional FP32 ONNX CPU inference."""

import argparse
from collections import Counter
import hashlib
import itertools
import json
from pathlib import Path
import sys
import tempfile

import numpy as np
import onnx
from onnx import helper as h, numpy_helper as nh, TensorProto as T
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.cgatr.layers.layer_norm import EquiLayerNorm
from src.cgatr.layers.linear import EquiLinear
from src.cgatr.layers.attention.self_attention import SelfAttention
from src.cgatr.primitives.bilinear import geometric_product
from src.cgatr.primitives.normalization import _GRADE_RANGES_BY_BLADES
from src.cgatr.interface.point import embed_point
from src.cgatr.interface.circle import embed_circle_ipns
from src.cgatr.interface.sphere import embed_sphere


def matrix_representation(gp):
    """Construct and exhaustively verify a real-packed Mat(4,C) representation."""
    table = gp.detach().cpu().to_dense().double().numpy()
    eye = np.eye(2, dtype=np.complex128)
    px = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    py = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
    pz = np.diag([1.0, -1.0]).astype(np.complex128)
    gamma = [
        np.kron(px, eye),
        np.kron(py, eye),
        np.kron(pz, px),
        np.kron(pz, py),
        1j * np.kron(pz, pz),
    ]
    basis = np.zeros((32, 4, 4), dtype=np.complex128)
    used = set()
    for grade in range(6):
        for indices in itertools.combinations(range(5), grade):
            value = np.eye(32)[0]
            matrix = np.eye(4, dtype=np.complex128)
            for index in indices:
                value = np.einsum("ijk,j,k->i", table, value, np.eye(32)[index + 1])
                matrix = matrix @ gamma[index]
            positions = np.flatnonzero(value)
            assert len(positions) == 1 and abs(value[positions[0]]) == 1
            blade = int(positions[0])
            assert blade not in used
            used.add(blade)
            basis[blade] = matrix / value[blade]
    assert len(used) == 32
    for j in range(32):
        for k in range(32):
            np.testing.assert_array_equal(
                basis[j] @ basis[k], np.einsum("i,ixy->xy", table[:, j, k], basis)
            )
    packed = np.concatenate([basis.real, basis.imag], axis=-1).reshape(32, 32)
    np.testing.assert_array_equal(packed @ packed.T, 4 * np.eye(32))
    block = np.concatenate(
        [
            np.concatenate([basis.real, basis.imag], -1),
            np.concatenate([-basis.imag, basis.real], -1),
        ],
        -2,
    ).reshape(32, 64)
    return torch.tensor(packed, dtype=torch.float32), torch.tensor(block, dtype=torch.float32)


class MatrixGP(nn.Module):
    """Faithful Cl(4,1) product, represented using ordinary real MatMul nodes."""

    def __init__(self, gp):
        super().__init__()
        left, right = matrix_representation(gp)
        self.register_buffer("left_matrix", left)
        self.register_buffer("right_matrix", right)
        self.register_buffer("out_matrix", left.T / 4)

    def forward(self, x, y):
        left, right = x @ self.left_matrix, y @ self.right_matrix
        product = (left.reshape(*x.shape[:-1], 4, 8) @ right.reshape(*y.shape[:-1], 8, 8)).flatten(
            -2
        )
        return product @ self.out_matrix


class PackedLinear(nn.Module):
    """Fold the fixed equivariant basis into its trained weights before export."""

    def __init__(self, source):
        super().__init__()
        self.out_channels = source.weight.shape[0]
        self.blades = source.basis.shape[-1]
        weight = torch.einsum("oib,bxy->oxiy", source.weight.detach(), source.basis.detach())
        self.register_buffer(
            "matrix", weight.reshape(self.out_channels * self.blades, -1).contiguous()
        )
        self.register_buffer(
            "bias", source.bias.detach().clone() if source.bias is not None else None
        )
        self.s2mvs = source.s2mvs
        self.mvs2s = source.mvs2s
        self.s2s = source.s2s

    def forward(self, multivectors, scalars=None):
        output = F.linear(multivectors.flatten(-2), self.matrix)
        output = output.reshape(*multivectors.shape[:-2], self.out_channels, self.blades)
        scalar_blade = output[..., :1]
        if self.bias is not None:
            scalar_blade = scalar_blade + self.bias
        if self.s2mvs is not None and scalars is not None:
            scalar_blade = scalar_blade + self.s2mvs(scalars).unsqueeze(-1)
        output = torch.cat([scalar_blade, output[..., 1:]], -1)
        s = self.mvs2s(multivectors[..., 0]) if self.mvs2s is not None else None
        if s is not None and self.s2s is not None and scalars is not None:
            s = s + self.s2s(scalars)
        return output, s


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
        squared = (
            squared.clamp_min(self.epsilon)
            if self.epsilon_mode == "clamp"
            else squared + self.epsilon
        )
        return (
            multivectors / squared.sqrt(),
            torch.nn.functional.layer_norm(scalars, [self.scalar_channels]),
        )


class GeometryEvent(nn.Module):
    """Measured [N,10] features to invariant coordinates and raw beta logit.

    Where replaces indexed mutation so hit count and detector mix remain dynamic.
    The supported checkpoint encoding is the tested 32-blade sphere/circle arm.
    """

    def __init__(self, model):
        super().__init__()
        if (
            model.projective
            or model.num_blades != 32
            or model.use_time
            or model.cga_encoding != "sphere_circle"
            or not model.invariant_output_head
        ):
            raise ValueError(
                "ONNX export requires 32-blade CGA sphere_circle, invariant output and 10 features"
            )
        self.model = model

    def forward(self, features):
        m = self.model
        hit_type = features[:, 3:4]
        is_dc, is_vtx = hit_type == 1, hit_type == 0
        position, wire = features[:, :3] / m.pos_scale, features[:, 4:7] / m.pos_scale
        drift = features[:, 7] / 5.0
        radius = features[:, 7] / m.pos_scale if m.physical_drift_geometry else drift
        cs, ss = torch.cos(features[:, 9]), torch.sin(features[:, 9])
        ca, sa = torch.cos(features[:, 8]), torch.sin(features[:, 8])
        direction = torch.stack(
            [ss * sa, -ss * ca, cs] if m.fix_wire_dir else [ss * ca, ss * sa, cs], -1
        )
        direction = direction / (direction.norm(dim=-1, keepdim=True) + 1e-8)
        vertex = embed_point(position)
        first = torch.where(
            is_dc,
            embed_sphere(wire, radius, fix_null=m.fix_cga_null),
            torch.where(is_vtx, vertex, torch.zeros_like(vertex)),
        )
        circle = embed_circle_ipns(wire, direction, radius, m.basis_outer, fix_null=m.fix_cga_null)
        mv = torch.stack([first, torch.where(is_dc, circle, torch.zeros_like(circle))], 1)
        if not m.separate_hit_type:
            first = torch.cat([mv[:, 0, :1] + hit_type, mv[:, 0, 1:]], -1)
            mv = torch.stack([first, mv[:, 1]], 1)
        if m.args.normalize_mv_inputs:
            mv = mv / mv.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        parts = []
        if m.separate_hit_metadata:
            parts.append(torch.where(is_dc, drift.unsqueeze(-1), torch.zeros_like(hit_type)))
        if m.separate_hit_type:
            parts.append(hit_type)
        scalars = torch.cat(parts, -1) if parts else None
        _, prediction = m.cgatr(mv, scalars=scalars, attention_mask=None)
        return prediction


def blocked_attention(source, output, block):
    """Bound score scratch memory while each query still attends to every key."""
    model = onnx.load(source)
    nodes = list(model.graph.node)
    producers = {out: node for node in nodes for out in node.output}
    replacements = {}
    removed = set()
    count = 0
    for softmax in nodes:
        if softmax.op_type != "Softmax":
            continue
        before = producers[softmax.input[0]]
        consumers = [n for n in nodes if softmax.output[0] in n.input]
        if before.op_type != "MatMul" or len(consumers) != 1 or consumers[0].op_type != "MatMul":
            raise ValueError("Unexpected attention pattern")
        after = consumers[0]
        assert after.input[0] == softmax.output[0]
        q, kt, v = before.input[0], before.input[1], after.input[1]
        prefix = f"bounded_attention_{count}/"
        new = []

        def const(name, value, dtype=np.int64):
            name = prefix + name
            model.graph.initializer.append(nh.from_array(np.array(value, dtype=dtype), name))
            return name

        def op(kind, inputs, name, **attrs):
            out = prefix + name
            new.append(h.make_node(kind, inputs, [out], name=out, **attrs))
            return out

        axes01 = const("axes01", [0, 1])
        axes0 = const("axes0", [0])
        zero = const("zero", 0)
        one = const("one", 1)
        tile = const("tile", block)
        tile_minus_one = const("tile_minus_one", block - 1)
        cond = const("condition", True, np.bool_)
        steps = const("steps", [1])
        starts = const("starts", [0])
        qshape = op("Shape", [q], "query_shape")
        flat = op("Flatten", [q], "flat_query", axis=3)
        # Single-event multi-query attention: K,V have singleton batch/head axes.
        keys = op("Squeeze", [kt, axes01], "keys_2d")
        values = op("Squeeze", [v, axes01], "values_2d")
        flatshape = op("Shape", [flat], "flat_shape")
        length = op("Gather", [flatshape, zero], "rows", axis=0)
        length_plus = op("Add", [length, tile_minus_one], "length_plus")
        trips = op("Div", [length_plus, tile], "trips")
        padded = op("Mul", [trips, tile], "padded_rows")
        padding = op("Sub", [padded, length], "padding")
        padding1 = op("Unsqueeze", [padding, axes0], "padding_vector")
        zeros2 = const("zeros2", [0, 0])
        zero1 = const("zero1", [0])
        pads = op("Concat", [zeros2, padding1, zero1], "pads", axis=0)
        padded_q = op("Pad", [flat, pads], "padded_query", mode="constant")
        iteration = prefix + "iteration"
        body_cond = prefix + "body_condition"
        start = prefix + "body_start"
        end = prefix + "body_end"
        start1 = prefix + "body_start_vector"
        end1 = prefix + "body_end_vector"
        query = prefix + "body_query"
        scores = prefix + "body_scores"
        probs = prefix + "body_probabilities"
        result = prefix + "body_output"
        body = h.make_graph(
            [
                h.make_node("Mul", [iteration, tile], [start]),
                h.make_node("Add", [start, tile], [end]),
                h.make_node("Unsqueeze", [start, axes0], [start1]),
                h.make_node("Unsqueeze", [end, axes0], [end1]),
                h.make_node("Slice", [padded_q, start1, end1, axes0, steps], [query]),
                h.make_node("MatMul", [query, keys], [scores]),
                h.make_node("Softmax", [scores], [probs], axis=-1),
                h.make_node("MatMul", [probs, values], [result]),
            ],
            prefix + "body",
            [
                h.make_tensor_value_info(iteration, T.INT64, []),
                h.make_tensor_value_info(body_cond, T.BOOL, []),
            ],
            [
                h.make_tensor_value_info(body_cond, T.BOOL, []),
                h.make_tensor_value_info(result, T.FLOAT, [block, None]),
            ],
        )
        scan = op("Loop", [trips, cond], "scan", body=body)
        vshape = op("Shape", [values], "value_shape")
        width = op("Gather", [vshape, one], "value_width", axis=0)
        width1 = op("Unsqueeze", [width, axes0], "value_width_vector")
        minus1 = const("minus1", [-1])
        scan_shape = op("Concat", [minus1, width1], "scan_shape", axis=0)
        flat_output = op("Reshape", [scan, scan_shape], "flat_output")
        length1 = op("Unsqueeze", [length, axes0], "length_vector")
        trimmed = op("Slice", [flat_output, starts, length1, axes0, steps], "trimmed")
        # Query/value widths are equal in this exported model's padded SDPA.
        new.append(
            h.make_node("Reshape", [trimmed, qshape], list(after.output), name=prefix + "output")
        )
        replacements[after.name] = new
        removed.update([before.name, softmax.name, after.name])
        count += 1
    assert count > 0
    final = []
    for node in nodes:
        if node.name in replacements:
            final.extend(replacements[node.name])
        elif node.name not in removed:
            final.append(node)
    del model.graph.node[:]
    model.graph.node.extend(final)
    # Drop stale internal shapes from the replaced dense attention.
    del model.graph.value_info[:]
    assert all(n.domain in ("", "ai.onnx") for n in model.graph.node)
    onnx.checker.check_model(model)
    onnx.save(model, output)
    return count


def runtime_attention(source, output, heads):
    """Use ORT built-in attention, keeping the exporter's existing Q/K scaling."""
    model = onnx.load(source)
    nodes = list(model.graph.node)
    prod = {o: n for n in nodes for o in n.output}
    removed = set()
    replacements = {}
    count = 0
    for softmax in nodes:
        if softmax.op_type != "Softmax":
            continue
        before = prod[softmax.input[0]]
        consumers = [n for n in nodes if softmax.output[0] in n.input]
        if before.op_type != "MatMul" or len(consumers) != 1 or consumers[0].op_type != "MatMul":
            raise ValueError("Unexpected attention pattern")
        after = consumers[0]
        if after.input[0] != softmax.output[0]:
            raise ValueError("Expected attention probabilities as the left MatMul input")
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
        k = op("Transpose", [kt], "key_untransposed", perm=[0, 1, 3, 2])
        k = op("Expand", [k, shape], "key_heads")
        v = op("Expand", [v, shape], "value_heads")
        flatshape = const("flat_shape", [0, 0, -1])
        q3 = op(
            "Reshape",
            [op("Transpose", [q], "query_transpose", perm=[0, 2, 1, 3]), flatshape],
            "query_3d",
        )
        k3 = op(
            "Reshape",
            [op("Transpose", [k], "key_transpose", perm=[0, 2, 1, 3]), flatshape],
            "key_3d",
        )
        v3 = op(
            "Reshape",
            [op("Transpose", [v], "value_transpose", perm=[0, 2, 1, 3]), flatshape],
            "value_3d",
        )
        # The exporter has already scaled Q and K by d^-1/4 each.
        attention = op(
            "MultiHeadAttention",
            [q3, k3, v3],
            "attention",
            domain="com.microsoft",
            num_heads=heads,
            scale=1.0,
            unidirectional=0,
        )
        head_shape = op(
            "Gather", [shape, const("shape_permutation", [0, 2, 1, 3])], "output_shape", axis=0
        )
        out4 = op("Reshape", [attention, head_shape], "output_4d")
        new.append(
            h.make_node(
                "Transpose", [out4], list(after.output), name=prefix + "output", perm=[0, 2, 1, 3]
            )
        )
        replacements[after.name] = new
        removed.update([before.name, softmax.name, after.name])
        count += 1
    final = []
    for n in nodes:
        if n.name in replacements:
            final.extend(replacements[n.name])
        elif n.name not in removed:
            final.append(n)
    assert count > 0
    del model.graph.node[:]
    model.graph.node.extend(final)
    del model.graph.value_info[:]
    model.opset_import.append(h.make_opsetid("com.microsoft", 1))
    onnx.checker.check_model(model)
    onnx.save(model, output)
    return count


def stabilize_attention(source, output):
    """Numerical change: suppress tiny values and floor relative logits at -30."""
    model = onnx.load(source)
    threshold = "stable_attention_threshold"
    zero = "stable_attention_zero"
    negative = "stable_attention_negative_range"
    for value in model.graph.initializer:
        x = nh.to_array(value)
        if x.dtype != np.float32:
            continue
        selected = (x != 0) & (np.abs(x) < 1e-12)
        if selected.any():
            y = x.copy()
            y[selected] = 0
            value.CopyFrom(nh.from_array(y, value.name))
    for name, value in [(threshold, 1e-12), (zero, 0), (negative, -30.0)]:
        model.graph.initializer.append(nh.from_array(np.array(value, dtype=np.float32), name))
    counts = {"qkv_guards": 0, "softmax": 0}

    def rewrite(graph):
        final = []
        for n in graph.node:
            for attr in n.attribute:
                if attr.type == onnx.AttributeProto.GRAPH:
                    rewrite(attr.g)
            if n.op_type == "Softmax":
                scores = n.input[0]
                maxval = scores + "/row_max"
                lower = scores + "/lower"
                clamped = scores + "/clamped"
                final.extend(
                    [
                        h.make_node("ReduceMax", [scores], [maxval], axes=[-1], keepdims=1),
                        h.make_node("Add", [maxval, negative], [lower]),
                        h.make_node("Max", [scores, lower], [clamped]),
                    ]
                )
                n.input[0] = clamped
                counts["softmax"] += 1
            final.append(n)
            if n.output[0].endswith(("/flat_query", "/keys_2d", "/values_2d")):
                output = n.output[0]
                raw = output + "/uncleaned"
                absolute = output + "/absolute"
                small = output + "/small"
                n.output[0] = raw
                final.extend(
                    [
                        h.make_node("Abs", [raw], [absolute]),
                        h.make_node("Less", [absolute, threshold], [small]),
                        h.make_node("Where", [small, zero, raw], [output]),
                    ]
                )
                counts["qkv_guards"] += 1
        del graph.node[:]
        graph.node.extend(final)

    rewrite(model.graph)
    assert counts["softmax"] > 0 and counts["qkv_guards"] == 3 * counts["softmax"]
    onnx.checker.check_model(model)
    onnx.save(model, output)

    return counts


def prepare(model):
    """Rewrite only the disposable export model, after its weights are loaded."""
    products = [m for m in model.modules() if isinstance(m, geometric_product)]
    if not products or any(tuple(m.gp.shape) != (32, 32, 32) for m in products):
        raise ValueError("Only Cl(4,1) checkpoints are supported")
    product = MatrixGP(products[0].gp)
    counts = Counter()

    def replace(parent):
        for name, child in list(parent.named_children()):
            if hasattr(child, "_use_fused"):
                child._use_fused = False
            if isinstance(child, geometric_product):
                setattr(parent, name, product)
                counts["geometric_product"] += 1
            elif isinstance(child, EquiLinear):
                setattr(parent, name, PackedLinear(child))
                counts["linear"] += 1
            elif isinstance(child, EquiLayerNorm):
                setattr(parent, name, _TraceableLayerNorm(child))
                counts["normalization"] += 1
            else:
                replace(child)

    replace(model)
    return counts


def export(checkpoint, features, output, runtime_tuned=False):
    """Export a trusted Lightning checkpoint; prefer EMA weights when present.

    Runtime tuning needs ONNX Runtime's built-in MultiHeadAttention operator and
    session.set_denormal_as_zero=1. Standard mode instead uses a numerical guard
    in the graph. Validate reconstruction for each new checkpoint in either mode.
    """
    from src.model import CGATrParquetModel
    from src.eval.forward_pass import _load_state_dict_tolerating_derived_buffers

    checkpoint, output = Path(checkpoint).resolve(), Path(output).resolve()
    if checkpoint == output:
        raise ValueError("Output must differ from the checkpoint")
    features = np.asarray(features)
    if (
        features.dtype != np.float32
        or features.ndim != 2
        or features.shape[1] != 10
        or not len(features)
        or not np.isfinite(features).all()
        or not np.isin(features[:, 3], [0, 1]).all()
    ):
        raise ValueError("Expected nonempty finite float32 [hits,10] features, hit_type 0 or 1")
    output.parent.mkdir(parents=True, exist_ok=True)
    with torch.random.fork_rng(devices=[]), torch.inference_mode():
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        config = dict(saved["hyper_parameters"], grad_checkpoint=False, equi_init="default")
        model = CGATrParquetModel(argparse.Namespace(**config)).cpu().float().eval()
        state = saved.get("ema_state_dict")
        weights = "ema_state_dict" if state else "state_dict"
        if not state:
            state = {k[6:]: v for k, v in saved["state_dict"].items() if k.startswith("model.")}
        _load_state_dict_tolerating_derived_buffers(model, state)
        model.requires_grad_(False)
        runner = GeometryEvent(model).eval()  # Reject unsupported layouts before rewriting.
        attention = [m for m in model.modules() if isinstance(m, SelfAttention)]
        heads = {m.config.num_heads for m in attention}
        if len(heads) != 1 or any(not m.config.multi_query for m in attention):
            raise ValueError("Expected consistent multi-query attention heads")
        parameters = sum(p.numel() for p in model.parameters())
        replacements = prepare(model)
        # A short trace is sufficient: hit count and detector mix stay dynamic.
        example = torch.from_numpy(
            np.concatenate([features[features[:, 3] == 0][:4], features[features[:, 3] == 1][:4]])
        )
        with tempfile.TemporaryDirectory(dir=output.parent) as directory:
            raw, transformed = Path(directory) / "raw.onnx", Path(directory) / "model.onnx"
            torch.onnx.export(
                runner,
                (example,),
                str(raw),
                dynamo=False,
                opset_version=17,
                input_names=["features"],
                output_names=["prediction"],
                dynamic_axes={"features": {0: "hits"}, "prediction": {0: "hits"}},
                do_constant_folding=True,
            )
            if runtime_tuned:
                count = runtime_attention(raw, transformed, heads=heads.pop())
            else:
                blocked = Path(directory) / "blocked.onnx"
                count = blocked_attention(raw, blocked, block=256)
                stabilize_attention(blocked, transformed)
            if count != len(attention):
                raise ValueError("Unexpected number of exported attention blocks")
            graph = onnx.load(transformed)
            onnx.checker.check_model(graph)
            record = dict(
                variant="runtime-tuned" if runtime_tuned else "standard",
                weights=weights,
                parameters=parameters,
                architecture={
                    k: config[k] for k in ("num_blocks", "hidden_mv_channels", "hidden_s_channels")
                },
                checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                replacements=dict(replacements),
                opset=17,
                input="float32 [hits,10]: x,y,z,hit_type,wire_x,wire_y,wire_z,drift,alpha,stereo",
                output=f"float32 [hits,{model.embed_dim + 1}]: coordinates,raw_beta_logit",
                required_session_setting=(
                    "session.set_denormal_as_zero=1" if runtime_tuned else None
                ),
                numerical_change=(
                    "flush subnormals"
                    if runtime_tuned
                    else "zero constants and Q/K/V below 1e-12; floor logits at row_max-30"
                ),
                reconstruction_validation="Required for each new checkpoint",
            )
            h.set_model_props(graph, {"circe.export": json.dumps(record)})
            onnx.save(graph, transformed)
            transformed.replace(output)
    record["model_sha256"] = hashlib.sha256(output.read_bytes()).hexdigest()
    record["file_bytes"] = output.stat().st_size
    output.with_suffix(".json").write_text(json.dumps(record, indent=2) + "\n")
    np.save(output.with_suffix(".example.npy"), features)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path, help="Trusted CIRCE Lightning checkpoint")
    parser.add_argument("--output", type=Path, required=True)
    sample = parser.add_mutually_exclusive_group(required=True)
    sample.add_argument(
        "--example-features", type=Path, help="One measured float32 [hits,10] .npy event"
    )
    sample.add_argument(
        "--data-dir", type=Path, help="IDEA parquet directory; also save one example event"
    )
    parser.add_argument("--seed", type=int, default=981)
    parser.add_argument("--event", type=int, default=0, help="Index within the selected seed")
    parser.add_argument(
        "--runtime-tuned", action="store_true", help="Use ONNX Runtime built-in attention + FTZ"
    )
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    if args.example_features:
        features = np.load(args.example_features, allow_pickle=False)
    else:
        from src.dataset.parquet_dataset import IDEAParquetDataset

        dataset = IDEAParquetDataset(
            str(args.data_dir), seed_list=[args.seed], max_hits_per_event=None
        )
        event = dataset[args.event]
        if event is None:
            parser.error("Selected event is empty")
        features = event["features"].numpy()
    print(json.dumps(export(args.checkpoint, features, args.output, args.runtime_tuned), indent=2))


if __name__ == "__main__":
    main()
