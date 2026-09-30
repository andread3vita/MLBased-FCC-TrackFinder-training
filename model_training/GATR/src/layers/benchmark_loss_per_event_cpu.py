"""CPU parity and timing probe for the block-sparse GGTF loss candidate."""

from __future__ import annotations

import argparse
import gc
import json
import os
import statistics
import time
from pathlib import Path

import torch

from src.dataset.parquet_ggtf_adapter import (
    ParquetGGTFDataset,
    collate_ggtf,
)
from src.layers.losses import object_condensation_loss_tracking
from src.layers.losses_per_event import (
    object_condensation_loss_tracking_per_event,
)


LOSS_KWARGS = dict(
    clust_loss_only=True,
    add_energy_loss=False,
    calc_e_frac_loss=False,
    q_min=3.0,
    frac_clustering_loss=0.1,
    attr_weight=1.0,
    repul_weight=1.0,
    fill_loss_weight=0.0,
    use_average_cc_pos=0.0,
    loss_type="hgcalimplementation",
    tracking=True,
)


def select_pack(sizes: list[int], budget: int) -> list[int]:
    selected = []
    tokens = 0
    for index, size in enumerate(sizes):
        if selected and tokens + size > budget:
            break
        selected.append(index)
        tokens += size
    if len(selected) < 2:
        # The optimization removes cross-event work, so exercise at least two
        # events even if the first pair slightly exceeds the requested budget.
        selected = list(range(min(2, len(sizes))))
    return selected


def evaluate(function, graph, truth, base_pred):
    pred = base_pred.detach().clone().requires_grad_(True)
    loss, components = function(graph, pred, truth, **LOSS_KWARGS)
    loss.backward()
    return (
        loss.detach(),
        tuple(component.detach() for component in components),
        pred.grad.detach(),
    )


def timed(function, graph, truth, base_pred, repeats: int) -> list[float]:
    values = []
    for _ in range(repeats):
        gc.collect()
        if base_pred.is_cuda:
            torch.cuda.synchronize(base_pred.device)
        start = time.perf_counter()
        evaluate(function, graph, truth, base_pred)
        if base_pred.is_cuda:
            torch.cuda.synchronize(base_pred.device)
        values.append(time.perf_counter() - start)
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--budget", type=int, default=16000)
    parser.add_argument("--events-to-index", type=int, default=16)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    torch.manual_seed(20260912)
    dataset = ParquetGGTFDataset(
        args.data_dir,
        (1, 1),
        min_num_hits=3,
        garbage_label=True,
        max_events_per_seed=args.events_to_index,
    )
    indices = select_pack(dataset.sizes, args.budget)
    graph, truth = collate_ggtf([dataset[index] for index in indices])
    event_sizes = [int(value) for value in graph.batch_num_nodes().tolist()]
    object_counts = []
    offset = 0
    for size in event_sizes:
        object_counts.append(
            int(graph.ndata["particle_number"][offset : offset + size].max())
        )
        offset += size

    device = torch.device(args.device)
    graph = graph.to(device)
    truth = truth.to(device)
    base_pred = torch.randn(
        graph.num_nodes(), 4, dtype=torch.float32, device=device
    )
    old_loss, old_components, old_grad = evaluate(
        object_condensation_loss_tracking, graph, truth, base_pred
    )
    new_loss, new_components, new_grad = evaluate(
        object_condensation_loss_tracking_per_event, graph, truth, base_pred
    )

    value_abs = float((old_loss - new_loss).abs())
    value_rel = value_abs / max(float(old_loss.abs()), 1e-12)
    component_abs = max(
        float((old - new).abs())
        for old, new in zip(old_components, new_components)
    )
    grad_delta = old_grad - new_grad
    grad_max_abs = float(grad_delta.abs().max())
    grad_rel_l2 = float(grad_delta.norm() / old_grad.norm().clamp(min=1e-12))

    torch.testing.assert_close(new_loss, old_loss, rtol=2e-5, atol=2e-6)
    for new, old in zip(new_components, old_components):
        torch.testing.assert_close(new, old, rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(new_grad, old_grad, rtol=5e-4, atol=2e-6)

    # Warm both paths once before collecting alternating timing samples.
    evaluate(object_condensation_loss_tracking, graph, truth, base_pred)
    evaluate(object_condensation_loss_tracking_per_event, graph, truth, base_pred)
    old_times = timed(
        object_condensation_loss_tracking,
        graph,
        truth,
        base_pred,
        args.repeats,
    )
    new_times = timed(
        object_condensation_loss_tracking_per_event,
        graph,
        truth,
        base_pred,
        args.repeats,
    )
    old_median = statistics.median(old_times)
    new_median = statistics.median(new_times)

    dense_pairs = sum(event_sizes) * sum(object_counts)
    block_pairs = sum(
        hits * objects for hits, objects in zip(event_sizes, object_counts)
    )
    report = {
        "schema_version": 1,
        "device": str(device),
        "torch_threads": args.threads,
        "event_sizes": event_sizes,
        "object_counts": object_counts,
        "total_hits": sum(event_sizes),
        "total_objects": sum(object_counts),
        "global_dense_pairs": dense_pairs,
        "event_block_pairs": block_pairs,
        "pair_reduction": dense_pairs / block_pairs,
        "parity": {
            "old_loss": float(old_loss),
            "new_loss": float(new_loss),
            "value_abs": value_abs,
            "value_rel": value_rel,
            "component_max_abs": component_abs,
            "gradient_max_abs": grad_max_abs,
            "gradient_relative_l2": grad_rel_l2,
        },
        "timing_forward_backward_seconds": {
            "old_samples": old_times,
            "new_samples": new_times,
            "old_median": old_median,
            "new_median": new_median,
            "speedup": old_median / new_median,
        },
        "decision_note": (
            "CPU parity establishes mathematical equivalence. A GPU microbenchmark "
            "is still required before changing queued production training."
        ),
    }
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        tmp = Path(f"{args.output}.tmp.{os.getpid()}")
        tmp.write_text(text)
        os.replace(tmp, args.output)
    print(text, end="")


if __name__ == "__main__":
    main()
