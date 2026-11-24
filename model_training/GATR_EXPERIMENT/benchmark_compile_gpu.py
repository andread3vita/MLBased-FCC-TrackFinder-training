#!/usr/bin/env python3
"""Compare eager and torch.compile'd GATr on one GPU with synthetic batches.

Builds the GATr network exactly as the tracking model does (same widths, the
xFormers block-diagonal mask, per-event join reference), then for each batch
of random events runs the eager and the compiled copy of the same weights and
reports agreement and time per batch for

* training steps (forward + backward), and
* validation inference (forward under no_grad).

No data, Lightning or DGL needed. Usage (GPU node, CUDA 12.1 module loaded):

    python benchmark_compile_gpu.py [--blocks 10] [--batches 12] [--events 8]
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
import sys
import time

import torch

ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT / "src" / "gatr_v142"), str(ROOT)]

from xformers.ops.fmha import BlockDiagonalMask  # noqa: E402

from gatr.interface import embed_point, embed_scalar, embed_translation  # noqa: E402
from gatr.layers.attention.config import SelfAttentionConfig  # noqa: E402
from gatr.layers.mlp.config import MLPConfig  # noqa: E402
from gatr.nets.gatr import GATr  # noqa: E402
from src.models.compiled_gatr import COMPILE_MODES, compile_gatr, pad_inputs, set_attention_mask, warm_up  # noqa: E402


def make_batch(generator, events, min_hits, max_hits, device):
    lengths = torch.randint(min_hits, max_hits + 1, (events,), generator=generator).tolist()
    n = sum(lengths)
    positions = torch.randn(n, 3, generator=generator) * 0.5
    vector = torch.randn(n, 3, generator=generator) * 0.01
    hit_type = torch.randint(0, 2, (n, 1), generator=generator).float()
    mv = (embed_point(positions) + embed_scalar(hit_type) + embed_translation(vector)).unsqueeze(-2)
    mv = mv.to(device)
    references, start = [], 0
    for length in lengths:  # per-event mean, as ExampleWrapper._join_reference
        mean = mv[start : start + length].mean(dim=0, keepdim=True)
        references.append(mean.expand(length, *mean.shape[1:]))
        start += length
    return mv, torch.cat(references), lengths


def run(network, compiled, mv, reference, lengths, train):
    n = mv.shape[0]
    inputs = mv.detach().requires_grad_(train)
    with torch.set_grad_enabled(train):
        hits, mask = inputs, BlockDiagonalMask.from_seqlens(lengths)
        if compiled:  # as in the tracking model: pad to a bucket, mask outside the graph
            hits, _, reference, padded_lengths = pad_inputs(inputs, None, reference, lengths)
            warm_up(network, hits, None, reference)
            set_attention_mask(BlockDiagonalMask.from_seqlens(padded_lengths))
            mask = None
        out, _ = network(hits, scalars=None, attention_mask=mask, join_reference=reference)
        out = out[:n]
        if train:
            out.square().mean().backward()
    return out.detach(), (inputs.grad if train else None)


def timed(fn):
    torch.cuda.synchronize()
    start = time.perf_counter()
    result = fn()
    torch.cuda.synchronize()
    return result, time.perf_counter() - start


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--blocks", type=int, default=10)
    parser.add_argument("--hidden-mv", type=int, default=16)
    parser.add_argument("--hidden-s", type=int, default=64)
    parser.add_argument("--events", type=int, default=8, help="events per batch")
    parser.add_argument("--min-hits", type=int, default=500)
    parser.add_argument("--max-hits", type=int, default=5000)
    parser.add_argument("--batches", type=int, default=12, help="timed batches (all different sizes)")
    parser.add_argument("--warmup-batches", type=int, default=3,
                        help="untimed batches (compilation; each new padding bucket also compiles once)")
    parser.add_argument("--compile-mode", choices=COMPILE_MODES, default="default")
    parser.add_argument("--checkpoint", action="store_true", help="gradient checkpointing per block")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        sys.exit("No GPU visible: run this on a GPU node (see benchmark_compile_gpu.sh)")
    device = torch.device("cuda")
    print(f"torch {torch.__version__} (CUDA {torch.version.cuda}), GPU {torch.cuda.get_device_name()}", flush=True)

    torch.manual_seed(0)
    eager = GATr(
        in_mv_channels=1, out_mv_channels=1, hidden_mv_channels=args.hidden_mv,
        in_s_channels=None, out_s_channels=None, hidden_s_channels=args.hidden_s,
        num_blocks=args.blocks, attention=SelfAttentionConfig(), mlp=MLPConfig(),
        checkpoint=["block"] if args.checkpoint else None,
    ).to(device)
    compiled = compile_gatr(copy.deepcopy(eager), mode=args.compile_mode)

    generator = torch.Generator().manual_seed(1)
    batches = [make_batch(generator, args.events, args.min_hits, args.max_hits, device)
               for _ in range(args.warmup_batches + args.batches)]

    for train in (True, False):
        label = "training step (fwd+bwd)" if train else "validation (fwd, no_grad)"
        times = {False: [], True: []}
        worst = 0.0
        for index, (mv, reference, lengths) in enumerate(batches):
            (out_e, grad_e), t_e = timed(lambda: run(eager, False, mv, reference, lengths, train))
            (out_c, grad_c), t_c = timed(lambda: run(compiled, True, mv, reference, lengths, train))
            diff = ((out_e - out_c).abs().max() / out_e.abs().max()).item()
            if train:
                diff = max(diff, ((grad_e - grad_c).abs().max() / grad_e.abs().max()).item())
            if index < args.warmup_batches:
                print(f"  [{label}] warm-up batch {index + 1}: {sum(lengths)} hits, "
                      f"eager {t_e:.3f} s, compiled {t_c:.3f} s (includes compilation)", flush=True)
                continue
            worst = max(worst, diff)
            times[False].append(t_e)
            times[True].append(t_c)
        eager_mean = sum(times[False]) / len(times[False])
        compiled_mean = sum(times[True]) / len(times[True])
        print(f"{label}: eager {eager_mean:.3f} s/batch, compiled {compiled_mean:.3f} s/batch, "
              f"speed-up x{eager_mean / compiled_mean:.2f}, max rel diff {worst:.1e} "
              f"({args.batches} batches of {args.events} events, {args.min_hits}-{args.max_hits} hits each)",
              flush=True)
    stats = torch._dynamo.utils.counters["stats"]
    print(f"compiled graphs: {stats['unique_graphs']} (one set per padding bucket and grad mode)")


if __name__ == "__main__":
    main()
