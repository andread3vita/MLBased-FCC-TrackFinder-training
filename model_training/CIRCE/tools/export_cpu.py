#!/usr/bin/env python3
"""Convert existing checkpoints; new training can save the CPU graph automatically."""

import argparse
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.cgatr.cpu_inference import SingleEvent, export_cpu_checkpoint
from src.dataset.parquet_dataset import IDEAParquetDataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--data-dir", type=Path, required=True, help="evaluation parquet root")
    parser.add_argument("--seed", type=int, default=981)
    parser.add_argument("--event", type=int, default=0, help="dataset index within the seed")
    parser.add_argument("--library", type=Path, default=ROOT / "build/libcirce_cpu.so")
    parser.add_argument("--output", type=Path, default=Path("circe_cpu.pt"))
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    dataset = IDEAParquetDataset(str(args.data_dir), seed_list=[args.seed], max_hits_per_event=None)
    features = dataset[args.event]["features"].cpu().float().contiguous()
    weights_source = export_cpu_checkpoint(args.checkpoint, features, args.library, args.output)
    example_path = args.output.with_suffix(".example.f32")
    features.numpy().tofile(example_path)
    print(f"Exported {weights_source} model: {args.output}")
    print(f"Example event: {example_path}; output ends in the raw beta logit")


if __name__ == "__main__":
    main()
