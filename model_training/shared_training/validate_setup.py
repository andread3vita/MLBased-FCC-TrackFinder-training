#!/usr/bin/env python3
"""Validate the common runtime and canonical train/validation file lists."""

import argparse
import glob
import importlib
import sys
from pathlib import Path

import pyarrow.parquet as pq


REQUIRED = {
    "event_number", "file_number", "n_hit", "n_part",
    "hit_x_true", "hit_y_true", "hit_z_true", "hit_time",
    "hit_x", "hit_y", "hit_z", "leftPosition_x", "leftPosition_y",
    "leftPosition_z", "rightPosition_x", "rightPosition_y",
    "rightPosition_z", "drift_distance", "wire_azimuthal_angle",
    "wire_stereo_angle", "hit_type", "hit_particle_index",
    "produced_by_secondary", "overlay", "part_p", "part_p_t", "part_theta",
    "part_phi", "part_m", "part_pid", "part_id", "gen_status", "part_parent",
    "part_vertex_x", "part_vertex_y", "part_vertex_z",
    "shared_schema_version",
}


def expand(patterns):
    paths = []
    for pattern in patterns:
        matches = glob.glob(pattern)
        if matches:
            paths.extend(str(Path(item).resolve()) for item in matches)
        elif Path(pattern).is_file():
            paths.append(str(Path(pattern).resolve()))
        else:
            raise FileNotFoundError(f"No files match {pattern!r}")
    return sorted(dict.fromkeys(paths))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", nargs="+", required=True)
    parser.add_argument("--val", nargs="+", required=True)
    parser.add_argument("--gatr", action="store_true")
    args = parser.parse_args()
    train, val = expand(args.train), expand(args.val)
    overlap = set(train) & set(val)
    if overlap:
        raise ValueError(f"Training and validation lists overlap: {sorted(overlap)[:3]}")
    for path in train + val:
        parquet = pq.ParquetFile(path)
        schema = parquet.schema_arrow.names
        missing = sorted(REQUIRED - set(schema))
        if missing:
            raise ValueError(f"{path} is missing shared columns: {missing}")
        versions = set(
            parquet.read_row_group(0, columns=["shared_schema_version"])[
                "shared_schema_version"
            ].to_pylist()
        )
        if versions != {1}:
            raise ValueError(f"{path} has unsupported shared schema versions: {versions}")
    modules = [
        "torch", "lightning", "torch_scatter", "awkward", "pyarrow", "scipy",
    ]
    if args.gatr:
        modules += ["dgl", "xformers"]
    versions = {}
    for name in modules:
        module = importlib.import_module(name)
        versions[name] = getattr(module, "__version__", "unknown")
    print(f"Python executable: {sys.executable}")
    print(f"Train files: {len(train)}; validation files: {len(val)}; overlap: 0")
    print("Runtime: " + ", ".join(f"{k}={v}" for k, v in versions.items()))


if __name__ == "__main__":
    main()
