"""Schema probe for offline debugging of dataset folders.

    python -m ssm.inspect_schema <folder> [--n 3]

Prints per parquet file: rows, columns with dtypes, layout detection, total hit
count. Run this on the GPU machine when EventParquetDataset fails.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl


def main():
    p = argparse.ArgumentParser()
    p.add_argument("folder")
    p.add_argument("--n", type=int, default=3, help="max files to print in detail")
    args = p.parse_args()
    folder = Path(args.folder)
    files = sorted(folder.rglob("*.parquet"))
    print(f"{len(files)} parquet files under {folder}")
    total_rows = 0
    for f in files:
        sch = pl.read_parquet_schema(f)
        n = pl.scan_parquet(f).select(pl.len()).collect().item()
        total_rows += n
        print(f"\n=== {f}\n  rows: {n}")
        if args.n > 0:
            for name, dt in sch.items():
                print(f"  {name:30s} {dt}")
            args.n -= 1
        else:
            print(f"  columns: {sorted(sch)}")
    print(f"\ntotal rows: {total_rows:,}")
    f0 = files[0]
    sch0 = pl.read_parquet_schema(f0)
    if "event_id" in sch0 and "hit_x" in sch0:
        layout = "flat_hits (rows = hits, grouped by event_id)"
    elif "hit_x" in sch0 and getattr(sch0["hit_x"], "inner", None) is not None:
        layout = "event_table (rows = events, list columns)"
    else:
        layout = "UNKNOWN"
    print(f"detected layout for {f0.name}: {layout}")


if __name__ == "__main__":
    main()
