#!/usr/bin/env python3
"""Build the optional Linux CPU operator against the active PyTorch installation."""

import argparse
import os
from pathlib import Path
import shlex
import subprocess

import torch


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=root / "build/libcirce_cpu.so")
    parser.add_argument("--native", action="store_true", help="target this CPU; rebuild for other hosts")
    args = parser.parse_args()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    torch_root = Path(torch.__file__).resolve().parent
    command = [
        *shlex.split(os.environ.get("CXX", "c++")),
        "-O3", "-DNDEBUG", "-ffp-contract=off", "-fopenmp", "-shared", "-fPIC", "-std=c++17",
        *(["-march=native"] if args.native else []),
        f"-D_GLIBCXX_USE_CXX11_ABI={int(torch._C._GLIBCXX_USE_CXX11_ABI)}",
        f"-I{torch_root / 'include'}", f"-I{torch_root / 'include/torch/csrc/api/include'}",
        str(root / "src/cgatr/primitives/geometric_product_cpu.cpp"),
        f"-L{torch_root / 'lib'}", f"-Wl,-rpath,{torch_root / 'lib'}",
        "-ltorch_cpu", "-lc10", "-o", str(output),
    ]
    subprocess.run(command, check=True)
    print(f"Built {output} for PyTorch {torch.__version__}")


if __name__ == "__main__":
    main()
