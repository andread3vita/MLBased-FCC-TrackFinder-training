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
    parser.add_argument("--scalar-only", action="store_true", help="omit the optional AVX-512 kernel")
    parser.add_argument("--runner", action="store_true", help="also build the standalone circe_cpu executable")
    args = parser.parse_args()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    torch_root = Path(torch.__file__).resolve().parent
    common = [
        *shlex.split(os.environ.get("CXX", "c++")),
        "-O3", "-DNDEBUG", "-ffp-contract=off", "-fopenmp", "-std=c++17",
        *(["-march=native"] if args.native else []),
        f"-D_GLIBCXX_USE_CXX11_ABI={int(torch._C._GLIBCXX_USE_CXX11_ABI)}",
        f"-I{torch_root / 'include'}", f"-I{torch_root / 'include/torch/csrc/api/include'}",
    ]
    libraries = [
        f"-L{torch_root / 'lib'}", f"-Wl,-rpath,{torch_root / 'lib'}",
        "-ltorch_cpu", "-lc10",
    ]
    command = [
        *common, "-shared", "-fPIC",
        *(["-DCIRCE_DISABLE_AVX512"] if args.scalar_only else []),
        str(root / "src/cgatr/primitives/geometric_product_cpu.cpp"),
        *libraries, "-o", str(output),
    ]
    subprocess.run(command, check=True)
    print(f"Built {output} for PyTorch {torch.__version__}")
    if args.runner:
        executable = output.parent / "circe_cpu"
        subprocess.run([
            *common, str(root / "tools/cpu_infer.cpp"),
            f"-L{output.parent}", "-Wl,--no-as-needed", f"-l:{output.name}",
            "-Wl,--as-needed", "-Wl,-rpath,$ORIGIN", *libraries,
            "-o", str(executable),
        ], check=True)
        print(f"Built {executable}; no LD_PRELOAD or Python needed at inference time")


if __name__ == "__main__":
    main()
