#!/usr/bin/env python3
"""Run the IDEA no-background Parquet production locally, without HTCondor.

Loops over the requested seed range and runs src/runSequence.sh once per seed
on this machine. Seeds whose exact graph/Graphs_<seed>_<type>.parquet output
already exists are skipped, so an interrupted batch can simply be rerun with
the same arguments. A failed seed never stops the others; rerun the same
command to retry only the missing outputs.
"""

import argparse
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Local (non-Condor) IDEA no-background Parquet production."
    )
    parser.add_argument("--outdir", required=True, help="output directory")
    parser.add_argument("--minseed", type=int, default=1, help="minimum seed (inclusive)")
    parser.add_argument("--maxseed", type=int, default=1, help="maximum seed (inclusive)")
    parser.add_argument("--type", choices=("train", "test"), default="train")
    parser.add_argument(
        "--key4hep_version",
        help="Key4hep release: date (e.g. 2026-10-20) for --key4hep_mode spack, "
        "weekday (Mon..Sat) or 'latest' for --key4hep_mode lcg",
        default="2026-10-20",
    )
    parser.add_argument(
        "--key4hep_mode",
        choices=("spack", "lcg"),
        default="spack",
        help="Key4hep stack: frozen spack nightlies (spack) or the rolling LCG view (lcg)",
    )
    parser.add_argument(
        "--nprocs",
        type=int,
        default=1,
        help="number of seeds processed in parallel (default: 1)",
    )
    parser.add_argument("--mainDir", help="pipeline directory", default="")
    return parser.parse_args()


def find_work_dir(start_dir):
    work_dir = start_dir.resolve()
    while not (work_dir / "data_creation").is_dir() and work_dir != work_dir.parent:
        work_dir = work_dir.parent
    if not (work_dir / "data_creation").is_dir():
        raise RuntimeError("Could not find WORK_DIR containing data_creation")
    return work_dir


def run_seed(script, outdir, train_or_test, seed, work_dir, key4hep_version, key4hep_mode, std_dir):
    out = std_dir / f"local_{seed}.out"
    err = std_dir / f"local_{seed}.err"
    with open(out, "w") as stdout, open(err, "w") as stderr:
        result = subprocess.run(
            [
                "bash",
                str(script),
                str(outdir),
                train_or_test,
                str(seed),
                str(work_dir),
                key4hep_version,
                key4hep_mode,
            ],
            stdout=stdout,
            stderr=stderr,
        )
    return seed, result.returncode


def main():
    parsed_args = parse_args()

    if parsed_args.nprocs < 1:
        raise ValueError("--nprocs must be positive")
    if parsed_args.minseed > parsed_args.maxseed:
        raise ValueError("--minseed must not exceed --maxseed")

    if parsed_args.mainDir:
        main_dir = Path(parsed_args.mainDir).resolve()
    else:
        main_dir = Path(__file__).resolve().parents[1]

    script = main_dir / "src" / "runSequence.sh"
    if not script.is_file():
        raise FileNotFoundError(f"runSequence.sh not found: {script}")

    work_dir = find_work_dir(main_dir)

    outdir = Path(parsed_args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "digi").mkdir(exist_ok=True)
    (outdir / "graph").mkdir(exist_ok=True)

    std_dir = main_dir / "std"
    std_dir.mkdir(exist_ok=True)

    train_or_val = parsed_args.type

    arguments_list = []
    for seed in range(parsed_args.minseed, parsed_args.maxseed + 1):
        output_file = outdir / "graph" / f"Graphs_{seed}_{train_or_val}.parquet"
        if not output_file.is_file():
            print(f"{output_file} : missing file")
            arguments_list.append(seed)

    job_count = len(arguments_list)
    if job_count == 0:
        print("")
        print("No missing jobs found.")
        return

    print("")
    print(f"[Running {job_count} jobs locally, {parsed_args.nprocs} in parallel] ...")

    failures = []
    with ThreadPoolExecutor(max_workers=parsed_args.nprocs) as executor:
        futures = [
            executor.submit(
                run_seed,
                script,
                outdir,
                train_or_val,
                seed,
                work_dir,
                parsed_args.key4hep_version,
                parsed_args.key4hep_mode,
                std_dir,
            )
            for seed in arguments_list
        ]
        for future in futures:
            seed, returncode = future.result()
            if returncode == 0:
                print(f"[done] seed {seed}")
            else:
                print(
                    f"[failed] seed {seed} (exit code {returncode}), "
                    f"see std/local_{seed}.out and std/local_{seed}.err"
                )
                failures.append(seed)

    if failures:
        print("")
        print(f"{len(failures)} job(s) failed: {failures}")
        print("Rerun the same command to retry only the missing outputs.")
        sys.exit(1)


if __name__ == "__main__":
    main()
