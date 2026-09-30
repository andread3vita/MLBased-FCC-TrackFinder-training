#!/usr/bin/env python3

"""Submit one ddsim Condor job for each missing background ROOT file."""

import argparse
import base64
import re
import subprocess
from pathlib import Path


SUPPORTED_QUEUES = (
    "espresso",
    "microcentury",
    "longlunch",
    "workday",
    "tomorrow",
    "testmatch",
    "nextweek",
)


def natural_sort_key(path: Path) -> list[tuple[int, object]]:
    """Sort names containing numbers in human order (2 before 10)."""
    return [
        (0, int(part)) if part.isdigit() else (1, part.casefold())
        for part in re.split(r"(\d+)", path.name)
    ]


def list_pairs_files(pairs_path: Path) -> list[Path]:
    """Return the regular .pairs files directly inside pairs_path."""
    return sorted(
        (path.resolve() for path in pairs_path.glob("*.pairs") if path.is_file()),
        key=natural_sort_key,
    )


def pair_identifier(pair_path: Path) -> str:
    """Match the historical output_N.pairs -> N naming convention."""
    stem = pair_path.stem
    if stem.startswith("output_") and stem != "output_":
        return stem[len("output_") :]
    return stem


def encode_path(path: Path) -> str:
    """Encode paths so Condor item-data parsing cannot split on whitespace."""
    return base64.b64encode(str(path).encode("utf-8")).decode("ascii")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Submit ddsim jobs for the first NUMFILE .pairs files."
    )
    parser.add_argument("--pairs-path", required=True, type=Path)
    parser.add_argument("--k4geo-path", required=True, type=Path)
    parser.add_argument("--outdir", required=True, type=Path)
    parser.add_argument("--num-files", required=True, type=int)
    parser.add_argument("--key4hep-version", required=True)
    parser.add_argument("--detector-version", type=int, default=3)
    parser.add_argument("--detector-option", type=int, default=1)
    parser.add_argument("--queue", choices=SUPPORTED_QUEUES, default="testmatch")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.num_files < 0:
        raise SystemExit("ERROR: --num-files must be non-negative")
    if not args.key4hep_version.strip():
        raise SystemExit("ERROR: --key4hep-version cannot be empty")
    if (args.detector_version, args.detector_option) != (3, 1):
        raise SystemExit("ERROR: this background workflow requires VERSION=3 and OPTION=1")

    pairs_path = args.pairs_path.expanduser().resolve()
    k4geo_path = args.k4geo_path.expanduser().resolve()
    outdir = args.outdir.expanduser().resolve()

    if not pairs_path.is_dir():
        raise SystemExit(f"ERROR: PAIRS_PATH is not a directory: {pairs_path}")
    if not k4geo_path.is_dir():
        raise SystemExit(f"ERROR: K4GEO_PATH is not a directory: {k4geo_path}")

    outdir.mkdir(parents=True, exist_ok=True)

    all_pairs = list_pairs_files(pairs_path)
    selected_pairs = all_pairs[: args.num_files]
    print(
        f"Found {len(all_pairs)} .pairs file(s); "
        f"checking the first {len(selected_pairs)}."
    )

    detector_tag = f"IDEA_o{args.detector_option}_v0{args.detector_version}"
    missing_jobs: list[tuple[Path, Path]] = []

    for pair_path in selected_pairs:
        output_path = outdir / (
            f"{detector_tag}_{pair_identifier(pair_path)}_background.root"
        )
        if output_path.is_file():
            print(f"SKIP: output already exists: {output_path}")
            continue
        if output_path.exists():
            raise SystemExit(
                f"ERROR: expected output path exists but is not a file: {output_path}"
            )
        missing_jobs.append((pair_path, output_path))

    if not missing_jobs:
        print("No missing ROOT files found; no Condor jobs submitted.")
        return

    project_dir = Path(__file__).resolve().parent.parent
    worker_script = project_dir / "src" / "run_background_IPC.sh"
    steering_file = project_dir / "utils" / "SteeringFile_IDEA_o1_v03_background.py"
    if not worker_script.is_file():
        raise SystemExit(f"ERROR: worker script not found: {worker_script}")
    if not steering_file.is_file():
        raise SystemExit(f"ERROR: steering file not found: {steering_file}")

    submit_dir = project_dir / "gun"
    log_dir = project_dir / "std"
    submit_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    submit_file = submit_dir / "background_IPC.sub"

    lines = [
        f"executable = {worker_script}",
        f"output = {log_dir}/condor.$(ClusterId).$(ProcId).out",
        f"error = {log_dir}/condor.$(ClusterId).$(ProcId).err",
        f"log = {log_dir}/condor.$(ClusterId).log",
        '+AccountingGroup = "group_u_FCC.local_gen"',
        f'+JobFlavour = "{args.queue}"',
        "RequestCpus = 3",
        "notification = Never",
        "should_transfer_files = NO",
        "arguments = $(PAIRS_B64) $(K4GEO_B64) $(OUTPUT_B64) $(KEY4HEP_B64) $(STEERING_B64)",
        "queue PAIRS_B64, K4GEO_B64, OUTPUT_B64, KEY4HEP_B64, STEERING_B64 from (",
    ]
    encoded_k4geo = encode_path(k4geo_path)
    encoded_steering = encode_path(steering_file)
    encoded_key4hep = base64.b64encode(
        args.key4hep_version.encode("utf-8")
    ).decode("ascii")
    for pair_path, output_path in missing_jobs:
        lines.append(
            f"{encode_path(pair_path)} {encoded_k4geo} {encode_path(output_path)} "
            f"{encoded_key4hep} {encoded_steering}"
        )
    lines.append(")")
    submit_file.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Submitting {len(missing_jobs)} missing background job(s).")
    subprocess.run(["condor_submit", str(submit_file)], check=True)


if __name__ == "__main__":
    main()
