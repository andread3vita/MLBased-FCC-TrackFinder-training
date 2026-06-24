#!/usr/bin/env python3

import sys
import subprocess
from pathlib import Path

def find_project_root(start_dir: Path) -> Path:
    """Find the project root containing data_creation."""
    current = start_dir.resolve()

    while current != current.parent:
        if (current / "data_creation").is_dir():
            return current
        current = current.parent

    raise RuntimeError(
        "Could not find project root containing data_creation"
    )


def main():

    TRAIN_OR_VAL = sys.argv[1]              # train or test
    TYPE  = sys.argv[2]                     # noBackground, background, loopers
    DETECTOR = sys.argv[3]                  # IDEA or CLD
    MINSEED = sys.argv[4]                   # min seed
    MAXSEED = sys.argv[5]                   # max seed
    OUTDIR = sys.argv[6]                    # path to the folder where the dataset will be saved
    KEY4HEP_VERSION = sys.argv[7]           # Key4hep version to use

    base_dir = find_project_root(Path(__file__).parent)
    main_dir = base_dir / f"data_creation/condor_pipeline/{DETECTOR}/"
    if TYPE == "background":
        OPTION = "background"
    elif TYPE == "loopers" or TYPE == "noBackground":
        OPTION = "noBackground"
    else:
        raise ValueError(f"Invalid background option: {TYPE}. Must be 'noBackground', 'background', or 'loopers'.")
    
    main_dir = main_dir / OPTION

    outdir = OUTDIR

    print(f"Running dataset creation with the following parameters:")
    print(f"  Type: {TRAIN_OR_VAL}")
    print(f"  TYPE: {TYPE}")
    print(f"  Detector: {DETECTOR}")
    print(f"  Min Seed: {MINSEED}")
    print(f"  Max Seed: {MAXSEED}")
    print(f"  Output Directory: {OUTDIR}")
    print(f"  Key4hep Version: {KEY4HEP_VERSION}")

    if TYPE.lower() == "loopers":

        subprocess.run([
            "python", f"{main_dir}/src/submit_jobs_loopers.py",
            "--mainDir", main_dir,
            "--queue", "testmatch",
            "--outdir", outdir,
            "--minseed", MINSEED,
            "--maxseed", MAXSEED,
            "--type", TRAIN_OR_VAL,
            "--key4hep_version", KEY4HEP_VERSION
        ])

    else:
        subprocess.run([
            "python", f"{main_dir}/src/submit_jobs.py",
            "--mainDir", main_dir,
            "--queue", "testmatch",
            "--outdir", outdir,
            "--minseed", MINSEED,
            "--maxseed", MAXSEED,
            "--type", TRAIN_OR_VAL,
            "--key4hep_version", KEY4HEP_VERSION
        ])


if __name__ == "__main__":

    main()