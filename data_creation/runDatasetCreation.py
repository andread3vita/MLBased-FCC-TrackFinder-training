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

    TRAIN_OR_TEST = sys.argv[1]              # train or test
    TYPE  = sys.argv[2]                     # noBackground, noBackground_parquet, background, background_parquet, loopers
    DETECTOR = sys.argv[3]                  # IDEA or CLD
    MINSEED = sys.argv[4]                   # min seed
    MAXSEED = sys.argv[5]                   # max seed
    OUTDIR = sys.argv[6]                    # path to the folder where the dataset will be saved
    KEY4HEP_VERSION = sys.argv[7]           # Key4hep version to use
    K4GEO_PATH = sys.argv[8]                # k4geo path to use
    K4FWCORE_PATH = sys.argv[9]             # k4fwcore path to use
    ACCOUNTING_GROUP = sys.argv[10]         # Condor accounting group


    base_dir = find_project_root(Path(__file__).parent)
    main_dir = base_dir / f"data_creation/condor_pipeline/{DETECTOR}/"
    type_normalized = TYPE.lower()
    if type_normalized == "background":
        OPTION = "background"
    elif type_normalized == "background_parquet":
        OPTION = "background_parquet"
    elif type_normalized in ("loopers", "nobackground"):
        OPTION = "noBackground"
    elif type_normalized == "nobackground_parquet":
        OPTION = "noBackground_parquet"
    else:
        raise ValueError(
            f"Invalid pipeline option: {TYPE}. Must be 'noBackground', "
            "'noBackground_parquet', 'background', 'background_parquet', "
            "or 'loopers'."
        )
    
    main_dir = main_dir / OPTION
    if not main_dir.is_dir():
        raise FileNotFoundError(f"Pipeline directory does not exist: {main_dir}")

    outdir = OUTDIR

    print(f"Running dataset creation with the following parameters:")
    print(f"  Type: {TRAIN_OR_TEST}")
    print(f"  Pipeline: {main_dir}")
    print(f"  Detector: {DETECTOR}")
    print(f"  Min Seed: {MINSEED}")
    print(f"  Max Seed: {MAXSEED}")
    print(f"  Output Directory: {OUTDIR}")
    print(f"  Key4hep Version: {KEY4HEP_VERSION}")

    if type_normalized == "loopers":

        subprocess.run([
            "python", f"{main_dir}/src/submit_jobs_loopers.py",
            "--mainDir", main_dir,
            "--queue", "testmatch",
            "--outdir", outdir,
            "--minseed", MINSEED,
            "--maxseed", MAXSEED,
            "--type", TRAIN_OR_TEST,
            "--key4hep_version", KEY4HEP_VERSION,
            "--accounting-group", ACCOUNTING_GROUP,
        ])

    elif type_normalized in ("nobackground", "nobackground_parquet"):
    
        subprocess.run([
            "python", f"{main_dir}/src/submit_jobs.py",
            "--mainDir", main_dir,
            "--queue", "testmatch",
            "--outdir", outdir,
            "--minseed", MINSEED,
            "--maxseed", MAXSEED,
            "--type", TRAIN_OR_TEST,
            "--key4hep_version", KEY4HEP_VERSION,
            "--accounting-group", ACCOUNTING_GROUP,
        ])

    elif type_normalized in ("background", "background_parquet"):

        subprocess.run([
            "python", f"{main_dir}/src/submit_jobs.py",
            "--mainDir", main_dir,
            "--queue", "testmatch",
            "--outdir", outdir,
            "--minseed", MINSEED,
            "--maxseed", MAXSEED,
            "--type", TRAIN_OR_TEST,
            "--key4hep_version", KEY4HEP_VERSION,
            "--k4geoPath", K4GEO_PATH,
            "--k4fwcorePath", K4FWCORE_PATH,
            "--accounting-group", ACCOUNTING_GROUP
        ])

if __name__ == "__main__":

    main()
