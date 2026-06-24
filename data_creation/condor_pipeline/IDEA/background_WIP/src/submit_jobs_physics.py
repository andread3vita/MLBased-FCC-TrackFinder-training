#!/usr/bin/env python

import os
import sys
import glob
import argparse
from pathlib import Path


# _____________________________________________________________________________________________________________
def main(base_path):

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--outdir",
        help="output directory",
        default="",
    )

    parser.add_argument("--njobs", help="number of seeds to check", default=2)
    parser.add_argument("--minseed", help="minimum seed", default=1)
    parser.add_argument("--k4geo_path", default=os.environ.get("K4GEO", ""), help="path to the K4GEO repository")

    parser.add_argument("--type", help="simulation type", default="Pythia")
    parser.add_argument("--config", help="Pythia configuration card name", default="")
    parser.add_argument("--detectorVersion", help="Detector Version", default=3)
    parser.add_argument("--detectorOption", help="Detector Option", default=1)
    parser.add_argument("--train_or_val", help="Dataset type", default="train")

    parser.add_argument(
        "--queue",
        help="queue for condor",
        choices=[
            "espresso",
            "microcentury",
            "longlunch",
            "workday",
            "tomorrow",
            "testmatch",
            "nextweek",
        ],
        default="longlunch",
    )

    args = parser.parse_args()

    queue = args.queue
    outdir = os.path.abspath(args.outdir)
    k4geo_path = os.path.abspath(args.k4geo_path)
    
    njobs = int(args.njobs)
    minseed = int(args.minseed)

    sim_type = args.type
    config = args.config
    detectorVersion = int(args.detectorVersion)
    detectorOption = int(args.detectorOption)
    train_or_val = args.train_or_val

    storage_path = f"{outdir}/{sim_type}/{config}"
    os.makedirs(storage_path, exist_ok=True)

    if sim_type == "Pythia":
        script = "src/run_sequence_global.sh"

    elif sim_type == "gun":
        print("gun mode not implemented yet")
        sys.exit(0)

    else:
        print(f"Unknown simulation type: {sim_type}")
        sys.exit(1)

    arguments_list = []

    # seeds: [minseed, minseed + njobs - 1]
    for seed in range(minseed, minseed + njobs):

        basename = f"{config}_graphs_{seed}_{train_or_val}*"
        output_pattern = os.path.join(storage_path, basename)

        matching_files = glob.glob(output_pattern)

        if not matching_files:

            print(f"{output_pattern} : missing output file")

            argts = (
                f"{outdir} {sim_type} {config} "
                f"{detectorVersion} {detectorOption} "
                f"{seed} {train_or_val} {base_path} {k4geo_path}"
            )

            arguments_list.append(argts)

            if len(arguments_list) == 1:
                print("")
                print(f"rm -rf job*; ./{script} {argts}")

    jobCount = len(arguments_list)

    os.makedirs("gun", exist_ok=True)
    os.makedirs("std", exist_ok=True)

    gun_name = f"gun/{sim_type}_{config}.sub"

    with open(gun_name, "w") as f:

        f.write(
f"""executable = {script}

output = std/condor.$(ClusterId).$(ProcId).out
error  = std/condor.$(ClusterId).$(ProcId).err
log    = std/condor.$(ClusterId).log

+AccountingGroup = "group_u_FCC.local_gen"
+JobFlavour      = "{queue}"

RequestCpus = 3

arguments = $(ARGS)

queue ARGS from (
"""
        )

        for args in arguments_list:
            f.write(f"{args}\n")

        f.write(")\n")

    if jobCount > 0:

        print("")
        print(f"[Submitting {jobCount} jobs] ...")

        os.system(f"condor_submit {gun_name}")

    else:
        print("")
        print("No missing jobs found.")


# _______________________________________________________________________________________
if __name__ == "__main__":

    script_dir = Path(__file__).resolve().parent
    work_dir = script_dir

    while not (work_dir / "data_creation").is_dir() and work_dir != work_dir.parent:
        work_dir = work_dir.parent

    if not (work_dir / "data_creation").is_dir():
        raise RuntimeError("Could not find WORK_DIR containing data_creation")

    print("Project root (WORK_DIR):", work_dir)

    main(work_dir)