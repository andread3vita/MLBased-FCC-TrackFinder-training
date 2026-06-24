#!/usr/bin/env python

import datetime
import os
import glob
import argparse
from pathlib import Path
from datetime import datetime


# _____________________________________________________________________________________________________________
def main(base_path):

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--outdir",
        help="output directory",
        default="",
    )

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

    parser.add_argument("--minseed", help="minimum seed", default=1)
    parser.add_argument("--maxseed", help="maximum seed", default=1)
    parser.add_argument("--type", help="train or test", default="train")
    parser.add_argument("--detector", help="IDEA or CLD", default="IDEA")
    parser.add_argument("--background", help="true or false", default="false")
    parser.add_argument(
        "--key4hep_version",
        help="Key4hep version to use",
        default="2026-05-19",
    )
    parser.add_argument("--mainDir", help="main directory", default="")

    parsed_args = parser.parse_args()

    queue = parsed_args.queue

    outdir = os.path.abspath(parsed_args.outdir)
    os.makedirs(outdir, exist_ok=True)

    mainDir = parsed_args.mainDir

    njobs = int(parsed_args.maxseed) - int(parsed_args.minseed) + 1
    minseed = int(parsed_args.minseed)
    train_or_val = parsed_args.type
    key4hep_version = parsed_args.key4hep_version

    script = f"{mainDir}/src/runSequence_loopers.sh"

    arguments_list = []

    # seeds: [minseed, minseed + njobs - 1]
    basename_graph = os.path.join(outdir, "graph")
    basename_digi = os.path.join(outdir, "digi")
    os.makedirs(basename_graph, exist_ok=True)
    os.makedirs(basename_digi, exist_ok=True)

    os.makedirs("gun", exist_ok=True)
    os.makedirs("std", exist_ok=True)

    for seed in range(minseed, minseed + njobs):

        fileName = f"Graphs_{seed}_{train_or_val}*"
        output_pattern = os.path.join(basename_graph, fileName)

        matching_files = glob.glob(output_pattern)
        if not matching_files:

            print(f"{output_pattern} : missing file")

            argts = (f"{outdir} {train_or_val} {seed} {base_path} {key4hep_version}")
            arguments_list.append(argts)


    jobCount = len(arguments_list)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    gun_name = f"gun/{parsed_args.detector}_{train_or_val}_{timestamp}.sub"

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

    main(work_dir)