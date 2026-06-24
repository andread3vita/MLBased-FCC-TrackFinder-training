#!/usr/bin/env python

# import os
# import sys
# import glob
# import random
# import argparse
# from pathlib import Path
# import re

# FILES_PER_SEED = 40

# def build_seed_file_map(seeds: range, all_files: list[str]) -> dict[int, list[str]]:
#     """
#     For every seed, deterministically sample FILES_PER_SEED file paths.
#     """
#     seed_file_map: dict[int, list[str]] = {}

#     for seed in seeds:
#         rng = random.Random(seed)
#         chosen = rng.sample(all_files, min(FILES_PER_SEED, len(all_files)))
#         seed_file_map[seed] = chosen

#     return seed_file_map

# def main(base_path):

#     parser = argparse.ArgumentParser()

#     parser.add_argument(
#         "--outdir",
#         help="output directory",
#         default="",
#     )

#     parser.add_argument("--njobs",   help="number of seeds to check", default=2)
#     parser.add_argument("--minseed", help="minimum seed",             default=1)

#     parser.add_argument("--type",            help="simulation type",               default="Pythia")
#     parser.add_argument("--config",          help="Pythia configuration card name", default="")
#     parser.add_argument("--detectorVersion", help="Detector Version",              default=3)
#     parser.add_argument("--detectorOption",  help="Detector Option",               default=1)
#     parser.add_argument("--train_or_val",    help="Dataset type",                  default="train")

#     parser.add_argument(
#         "--pairs_path",
#         help="path to the file repository from which 40 files are drawn per seed",
#         required=True,
#     )
#     parser.add_argument(
#         "--k4geo_path",
#         default=os.environ.get("K4GEO", ""),
#         help="path to the K4GEO repository"
#     )

#     parser.add_argument(
#         "--queue",
#         help="queue for condor",
#         choices=[
#             "espresso",
#             "microcentury",
#             "longlunch",
#             "workday",
#             "tomorrow",
#             "testmatch",
#             "nextweek",
#         ],
#         default="longlunch",
#     )

#     args = parser.parse_args()

#     queue      = args.queue
#     outdir     = os.path.abspath(args.outdir)
#     pairs_path = os.path.abspath(args.pairs_path)
#     k4geo_path = os.path.abspath(args.k4geo_path)

#     njobs   = int(args.njobs)
#     minseed = int(args.minseed)

#     sim_type        = args.type
#     config          = args.config
#     detectorVersion = int(args.detectorVersion)
#     detectorOption  = int(args.detectorOption)
#     train_or_val    = args.train_or_val

#     storage_path = Path(outdir) / sim_type / config
#     storage_path.mkdir(parents=True, exist_ok=True)

#     bkg_storage_path = Path(outdir) / sim_type / "background"
#     bkg_storage_path.mkdir(parents=True, exist_ok=True)

#     if sim_type == "Pythia":
#         script_background   = "src/run_background_IPC.sh"
#         script   = "src/run_background.sh"

#     elif sim_type == "gun":
#         print("gun mode not implemented yet")
#         sys.exit(0)
#     else:
#         print(f"Unknown simulation type: {sim_type}")
#         sys.exit(1)

#     seeds         = range(minseed, minseed + njobs)
#     all_files     = get_repo_files(pairs_path)
#     seed_file_map = build_seed_file_map(seeds, all_files)

#     arguments_list_background = []

#     # seeds: [minseed, minseed + njobs - 1]
#     background_events = []
#     arguments_list_background = []
#     arguments_list = []
#     for seed in seeds:                    

#         basename       = f"{config}_graphs_{seed}_{train_or_val}*"
#         output_pattern = os.path.join(storage_path, basename)
#         matching_files = glob.glob(output_pattern)

#         if not matching_files:

#             print(f"{output_pattern} : missing output file")

#             argts = (
#                 f"{outdir} {sim_type} {config} "
#                 f"{detectorVersion} {detectorOption} "
#                 f"{seed} {train_or_val} {base_path}"
#             )

#             arguments_list.append(argts)

#             if len(arguments_list) == 1:
#                 print("")
#                 print(f"rm -rf job*; ./{script} {argts}")


#             for f in seed_file_map[seed]:  
                
#                 match = re.search(r'output_(\d+)', f.stem)
#                 pair_id = match.group(1) if match else "unknown"
#                 output_bkg_path = (bkg_storage_path / f"IDEA_o{detectorOption}_v0{detectorVersion}_{pair_id}_background.root")
#                 if not output_bkg_path.is_file():

#                     background_events.append(f)

#                     argts_bkg = (
#                         f"{outdir} {detectorVersion} {detectorOption} {k4geo_path} {f}"
#                     )
#                     arguments_list_background.append(argts_bkg)

#     jobCount_background = len(arguments_list_background)
#     jobCount = len(arguments_list)

#     os.makedirs("gun", exist_ok=True)
#     os.makedirs("std", exist_ok=True)
#     gun_name_background = f"gun/background.sub"
#     gun_name = f"gun/{sim_type}_{config}.sub"


#     with open(gun_name_background, "w") as f:

#         f.write(
#             f"""executable = {script_background}

#             output = std/condor.$(ClusterId).$(ProcId).out
#             error  = std/condor.$(ClusterId).$(ProcId).err
#             log    = std/condor.$(ClusterId).log

#             +AccountingGroup = "group_u_FCC.local_gen"
#             +JobFlavour      = "{queue}"

#             RequestCpus = 3

#             arguments = $(ARGS)

#             queue ARGS from (
#             """)

#         for args in arguments_list_background:
#             f.write(f"{args}\n")

#         f.write(")\n")

#     if jobCount_background > 0:

#         print("")
#         print(f"[Submitting {jobCount_background} jobs] ...")

#         os.system(f"condor_submit {gun_name_background}")
#     else:
#         print("")
#         print("No missing jobs found.")

#     # with open(gun_name, "w") as f:

#     #     f.write(
#     #         f"""executable = {script}

#     #         output = std/condor.$(ClusterId).$(ProcId).out
#     #         error  = std/condor.$(ClusterId).$(ProcId).err
#     #         log    = std/condor.$(ClusterId).log

#     #         +AccountingGroup = "group_u_FCC.local_gen"
#     #         +JobFlavour      = "{queue}"

#     #         RequestCpus = 3

#     #         arguments = $(ARGS)

#     #         queue ARGS from (
#     #         """
#     #     )

#     #     for args in arguments_list:
#     #         f.write(f"{args}\n")

#     #     f.write(")\n")

#     # if jobCount > 0:

#     #     print("")
#     #     print(f"[Submitting {jobCount} jobs] ...")

#     #     os.system(f"condor_submit {gun_name}")
#     # else:
#         print("")
#         print("No missing jobs found.")

# if __name__ == "__main__":

#     script_dir = Path(__file__).resolve().parent
#     work_dir   = script_dir

#     while not (work_dir / "data_creation").is_dir() and work_dir != work_dir.parent:
#         work_dir = work_dir.parent

#     if not (work_dir / "data_creation").is_dir():
#         raise RuntimeError("Could not find WORK_DIR containing data_creation")

#     print("Project root (WORK_DIR):", work_dir)

#     main(work_dir)


import os
import sys
import glob
import random
import argparse
from pathlib import Path
import re

def get_repo_files(repo_path: str) -> list[str]:

    """Return a sorted list of all file paths found under repo_path."""
    repo = Path(repo_path)

    if not repo.is_dir():
        print(f"ERROR: repo_path '{repo_path}' is not a valid directory.")
        sys.exit(1)

    all_files = [f.resolve() for f in repo.iterdir() if f.is_file()]

    return all_files


def main(base_path):

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--outdir",
        help="output directory",
        default="",
    )

    parser.add_argument("--maxFile",   help="maximum file number", default=2)
    parser.add_argument("--minFile", help="minimum file number", default=1)
    parser.add_argument("--detectorVersion", help="Detector Version",              default=3)
    parser.add_argument("--detectorOption",  help="Detector Option",               default=1)

    parser.add_argument(
        "--pairs_path",
        help="path to the file repository from which 40 files are drawn per seed",
        required=True,
    )
    parser.add_argument(
        "--k4geo_path",
        default=os.environ.get("K4GEO", ""),
        help="path to the K4GEO repository"
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

    args = parser.parse_args()

    queue      = args.queue
    outdir     = os.path.abspath(args.outdir)
    pairs_path = os.path.abspath(args.pairs_path)
    k4geo_path = os.path.abspath(args.k4geo_path)

    maxFile = int(args.maxFile)
    minFile = int(args.minFile)

    detectorVersion = int(args.detectorVersion)
    detectorOption  = int(args.detectorOption)

    bkg_storage_path = Path(outdir)
    bkg_storage_path.mkdir(parents=True, exist_ok=True)

    script_background   = "src/run_background_IPC.sh"
    all_files     = get_repo_files(pairs_path)

    arguments_list_background = []
    for idx, file in enumerate(all_files):     

        if not (minFile <= idx <= maxFile):
            continue

        match = re.search(r'output_(\d+)', file.stem)
        pair_id = match.group(1) if match else "unknown"
        output_bkg_path = (bkg_storage_path / f"IDEA_o{detectorOption}_v0{detectorVersion}_{pair_id}_background.root")
        
        if not output_bkg_path.is_file():
            
            argts_bkg = (f"{outdir} {detectorVersion} {detectorOption} {k4geo_path} {file}")
            arguments_list_background.append(argts_bkg)

    jobCount_background = len(arguments_list_background)

    os.makedirs("gun", exist_ok=True)
    os.makedirs("std", exist_ok=True)
    gun_name_background = f"gun/background.sub"

    with open(gun_name_background, "w") as f:

        f.write(
            f"""executable = {script_background}

            output = std/condor.$(ClusterId).$(ProcId).out
            error  = std/condor.$(ClusterId).$(ProcId).err
            log    = std/condor.$(ClusterId).log

            +AccountingGroup = "group_u_FCC.local_gen"
            +JobFlavour      = "{queue}"

            RequestCpus = 3

            arguments = $(ARGS)

            queue ARGS from (
            """)

        for args in arguments_list_background:
            f.write(f"{args}\n")

        f.write(")\n")

    if jobCount_background > 0:

        print("")
        print(f"[Submitting {jobCount_background} jobs] ...")

        os.system(f"condor_submit {gun_name_background}")
    else:
        print("")
        print("No missing jobs found.")

if __name__ == "__main__":

    script_dir = Path(__file__).resolve().parent
    work_dir   = script_dir

    while not (work_dir / "data_creation").is_dir() and work_dir != work_dir.parent:
        work_dir = work_dir.parent

    if not (work_dir / "data_creation").is_dir():
        raise RuntimeError("Could not find WORK_DIR containing data_creation")

    print("Project root (WORK_DIR):", work_dir)

    main(work_dir)