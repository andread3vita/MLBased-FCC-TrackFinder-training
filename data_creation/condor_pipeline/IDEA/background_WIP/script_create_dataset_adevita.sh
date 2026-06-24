#!/bin/bash

TYPE=${1}           # type: Pythia or Gun
CONFIG=${2}         # config file
VERSION=${3}        # IDEA version (2 or 3)
OPTION=${4}         # IDEA option
MINSEED=${5}        # minimum seed to process
NFILE=${6}          # number of files
TRAIN_OR_VAL=${7}   # training or validation ('train' or 'val')
OUTDIR=${8}         # output directory
K4GEO_DIR=${9}      # path to the K4GEO repository

CURRPATH=$(pwd)
ORIG_PARAMS=("$@")
set --
source /cvmfs/sw-nightlies.hsf.org/key4hep/setup.sh -r 2026-05-19 # if you need to fix a specific nightly: source /cvmfs/sw-nightlies.hsf.org/key4hep/setup.sh -r your_version
set -- "${ORIG_PARAMS[@]}"

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
while [[ ! -d "$BASE_DIR/data_creation" && "$BASE_DIR" != "/" ]]; do
    BASE_DIR="$(dirname "$BASE_DIR")"
done
[[ "$BASE_DIR" == "/" ]] && { echo "ERROR: could not find project root containing data_creation"; exit 1; }

outdir="$OUTDIR"
mkdir -p "$BASE_DIR/data_creation/condor_pipeline/IDEA/background/gun"

if [[ "${VERSION}" -eq 3 ]]; then

    STEERING_FILE=utils/SteeringFile_IDEA_o1_v03.py

    src_file="$FCCCONFIG/FullSim/IDEA/IDEA_o1_v03/SteeringFile_IDEA_o1_v03.py"
    cp "$src_file" "$STEERING_FILE"
    sed -i 's/simulateCalo *= *True/simulateCalo = False/' "$STEERING_FILE"
    
fi

python src/submit_jobs_physics.py  --queue testmatch --outdir $outdir --njobs $NFILE --minseed $MINSEED --type $TYPE --config $CONFIG --detectorVersion $VERSION --detectorOption $OPTION --train_or_val $TRAIN_OR_VAL --k4geo_path $K4GEO_DIR