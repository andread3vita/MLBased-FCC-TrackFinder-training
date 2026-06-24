#!/bin/bash

VERSION=${1}        # IDEA version (2 or 3)
OPTION=${2}         # IDEA option
MINFILE=${3}        # minimum file number
MAXFILE=${4}        # maximum file number
OUTDIR=${5}         # output directory
PAIRS_PATH=${6}     # path to the Bkg repository
K4GEO_PATH=${7}    # path to the K4GEO repository

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

    STEERING_FILE=utils/SteeringFile_IDEA_o1_v03_background.py
    sed -i 's/simulateCalo *= *True/simulateCalo = False/' "$STEERING_FILE"
    
fi

python src/submit_jobs_IPC.py  --queue testmatch --outdir $outdir --maxFile $MAXFILE --minFile $MINFILE --pairs_path $PAIRS_PATH --k4geo_path $K4GEO_PATH --detectorVersion $VERSION --detectorOption $OPTION