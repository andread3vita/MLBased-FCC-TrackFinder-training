#!/bin/bash

OUTDIR=${1}
DETECTOR_VERSION=${2}
DETECTOR_OPTION=${3}
K4GEO_PATH=${4}
PAIR_PATH=${5}
WORK_DIR=${6}

source /cvmfs/sw-nightlies.hsf.org/key4hep/setup.sh -r 2026-05-19

PAIR_FILE=$(basename ${PAIR_PATH})
PAIR_ID=${PAIR_FILE#output_}
PAIR_ID=${PAIR_ID%.pairs}

# For using a local version of K4GEO
cd ${K4GEO_PATH}
k4_local_repo
cd -

ddsim --compactFile ${K4GEO_PATH}/FCCee/IDEA/compact/IDEA_o${DETECTOR_OPTION}_v0${DETECTOR_VERSION}/IDEA_o${DETECTOR_OPTION}_v0${DETECTOR_VERSION}.xml \
    -I ${PAIR_PATH} \
    -O ${OUTDIR}/IDEA_o${DETECTOR_OPTION}_v0${DETECTOR_VERSION}_${PAIR_ID}_background.root \
    -N -1 \
    --crossingAngleBoost 0.015 \
    --part.keepAllParticles True \
    --steeringFile $WORK_DIR/data_creation/condor_pipeline/IDEA/background/utils/SteeringFile_IDEA_o1_v03_background.py
