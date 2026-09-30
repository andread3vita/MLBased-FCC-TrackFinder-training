#!/bin/bash

set -eo pipefail

if [[ $# -ne 5 ]]; then
    echo "Usage: $0 PAIRS_PATH_BASE64 K4GEO_PATH_BASE64 OUTPUT_PATH_BASE64 KEY4HEP_VERSION_BASE64 STEERING_FILE_BASE64" >&2
    exit 2
fi

decode_path() {
    printf '%s' "$1" | base64 --decode
}

PAIR_PATH=$(decode_path "$1")
K4GEO_PATH=$(decode_path "$2")
OUTPUT_PATH=$(decode_path "$3")
KEY4HEP_VERSION=$(decode_path "$4")
STEERING_FILE=$(decode_path "$5")

readonly DETECTOR_VERSION=3
readonly DETECTOR_OPTION=1

COMPACT_FILE="$K4GEO_PATH/FCCee/IDEA/compact/IDEA_o${DETECTOR_OPTION}_v0${DETECTOR_VERSION}/IDEA_o${DETECTOR_OPTION}_v0${DETECTOR_VERSION}.xml"

# Recheck on the worker in case the file appeared after job submission.
if [[ -f "$OUTPUT_PATH" ]]; then
    echo "SKIP: output already exists: $OUTPUT_PATH"
    exit 0
fi

if [[ ! -f "$PAIR_PATH" ]]; then
    echo "ERROR: input .pairs file not found: $PAIR_PATH" >&2
    exit 1
fi
if [[ ! -f "$COMPACT_FILE" ]]; then
    echo "ERROR: compact geometry file not found: $COMPACT_FILE" >&2
    exit 1
fi
if [[ ! -f "$STEERING_FILE" ]]; then
    echo "ERROR: steering file not found: $STEERING_FILE" >&2
    exit 1
fi

mkdir -p "$(dirname "$OUTPUT_PATH")"
source /cvmfs/sw-nightlies.hsf.org/key4hep/setup.sh -r "$KEY4HEP_VERSION"

# Register the supplied modified K4GEO checkout in this job's environment.
ORIGINAL_DIR=$PWD
cd "$K4GEO_PATH"
k4_local_repo
cd "$ORIGINAL_DIR"

ddsim \
    --compactFile "$COMPACT_FILE" \
    -I "$PAIR_PATH" \
    -O "$OUTPUT_PATH" \
    -N -1 \
    --crossingAngleBoost 0.015 \
    --part.keepAllParticles True \
    --steeringFile "$STEERING_FILE"
