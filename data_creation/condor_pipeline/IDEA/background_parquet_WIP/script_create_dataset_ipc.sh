#!/bin/bash

set -eo pipefail

usage() {
    echo "Usage: $0 PAIRS_PATH K4GEO_PATH OUTDIR NUMFILE KEY4HEP_VERSION" >&2
}

if [[ $# -ne 5 ]]; then
    usage
    exit 2
fi

PAIRS_PATH=$1
K4GEO_PATH=$2
OUTDIR=$3
NUMFILE=$4
KEY4HEP_VERSION=$5

readonly VERSION=3
readonly OPTION=1

if [[ ! -d "$PAIRS_PATH" ]]; then
    echo "ERROR: PAIRS_PATH is not a directory: $PAIRS_PATH" >&2
    exit 1
fi

if [[ ! -d "$K4GEO_PATH" ]]; then
    echo "ERROR: K4GEO_PATH is not a directory: $K4GEO_PATH" >&2
    exit 1
fi

if [[ ! "$NUMFILE" =~ ^[0-9]+$ ]]; then
    echo "ERROR: NUMFILE must be a non-negative integer, got: $NUMFILE" >&2
    exit 2
fi

if [[ -z "$KEY4HEP_VERSION" ]]; then
    echo "ERROR: KEY4HEP_VERSION cannot be empty" >&2
    exit 2
fi

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

# Set up both condor_submit and the Key4hep environment used by the jobs.
source /cvmfs/sw-nightlies.hsf.org/key4hep/setup.sh -r "$KEY4HEP_VERSION"

python3 "$SCRIPT_DIR/src/submit_jobs_IPC.py" \
    --pairs-path "$PAIRS_PATH" \
    --k4geo-path "$K4GEO_PATH" \
    --outdir "$OUTDIR" \
    --num-files "$NUMFILE" \
    --key4hep-version "$KEY4HEP_VERSION" \
    --detector-version "$VERSION" \
    --detector-option "$OPTION" \
    --queue testmatch
