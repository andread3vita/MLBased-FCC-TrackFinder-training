#!/usr/bin/env bash

if [ "$#" -ne 7 ]; then
echo "Usage:"
echo " $0 TRAIN_OR_VAL BACKGROUND DETECTOR MINSEED MAXSEED OUTDIR KEY4HEP_VERSION"
exit 1
fi

TRAIN_OR_VAL=$1
BACKGROUND=$2
DETECTOR=$3
MINSEED=$4
MAXSEED=$5
OUTDIR=$6
KEY4HEP_VERSION=$7

mkdir -p ${OUTDIR}
mkdir -p ${OUTDIR}/digi/
mkdir -p ${OUTDIR}/graph/

ORIG_PARAMS=("$@")
set --
source /cvmfs/sw-nightlies.hsf.org/key4hep/setup.sh -r ${KEY4HEP_VERSION} # if you need to fix a specific nightly: source /cvmfs/sw-nightlies.hsf.org/key4hep/setup.sh -r your_version
set -- "${ORIG_PARAMS[@]}"

echo "Loading Key4hep nightly: ${KEY4HEP_VERSION}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

python "${SCRIPT_DIR}/runDatasetCreation.py" \
    "${TRAIN_OR_VAL}" \
    "${BACKGROUND}" \
    "${DETECTOR}" \
    "${MINSEED}" \
    "${MAXSEED}" \
    "${OUTDIR}" \
    "${KEY4HEP_VERSION}" \