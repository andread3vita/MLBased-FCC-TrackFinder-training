#!/usr/bin/env bash

# Inputs:
#   TRAIN_OR_TEST: train, val or test (val simulates like train, test keeps all particles)
#   PIPELINE: noBackground, noBackground_parquet, background_parquet
#   DETECTOR: IDEA or CLD
#   MINSEED/MAXSEED: non-negative seed range
#   OUTDIR: output directory
#   KEY4HEP_VERSION: Key4hep release; a date for KEY4HEP_MODE=spack (for example 2026-10-20),
#                    a weekday Mon..Sat or "latest" for KEY4HEP_MODE=lcg
#   K4GEO_PATH: k4geo installation path (not necessary if no background)
#   K4FWCORE_PATH: k4FWCore installation path (not necessary if no background)
#   ACCOUNTING_GROUP: Condor accounting group, for example group_u_FCC.local_gen
#   KEY4HEP_MODE: optional, spack (default) or lcg; supported by the noBackground_parquet pipeline

if [ "$#" -ne 10 ] && [ "$#" -ne 11 ]; then
    echo "Usage:"
    echo " $0 TRAIN_OR_TEST PIPELINE DETECTOR MINSEED MAXSEED OUTDIR KEY4HEP_VERSION K4GEO_PATH K4FWCORE_PATH ACCOUNTING_GROUP [KEY4HEP_MODE]"
    echo ""
    echo "PIPELINE choices: noBackground, noBackground_parquet, noBackgroundDisplacement_parquet, background, background_parquet, loopers"
    exit 1
fi

TRAIN_OR_TEST=$1
PIPELINE=$2
DETECTOR=$3
MINSEED=$4
MAXSEED=$5
OUTDIR=$6
KEY4HEP_VERSION=$7
K4GEO_PATH=$8
K4FWCORE_PATH=$9
ACCOUNTING_GROUP=${10}
KEY4HEP_MODE=${11:-spack}

case "${PIPELINE}" in
    noBackground|noBackground_parquet|noBackgroundDisplacement_parquet|background|background_parquet|loopers)
        ;;
    *)
        echo "Invalid pipeline '${PIPELINE}'." >&2
        echo "Choose one of: noBackground, noBackground_parquet, noBackgroundDisplacement_parquet, background, background_parquet, loopers" >&2
        exit 2
        ;;
esac

if [[ "${PIPELINE}" == *_parquet ]] && [ "${DETECTOR}" != "IDEA" ]; then
    echo "Parquet pipelines are currently available only for detector IDEA." >&2
    exit 2
fi

mkdir -p "${OUTDIR}"
mkdir -p "${OUTDIR}/digi/"
mkdir -p "${OUTDIR}/graph/"

ORIG_PARAMS=("$@")
set --
source /cvmfs/sw-nightlies.hsf.org/key4hep/setup.sh --${KEY4HEP_MODE} -r ${KEY4HEP_VERSION} # if you need to fix a specific nightly: source /cvmfs/sw-nightlies.hsf.org/key4hep/setup.sh -r your_version
set -- "${ORIG_PARAMS[@]}"

echo "Loading Key4hep nightly: ${KEY4HEP_VERSION}"
echo "Selected pipeline: condor_pipeline/${DETECTOR}/${PIPELINE}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

python "${SCRIPT_DIR}/runDatasetCreation.py" \
    "${TRAIN_OR_TEST}" \
    "${PIPELINE}" \
    "${DETECTOR}" \
    "${MINSEED}" \
    "${MAXSEED}" \
    "${OUTDIR}" \
    "${KEY4HEP_VERSION}" \
    "${K4GEO_PATH}" \
    "${K4FWCORE_PATH}" \
    "${ACCOUNTING_GROUP}" \
    "${KEY4HEP_MODE}"
