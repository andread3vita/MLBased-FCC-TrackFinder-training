#!/bin/bash

OUTDIR=${1} 
TRAIN_OR_TEST=${2} 
SEED=${3}
WORK_DIR=${4}
KEY4HEP_VERSION=${5}
NEV=500

ORIG_PARAMS=("$@")
set --
source /cvmfs/sw-nightlies.hsf.org/key4hep/setup.sh --spack -r "$KEY4HEP_VERSION"
setup_status=$?
set -- "${ORIG_PARAMS[@]}"

if (( setup_status != 0 )); then
      echo "ERROR: Key4hep setup failed, probably due to CVMFS access." >&2
      exit 75
fi

if ! python -c 'import ROOT'; then
      echo "ERROR: Key4hep runtime validation failed, probably due to incomplete CVMFS access." >&2
      exit 75
fi

# Stop immediately if any later processing command fails.
set -e

cd $OUTDIR

cp $WORK_DIR/data_creation/utils/Pythia_generation/Zcard.cmd Zcard_${SEED}.cmd
printf '\nRandom:setSeed=on\nRandom:seed=%s\n' "$SEED" >> Zcard_${SEED}.cmd

k4run $WORK_DIR/data_creation/utils/Pythia_generation/pythia.py \
      -n $NEV \
      --IOSvc.Output out_${SEED}.root \
      --HepMCFileWriter.Filename out_${SEED}.hepmc \
      --Pythia8.PythiaInterface.pythiacard Zcard_${SEED}.cmd
rm Zcard_${SEED}.cmd

if [[ "${TRAIN_OR_TEST}" == "train" ]]
then
      
      ddsim --compactFile $K4GEO/FCCee/IDEA/compact/IDEA_o1_v04/IDEA_o1_v04.xml \
            --outputFile out_sim_edm4hep_${SEED}.root \
            --inputFiles out_${SEED}.hepmc \
            --numberOfEvents $NEV \
            --random.seed $SEED \
            --steeringFile  $WORK_DIR/data_creation/condor_pipeline/IDEA/noBackground_parquet/utils/SteeringFile_IDEA_o1_v04.py \
            --part.minimalKineticEnergy "0.00*MeV"   
fi

if [[ "${TRAIN_OR_TEST}" == "test" ]]
then

      ddsim --compactFile $K4GEO/FCCee/IDEA/compact/IDEA_o1_v04/IDEA_o1_v04.xml \
            --outputFile out_sim_edm4hep_${SEED}.root \
            --inputFiles out_${SEED}.hepmc \
            --numberOfEvents $NEV \
            --random.seed $SEED \
            --steeringFile $WORK_DIR/data_creation/condor_pipeline/IDEA/noBackground_parquet/utils/SteeringFile_IDEA_o1_v04.py \
            --part.userParticleHandler='' \
            --part.keepAllParticles true 
fi        
rm out_${SEED}.root out_${SEED}.hepmc
      
k4run $WORK_DIR/data_creation/condor_pipeline/IDEA/noBackground_parquet/utils/runIDEA_v4o1_trackerDigitizer.py --inputFile out_sim_edm4hep_${SEED}.root --outputFile digi/output_IDEA_DIGI_${SEED}_${TRAIN_OR_TEST}.root
rm out_sim_edm4hep_${SEED}.root

python $WORK_DIR/data_creation/condor_pipeline/IDEA/noBackground_parquet/src/process_tree.py \
      digi/output_IDEA_DIGI_${SEED}_${TRAIN_OR_TEST}.root \
      graph/Graphs_${SEED}_${TRAIN_OR_TEST}.parquet \
      --file-number "$SEED" \
      --row-group-size 25
