#!/bin/bash

OUTDIR=${1} 
TRAIN_OR_TEST=${2} 
SEED=${3}
WORK_DIR=${4}
KEY4HEP_VERSION=${5}
NEV=500

ORIG_PARAMS=("$@")
set --
source /cvmfs/sw-nightlies.hsf.org/key4hep/setup.sh -r ${KEY4HEP_VERSION} # if you need to fix a specific nightly: source /cvmfs/sw-nightlies.hsf.org/key4hep/setup.sh -r your_version
set -- "${ORIG_PARAMS[@]}"

cd $OUTDIR

cp $WORK_DIR/data_creation/utils/Pythia_generation/Zcard.cmd Zcard_${SEED}.cmd
echo "Random:seed=${SEED}" >> Zcard_${SEED}.cmd

k4run $WORK_DIR/data_creation/utils/Pythia_generation/pythia.py -n $NEV --IOSvc.Output out_${SEED}.root --Pythia8.PythiaInterface.pythiacard Zcard_${SEED}.cmd
rm Zcard_${SEED}.cmd

if [[ "${TRAIN_OR_TEST}" == "train" ]]
then
      
      ddsim --compactFile $K4GEO/FCCee/IDEA/compact/IDEA_o1_v04/IDEA_o1_v04.xml \
            --outputFile out_sim_edm4hep_${SEED}.root \
            --inputFiles out_${SEED}.root \
            --numberOfEvents $NEV \
            --random.seed $SEED \
            --steeringFile  $WORK_DIR/data_creation/condor_pipeline/IDEA/noBackground/utils/SteeringFile_IDEA_o1_v04.py \
            --part.minimalKineticEnergy "0.00*MeV"   
fi

if [[ "${TRAIN_OR_TEST}" == "test" ]]
then

      ddsim --compactFile $K4GEO/FCCee/IDEA/compact/IDEA_o1_v04/IDEA_o1_v04.xml \
            --outputFile out_sim_edm4hep_${SEED}.root \
            --inputFiles out_${SEED}.root \
            --numberOfEvents $NEV \
            --random.seed $SEED \
            --steeringFile $WORK_DIR/data_creation/condor_pipeline/IDEA/noBackground/utils/SteeringFile_IDEA_o1_v04.py \
            --part.userParticleHandler='' \
            --part.keepAllParticles true 
fi        
rm out_${SEED}.root
      
k4run $WORK_DIR/data_creation/condor_pipeline/IDEA/noBackground/utils/runIDEA_v4o1_trackerDigitizer.py --inputFile out_sim_edm4hep_${SEED}.root --outputFile digi/output_IDEA_DIGI_${SEED}_${TRAIN_OR_TEST}.root
rm out_sim_edm4hep_${SEED}.root

python $WORK_DIR/data_creation/condor_pipeline/IDEA/noBackground/src/process_tree.py digi/output_IDEA_DIGI_${SEED}_${TRAIN_OR_TEST}.root graph/Graphs_${SEED}_${TRAIN_OR_TEST}.root
