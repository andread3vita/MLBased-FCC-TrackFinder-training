#!/bin/bash

OUTDIR=${1} 
TRAIN_OR_TEST=${2} 
SEED=${3}
WORK_DIR=${4}
KEY4HEP_VERSION=${5}
K4GEO_PATH=${6}
K4FWCORE_PATH=${7}

NEV=50

ORIG_PARAMS=("$@")
set --
source /cvmfs/sw-nightlies.hsf.org/key4hep/setup.sh -r ${KEY4HEP_VERSION} # if you need to fix a specific nightly: source /cvmfs/sw-nightlies.hsf.org/key4hep/setup.sh -r your_version
set -- "${ORIG_PARAMS[@]}"

cd $OUTDIR

ORIGINAL_DIR=$PWD
cd "$K4GEO_PATH" # this k4geo version should include the modification to run background events: https://github.com/HEP-FCC/bib-studies/blob/main/simulation/README.md
k4_local_repo
cd "$ORIGINAL_DIR"

ORIGINAL_DIR=$PWD
cd "$K4FWCORE_PATH" # this k4FWCore version should include the modification to overlay background with random pick of the root files: https://github.com/ArinaPon/k4FWCore/tree/fix-random-mix-file-cursor
k4_local_repo
cd "$ORIGINAL_DIR"

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
      
      ddsim --compactFile $K4GEO/FCCee/IDEA/compact/IDEA_o1_v03/IDEA_o1_v03.xml \
            --outputFile out_sim_edm4hep_${SEED}.root \
            --inputFiles out_${SEED}.hepmc \
            --numberOfEvents $NEV \
            --random.seed $SEED \
            --steeringFile  $WORK_DIR/data_creation/condor_pipeline/IDEA/background_parquet/utils/SteeringFile_IDEA_o1_v03_background.py \
            --part.minimalKineticEnergy "0.00*MeV"   
fi

if [[ "${TRAIN_OR_TEST}" == "test" ]]
then

      ddsim --compactFile $K4GEO/FCCee/IDEA/compact/IDEA_o1_v03/IDEA_o1_v03.xml \
            --outputFile out_sim_edm4hep_${SEED}.root \
            --inputFiles out_${SEED}.hepmc \
            --numberOfEvents $NEV \
            --random.seed $SEED \
            --steeringFile $WORK_DIR/data_creation/condor_pipeline/IDEA/background_parquet/utils/SteeringFile_IDEA_o1_v03_background.py \
            --part.userParticleHandler='' \
            --part.keepAllParticles true 
fi        
rm out_${SEED}.root out_${SEED}.hepmc

k4run $WORK_DIR/data_creation/condor_pipeline/IDEA/background_parquet/utils/overlay.py --inputFile out_sim_edm4hep_${SEED}.root --outputFile out_sim_edm4hep_${SEED}_overlay.root --backgroundFilesPath /eos/experiment/fcc/ee/simulation/key4hep_2026_07_29/91GeV/IDEA_o1_v3/test/testV106.2/IPC
      
k4run $WORK_DIR/data_creation/condor_pipeline/IDEA/background_parquet/utils/runIDEAv3o1_trackerDigitizer.py --inputFile out_sim_edm4hep_${SEED}_overlay.root --outputFile digi/output_IDEA_DIGI_${SEED}_${TRAIN_OR_TEST}.root --inputCollectionPrefix Overlay
rm out_sim_edm4hep_${SEED}.root
rm out_sim_edm4hep_${SEED}_overlay.root

python $WORK_DIR/data_creation/condor_pipeline/IDEA/background_parquet/src/process_tree.py \
      digi/output_IDEA_DIGI_${SEED}_${TRAIN_OR_TEST}.root \
      graph/Graphs_${SEED}_${TRAIN_OR_TEST}.parquet \
      --file-number "$SEED" \
      --row-group-size 25
