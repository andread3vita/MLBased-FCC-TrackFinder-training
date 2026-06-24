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

if [[ "${TRAIN_OR_TEST}" == "train" ]]
then
      
      MIN=3
      MAX=10
      NPARTICLES=$(( RANDOM % ($MAX - $MIN + 1) + $MIN ))
      NPARTICLES=$((NPARTICLES))

      ddsim --enableGun --gun.distribution uniform \
            --gun.momentumMin "0.01*GeV" --gun.momentumMax "1.00*GeV" \
            --gun.particle pi+ \
            --gun.multiplicity $NPARTICLES \
            --random.enableEventSeed --random.seed $SEED \
            --numberOfEvents $NEV \
            --compactFile $K4GEO/FCCee/IDEA/compact/IDEA_o1_v04/IDEA_o1_v04.xml \
            --steeringFile $WORK_DIR/data_creation/condor_pipeline/IDEA/noBackground/utils/SteeringFile_IDEA_o1_v04.py \
            --part.minimalKineticEnergy "0.00*MeV" \
            --outputFile out_sim_edm4hep_${SEED}.root   
fi

if [[ "${TRAIN_OR_TEST}" == "test" ]]
then

      MIN=3
      MAX=10
      NPARTICLES=$(( RANDOM % ($MAX - $MIN + 1) + $MIN ))
      NPARTICLES=$((NPARTICLES))

      ddsim --enableGun --gun.distribution uniform \
            --gun.momentumMin "0.01*GeV" --gun.momentumMax "1.00*GeV" \
            --gun.particle pi+ \
            --gun.multiplicity $NPARTICLES \
            --random.enableEventSeed --random.seed $SEED \
            --numberOfEvents $NEV \
            --compactFile $K4GEO/FCCee/IDEA/compact/IDEA_o1_v04/IDEA_o1_v04.xml \
            --steeringFile $WORK_DIR/data_creation/condor_pipeline/IDEA/noBackground/utils/SteeringFile_IDEA_o1_v04.py \
            --outputFile out_sim_edm4hep_${SEED}.root \
            --part.userParticleHandler='' \
            --part.keepAllParticles true 

fi
      
k4run $WORK_DIR/data_creation/condor_pipeline/IDEA/noBackground/utils/runIDEAv4_o1_trackerDigitizer.py --inputFile out_sim_edm4hep_${SEED}.root --outputFile digi/output_IDEA_DIGI_${SEED}_${TRAIN_OR_TEST}.root
rm out_sim_edm4hep_${SEED}.root

python $WORK_DIR/data_creation/condor_pipeline/IDEA/noBackground/src/process_tree.py digi/output_IDEA_DIGI_${SEED}_${TRAIN_OR_TEST}.root graph/Graphs_${SEED}_${TRAIN_OR_TEST}.root
