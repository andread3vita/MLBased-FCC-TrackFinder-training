#!/bin/bash
# the code comes from here: https://zenodo.org/records/8260741
#SBATCH -p main
#SBATCH --mem-per-cpu=6G
#SBATCH --cpus-per-task=1
#SBATCH -o logs/slurm-%x-%j-%N.out
# set -e
# set -x

# env
# df -h

OUTDIR=${1} 
TYPE=${2} 
CONFIG=${3} 
VERSION=${4} 
OPTION=${5} 
SEED=${6}
TRAIN_OR_VAL=${7}
WORK_DIR=${8}
K4GEO_DIR=${9}

NEV=500

ORIG_PARAMS=("$@")
set --
source /cvmfs/sw-nightlies.hsf.org/key4hep/setup.sh -r 2026-05-19 # if you need to fix a specific nightly: source /cvmfs/sw-nightlies.hsf.org/key4hep/setup.sh -r your_version
set -- "${ORIG_PARAMS[@]}"

cd ${K4GEO_DIR}
k4_local_repo
cd -

TEMP_DIR=${OUTDIR}/${TYPE}/temp/
FULLOUTDIR=${OUTDIR}/${TYPE}/${CONFIG}

mkdir -p $TEMP_DIR
cd $TEMP_DIR
mkdir -p out_hepmc/
mkdir -p out_edm4hep/
mkdir -p out_digi/
mkdir -p ${FULLOUTDIR}


if [[ "${TYPE}" == "Pythia" ]]
then 

      cp $WORK_DIR/data_creation/utils/Pythia_generation/${CONFIG}.cmd ${CONFIG}_${SEED}.cmd
      echo "Random:seed=${SEED}" >> ${CONFIG}_${SEED}.cmd

      k4run $WORK_DIR/data_creation/utils/Pythia_generation/pythia.py -n $NEV --Dumper.Filename out_hepmc/out_${SEED}.hepmc --Pythia8.PythiaInterface.pythiacard ${CONFIG}_${SEED}.cmd
      rm ${CONFIG}_${SEED}.cmd

      if [[ $VERSION -eq 3 ]]
      then

            if [[ "${TRAIN_OR_VAL}" == "train" ]]
            then

                  ddsim --compactFile $K4GEO/FCCee/IDEA/compact/IDEA_o${OPTION}_v0${VERSION}/IDEA_o${OPTION}_v0${VERSION}.xml \
                        --outputFile out_edm4hep/out_sim_edm4hep_${SEED}.root \
                        --inputFiles out_hepmc/out_${SEED}.hepmc \
                        --numberOfEvents $NEV \
                        --random.seed $SEED \
                        --steeringFile  $WORK_DIR/data_creation/condor_pipeline/IDEA/background/utils/SteeringFile_IDEA_o1_v03.py \
                        --part.minimalKineticEnergy "0.00*MeV"   
            fi

            if [[ "${TRAIN_OR_VAL}" == "val" ]]
            then

                  ddsim --compactFile $K4GEO/FCCee/IDEA/compact/IDEA_o${OPTION}_v0${VERSION}/IDEA_o${OPTION}_v0${VERSION}.xml \
                        --outputFile out_edm4hep/out_sim_edm4hep_${SEED}.root \
                        --inputFiles out_hepmc/out_${SEED}.hepmc \
                        --numberOfEvents $NEV \
                        --random.seed $SEED \
                        --steeringFile $WORK_DIR/data_creation/condor_pipeline/IDEA/background/utils/SteeringFile_IDEA_o1_v03.py \
                        --part.userParticleHandler='' \
                        --part.keepAllParticles true 
            fi            
      fi   

      if [[ $VERSION -eq 2 ]]
      then
            if [[ "${TRAIN_OR_VAL}" == "train" ]]
            then

                  ddsim --compactFile $K4GEO/FCCee/IDEA/compact/IDEA_o${OPTION}_v0${VERSION}/IDEA_o${OPTION}_v0${VERSION}.xml \
                        --outputFile out_edm4hep/out_sim_edm4hep_${SEED}.root \
                        --inputFiles out_hepmc/out_${SEED}.hepmc \
                        --numberOfEvents $NEV \
                        --random.seed $SEED \
                        --part.minimalKineticEnergy "0.00*MeV"
                  
            fi

            if [[ "${TRAIN_OR_VAL}" == "val" ]]
            then

                  ddsim --compactFile $K4GEO/FCCee/IDEA/compact/IDEA_o${OPTION}_v0${VERSION}/IDEA_o${OPTION}_v0${VERSION}.xml \
                        --outputFile out_edm4hep/out_sim_edm4hep_${SEED}.root \
                        --inputFiles out_hepmc/out_${SEED}.hepmc \
                        --numberOfEvents $NEV \
                        --random.seed $SEED \
                        --part.keepAllParticles true \
                        --part.userParticleHandler=''
            fi     
      fi
      rm out_hepmc/out_${SEED}.hepmc
      
      # k4run $WORK_DIR/data_creation/condor_pipeline/IDEA/background/utils/runIDEAv${VERSION}o${OPTION}_trackerDigitizer.py --inputFile out_edm4hep/out_sim_edm4hep_${SEED}.root --outputFile out_digi/output_IDEA_DIGI_${SEED}_${TRAIN_OR_VAL}.root
      # rm out_edm4hep/out_sim_edm4hep_${SEED}.root

      # python $WORK_DIR/data_creation/condor_pipeline/IDEA/background/src/process_tree.py \
      # out_digi/output_IDEA_DIGI_${SEED}_${TRAIN_OR_VAL}.root \
      # "${FULLOUTDIR}/${CONFIG}_graphs_${SEED}_${TRAIN_OR_VAL}.root" \
      # ${VERSION} \
      # ${OPTION}

fi