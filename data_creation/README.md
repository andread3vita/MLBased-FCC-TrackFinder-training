# How to Create the Dataset

## Prerequisites
To create the dataset for training or validation, ensure that you are working in an environment with access to **HTCondor**.  
This repository is designed to be used with Condor, but you can adapt it to best suit your setup.

### 📦 Condor Jobs
Inside the `condor_pipeline/` directory, you will find several detector configurations.  
You can choose whether to include background or not.  

At the moment, the only fully tested configuration is `IDEA/noBackground`.  

You can run the script `scriptDatasetCreation.sh`, which submits Condor jobs to generate the dataset.
