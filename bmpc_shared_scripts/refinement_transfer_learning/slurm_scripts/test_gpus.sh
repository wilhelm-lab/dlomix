#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --job-name=tf_test
#SBATCH --mem-per-cpu=10000MB
#SBATCH --gpus=1
#SBATCH --partition=compms-gpu-a40
 
source $HOME/condaInit.sh
conda activate dlomix

set -x
which python
pwd
# python test_gpus.py