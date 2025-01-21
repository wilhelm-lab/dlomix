#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --job-name=dlomix_train_gpu
#SBATCH --mem=16GB
#SBATCH --partition=shared-gpu
#SBATCH --gpus=1

source $HOME/condaInit.sh

conda run -n bsc python models/dlomix/train_dlomix.py -j models/wandb_configs/prospect_gorshkov_dlomix.json
# location equivalents in dlomix repo:
# models/dlomix/train_dlomix.py  -->  dlomix/src/dlomix/train_dlomix.py
# models/wandb_configs/prospect_gorshkov_dlomix.json  -->  prospect_gorshkov_dlomix.json

