#!/bin/bash

#SBATCH --job-name=prosit_cid_only_model_training
#SBATCH --out=R-%x.%j.out
#SBATCH --partition=compms-gpu-a40
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16

# script
source $HOME/condaInit.sh
conda activate dlomix


python ../baseline_model_training.py --config ../config_files/train_etd.yaml
