#!/bin/bash

#SBATCH --job-name=prosit_hcd_only_model_training
#SBATCH --out=R-%x.%j.out
#SBATCH --partition=compms-cpu-big
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --time=10:00:00

# script
source $HOME/condaInit.sh
conda activate dlomix

python ../prepare_dataset.py --config ../config_files/noptm_baseline_full_bs1024_unmod_extended_hcd.yaml
