#!/bin/bash

#SBATCH --job-name=bmpc-sweep-finn
#SBATCH --out=R-%x.%j.out
#SBATCH --partition=compms-gpu-a40
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=2:00:00

# script
source $HOME/condaInit.sh
conda activate dlomix

python ../rl_tl_training_agent.py --config ../config_files/baseline_no_ptm_to_ptm_tum_mod_monomethyl.yaml --sweep-id 7kmwef47 --sweep-count 15 
