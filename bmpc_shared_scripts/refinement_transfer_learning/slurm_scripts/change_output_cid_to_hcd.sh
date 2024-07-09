#!/bin/bash

#SBATCH --job-name=bmpc-change-output
#SBATCH --out=R-%x.%j.out
#SBATCH --partition=compms-gpu-a40
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1


# script
source $HOME/condaInit.sh
conda activate dlomix

python ../rl_tl_training_agent.py --config ../config_files/noptm_cid_to_hcd.yaml --sweep-id z3g6l7jk --sweep-count 10
