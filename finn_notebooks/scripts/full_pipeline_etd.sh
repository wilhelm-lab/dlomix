#!/bin/bash

#SBATCH --job-name=oktoberfest_w_dlomix
#SBATCH --out=R-%x.%j.out
#SBATCH --partition=compms-gpu-a40
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8


# script
source $HOME/condaInit.sh
conda activate mapra_pipeline

module load percolator/3.6.1 thermorawfileparser/1.4.3
 
# DO NOT CHANGE THIS, only change with --cpus-per-task above!
NUM_THREADS=$SLURM_CPUS_PER_TASK

# =========== Run oktoberfest =========== #
 
python -m oktoberfest --config_path=../configs/refinement_etd_config_3.json
