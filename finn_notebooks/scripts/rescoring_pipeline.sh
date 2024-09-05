#!/bin/bash

#SBATCH --job-name=oktoberfest
#SBATCH --out=R-%x.%j.out
#SBATCH --partition=compms-gpu-a40
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=5
#SBATCH --mem=16G


# script
source $HOME/condaInit.sh
conda activate mapra_pipeline

module load percolator/3.6.1 thermorawfileparser/1.4.3
 
# DO NOT CHANGE THIS, only change with --cpus-per-task above!
NUM_THREADS=$SLURM_CPUS_PER_TASK

python -m oktoberfest --config_path=../configs/rescoring_etd_sage_wo_refinement_improved_model_config.json
