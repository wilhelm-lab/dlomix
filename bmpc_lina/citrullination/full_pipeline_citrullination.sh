#!/bin/bash

#SBATCH --job-name=citrullination_pipeline_cont
#SBATCH --out=/nfs/home/students/l.willruth/mapra/dlomix/bmpc_lina/citrullination/out_files/R-%x.%j.out
#SBATCH --partition=compms-gpu-a40
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8


# script
source /nfs/home/students/l.willruth/miniconda3/etc/profile.d/conda.sh
conda activate okdl

module load percolator/3.6.1 thermorawfileparser/1.4.3
 
# DO NOT CHANGE THIS, only change with --cpus-per-task above!
NUM_THREADS=$SLURM_CPUS_PER_TASK

# =========== Run oktoberfest =========== #
 
python -m oktoberfest --config_path=/nfs/home/students/l.willruth/mapra/dlomix/bmpc_lina/citrullination/citrullination_configs/tl_citrullination_data2.config.json
