#!/bin/bash

#SBATCH --job-name=dlomix_test_logging
#SBATCH --out=R-%x.%j.out
#SBATCH --partition=compms-gpu-a40
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16

# script
source /nfs/home/students/s.baier/miniconda3/etc/profile.d/conda.sh
conda activate okt_dlomix
python /nfs/home/students/s.baier/mapra/dlomix/bmpc_shared_scripts/refinement_transfer_learning/usage_example_automatic_refinement_learning.py

