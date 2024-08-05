#!/bin/bash

#SBATCH --job-name=DLOmix_train
#SBATCH --partition shared-gpu
#SBATCH --gres=gpu:a40:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --ntasks-per-node=1
#SBATCH --mem=32GB
#SBATCH --time=2-0
#SBATCH --output=/nfs/home/students/j.poschenrieder/MaPra_Wilhelm/logs/slurm_%A_%a.log

echo $HOSTNAME
echo $0


source /nfs/home/students/j.poschenrieder/MaPra_Wilhelm/start_miniconda.sh
conda activate dlomix

cd /nfs/home/students/j.poschenrieder/MaPra_Wilhelm/dlomix2
pip install -e .

cd /nfs/home/students/j.poschenrieder/MaPra_Wilhelm/dlomix2/bmpc_scripts_juli

python usage_example_automatic_rl_tl.py \
    --dataset-path /cmnfs/proj/bmpc_dlomix/datasets/processed/noptm_baseline_full_bs1024_unmod_extended
