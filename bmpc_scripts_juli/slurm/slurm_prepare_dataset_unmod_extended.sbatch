#!/bin/bash

#SBATCH --job-name=DLOmix_train
#SBATCH --partition shared-cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --ntasks-per-node=1
#SBATCH --mem=32GB
#SBATCH --time=2-0
#SBATCH --output=/nfs/home/students/j.poschenrieder/MaPra_Wilhelm/dlomix2/bmpc_shared_scripts/prepare_dataset/logs/slurm_%A_%a.log

echo $HOSTNAME
echo $0

cd /nfs/home/students/j.poschenrieder/MaPra_Wilhelm/dlomix2/bmpc_shared_scripts/prepare_dataset

source /nfs/home/students/j.poschenrieder/MaPra_Wilhelm/start_miniconda.sh
conda activate dlomix

python prepare_dataset.py --config config_files/noptm_baseline_full_bs1024_unmod_extended.yaml --num-proc 32
