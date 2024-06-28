#!/bin/bash

#SBATCH --job-name=DLOmix_train
#SBATCH --partition shared-gpu
#SBATCH --nodelist=compms-gpu-1.exbio.wzw.tum.de,compms-gpu-2.exbio.wzw.tum.de
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --ntasks-per-node=1
#SBATCH --mem=32GB
#SBATCH --time=2-0
#SBATCH --output=/nfs/home/students/j.poschenrieder/MaPra_Wilhelm/dlomix2/bmpc_shared_scripts/baseline_training/logs/slurm_%A_%a.log

echo $HOSTNAME
echo $0

cd /nfs/home/students/j.poschenrieder/MaPra_Wilhelm/dlomix2/bmpc_shared_scripts/baseline_training

source /nfs/home/students/j.poschenrieder/MaPra_Wilhelm/start_miniconda.sh
conda activate dlomix

python baseline_model_training.py --config config_files/noptm_baseline_full_bs1024_unmod_extended_full_train3.yaml --cuda-device-nr 0
