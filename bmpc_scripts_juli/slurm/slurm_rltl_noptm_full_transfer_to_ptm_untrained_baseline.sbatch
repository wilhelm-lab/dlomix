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
#SBATCH --output=/nfs/home/students/j.poschenrieder/MaPra_Wilhelm/dlomix2/bmpc_shared_scripts/refinement_transfer_learning/logs/slurm_%A_%a.log

echo $HOSTNAME
echo $0

cd /nfs/home/students/j.poschenrieder/MaPra_Wilhelm/dlomix2/bmpc_shared_scripts/refinement_transfer_learning

source /nfs/home/students/j.poschenrieder/MaPra_Wilhelm/start_miniconda.sh
conda activate dlomix

python rl_tl_training.py \
    --config config_files/noptm_baseline_full_bs1024_unmod_extended_transfer_to_ptm_untrained_baseline.yaml \
    --cuda-device-nr 0 \
    --cpu-threads 16


