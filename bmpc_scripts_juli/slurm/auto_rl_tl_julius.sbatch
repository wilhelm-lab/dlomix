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

python usage_example_automatic_rl_tl_parquet.py \
    --dataset-path /cmnfs/home/students/j.schlensok/oktoberfest/tutorials/out/data/dlomix/refinement_dataset/processed_dataset.parquet \
    --model-path /cmnfs/proj/bmpc_dlomix/models/baseline_models/noptm_baseline_full_bs1024_unmod_extended/7ef3360f-2349-46c0-a905-01187d4899e2.keras \
    --modifications A C[UNIMOD:4] D E F G H I K L M N P Q R S T V W Y M[UNIMOD:35]
