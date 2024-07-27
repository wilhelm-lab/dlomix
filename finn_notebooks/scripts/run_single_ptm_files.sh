#!/bin/bash

#SBATCH --job-name=prosit_cid_only_model_training
#SBATCH --out=R-%x.%j.out
#SBATCH --partition=compms-gpu-a40
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1


# script
source $HOME/condaInit.sh
conda activate dlomix

files=(
    "Kmod_Acetyl.parquet"
    "Kmod_Formyl.parquet"
    "Rmod_Citrullin.parquet"
    "Ymod_Nitrotyr.parquet"
    "Pmod_Hydroxypro.parquet"
)

for file in "${files[@]}"; do
    python test_single_small_ptms.py --parquet /cmnfs/data/proteomics/Prosit_PTMs/21PTMs/"$file"
done