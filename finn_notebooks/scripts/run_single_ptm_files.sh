#!/bin/bash

#SBATCH --job-name=prosit_single_ptm_transfer
#SBATCH --out=R-%x.%j.out
#SBATCH --partition=compms-gpu-a40
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8

# script
source $HOME/condaInit.sh
conda activate dlomix

files=(
    #"Kmod_Acetyl.parquet"
    #"Kmod_Formyl.parquet"
    #"Rmod_Citrullin.parquet"
    #"Ymod_Nitrotyr.parquet"
    #"Pmod_Hydroxypro.parquet"
    #"TUM_mod_acetylated.parquet"
    "TUM_mod_citrullination_2.parquet"
)

for file in "${files[@]}"; do
    python run_single_ptm_transfer.py --parquet /cmnfs/data/proteomics/Prosit_PTMs/"$file" --model_path /cmnfs/proj/bmpc_dlomix/models/refinement_transfer_learning/single_ptm_models/refined_to_"${file%.*}".keras --improve
done
