#!/bin/bash

#SBATCH --job-name=prosit-train-etd
#SBATCH --out=R-%x.%j.out
#SBATCH --partition=compms-gpu-a40
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16


# script
source $HOME/condaInit.sh
conda activate dlomix

python train_etd_from_scratch.py --parquet /cmnfs/proj/bmpc_dlomix/oktoberfest_output/refinement_etd_out/data/dlomix/refinement_dataset/processed_dataset.parquet --model_path /cmnfs/proj/bmpc_dlomix/models/refinement_transfer_learning/etd_models/etd_from_scratch.keras
