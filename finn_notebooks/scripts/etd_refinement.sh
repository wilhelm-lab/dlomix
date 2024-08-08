#!/bin/bash

#SBATCH --job-name=run_etd_refinement
#SBATCH --out=R-%x.%j.out
#SBATCH --partition=compms-gpu-a40
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16


# script
source $HOME/condaInit.sh
conda activate dlomix

python run_etd_refinement.py --parquet /cmnfs/data/proteomics/ProteomeTools/ETD/parquet/etd_data --model_path /cmnfs/proj/bmpc_dlomix/models/refinement_transfer_learning/etd_models/new_etd_model.keras
