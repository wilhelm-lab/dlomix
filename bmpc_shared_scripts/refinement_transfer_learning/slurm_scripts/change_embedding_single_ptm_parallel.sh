#!/bin/bash

#SBATCH --job-name=bmpc-sweep-finn
#SBATCH --out=R-%x.%j.out
#SBATCH --partition=compms-gpu-a40
#SBATCH --nodes=1
#SBATCH --ntasks=3
#SBATCH --gres=gpu:3
#SBATCH --time=2:00:00

# script
source $HOME/condaInit.sh
conda activate dlomix

for num in {0..2}
do
srun --exact --mpi=pmi2 --ntasks=1 --cpus-per-task=1 --mem=10G --gpus-per-task=1 python ../rl_tl_training_agent.py --cuda-device-nr "0" --config ../config_files/baseline_no_ptm_to_ptm_tum_mod_monomethyl.yaml --sweep-id 7kmwef47 --sweep-count 1 & 
done

wait

