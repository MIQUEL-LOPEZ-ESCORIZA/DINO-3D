#!/bin/bash

#SBATCH --job-name=all_training_dataset_validation
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h100             
#SBATCH --time=48:00:00
#SBATCH --mem-per-cpu=100G
#SBATCH --output=slurm/%x-%j.out  
#SBATCH --error=slurm/%x-%j.err  

source /share/apps/anaconda3/2020.07/etc/profile.d/conda.sh;
conda activate /scratch/ml9715/CLASSIFIER/myenv
cd /scratch/ml9715/DINO-3D/LINEAR_PROBIBNG

# debugging flags (optional)
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1


torchrun --nproc_per_node=1 --master_port=$((12000 + $RANDOM % 1000)) validation.py 