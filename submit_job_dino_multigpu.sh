#!/bin/bash
#SBATCH --job-name=RUN_4
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:v100:2            
#SBATCH --time=48:00:00
#SBATCH --mem-per-cpu=100G
#SBATCH --output=slurm/%x-%A-%a-%j.out
#SBATCH --error=slurm/%x-%A-%a-%j.err

source /share/apps/anaconda3/2024.02/etc/profile.d/conda.sh;
conda activate /scratch/ml9715/CLASSIFIER/myenv
cd /scratch/ml9715/DINO-3D

# debugging flags (optional)
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1

torchrun --nproc_per_node=1 --master_port=$((12000 + $RANDOM % 1000)) train_multigpu_2.py --batch_size_per_gpu 64 --local_crops_number 2 --lr 0.00003 --output_dir "/scratch/ml9715/DINO-3D/checkpoints/RUN_4/"