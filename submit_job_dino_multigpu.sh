#!/bin/bash
#SBATCH --job-name=dino-pretrain
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:rtx8000:4             # Request 4 V100 GPUs
#SBATCH --time=48:00:00
#SBATCH --mem-per-cpu=70G
#SBATCH --output=slurm/%x-%A-%a-%j.out
#SBATCH --error=slurm/%x-%A-%a-%j.err

source /share/apps/anaconda3/2020.07/etc/profile.d/conda.sh;
conda activate /scratch/ml9715/CLASSIFIER/myenv
cd /scratch/ml9715/DINO-3D

# debugging flags (optional)
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1

# Run with 4 GPUs
torchrun --nproc_per_node=4 --master_port 12346 train_multigpu.py --batch_size_per_gpu 128 --local_crops_number 0 --lr 0.00003 --output_dir "./checkpoints/RUN_3/"