#!/bin/bash
#SBATCH --job-name=vae_test_lr1.6e-3
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:v100:1
#SBATCH --time=12:00:00
#SBATCH --mem-per-cpu=50G
#SBATCH --partition=gpu8_short
#SBATCH --output=./log/vae_test_lr1.6e-3.out
#SBATCH --error=./log/vae_test_lr1.6e-3.err

# activate conda env
source activate head_ct

# debugging flags (optional)
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1

torchrun --nnodes 1 --nproc_per_node 1 --master_port 12340 main_pretrain.py --local_rank 0 \
    --model_name "vae" --batch_size 8 --num_workers 4 --max_epochs 100 --base_lr 1e-2 \
    --cfg /gpfs/data/denizlab/Users/hh2740/git_backups/MedSSL-3D/configs/vae/vae_HeadCT.yaml \
    --use_wandb