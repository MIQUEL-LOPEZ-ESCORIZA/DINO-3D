#!/bin/bash
#SBATCH --job-name=mae_10%_lr1.5e-4_mask0.75_sincos_pflash_ep1600_adamw_clip3.0_gpu4_s42
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
##SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:4
#SBATCH --time=15-00:00:00
#SBATCH --mem-per-cpu=50G
#SBATCH --partition=radiology
#SBATCH --output=./log/mae_10%_lr1.5e-4_mask0.75_sincos_pflash_ep1600_adamw_clip3.0_gpu4_s42.out
#SBATCH --error=./log/mae_10%_lr1.5e-4_mask0.75_sincos_pflash_ep1600_adamw_clip3.0_gpu4_s42.err

# activate conda env
source activate head_ct

module load cuda/11.8
module load gcc/10.2.0

# debugging flags (optional)
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1

torchrun --nnodes 1 --nproc_per_node 4 --master_port 12348 main_pretrain.py --local_rank 0 \
    --model_name "mae" --batch_size 256 --num_workers 16 --max_epochs 1600 --base_lr 1.5e-4 \
    --cfg /gpfs/data/denizlab/Users/hh2740/git_backups/MedSSL-3D/configs/mae/mae_HeadCT.yaml \
    --use_amp --use_wandb --optimizer "AdamW" --scheduler "cosine" --weight_decay 5e-3 \
    --grad_clip 3.0