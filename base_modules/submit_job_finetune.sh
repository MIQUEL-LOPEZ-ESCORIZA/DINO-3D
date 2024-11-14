#!/bin/bash
#SBATCH --job-name=vit_finetune_stroke_100%
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
##SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:1
#SBATCH --time=2-00:00:00
#SBATCH --mem-per-cpu=50G
#SBATCH --partition=a100_long
#SBATCH --output=./log/vit_finetune_stroke_100%.out
#SBATCH --error=./log/vit_finetune_stroke_100%.err

# activate conda env
source activate head_ct

module load cuda/11.8
module load gcc/10.2.0

# debugging flags (optional)
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1

torchrun --nnodes 1 --nproc_per_node 1 --master_port 12345 main_finetune.py --local_rank 0 \
    --model_name "vit" --batch_size 16 --num_workers 4 --max_epochs 10 --base_lr 1.0e-5 \
    --cfg /gpfs/data/denizlab/Users/hh2740/git_backups/MedSSL-3D/configs/vit/vit_finetune_HeadCT.yaml \
    --use_amp --optimizer "AdamW" --scheduler "cosine" --weight_decay 0.01 \
    --grad_clip 3.0 --percent "100%"