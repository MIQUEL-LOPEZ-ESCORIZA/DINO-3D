#!/bin/bash
#SBATCH --job-name=vit_extract_embed_val
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
##SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:1
#SBATCH --time=3-00:00:00
#SBATCH --mem-per-cpu=50G
#SBATCH --partition=a100_short
#SBATCH --output=./log/vit_extract_embed_val.out
#SBATCH --error=./log/vit_extract_embed_val.err

# activate conda env
source activate head_ct

module load cuda/11.7
module load gcc/10.2.0

# debugging flags (optional)
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1

torchrun --nnodes 1 --nproc_per_node 1 --master_port 12303 extract_embed.py --local_rank 0 \
    --model_name "vit" --batch_size 32 --num_workers 4 \
    --cfg /gpfs/data/denizlab/Users/hh2740/git_backups/MedSSL-3D/configs/vit/vit_extract_HeadCT.yaml \
    --use_amp --pooling "cls"