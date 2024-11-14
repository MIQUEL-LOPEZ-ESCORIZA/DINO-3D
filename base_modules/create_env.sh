#!/bin/bash
#SBATCH --job-name=conda_env_setup
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/ml9715/DINO-3D/slurm/%x-%A-%a-%j.out
#SBATCH --error=/scratch/ml9715/DINO-3D/slurm/%x-%A-%a-%j.err

rm -r /scratch/ml9715/DINO-3D/dinoenv

module load anaconda3/2024.02
conda env create -p /scratch/ml9715/DINO-3D/dinoenv -f /scratch/ml9715/DINO-3D/base_modules/conda_env.yml