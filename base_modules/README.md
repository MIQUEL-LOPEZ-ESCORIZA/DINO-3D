# MedSSL-3D

# To install conda environment in local
```
conda env create -f conda_env.yml
```

# To Run with local environment
```
torchrun --nnodes 1 --nproc_per_node 1 --master_port 12345 main_pretrain.py --local_rank 0 \
        --backbone "vit" --batch_size 8 --num_workers 2 --max_epochs 800 --base_lr 1.5e-4 \
        --cfg /gpfs/data/denizlab/Users/hh2740/git_backups/MedSSL-3D/configs/mae/mae_HeadCT.yaml \
        --use_amp --use_wandb
```

# To Run with Singularity
- Copy container from /gpfs/scratch/huangh13/environment.sif
- Run the following commands in root directory of this repo (--bind dataset folder if it's not in the path of current root directory)
```
module load singularity/3.7.1
singularity exec --nv --bind $(pwd) --bind /gpfs/data/razavianlab/data/headCT environment.sif torchrun --nnodes 1 --nproc_per_node 1 --master_port 12345 main_pretrain.py --local_rank 0 \
    --model_name "mae" --batch_size 64 --num_workers 8 --max_epochs 800 --base_lr 1.6e-3 \
    --cfg ./configs/mae/mae_HeadCT.yaml
```
