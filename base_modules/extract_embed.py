import os
import json
import random
import argparse
import wandb
import numpy as np
from logger import create_logger
from config import get_config

from src.utils.misc import *
from src.utils.pos_embed import interpolate_pos_embed

from src.data.datasets import get_one_dataloader
from src.data.transforms import vit_transforms

from src.models.vit import ViT

from monai.config import print_config

from engine_extract import extract_embedding

import torch
import torch.distributed as dist

print_config()

def parse_option():
    parser = argparse.ArgumentParser('MONAI training and evaluation script', add_help=False)
    parser.add_argument('--cfg', type=str, required=True, metavar="FILE", help='path to config file')
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line",
        default=None,
        nargs='+',
    )

    # distributed training
    parser.add_argument("--local_rank", type=int, required=True, help='local rank for DistributedDataParallel')
    parser.add_argument('--dist-backend', default='nccl', help='used to set up distributed backend')
    parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')
    parser.add_argument("--seed", type=int, help='seed')
    parser.add_argument("--use_amp", action='store_true')

    # wandb configs
    parser.add_argument("--use_wandb", action='store_true')
    parser.add_argument("--wandb_project", type=str, default="monai-test")

    # model parameters
    parser.add_argument("--model_name", type=str, help='model name')
    parser.add_argument("--model_load_path", type=str, help='path to trained model')
    parser.add_argument("--batch_size", type=int, help='batch size')
    parser.add_argument("--num_workers", type=int, help='number of workers for dataloader')
    parser.add_argument("--max_epochs", type=int, help='max epoch')
    parser.add_argument("--pooling", type=str, help='pooling method for vit representation')

    # dataset parameters
    parser.add_argument('--train_csv_path', type=str, help='path to train csv file')
    parser.add_argument('--val_csv_path', type=str, help='path to val csv file')
    parser.add_argument('--test_csv_path', type=str, help='path to test csv file')

    args, unparsed = parser.parse_known_args()

    config = get_config(args)

    return args, config


def main(config, wandb_run):
    # Define transforms for image and segmentation
    aug = vit_transforms(config, mode='test')
    
    # Create dataloaders
    data_loader = get_one_dataloader(config, aug)

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if config.MODEL.NAME == "vit":
        model = ViT(
            img_size=config.VIT.INPUT_SIZE,
            patch_size=config.VIT.PATCH_SIZE,
            hidden_size=config.VIT.HIDDEN_SIZE,
            mlp_dim=config.VIT.MLP_DIM,
            num_layers=config.VIT.NUM_LAYERS,
            num_heads=config.VIT.NUM_HEADS,
            in_chans=config.VIT.IN_CHANS,
            dropout_rate=config.VIT.DROPOUT_RATE,
            spatial_dims=config.VIT.SPATIAL_DIMS,
            patch_embed=config.VIT.PATCH_EMBED,
            pos_embed=config.VIT.POS_EMBED,
            classification=config.VIT.CLASSIFICATION,
            num_classes=config.VIT.NUM_CLASSES,
            post_activation="Tanh",
            qkv_bias=config.VIT.USE_BIAS,
            use_flash_attn=config.VIT.USE_FLASH_ATTN,
        ).to(device)
    else:
        raise ValueError(f"Backbone {config.MODEL.NAME} not supported")
    
    # Load model with wrong size weights unloaded
    if config.MODEL.PRETRAINED != None:
        loaded_state_dict = torch.load(config.MODEL.PRETRAINED, map_location=torch.device('cpu'))['state_dict']
        #current_model_dict = model.state_dict()
        # new_state_dict = {k:v if v.size()==current_model_dict[k].size() else current_model_dict[k] \
        #                 for k,v in zip(current_model_dict.keys(), loaded_state_dict.values())}
        new_state_dict = {k.replace("module.", ""): v for k, v in loaded_state_dict.items()}
        # interpolate position embedding
        interpolate_pos_embed(model, new_state_dict)
        msg = model.load_state_dict(new_state_dict, strict=False)
        logger.info(f"Load Pretrained Model: {msg} for Achitecture: {config.MODEL.NAME}")
        
    # Convert all BatchNorm layers to SyncBatchNorm layers
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    # Use DistributedDataParallel for distributed training
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device], \
        broadcast_buffers=False, find_unused_parameters=False)
    
    torch.backends.cudnn.benchmark = True
    
    world_size = dist.get_world_size()
    use_amp = config.AMP_ENABLE
    pooling = config.VIT.POOLING
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    # calculate effective total training steps in distributed setting
    effective_batch_size = config.DATA.BATCH_SIZE * world_size

    logger.info(f"Effective Batch Size: {effective_batch_size}")
 
    extract_embedding(
        config=config,
        model=model,
        data_loader=data_loader,
        pooling=pooling,
        logger=logger,
        device=device,
        use_amp=use_amp,
        scaler=scaler,
    )

    logger.info(f"Extracting Embeddings Done, Saved ...")


def init_seed(seed):
    random_seed = seed
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)


if __name__ == "__main__":
    args, config = parse_option()
    # init distributed training
    init_distributed_mode(args)
    seed = config.SEED + dist.get_rank()
    init_seed(seed)
    # create logger
    logger = create_logger(output_dir='./log', dist_rank=dist.get_rank(), name=config.LOG.FILENAME)

    # output config settings
    if dist.get_rank() == 0:
        path = os.path.join(config.OUTPUT, f"{config.LOG.FILENAME}.json")
        with open(path, "w") as f:
            f.write(config.dump())
        logger.info(f"Full config saved to {path}")

    # print config
    logger.info(config.dump())
    logger.info(json.dumps(vars(args)))

    # init wandb
    wandb_run = None
    if config.WANDB.WANDB_ENABLE and dist.get_rank() == 0:
        wandb_run = wandb.init(
                # Set the project where this run will be logged
                name = config.LOG.FILENAME,
                project=config.WANDB.PROJECT,
                # Track hyperparameters and run metadata
                config={
                    "learning_rate": config.TRAIN.BASE_LR,
                    "batch_size": config.DATA.BATCH_SIZE,
                    "epochs": config.TRAIN.MAX_EPOCHS,
                    "backbone": config.MODEL.NAME,
                }
            )

    # run main training
    main(config, wandb_run)