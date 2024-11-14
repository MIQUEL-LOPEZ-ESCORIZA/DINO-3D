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
from src.utils.optimizers import get_optimizer
from src.utils.lr_sched import get_scheduler

from src.data.datasets import get_dataloaders
from src.data.transforms import mae3d_transforms

from src.models.mae import MaskedAutoencoderViT
from src.models.vae import VAE

from monai.config import print_config

from engine_pretrain import *

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
    parser.add_argument("--optimizer", type=str, help='training optimizer')
    parser.add_argument("--scheduler", type=str, help='learning rate scheduler')
    parser.add_argument("--base_lr", type=float, help='base learning rate')
    parser.add_argument("--min_lr", type=float, help='minimum learning rate')
    parser.add_argument("--weight_decay", type=float, help='max epoch')
    parser.add_argument("--grad_clip", type=float, help='gradient clipping')
    parser.add_argument("--batch_size", type=int, help='batch size')
    parser.add_argument("--num_workers", type=int, help='number of workers for dataloader')
    parser.add_argument("--max_epochs", type=int, help='max epoch')

    # dataset parameters
    parser.add_argument('--train_csv_path', type=str, help='path to train csv file')
    parser.add_argument('--val_csv_path', type=str, help='path to val csv file')
    parser.add_argument('--test_csv_path', type=str, help='path to test csv file')

    args, unparsed = parser.parse_known_args()

    config = get_config(args)

    return args, config


def main(config, wandb_run):
    # Define parameters
    max_epochs = config.TRAIN.MAX_EPOCHS
    val_every = config.TRAIN.VAL_EVERY

    # Define transforms for image and segmentation
    imtrans = mae3d_transforms(config, mode='train')
    imvals = mae3d_transforms(config, mode='val')
    imtests = mae3d_transforms(config, mode='test')
    
    augs = [imtrans, imvals, imtests]
    
    # Create dataloaders
    train_loader, val_loader, test_loader  = get_dataloaders(config, augs)

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if config.MODEL.NAME == "mae":
        model = MaskedAutoencoderViT(
            input_size=config.MAE.INPUT_SIZE,
            patch_size=config.MAE.PATCH_SIZE,
            mask_ratio=config.MAE.MASK_RATIO,
            in_chans=config.MAE.IN_CHANS,
            dropout_rate=config.MAE.DROPOUT_RATE,
            spatial_dims=config.MAE.SPATIAL_DIMS,
            patch_embed=config.MAE.PATCH_EMBED,
            pos_embed=config.MAE.POS_EMBED,
            encoder_depth=config.MAE.ENCODER_DEPTH,
            encoder_embed_dim=config.MAE.ENCODER_EMBED_DIM,
            encoder_mlp_dim=config.MAE.ENCODER_MLP_DIM,
            encoder_num_heads=config.MAE.ENCODER_NUM_HEADS,
            decoder_depth=config.MAE.DECODER_DEPTH,
            decoder_embed_dim=config.MAE.DECODER_EMBED_DIM,
            decoder_mlp_dim=config.MAE.DECODER_MLP_DIM,
            decoder_num_heads=config.MAE.DECODER_NUM_HEADS,
            norm_pix_loss=config.MAE.NORM_PIX_LOSS,
            use_bias=config.MAE.USE_BIAS,
            use_flash_attn=config.MAE.USE_FLASH_ATTN,
        ).to(device)
    elif config.MODEL.NAME == "vae":
        model = VAE(
            latent_dim=config.VAE.LATENT_DIM
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
    # calculate effective total training steps in distributed setting
    effective_batch_size = config.DATA.BATCH_SIZE * world_size
    total_steps = len(train_loader) * config.TRAIN.MAX_EPOCHS
    # warmup and scheduler setup
    num_warmup_steps = (int)(config.TRAIN.PER_WARMUP * total_steps)
    
    # learning rate scaling
    config.defrost()
    config.TRAIN.BASE_LR = config.TRAIN.BASE_LR * effective_batch_size / 256
    config.TRAIN.MIN_LR = config.TRAIN.BASE_LR * 1e-3
    config.freeze()

    logger.info(f"Effective Learning Rate: {config.TRAIN.BASE_LR}, Effective Batch Size: {effective_batch_size}, Max Epochs: {config.TRAIN.MAX_EPOCHS}")
    logger.info(f"Number of Warmup Steps: {num_warmup_steps}, Total Steps: {total_steps}")

    # optimizer
    optimizer = get_optimizer(config, [model])
    # scheduler
    scheduler = get_scheduler(config, optimizer, num_warmup_steps, total_steps)

    start_epoch = 0
 
    train_loss = trainer(
        config=config,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        start_epoch=start_epoch,
        max_epochs=max_epochs,
        val_every=val_every,
        logger=logger,
        device=device,
        wandb_run=wandb_run,
    )

    logger.info(f"train completed, best train reconstruction loss: {train_loss:.4f} ")

    test_loss = tester(
        config=config,
        model=model,
        test_loader=test_loader,
        logger=logger,
        device=device,
        wandb_run=wandb_run,
    )
    
    logger.info(f"test completed, best test reconstruction loss: {test_loss:.4f} ")


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