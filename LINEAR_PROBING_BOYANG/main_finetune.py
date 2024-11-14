import os
import json
import random
import argparse
import wandb
import pickle
import numpy as np
from logger import create_logger
from config import get_config

from src.utils.misc import *
from src.utils.pos_embed import interpolate_pos_embed
from src.utils.optimizers import get_optimizer
from src.utils.lr_sched import get_scheduler

from src.data.transforms import vit_transforms, resnet_transforms
from src.data.datasets import get_dataloaders

from src.models.vit import ViT
from src.models.classifier import Linear_Classifier, Attention_Classifier
from src.models.unetr import UNETR_decoder
from src.models import resnet

from monai.config import print_config

from engine_finetune import *

import torch
import torch.nn as nn
import torch.distributed as dist
from monai.losses import DiceCELoss, DiceLoss

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
    
    parser.add_argument("--percent", type=str, help='percent tag for plotting')

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
    parser.add_argument("--lock", action='store_true')

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
    class_idx = config.DATA.CLASS_INDEX
    segmentation_suffix = config.DATA.SEGMENTATION_SUFFIX
    # Create dataloaders
    if 'centercrop' in config.DATA.TRANSFORM:
        imtrans = resnet_transforms(config, mode='train')
        imvals = resnet_transforms(config, mode='val')
        imtests = resnet_transforms(config, mode='test')
    else:
        if max_epochs<1 and ( '_10' not in config.DATA.TEST_CSV_PATH):
            imtrans = vit_transforms(config, mode='train')
            imvals = vit_transforms(config, mode='val')
            imtests = vit_transforms(config, mode='in_the_wild')
        else:
            imtrans = vit_transforms(config, mode='train')
            imvals = vit_transforms(config, mode='val')
            imtests = vit_transforms(config, mode='test')


    augs = [imtrans, imvals, imtests]
    # Create dataloaders
    train_loader, val_loader, test_loader  = get_dataloaders(config, augs, class_idx=class_idx, segmentation_suffix = segmentation_suffix)
    
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
            pooling=config.VIT.POOLING,
        ).to(device)
        classifier = Linear_Classifier(config.VIT.HIDDEN_SIZE * config.VIT.NUM_PATCHES, \
        config.VIT.NUM_CLASSES).to(device)
    elif config.MODEL.NAME == "resnet":
        model = resnet.generate_model(
            model_depth=config.RESNET.DEPTH,
            n_classes=config.RESNET.NUM_CLASSES,
            n_input_channels=config.RESNET.N_INPUT_CHANNELS,
            shortcut_type=config.RESNET.SHORTCUT_TYPE,
            conv1_t_size=config.RESNET.CONV1_T_SIZE,
            conv1_t_stride=config.RESNET.CONV1_T_STRIDE,
            no_max_pool=config.RESNET.NO_MAX_POOL,
            widen_factor=config.RESNET.WIDEN_FACTOR
        ).to(device)
        classifier = Linear_Classifier(config.RESNET.HIDDEN_SIZE * config.RESNET.NUM_PATCHES, \
        config.RESNET.NUM_CLASSES).to(device)
    elif config.MODEL.NAME == "unetr":
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
            pooling=config.VIT.POOLING,
        ).to(device)
        classifier = UNETR_decoder(
            in_channels=config.VIT.IN_CHANS,
            out_channels=config.VIT.NUM_CLASSES,
            img_size=config.VIT.INPUT_SIZE,
            patch_size=config.VIT.PATCH_SIZE,
            feature_size=16,
            hidden_size=config.VIT.HIDDEN_SIZE,
            spatial_dims=config.VIT.SPATIAL_DIMS).to(device)
    else:
        raise ValueError(f"Backbone {config.MODEL.NAME} not supported")
    
    # Load model with wrong size weights unloaded
    if config.MODEL.PRETRAINED != None:
        loaded_state_dict = torch.load(config.MODEL.PRETRAINED, map_location=torch.device('cpu'))['state_dict']
        new_state_dict = {k.replace("module.", ""): v for k, v in loaded_state_dict.items()}
        # interpolate position embedding
        interpolate_pos_embed(model, new_state_dict)
        msg = model.load_state_dict(new_state_dict, strict=False)
        #print('model', model.state_dict().keys(), new_state_dict.keys(), loaded_state_dict.keys())
        logger.info(f"Load Pretrained Model: {msg} for Achitecture: {config.MODEL.NAME}")
        #print(model.state_dict()['blocks.11.attn.qkv.bias'])

        if 'all_gather' in config.MODEL.SAVE_NAME or 'debug' in config.MODEL.SAVE_NAME or 'from_cognitive' in config.MODEL.SAVE_NAME:
            classifier_ckpt_path = os.path.join(config.MODEL.DIR, config.MODEL.PRETRAINED.replace('.pt', '_classifier.pt'))
        else:
            classifier_ckpt_path = os.path.join(config.MODEL.DIR, config.MODEL.SAVE_NAME.replace('.pt', '_classifier.pt'))
        print(config.MODEL.PRETRAINED, classifier_ckpt_path)
        #classifier_ckpt_path = '/gpfs/scratch/by2026/BrainATLAS/mae/PRETRAINED_MODEL/unetr_nacc_full_finetune_patch_T1_mni152_classifier.pt'
        if os.path.exists(classifier_ckpt_path) and 'embedding' not in config.MODEL.SAVE_NAME:
            loaded_state_dict = torch.load(classifier_ckpt_path, map_location=torch.device('cpu'))['state_dict']
            new_state_dict = {k.replace("module.", ""): v for k, v in loaded_state_dict.items()}
            #print('classifier', classifier.state_dict().keys(), new_state_dict.keys(), loaded_state_dict.keys())
            #print(classifier.state_dict()['linear.weight'][0][:5], classifier.state_dict()['ln.weight'][:5])
            #print(new_state_dict['linear.weight'][0][:5], new_state_dict['ln.weight'][:5])
            msg = classifier.load_state_dict(new_state_dict, strict=False)
            #print(classifier.state_dict()['linear.weight'][0][:5], classifier.state_dict()['ln.weight'][:5])
            logger.info(f"Load Pretrained Classifier: {msg} for Achitecture: {config.MODEL.NAME}")
            # print(classifier.state_dict()['blocks.11.attn.qkv.bias'])
        
    # Convert all BatchNorm layers to SyncBatchNorm layers
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    # Use DistributedDataParallel for distributed training
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device], \
        broadcast_buffers=False, find_unused_parameters=True) #find_unused_parameters=False
    print(model)
    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         print(name)
    # Lock backbone
    if config.TRAIN.LOCK:
        if not config.TRAIN.LOCK_LAST_ATTENTION_MODULE:
            set_requires_grad_false_except_last_attention_module(model)
        else:
            set_requires_grad_false(model)
    
    
    # feat_dim = 768
    # d_model = 768
    # n_heads = 12
    # num_layers = 3
    # dim_feedforward = d_model * 4
    
    # classifier = Attention_Classifier(feat_dim, d_model, n_heads, num_layers, dim_feedforward, \
    #     config.VIT.NUM_CLASSES, dropout=0., activation='gelu', norm='BatchNorm').to(device)
    
    # Convert all BatchNorm layers to SyncBatchNorm layers
    classifier = torch.nn.SyncBatchNorm.convert_sync_batchnorm(classifier)
    # Use DistributedDataParallel for distributed training
    classifier = torch.nn.parallel.DistributedDataParallel(classifier, device_ids=[device], \
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
    #models = [model, classifier]
    lr = config.TRAIN.BASE_LR
    optimizer_model = get_optimizer(config, lr, [model])
    optimizer_classifier = get_optimizer(config, lr * 1e2, [classifier])
    optimizers = [optimizer_model, optimizer_classifier]
    # scheduler
    scheduler_model = get_scheduler(config, optimizer_model, num_warmup_steps, total_steps)
    scheduler_classifier = get_scheduler(config, optimizer_classifier, num_warmup_steps, total_steps)
    schedulers = [scheduler_model, scheduler_classifier]
    # loss
    if config.MODEL.NAME == "vit":
        if config.VIT.NUM_CLASSES != 1:#==2:
            criterion = nn.CrossEntropyLoss(ignore_index=-99)
            #criterion = nn.CrossEntropyLoss(weight=torch.tensor([1., 2., 3.]).to(device))
        else:
            criterion = nn.MSELoss()
    elif config.MODEL.NAME == "unetr" or config.MODEL.NAME == "unet":
        criterion = DiceCELoss(to_onehot_y=True,
                               softmax=True,
                               squared_pred=True)
    
    start_epoch = 0
 
    train_loss, best_model, best_classifier = trainer(
        config=config,
        model=model,
        classifier=classifier,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizers=optimizers,
        schedulers=schedulers,
        criterion=criterion,
        start_epoch=start_epoch,
        max_epochs=max_epochs,
        val_every=val_every,
        logger=logger,
        device=device,
        wandb_run=wandb_run,
    )

    logger.info(f"train completed, best train loss: {train_loss:.4f} ")
    logger.info(f"save model/classifier to: {config.MODEL.SAVE_NAME} and {config.MODEL.SAVE_NAME.replace('.pt', '_classifier.pt')}")
    #print(best_model.state_dict()['module.blocks.11.attn.qkv.bias'])
    print(best_classifier.state_dict()['module.linear.weight'][0][:5], classifier.state_dict()['module.ln.weight'][:5])
    save_checkpoint(
        best_model,
        max_epochs,
        optimizers[0],
        schedulers[0],
        best_loss=train_loss,
        dir_add=config.MODEL.DIR,
        filename=config.MODEL.SAVE_NAME,
        logger=logger,
    )
    save_checkpoint(
        best_classifier,
        max_epochs,
        optimizers[1],
        schedulers[1],
        best_loss=train_loss,
        dir_add=config.MODEL.DIR,
        filename=config.MODEL.SAVE_NAME.replace('.pt', '_classifier.pt'),
        logger=logger,
    )
    

    test_loss = tester(
        config=config,
        model=best_model,
        classifier=best_classifier,
        test_loader=test_loader,
        criterion=criterion,
        logger=logger,
        device=device,
        wandb_run=wandb_run,
    )
    
    logger.info(f"test completed, best test loss: {test_loss:.4f} ")


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