
# Modified from https://github.com/jankrepl/mildlyoverfitted/tree/master/github_adventures/dino
import sys
import os



# Add the `src` directory to the Python path
sys.path.append(os.path.abspath("/scratch/ml9715/DINO-3D/base_modules"))

sys.path.append(os.path.abspath("/scratch/ml9715/DINO-3D"))

import wandb
import math
import datetime
import os
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import torch.multiprocessing as mp
import time
from dataloader_mri import HeadDataset
from augmentations import DataAugmentationDINO3D, MY_AUGMENTATION
from utils_multigpu import DINOHead, Loss, clip_gradients, get_dataloaders, init_distributed_mode, MultiCropWrapper
from src.models.vit import ViT
import argparse
import json
import pathlib
import pickle
from pathlib import Path
import numpy as np
import timm
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torchvision.transforms as transforms
import tqdm
from torch.utils.data import DataLoader, SubsetRandomSampler
#from torch.utils.tensorboard import SummaryWriter
from torchvision.datasets import ImageFolder
import utils
sys.path.append("/scratch/ml9715/DINO-3D/LINEAR_PROBING/")

from save_features import compute_features
sys.path.append("/../../../DINO3D/base_modules/")
# from utils_multigpu import ViT


def get_args_parser():
    parser = argparse.ArgumentParser('DINO', add_help=False)

    # Model parameters
    # parser.add_argument('--arch', default='vit_small', type=str,
    #     choices=['vit_tiny', 'vit_small', 'vit_base', 'xcit', 'deit_tiny', 'deit_small'] \
    #             + torchvision_archs + torch.hub.list("facebookresearch/xcit:main"),
    #     help="""Name of architecture to train. For quick experiments with ViTs,
    #     we recommend using vit_tiny or vit_small.""")
    parser.add_argument('--patch_size', default=16, type=int, help="""Size in pixels
        of input square patches - default 16 (for 16x16 patches). Using smaller
        values leads to better performance but requires more memory. Applies only
        for ViTs (vit_tiny, vit_small and vit_base). If <16, we recommend disabling
        mixed precision training (--use_fp16 false) to avoid unstabilities.""")
    parser.add_argument('--out_dim', default=65536, type=int, help="""Dimensionality of
        the DINO head output. For complex and large datasets large values (like 65k) work well.""")
    parser.add_argument('--norm_last_layer', default=True, type=utils.bool_flag,
                        help="""Whether or not to weight normalize the last layer of the DINO head.
        Not normalizing leads to better performance but can make the training unstable.
        In our experiments, we typically set this paramater to False with vit_small and True with vit_base.""")
    parser.add_argument('--momentum_teacher', default=0.996, type=float, help="""Base EMA
        parameter for teacher update. The value is increased to 1 during training with cosine schedule.
        We recommend setting a higher value with small batches: for example use 0.9995 with batch size of 256.""")
    parser.add_argument('--use_bn_in_head', default=False, type=utils.bool_flag,
                        help="Whether to use batch normalizations in projection head (Default: False)")

    # Temperature teacher parameters
    parser.add_argument('--warmup_teacher_temp', default=0.04, type=float,
                        help="""Initial value for the teacher temperature: 0.04 works well in most cases.
        Try decreasing it if the training loss does not decrease.""")
    parser.add_argument('--teacher_temp', default=0.04, type=float, help="""Final value (after linear warmup)
        of the teacher temperature. For most experiments, anything above 0.07 is unstable. We recommend
        starting with the default value of 0.04 and increase this slightly if needed.""")
    parser.add_argument('--warmup_teacher_temp_epochs', default=0, type=int,
                        help='Number of warmup epochs for the teacher temperature (Default: 30).')

    # Training/Optimization parameters
    parser.add_argument('--use_fp16', type=utils.bool_flag, default=True, help="""Whether or not
        to use half precision for training. Improves training time and memory requirements,
        but can provoke instability and slight decay of performance. We recommend disabling
        mixed precision if the loss is unstable, if reducing the patch size or if training with bigger ViTs.""")
    parser.add_argument('--weight_decay', type=float, default=0.04, help="""Initial value of the
        weight decay. With ViT, a smaller value at the beginning of training works well.""")
    parser.add_argument('--weight_decay_end', type=float, default=0.4, help="""Final value of the
        weight decay. We use a cosine schedule for WD and using a larger decay by
        the end of training improves performance for ViTs.""")
    parser.add_argument('--clip_grad', type=float, default=3.0, help="""Maximal parameter
        gradient norm if using gradient clipping. Clipping with norm .3 ~ 1.0 can
        help optimization for larger ViT architectures. 0 for disabling.""")
    parser.add_argument('--batch_size_per_gpu', default=32, type=int,
                        help='Per-GPU batch-size : number of distinct images loaded on one GPU.')#32
    parser.add_argument('--epochs', default=200, type=int, help='Number of epochs of training.') #1001
    parser.add_argument('--freeze_last_layer', default=1, type=int, help="""Number of epochs
        during which we keep the output layer fixed. Typically doing so during
        the first epoch helps training. Try increasing this value if the loss does not decrease.""")
    parser.add_argument("--lr", default=0.0001, type=float, help="""Learning rate at the end of
        linear warmup (highest LR used during training). The learning rate is linearly scaled
        with the batch size, and specified here for a reference batch size of 256.""")
    parser.add_argument("--warmup_epochs", default=4, type=int,
                        help="Number of epochs for the linear learning-rate warm up.")
    parser.add_argument('--min_lr', type=float, default=3e-6, help="""Target LR at the
        end of optimization. We use a cosine LR schedule with linear warmup.""")
    parser.add_argument('--optimizer', default='adamw', type=str,
                        choices=['adamw', 'sgd', 'lars'], help="""Type of optimizer. We recommend using adamw with ViTs.""")
    parser.add_argument('--drop_path_rate', type=float, default=0.1, help="stochastic depth rate")

    # Multi-crop parameters
    parser.add_argument('--global_crops_scale', type=float, nargs='+', default=(0.4, 1.),
                        help="""Scale range of the cropped image before resizing, relatively to the origin image.
        Used for large global view cropping. When disabling multi-crop (--local_crops_number 0), we
        recommand using a wider range of scale ("--global_crops_scale 0.14 1." for example)""")
    parser.add_argument('--local_crops_number', type=int, default=0, help="""Number of small
        local views to generate. Set this parameter to 0 to disable multi-crop training.
        When disabling multi-crop we recommend to use "--global_crops_scale 0.14 1." """)
    parser.add_argument('--local_crops_scale', type=float, nargs='+', default=(0.05, 0.4),
                        help="""Scale range of the cropped image before resizing, relatively to the origin image.
        Used for small local view cropping of multi-crop.""")

    # Misc
    parser.add_argument('--data_path', default='/path/to/imagenet/train/', type=str,
                        help='Please specify path to the ImageNet training data.')
    parser.add_argument('--output_dir', default="./checkpoints/", type=str,
                        help='Path to save logs and checkpoints.')
    parser.add_argument('--saveckp_freq', default=5, type=int,
                        help='Save checkpoint every x epochs.')
    parser.add_argument('--seed', default=0, type=int, help='Random seed.')
    parser.add_argument('--num_workers', default=6, type=int,
                        help='Number of data loading workers per GPU.')
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
        distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument("--local_rank", default=0, type=int,
                        help="Please ignore and do not set this argument.")
    parser.add_argument('--model', default='ViT3D', type=str,)

    return parser


def train_dino(args):
    utils.init_distributed_mode(args)
    utils.fix_random_seeds(args.seed)
    # print("git:\n  {}\n".format(utils.get_sha()))
    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    # cudnn.benchmark = True

    # csv_path = "/gpfs/home/ds5396/scratch/MedSSL-3D/datasets/debug.csv"
    # csv_path = "./dataset/train.csv"
    # csv_path = "/gpfs/scratch/huangh13/datasets/train_denoise.csv"
    # csv_path = "/gpfs/scratch/huangh13/datasets_large_exclude_errors/train_large_denoise.csv"
    # csv_path = "/gpfs/data/razavianlab/data/headCT/split_final/train_final_corrected.csv"
    #csv_path = "/gpfs/scratch/huangh13/datasets/train_final_label_10%.csv"
    csv_path = "/scratch/by2026/BrainATLAS/mae/mae/data_csv/mae_with_UKB_train.csv"  
    
    # LOAD DATASETS AND DATALOADERS
    #data_augmentation_3d = DataAugmentation3D()
    # data_augmentation_3d = DataAugmentationDINO3D(global_crops_size=(96, 96, 96),
    #                            local_crops_size=(96, 96, 96),
    #                            global_crops_scale=(0.8, 1.),
    #                            local_crops_scale=(0.4, 0.7),
    #                            local_crops_number=args.local_crops_number)

    # data_augmentation_3d = DataAugmentationDINO3D(
    #     final_size=(96, 96, 96),
    #     local_crops_size=(64, 64, 64),
    #     global_crops_size=(112, 112, 112),
    #     local_crops_number=args.local_crops_number)

    # data_augmentation_3d = DataAugmentationDINO3D(
    #     final_size=(128, 128, 128),
    #     local_crops_size=(96, 96, 96),
    #     global_crops_size=(192, 192, 192),
    #     local_crops_number=args.local_crops_number)
    data_augmentation_3d = MY_AUGMENTATION(
        final_size=(128, 128, 128),
        local_crops_size=(64,64,64),
        global_crops_size=(128, 128, 128),
        local_crops_number=args.local_crops_number)
    
    # dataset = HeadDataset(csv_path, data_augmentation_3d)
    dataset = HeadDataset(csv_path, data_augmentation_3d,
                          cache_dir="/scratch/ml9715/DINO-3D/temp_cache_large_3_channel/")
    sampler = torch.utils.data.DistributedSampler(
        dataset, num_replicas=utils.get_world_size(), rank=utils.get_rank(), shuffle=True)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=False,
        drop_last=True,
    )
    print(f"Data loaded: there are {len(dataset)} images.")

    # student_vit = ViT(input_size=96, patch_size=16)
    # teacher_vit = ViT(input_size=96, patch_size=16)
    # student_vit = ViT(in_chans=1, img_size=(96, 96, 96), patch_size=(16, 16, 16), patch_embed='conv', pos_embed='learnable')
    # teacher_vit = ViT(in_chans=1, img_size=(96, 96, 96), patch_size=(16, 16, 16), patch_embed='conv', pos_embed='learnable')
    # student_vit = ViT(in_chans=1, img_size=(96, 96, 96), patch_size=(12, 12, 12),
    #                   patch_embed='conv', pos_embed='sincos', use_flash_attn=True)
    # teacher_vit = ViT(in_chans=1, img_size=(96, 96, 96), patch_size=(12, 12, 12),
    #                   patch_embed='conv', pos_embed='sincos', use_flash_attn=True)

    student_vit = ViT(in_chans=1, img_size=(128, 128, 128), patch_size=(16, 16, 16),
                      patch_embed='conv', pos_embed='sincos', use_flash_attn=False)
    teacher_vit = ViT(in_chans=1, img_size=(128, 128, 128), patch_size=(16, 16, 16),
                      patch_embed='conv', pos_embed='sincos', use_flash_attn=False)
    embed_dim = 768
    print("student vit initialiized")
    print(student_vit)
    student = MultiCropWrapper(model=student_vit, head=DINOHead(
        embed_dim,
        args.out_dim,
        use_bn=args.use_bn_in_head,
        norm_last_layer=args.norm_last_layer,
    ))
    teacher = MultiCropWrapper(model=teacher_vit, head=DINOHead(
        embed_dim,
        args.out_dim,
        use_bn=args.use_bn_in_head,
        norm_last_layer=args.norm_last_layer,
    ))
    print("student MultiCropWrapper initialiized")
    print(student)
    
    # move networks to gpu
    student, teacher = student.cuda(), teacher.cuda()
    if utils.has_batchnorms(student):
        print("student before in utils.has_batchnorms(student) nn.parallel.DistributedDataParalled")
    
        student = nn.SyncBatchNorm.convert_sync_batchnorm(student)
        teacher = nn.SyncBatchNorm.convert_sync_batchnorm(teacher)
        print("student after in utils.has_batchnorms(student) nn.parallel.DistributedDataParalled")
    

        # we need DDP wrapper to have synchro batch norms working...
        teacher = nn.parallel.DistributedDataParallel(
            teacher, device_ids=[args.gpu], find_unused_parameters=True)
        teacher_without_ddp = teacher.module
    else:
        # teacher_without_ddp and teacher are the same thing
        teacher_without_ddp = teacher
    
    print("student before  nn.parallel.DistributedDataParalled")
    student = nn.parallel.DistributedDataParallel(
        student, device_ids=[args.gpu], find_unused_parameters=True)
    print("student after  nn.parallel.DistributedDataParalled")
    # teacher and student start with the same weights
    teacher_without_ddp.load_state_dict(student.module.state_dict())
    # there is no backpropagation through the teacher, so no need for gradients
    for p in teacher.parameters():
        p.requires_grad = False
    print(f"Student and Teacher are built: they are both ViT network.")

    dino_loss = DINOLoss(
        args.out_dim,
        args.local_crops_number + 2,  # total number of crops = 2 global crops + local_crops_number
        args.warmup_teacher_temp,
        args.teacher_temp,
        args.warmup_teacher_temp_epochs,
        args.epochs,
    ).cuda()

    # ============ preparing optimizer ... ============
    params_groups = utils.get_params_groups(student)
    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(params_groups)  # to use with ViTs
    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(params_groups, lr=0, momentum=0.9)  # lr is set by scheduler
    elif args.optimizer == "lars":
        optimizer = utils.LARS(params_groups)  # to use with convnet and large batches
    # for mixed precision training
    fp16_scaler = None
    if args.use_fp16:
        fp16_scaler = torch.cuda.amp.GradScaler()

    # ============ init schedulers ... ============
    lr_schedule = utils.cosine_scheduler(
        args.lr * (args.batch_size_per_gpu * utils.get_world_size()) / 256.,  # linear scaling rule
        args.min_lr,
        args.epochs, len(data_loader),
        warmup_epochs=args.warmup_epochs,
    )
    wd_schedule = utils.cosine_scheduler(
        args.weight_decay,
        args.weight_decay_end,
        args.epochs, len(data_loader),
    )
    # momentum parameter is increased to 1. during training with a cosine schedule
    momentum_schedule = utils.cosine_scheduler(args.momentum_teacher, 1,
                                               args.epochs, len(data_loader))
    print(f"Loss, optimizer and schedulers ready.")

    # ============ optionally resume training ... ============
    to_restore = {"epoch": 0}
    # utils.restart_from_checkpoint(
    #     os.path.join(args.output_dir, "checkpoint.pth"),
    #     run_variables=to_restore,
    #     student=student,
    #     teacher=teacher,
    #     optimizer=optimizer,
    #     fp16_scaler=fp16_scaler,
    #     dino_loss=dino_loss,
    # )
    start_epoch = to_restore["epoch"]

    start_time = time.time()
    print("Starting DINO training !")

    if utils.is_main_process() and args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)
        log_writer = wandb.init(project="DINO_CT", name=args.output_dir)

    else:
        log_writer = None
    print(teacher)
    for epoch in range(start_epoch, args.epochs):
        data_loader.sampler.set_epoch(epoch)

        # ============ training one epoch of DINO ... ============
        train_stats = train_one_epoch(student, teacher, teacher_without_ddp, dino_loss,
                                      data_loader, optimizer, lr_schedule, wd_schedule, momentum_schedule,
                                      epoch, fp16_scaler, log_writer, args)

        # ============ writing logs ... ============
        save_dict = {
            'student': student.state_dict(),
            'teacher': teacher.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch + 1,
            'args': args,
            'dino_loss': dino_loss.state_dict(),
        }
        if fp16_scaler is not None:
            save_dict['fp16_scaler'] = fp16_scaler.state_dict()
        utils.save_on_master(save_dict, os.path.join(args.output_dir, 'checkpoint.pth'))
        if args.saveckp_freq and epoch % args.saveckp_freq == 0 and epoch != 0:
            utils.save_on_master(save_dict, os.path.join(
                args.output_dir, f'checkpoint{epoch:04}.pth'))
            CHECKPOINT =  os.path.join(args.output_dir, f'checkpoint{epoch:04}.pth')
            VIT_PARAMS = {
                    "in_chans": 1,
                    "img_size": (128, 128, 128),
                    "patch_size": (16, 16, 16),
                    "patch_embed": "conv",
                    "pos_embed": "sincos",
                    "use_flash_attn": True
                }
            OUTPUT_DIR= '/scratch/ml9715/DINO-3D/LINEAR_PROBING/DATAFRAMES/RUN_3'
            CACHE_DIRECTORY = '/scratch/ml9715/DINO-3D/LINEAR_PROBING/cache_dataset'
            TRAIN_CSV_PATH = '/scratch/by2026/BrainATLAS/mae/mae/data_csv/mae_UKB_ADNI_HCP_CamCAN_IXI_train_with_stat.csv'
            compute_features(TRAIN_CSV_PATH, CHECKPOINT, VIT_PARAMS, epoch, OUTPUT_DIR, CACHE_DIRECTORY, args.batch_size_per_gpu, ['age', 'gender'])
            
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch}

        if utils.is_main_process():
            with (Path(args.output_dir) / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


def train_one_epoch(student, teacher, teacher_without_ddp, dino_loss, data_loader,
                    optimizer, lr_schedule, wd_schedule, momentum_schedule, epoch,
                    fp16_scaler, log_writer, args):
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Epoch: [{}/{}]'.format(epoch, args.epochs)
    print(header)
    for it, images in enumerate(metric_logger.log_every(data_loader, 10, header)):
        # update weight decay and learning rate according to their schedule
        it = len(data_loader) * epoch + it  # global training iteration
        for i, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = lr_schedule[it]
            if i == 0:  # only the first group is regularized
                param_group["weight_decay"] = wd_schedule[it]

        # move images to gpu
        images = [im.cuda(non_blocking=True) for im in images]
        # if utils.is_main_process():
        #     print(len(images))
        # teacher and student forward passes + compute dino loss
        with torch.cuda.amp.autocast(fp16_scaler is not None):
            teacher_output = teacher(images[:2])  # only the 2 global views pass through the teacher
            student_output = student(images)
            loss = dino_loss(student_output, teacher_output, epoch)

        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()), force=True)
            sys.exit(1)

        # student update
        optimizer.zero_grad()
        param_norms = None
        if fp16_scaler is None:
            loss.backward()
            if args.clip_grad:
                param_norms = utils.clip_gradients(student, args.clip_grad)
            utils.cancel_gradients_last_layer(epoch, student,
                                              args.freeze_last_layer)
            optimizer.step()
        else:
            fp16_scaler.scale(loss).backward()
            if args.clip_grad:
                # unscale the gradients of optimizer's assigned params in-place
                fp16_scaler.unscale_(optimizer)
                param_norms = utils.clip_gradients(student, args.clip_grad)
            utils.cancel_gradients_last_layer(epoch, student, args.freeze_last_layer)
            fp16_scaler.step(optimizer)
            fp16_scaler.update()

        # EMA update for the teacher
        with torch.no_grad():
            m = momentum_schedule[it]  # momentum parameter
            for param_q, param_k in zip(student.module.parameters(), teacher_without_ddp.parameters()):
                param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

        # logging
        torch.cuda.synchronize()
        # if utils.is_main_process():
        #     if it % 10 == 0:
        #         print("Step: {}, loss: {}".format(it, loss.item()))
        metric_logger.update(loss=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(wd=optimizer.param_groups[0]["weight_decay"])

        if log_writer is not None and (it + 1) % 10 == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((it / len(data_loader)) * 100)
            log_writer.log({'train_loss': loss.item(),
                            'lr': optimizer.param_groups[0]["lr"],
                            'wd': optimizer.param_groups[0]["weight_decay"],
                            }, epoch_1000x)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    torch.cuda.empty_cache()
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


class DINOLoss(nn.Module):
    def __init__(self, out_dim, ncrops, warmup_teacher_temp, teacher_temp,
                 warmup_teacher_temp_epochs, nepochs, student_temp=0.1,
                 center_momentum=0.9):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.ncrops = ncrops
        self.register_buffer("center", torch.zeros(1, out_dim))
        # we apply a warm up for the teacher temperature because
        # a too high temperature makes the training instable at the beginning
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp,
                        teacher_temp, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        ))

    def forward(self, student_output, teacher_output, epoch):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        student_out = student_output / self.student_temp
        student_out = student_out.chunk(self.ncrops)

        # teacher centering and sharpening
        temp = self.teacher_temp_schedule[epoch]
        teacher_out = F.softmax((teacher_output - self.center) / temp, dim=-1)
        teacher_out = teacher_out.detach().chunk(2)

        total_loss = 0
        n_loss_terms = 0
        for iq, q in enumerate(teacher_out):
            for v in range(len(student_out)):
                if v == iq:
                    # we skip cases where student and teacher operate on the same view
                    continue
                loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1
        total_loss /= n_loss_terms
        self.update_center(teacher_output)
        return total_loss

    @torch.no_grad()
    def update_center(self, teacher_output):
        """
        Update center used for teacher output.
        """
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        dist.all_reduce(batch_center)
        batch_center = batch_center / (len(teacher_output) * dist.get_world_size())

        # ema update
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)

    # # Loss related
    # loss_inst = Loss(
    #     args.out_dim,
    #     teacher_temp=args.teacher_temp,
    #     student_temp=args.student_temp,
    # ).to(device)
    # lr = 0.0005 * args.batch_size / 256
    # optimizer = torch.optim.AdamW(
    #     student.parameters(),
    #     lr=lr,
    #     weight_decay=args.weight_decay,
    # )

    # # Training loop
    # n_batches = len(train_ds) // args.batch_size # need to update
    # n_steps = 0

    # losses = []
    # total_loss = 0
    # start = time.time()
    # for e in range(2):
    #     print("Starting epoch", e)
    #     train_loader.sampler.set_epoch(e)
    #     for i, images in tqdm.tqdm(
    #         enumerate(train_loader), total=n_batches
    #     ):
    #         student.train()

    #         images = [img.to(device) for img in images]

    #         print("Got images to device")

    #         teacher_output = teacher(images[:2])
    #         student_output = student(images)

    #         print("Passed through the model")

    #         loss = loss_inst(student_output, teacher_output)
    #         total_loss += loss.item()

    #         print("Loss calculated")

    #         optimizer.zero_grad()
    #         loss.backward()
    #         clip_gradients(student, args.clip_grad)
    #         optimizer.step()

    #         with torch.no_grad():
    #             for student_ps, teacher_ps in zip(
    #                 student.parameters(), teacher.parameters()
    #             ):
    #                 teacher_ps.data.mul_(args.momentum_teacher)
    #                 teacher_ps.data.add_(
    #                     (1 - args.momentum_teacher) * student_ps.detach().data
    #                 )

    #         n_steps += 1
    #     losses.append(total_loss)
    #     total_loss = 0

    # # torch.save(student.module.state_dict(), "./student_newaugs.pth")
    # # torch.save(teacher.module.state_dict(), "./teacher_newaugs.pth")
    # # with open("losses", "wb") as fp:   #Pickling
    # #     pickle.dump(losses, fp)

    # print("total time = ", time.time()-start)

    # destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser('DINO', parents=[get_args_parser()])
    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    train_dino(args)
