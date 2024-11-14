import os
import yaml
from yacs.config import CfgNode as CN

_C = CN()

# Base config files
_C.BASE = ['']

# -----------------------------------------------------------------------------
# Data Settings
# -----------------------------------------------------------------------------
_C.DATA = CN()
# Batch size for a single GPU
_C.DATA.BATCH_SIZE = 8
# Data loader

# -----------------------------------------------------------------------------
# General Model Settings
# -----------------------------------------------------------------------------
_C.MODEL = CN()
# Model type
_C.MODEL.NAME  = 'mae'
# Pretrained model path to load
_C.MODEL.PRETRAINED = None
# Model save directoy
_C.MODEL.DIR = ''
# Model save name
_C.MODEL.SAVE_NAME = ''
# Region of Interest
_C.MODEL.ROI = [96, 96, 96]
# Sliding windows batch size
_C.MODEL.SW_BATCH_SIZE = 2
# Sliding windows overlap
_C.MODEL.INFER_OVERLAP = 0.5

# -----------------------------------------------------------------------------
# MAE Settings
# -----------------------------------------------------------------------------
_C.MAE = CN()
# Input size
_C.MAE.INPUT_SIZE = 96
# Patch size
_C.MAE.PATCH_SIZE = 16
# Mask Ratio
_C.MAE.MASK_RATIO = 0.75
# Input channels
_C.MAE.IN_CHANS = 1
# Dropout rate
_C.MAE.DROPOUT_RATE = 0.
# Patch embedding layer type
_C.MAE.PATCH_EMBED = 'conv'
# Position embedding layer type
_C.MAE.POS_EMBED = 'sincos'
# Spatial dimension of input
_C.MAE.SPATIAL_DIMS = 3
# Patch pixels normalization
_C.MAE.NORM_PIX_LOSS = False
# Return image
_C.MAE.RETURN_IMAGE = False

# Encoder embedding dimension
_C.MAE.ENCODER_EMBED_DIM = 768
# Encoder depth
_C.MAE.ENCODER_DEPTH = 12
# Encoder MLP layer dimension
_C.MAE.ENCODER_MLP_DIM = 3072
# Encoder number of multi-heads
_C.MAE.ENCODER_NUM_HEADS = 12

# Decoder embedding dimension
_C.MAE.DECODER_EMBED_DIM = 768
# Decoder depth
_C.MAE.DECODER_DEPTH = 8
# Decoder MLP layer dimension
_C.MAE.DECODER_MLP_DIM = 2048
# Decoder MLP layer dimension
_C.MAE.DECODER_NUM_HEADS = 16

# Bias
_C.MAE.USE_BIAS = False
# Flash Attention
_C.MAE.USE_FLASH_ATTN = False

# -----------------------------------------------------------------------------
# VIT Settings
# -----------------------------------------------------------------------------
_C.VIT = CN()
# Input size
_C.VIT.INPUT_SIZE = 96
# Patch size
_C.VIT.PATCH_SIZE = 16
# Input channels
_C.VIT.IN_CHANS = 1
# Dropout rate
_C.VIT.DROPOUT_RATE = 0.
# Patch embedding layer type
_C.VIT.PATCH_EMBED = 'conv'
# Position embedding layer type
_C.VIT.POS_EMBED = 'sincos'
# Spatial dimension of input
_C.VIT.SPATIAL_DIMS = 3

# ViT depth
_C.VIT.NUM_LAYERS = 12
# ViT number of multi-heads
_C.VIT.NUM_HEADS = 12
# ViT hidden dimension
_C.VIT.HIDDEN_SIZE = 768
# ViT MLP layer dimension
_C.VIT.MLP_DIM = 3072
# ViT number of patches splitting for inference
_C.VIT.NUM_PATCHES = 8
# ViT patches splitting overlap
_C.VIT.PATCHES_OVERLAP = 0.2
# ViT pooling
_C.VIT.POOLING = 'cls'

# ViT classification
_C.VIT.CLASSIFICATION = False
# ViT number of classes
_C.VIT.NUM_CLASSES = 2

# Bias
_C.VIT.USE_BIAS = False
# Flash Attention
_C.VIT.USE_FLASH_ATTN = False

# -----------------------------------------------------------------------------
# Training Settings
# -----------------------------------------------------------------------------
_C.TRAIN = CN()
# Epoch
_C.TRAIN.MAX_EPOCHS = 100
# Validation every N epochs
_C.TRAIN.VAL_EVERY = 10
# Base learning rate
_C.TRAIN.BASE_LR = 1.5e-3
# Minimum learning rate
_C.TRAIN.MIN_LR = 1.5e-7
# Weight decay
_C.TRAIN.WEIGHT_DECAY = 0.05
# AdawmW beta values
_C.TRAIN.BETA1 = 0.9
_C.TRAIN.BETA2 = 0.95
# Momentum
_C.TRAIN.MOMENTUM = 0.9
# Loss type
_C.TRAIN.LOSS = 'l1'
# Contrastive loss temperature
_C.TRAIN.TEMPERATURE = 0.5
# Optimizer type
_C.TRAIN.OPTIMIZER = 'AdamW'
# Schduler type
_C.TRAIN.SCHEDULER = 'cosine'
# Percentage of linear warmup
_C.TRAIN.PER_WARMUP = 0.05
# Gradient clip
_C.TRAIN.GRAD_CLIP = 1.0
# Lock backbone
_C.TRAIN.LOCK = False

# -----------------------------------------------------------------------------
# Logging Settings
# -----------------------------------------------------------------------------
_C.LOG = CN()
# Logging file save name
_C.LOG.FILENAME = 'MAE3D_HeadCT'

# -----------------------------------------------------------------------------
# wandb Settings
# -----------------------------------------------------------------------------
_C.WANDB = CN()
# enable wandb
_C.WANDB.WANDB_ENABLE = False
# wandb project name
_C.WANDB.PROJECT = 'MAE3D'

# -----------------------------------------------------------------------------
# Misc Settings
# -----------------------------------------------------------------------------
# Seed to ensure reproducibility
_C.SEED = 42
# Enable Pytorch automatic mixed precision
_C.AMP_ENABLE = False
# local rank for distributed training
_C.LOCAL_RANK = 0
# Path to output folder
_C.OUTPUT = ''
# Tag of experiment
_C.TAG = 'default'
# Percentage Tag
_C.PERCENT = 'None'


def _update_config_from_file(config, cfg_file):
    config.defrost()
    with open(cfg_file, 'r') as f:
        yaml_cfg = yaml.load(f, Loader=yaml.FullLoader)

    for cfg in yaml_cfg.setdefault('BASE', ['']):
        if cfg:
            _update_config_from_file(
                config, os.path.join(os.path.dirname(cfg_file), cfg)
            )
    print('=> merge config from {}'.format(cfg_file))
    config.merge_from_file(cfg_file)
    config.freeze()


def update_config(config, args):
    _update_config_from_file(config, args.cfg)

    config.defrost()
    if args.opts:
        config.merge_from_list(args.opts)

    def _check_args(name):
        if hasattr(args, name) and eval(f'args.{name}'):
            return True
        return False

    # merge from specific arguments
    if _check_args('percent'):
        config.PERCENT = args.percent
    if _check_args('batch_size'):
        config.DATA.BATCH_SIZE = args.batch_size
    if _check_args('num_workers'):
        config.DATA.NUM_WORKERS = args.num_workers
    if _check_args('train_csv_path'):
        config.DATA.TRAIN_CSV_PATH = args.train_csv_path
    if _check_args('val_csv_path'):
        config.DATA.VAL_CSV_PATH = args.val_csv_path
    if _check_args('test_csv_path'):
        config.DATA.TEST_CSV_PATH = args.test_csv_path
    if _check_args('optimizer'):
        config.TRAIN.OPTIMIZER = args.optimizer
    if _check_args('scheduler'):
        config.TRAIN.SCHEDULER = args.scheduler
    if _check_args('max_epochs'):
        config.TRAIN.MAX_EPOCHS = args.max_epochs
    if _check_args('grad_clip'):
        config.TRAIN.GRAD_CLIP = args.grad_clip
    if _check_args('base_lr'):
        config.TRAIN.BASE_LR = args.base_lr
    if _check_args('min_lr'):
        config.TRAIN.MIN_LR = args.min_lr
    if _check_args('weight_decay'):
        config.TRAIN.WEIGHT_DECAY = args.weight_decay
    if _check_args('lock'):
        config.TRAIN.LOCK = args.lock
    if _check_args('pooling'):
        config.VIT.POOLING = args.pooling
    if _check_args('seed'):
        config.SEED = args.seed
    if _check_args('use_amp'):
        config.AMP_ENABLE = args.use_amp
    if _check_args('use_wandb'):
        config.WANDB.WANDB_ENABLE = args.use_wandb
    if _check_args('wandb_project'):
        config.WANDB.PROJECT = args.wandb_project
    if _check_args('model_name'):
        config.MODEL.NAME = args.model_name
    if _check_args('model_load_path'):
        config.MODEL.PRETRAINED = args.model_load_path

    # set local rank for distributed training
    config.LOCAL_RANK = args.local_rank

    # output folder
    #config.OUTPUT = os.path.join(config.OUTPUT, config.MODEL.TYPE, config.TAG)
    config.OUTPUT = os.path.join(config.OUTPUT)

    config.freeze()


def get_config(args):
    """Get a yacs CfgNode object with default values."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    config = _C.clone()
    update_config(config, args)

    return config
