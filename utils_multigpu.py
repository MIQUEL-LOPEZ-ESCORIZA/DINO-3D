# pip install 'monai[einops]' -> needs to be installed like this
# Modified from https://github.com/jankrepl/mildlyoverfitted/tree/master/github_adventures/dino
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from timm.models.layers import to_3tuple
from monai.networks.blocks.transformerblock import TransformerBlock
from monai.networks.layers import trunc_normal_
from collections.abc import Sequence
import numpy as np
from torch.nn import LayerNorm
from monai.networks.layers import Conv, trunc_normal_
from monai.utils import ensure_tuple_rep, optional_import
from monai.utils.module import look_up_option
import nibabel as ni
import pandas as pd
import torch.distributed as dist
from monai import data
from monai import transforms
import os
import datetime
from torch.utils.data import Dataset, DataLoader
import numpy as np


csv_path = "/gpfs/home/ds5396/scratch/MedSSL-3D/datasets/train_new.csv"

# class DataAugmentation3D:
#     def __init__(self):

#         self.base_trans = transforms.Compose(
#             [
#                 transforms.LoadImage(),
#                 transforms.EnsureChannelFirst(),
#                 transforms.Orientation(
#                     axcodes="RAS"
#                 ),
#                 transforms.Spacing(
#                     pixdim=(1.0, 1.0, 1.0),
#                     mode="bilinear"
#                 ),
#                 transforms.CropForeground(
#                     source_key="image",
#                     allow_smaller=True,
#                 ),
#                 transforms.ScaleIntensityRange(
#                     a_min=40-150,
#                     a_max=40+150,
#                     b_min=0.0,
#                     b_max=1.0,
#                     clip=True,
#                 ),
#                 transforms.RandSpatialCrop(
#                     roi_size=(96, 96, 96),
#                     random_center=True,
#                     random_size=False,
#                 ),
#                 transforms.ResizeWithPadOrCrop(
#                     spatial_size=(96, 96, 96),
#                     method='symmetric',
#                     mode='constant',
#                     value=0,
#                 ),
#                 transforms.RandFlip(
#                     prob=0.1,
#                     spatial_axis=0,
#                 ),
#                 transforms.RandFlip(
#                     prob=0.3,
#                     spatial_axis=0,
#                 ),
#                 transforms.RandGaussianSharpen(
#                     prob=0.5,
#                 ),
#                 transforms.RandShiftIntensity(
#                     offsets=0.1,
#                     prob=0.5,
#                 ),
#                 transforms.ToTensor(),
#             ]
#         )


#     def __call__(self, img):
#         all_crops = []

#         all_crops.extend([self.base_trans(img) for _ in range(4)])


#         return all_crops
    

class DataAugmentationDINO3D(object):
    def __init__(self, global_crops_size, local_crops_size, global_crops_scale, local_crops_scale, local_crops_number):
        # Define transforms for flipping and random affine transformations
        # flip_and_noise = transforms.Compose([
        #     transforms.RandFlip(prob=0.5, spatial_axis=[0, 1, 2]),  # Random flip across different axes
        #     transforms.RandBiasField(prob=0.8),
        # ])
        flip_and_noise = transforms.Compose([transforms.RandFlip(prob=0.3, spatial_axis=0), 
                                    transforms.RandFlip(prob=0.3, spatial_axis=1), 
                                    transforms.RandFlip(prob=0.3, spatial_axis=2), 
                                    transforms.RandShiftIntensity(offsets=0.2, prob=0.5)
                                    ])

        # Normalization transform - adjust mean and std as per your dataset
        # normalize = transforms.Compose([transforms.ToTensor(), 
        #                                 transforms.NormalizeIntensity()])
        normalize = transforms.ToTensor()

        # Global crop transforms
        self.global_transfo1 = transforms.Compose([
            # transforms.LoadImage(image_only=True),
            # transforms.EnsureChannelFirst(),
            transforms.CastToType(dtype=np.float32),
            transforms.ResizeWithPadOrCrop(spatial_size=(224, 224, 224)),
            transforms.RandSpatialCrop((160, 160, 160), random_center=True, random_size=True),
            transforms.Resize(spatial_size=(96, 96, 96)),
            # transforms.RandScaleCrop(roi_scale=global_crops_scale[0], max_roi_scale=global_crops_scale[1], random_size=True),
            # transforms.Resize(spatial_size=global_crops_size),  # Resize to a specified size
            flip_and_noise, 
            transforms.RandGaussianSmooth(sigma_x=(0.5,1.0), sigma_y=(0.5,1.0), sigma_z=(0.5,1.0), prob=0.2), 
            normalize, 
        ])
        self.global_transfo2 = transforms.Compose([
            # transforms.LoadImage(image_only=True),
            # transforms.EnsureChannelFirst(),
            transforms.CastToType(dtype=np.float32),
            transforms.ResizeWithPadOrCrop(spatial_size=(224, 224, 224)),
            transforms.RandSpatialCrop((160, 160, 160), random_center=True, random_size=True),
            transforms.Resize(spatial_size=(96, 96, 96)),
            # transforms.RandScaleCrop(roi_scale=global_crops_scale[0], max_roi_scale=global_crops_scale[1], random_size=True), 
            # transforms.Resize(spatial_size=global_crops_size), 
            flip_and_noise, 
            # transforms.RandGaussianSmooth(sigma_x=(0.05,0.1), sigma_y=(0.05,0.1), sigma_z=(0.05,0.1)),
            transforms.RandAdjustContrast(gamma=(0.2,1.),prob=0.2),
            normalize,
        ])

        # Local crop transform
        self.local_crops_number = local_crops_number
        self.local_transfo = transforms.Compose([
            transforms.CastToType(dtype=np.float32),
            transforms.ResizeWithPadOrCrop(spatial_size=(224, 224, 224)),
            transforms.RandSpatialCrop((128, 128, 128), max_roi_size=(160, 160, 160), random_center=True, random_size=True),
            # transforms.ResizeWithPadOrCrop(spatial_size=(96, 96, 96)),
            transforms.Resize(spatial_size=(96, 96, 96)),
            # transforms.RandScaleCrop(roi_scale=local_crops_scale[0], max_roi_scale=local_crops_scale[1], random_size=True),
            # transforms.Resize(spatial_size=local_crops_size),
            # flip_and_noise,
            # transforms.RandGaussianSmooth(sigma_x=(0.05,0.1), sigma_y=(0.05,0.1), sigma_z=(0.05,0.1), prob=0.2), 
            normalize
        ])
        # self.local_transfo = transforms.Compose([
        #     transforms.LoadImage(image_only=True),
        #     transforms.EnsureChannelFirst(),
        #     transforms.RandScaleCrop(roi_scale=local_crops_scale[0], max_roi_scale=local_crops_scale[1], random_size=True),
        #     transforms.Resize(spatial_size=local_crops_size),
        #     flip_and_noise,
        #     transforms.RandGaussianSmooth(prob=0.5),
        #     normalize,
        # ])

    def __call__(self, image):
        crops = []
        # crops.append(torch.FloatTensor(self.global_transfo1(image)))
        # crops.append(torch.FloatTensor(self.global_transfo2(image)))
        crops.append(self.global_transfo1(image))
        crops.append(self.global_transfo2(image))
        for _ in range(self.local_crops_number):
            crops.append(self.local_transfo(image))
        return crops
    

class DataAugmentationTest:
    def __init__(self):

        self.base_trans = transforms.Compose(
            [
                transforms.LoadImage(),
                transforms.EnsureChannelFirst(),
                transforms.Orientation(
                    axcodes="RAS"
                ),
                transforms.Spacing(
                    pixdim=(1.0, 1.0, 1.0),
                    mode="bilinear"
                ),
                transforms.CropForeground(
                    source_key="image",
                    allow_smaller=True,
                ),
                transforms.ScaleIntensityRange(
                    a_min=40-150,
                    a_max=40+150,
                    b_min=0.0,
                    b_max=1.0,
                    clip=True,
                ),
                transforms.Resize(spatial_size=(96, 96, 96)),
                transforms.ToTensor(),
            ]
        )


    def __call__(self, img):

        return [self.base_trans(img)]


Rearrange, _ = optional_import("einops.layers.torch", name="Rearrange")
SUPPORTED_EMBEDDING_TYPES = {"conv", "perceptron"}


class HeadDataset(Dataset):
    def __init__(self, csv_file, data_augmentation, cache_dir="/gpfs/scratch/wz727/temp_cache/"):
        self.data = pd.read_csv(csv_file)
        self.load = transforms.Compose([
            transforms.LoadImage(image_only=True),
            transforms.EnsureChannelFirst(), 
            transforms.Orientation(
                axcodes="RAS"
            ),
            transforms.Spacing(
                pixdim=(1.0, 1.0, 1.0),
                mode="bilinear"
            ),
            # transforms.CropForeground(
            #     source_key="image",
            #     allow_smaller=True,
            # ),
            transforms.ScaleIntensityRange(
                    a_min=40-150,
                    a_max=40+150,
                    b_min=0.0,
                    b_max=1.0,
                    clip=True,
                ),
            transforms.CastToType(dtype=np.float16)
        ])

        self.cache_dir = cache_dir
        self.cache_dataset = data.PersistentDataset(
            data=list(self.data['img_path']), 
            transform=self.load, 
            cache_dir=self.cache_dir,
        )

        self.data_augmentation = data_augmentation

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data.loc[idx, 'img_path']
        # image = self.load(img_path)
        image = self.cache_dataset.__getitem__(idx)
        if self.data_augmentation:
            # print(img_path)
            image = self.data_augmentation(image)

        return image


class PatchEmbeddingBlock(nn.Module):
    """
    A patch embedding block, based on: "Dosovitskiy et al.,
    An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale <https://arxiv.org/abs/2010.11929>"

    Example::

        >>> from monai.networks.blocks import PatchEmbeddingBlock
        >>> PatchEmbeddingBlock(in_channels=4, img_size=32, patch_size=8, hidden_size=32, num_heads=4,
        >>>                     patch_embed="conv", pos_embed="sincos")

    """

    def __init__(
        self,
        in_channels: int,
        img_size: Sequence[int] | int,
        patch_size: Sequence[int] | int,
        hidden_size: int,
        num_heads: int,
        patch_embed: str = 'conv',
        pos_embed: str = "learnable",
        dropout_rate: float = 0.0,
        spatial_dims: int = 3,
    ) -> None:
        """
        Args:
            in_channels: dimension of input channels.
            img_size: dimension of input image.
            patch_size: dimension of patch size.
            hidden_size: dimension of hidden layer.
            num_heads: number of attention heads.
            patch_embed: patch embedding layer type.
            pos_embed: position embedding layer type.
            dropout_rate: faction of the input units to drop.
            spatial_dims: number of spatial dimensions.
        """

        super().__init__()

        if not (0 <= dropout_rate <= 1):
            raise ValueError(f"dropout_rate {dropout_rate} should be between 0 and 1.")

        if hidden_size % num_heads != 0:
            raise ValueError(f"hidden size {hidden_size} should be divisible by num_heads {num_heads}.")

        self.patch_embed = look_up_option(patch_embed, SUPPORTED_EMBEDDING_TYPES)

        img_size = ensure_tuple_rep(img_size, spatial_dims)
        patch_size = ensure_tuple_rep(patch_size, spatial_dims)
        for m, p in zip(img_size, patch_size):
            if m < p:
                raise ValueError("patch_size should be smaller than img_size.")
            if self.patch_embed == "perceptron" and m % p != 0:
                raise ValueError("patch_size should be divisible by img_size for perceptron.")
        self.n_patches = np.prod([im_d // p_d for im_d, p_d in zip(img_size, patch_size)])
        self.patch_dim = int(in_channels * np.prod(patch_size))

        grid_size = []
        for in_size, pa_size in zip(img_size, patch_size):
            assert in_size % pa_size == 0, "input size and patch size are not proper"
            grid_size.append(in_size // pa_size)

        self.patch_embeddings: nn.Module
        if self.patch_embed == "conv":
            self.patch_embeddings = Conv[Conv.CONV, spatial_dims](
                in_channels=in_channels, out_channels=hidden_size, kernel_size=patch_size, stride=patch_size
            )
        elif self.patch_embed == "perceptron":
            # for 3d: "b c (h p1) (w p2) (d p3)-> b (h w d) (p1 p2 p3 c)"
            chars = (("h", "p1"), ("w", "p2"), ("d", "p3"))[:spatial_dims]
            from_chars = "b c " + " ".join(f"({k} {v})" for k, v in chars)
            to_chars = f"b ({' '.join([c[0] for c in chars])}) ({' '.join([c[1] for c in chars])} c)"
            axes_len = {f"p{i+1}": p for i, p in enumerate(patch_size)}
            self.patch_embeddings = nn.Sequential(
                Rearrange(f"{from_chars} -> {to_chars}", **axes_len), nn.Linear(self.patch_dim, hidden_size)
            )
        self.position_embeddings = nn.Parameter(torch.zeros(1, self.n_patches, hidden_size))
        self.dropout = nn.Dropout(dropout_rate)

        if pos_embed == "none":
            pass
        elif pos_embed == "learnable":
            trunc_normal_(self.position_embeddings, mean=0.0, std=0.02, a=-2.0, b=2.0)
        else:
            raise ValueError(f"pos_embed type {pos_embed} not supported.")

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, mean=0.0, std=0.02, a=-2.0, b=2.0)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        x = self.patch_embeddings(x)
        if self.patch_embed == "conv":
            x = x.flatten(2).transpose(-1, -2)
        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings
    

class ViT(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self,
                 input_size,
                 patch_size,
                 in_chans=1,
                 dropout_rate=0.,
                 spatial_dims=3,
                 patch_embed='conv',
                 pos_embed='learnable',
                 encoder_depth=12,
                 encoder_embed_dim=768,
                 encoder_mlp_dim=3072,
                 encoder_num_heads=12,
                 decoder_depth=8,
                 decoder_embed_dim=768,
                 decoder_mlp_dim=3072,
                 decoder_num_heads=16,
                 norm_pix_loss=False,
                 ):
        super().__init__()

        input_size = to_3tuple(input_size)
        patch_size = to_3tuple(patch_size)

        # register parameters
        self.input_size = input_size
        self.patch_size = patch_size
        self.spatial_dims = spatial_dims
        self.pos_embed = pos_embed
        self.norm_pix_loss = norm_pix_loss

        self.encoder_embed_dim = encoder_embed_dim
        # self.decoder_embed_dim = decoder_embed_dim

        # calculate grid size
        out_chans = in_chans * np.prod(patch_size)
        self.out_chans = out_chans

        grid_size = []
        for in_size, pa_size in zip(input_size, patch_size):
            assert in_size % pa_size == 0, "input size and patch size are not proper"
            grid_size.append(in_size // pa_size)
        self.grid_size = grid_size

        num_patches = np.prod(grid_size)
        patch_dim = np.prod(patch_size)

        # class token and position embedding
        self.cls_token = nn.Parameter(torch.zeros(1, 1, encoder_embed_dim))
        # self.decoder_cls_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        # self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding

        # transformer encoder
        self.patch_embedding = PatchEmbeddingBlock(
            img_size=input_size,
            patch_size=patch_size,
            in_channels=in_chans,
            hidden_size=encoder_embed_dim,
            num_heads=encoder_num_heads,
            patch_embed=patch_embed,
            pos_embed=pos_embed,
            dropout_rate=dropout_rate,
            spatial_dims=spatial_dims,
        )

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(encoder_embed_dim, encoder_mlp_dim, encoder_num_heads, \
                    dropout_rate, qkv_bias=True, save_attn=False)
                for i in range(encoder_depth)
            ]
        )

        self.norm = nn.LayerNorm(encoder_embed_dim)

        # transformer decoder
        # self.decoder_blocks = nn.ModuleList(
        #     [
        #         TransformerBlock(decoder_embed_dim, decoder_mlp_dim, decoder_num_heads, \
        #             dropout_rate, qkv_bias=True, save_attn=False)
        #         for i in range(decoder_depth)
        #     ]
        # )

        # self.decoder_norm = nn.LayerNorm(decoder_embed_dim)
        # self.decoder_embed = nn.Linear(encoder_embed_dim, decoder_embed_dim, bias=True)
        # self.decoder_pred = nn.Linear(decoder_embed_dim, patch_dim * in_chans, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # trunc_normal_(self.decoder_pos_embed, std=.02)

        # trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        trunc_normal_(self.cls_token, std=.02)
        # trunc_normal_(self.decoder_cls_token, std=.02)
        # trunc_normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, x):
        """
        Different from 2D version patchification: The final axis follows the order of [ph, pw, pd, c] instead of [c, ph, pw, pd]
        """
        # patchify input, [B,C,H,W,D] --> [B,C,gh,ph,gw,pw,gd,pd] --> [B,gh*gw*gd,ph*pw*pd*C]
        B, C, H, W, D = x.shape
        patch_size = self.patch_size
        grid_size = (H // patch_size[0], W // patch_size[1], D // patch_size[2])

        gh, gw, gd = grid_size[0], grid_size[1], grid_size[2]
        ph, pw, pd = patch_size[0], patch_size[1], patch_size[2]

        x = x.reshape(B, C, gh, ph, gw, pw, gd, pd) # [B,C,gh,ph,gw,pw,gd,pd]
        x = x.permute(0, 2, 4, 6, 3, 5, 7, 1).reshape(B, np.prod(grid_size), np.prod(patch_size) * C) # [B,gh*gw*gd,ph*pw*pd*C]

        return x

    def unpatchify(self, x, x_ori):
        """
        Different from 2D version patchification: The final axis follows the order of [ph, pw, pd, c] instead of [c, ph, pw, pd]
        """
        # unpatchify input, [B,gh*gw*gd,ph*pw*pd*C] --> [B,gh,ph,gw,pw,gd,pd, C] --> [B,C,gh*ph,gw*pw,gd*pd]
        B, C, H, W, D = x_ori.shape
        patch_size = self.patch_size

        ph, pw, pd = patch_size[0], patch_size[1], patch_size[2]
        gh, gw, gd = H // ph, W // pw, D // pd

        x = x.reshape(B, gh, gw, gd, ph, pw, pd, C) # [B,gh,ph,gw,pw,gd,pd, C]
        x = x.permute(0, 7, 1, 4, 2, 5, 3, 6).reshape(B, C, gh*ph, gw*pw, gd*pd) # [B,C,gh*ph,gw*pw,gd*pd]

        return x

    # def random_masking(self, x):
    #     # compute length for selected and masked
    #     N, L, D = x.shape
    #     len_keep = int(L * (1 - self.mask_ratio))
    #     noise = torch.rand(N, L, device=x.device)

    #     # sort noise for each sample
    #     ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
    #     ids_restore = torch.argsort(ids_shuffle, dim=1)

    #     ids_keep = ids_shuffle[:, :len_keep]
    #     x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

    #     # generate the binary mask: 0 is keep, 1 is remove
    #     mask = torch.ones([N, L], device=x.device)
    #     mask[:, :len_keep] = 0
    #     # unshuffle to get the binary mask
    #     mask = torch.gather(mask, dim=1, index=ids_restore)

    #     return x_masked, mask, ids_restore, ids_keep

    def forward_encoder(self, x):
        # embed patches with positional embedding added
        x = self.patch_embedding(x)

        # append cls token
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)

        # apply normalization
        x = self.norm(x)

        return x

    # def forward_decoder(self, x, ids_restore):
    #     # embed tokens
    #     x = self.decoder_embed(x)

    #     # append mask tokens to sequence
    #     mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
    #     x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
    #     x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
    #     x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

    #     # add pos embed
    #     decoder_cls_token = self.decoder_cls_token.expand(x.shape[0], -1, -1)
    #     decoder_pos_embed = self.decoder_pos_embed.expand(x.shape[0], -1, -1)
    #     decoder_pos_embed = torch.cat((decoder_cls_token, decoder_pos_embed), dim=1)
    #     x = x + decoder_pos_embed

    #     # apply Transformer blocks
    #     for blk in self.decoder_blocks:
    #         x = blk(x)

    #     # apply normalization
    #     x = self.decoder_norm(x)

    #     # predictor projection
    #     x = self.decoder_pred(x)

    #     # remove cls token
    #     x = x[:, 1:, :]

    #     return x

    # def forward_loss(self, imgs, pred, mask):
    #     target = self.patchify(imgs)
    #     if self.norm_pix_loss:
    #         mean = target.mean(dim=-1, keepdim=True)
    #         var = target.var(dim=-1, keepdim=True)
    #         target = (target - mean) / (var + 1.e-6)**.5

    #     #pred, target = all_gather(pred), all_gather(target)

    #     loss = (pred - target) ** 2
    #     loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

    #     mask = mask.view(loss.shape)

    #     loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
    #     return loss

    def forward(self, x):
        latent = self.forward_encoder(x)

        return latent[:, 0, :]

class DINOHead(nn.Module):
    def __init__(self, in_dim, out_dim, use_bn=False, norm_last_layer=True, nlayers=3, hidden_dim=2048, bottleneck_dim=256):
        super().__init__()
        nlayers = max(nlayers, 1)
        if nlayers == 1:
            self.mlp = nn.Linear(in_dim, bottleneck_dim)
        else:
            layers = [nn.Linear(in_dim, hidden_dim)]
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
            for _ in range(nlayers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if use_bn:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            self.mlp = nn.Sequential(*layers)
        self.apply(self._init_weights)
        self.last_layer = nn.utils.weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1)
        if norm_last_layer:
            self.last_layer.weight_g.requires_grad = False

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.mlp(x)
        x = nn.functional.normalize(x, dim=-1, p=2)
        x = self.last_layer(x)
        return x

# class MultiCropWrapper(nn.Module):
#     """Convenience class for forward pass of multiple crops.

#     """
#     def __init__(self, model, head):
#         super().__init__()
#         self.model = model
#         self.head = head

#     def forward(self, x):
#         """Run the forward pass.

#         The different crops are concatenated along the batch dimension
#         and then a single forward pass is fun. The resulting tensor
#         is then chunked back to per crop tensors.

#         Parameters
#         ----------
#         x : list
#             List of `torch.Tensor` each of shape `(n_samples, 3, size, size)`.

#         Returns
#         -------
#         tuple
#             Tuple of `torch.Tensor` each of shape `(n_samples, out_dim)` where
#             `output_dim` is determined by `Head`.
#         """
#         n_crops = len(x)
#         concatenated = torch.cat(x, dim=0)  # (n_samples * n_crops, 3, size, size)
#         logits = self.model(concatenated)  # (n_samples * n_crops, in_dim)
#         logits = self.head(logits)
#         chunks = logits.chunk(n_crops)  # n_crops * (n_samples, out_dim)

#         return chunks

class MultiCropWrapper(nn.Module):
    """
    Perform forward pass separately on each resolution input.
    The inputs corresponding to a single resolution are clubbed and single
    forward is run on the same resolution inputs. Hence we do several
    forward passes = number of different resolutions used. We then
    concatenate all the output features and run the head forward on these
    concatenated features.
    """
    def __init__(self, model, head):
        super(MultiCropWrapper, self).__init__()
        # disable layers dedicated to ImageNet labels classification
        # backbone.fc, backbone.head = nn.Identity(), nn.Identity()
        self.backbone = model
        self.head = head

    def forward(self, x):
        # convert to list
        if not isinstance(x, list):
            x = [x]
        idx_crops = torch.cumsum(torch.unique_consecutive(
            torch.tensor([inp.shape[-1] for inp in x]),
            return_counts=True,
        )[1], 0)
        start_idx, output = 0, torch.empty(0).to(x[0].device)
        for end_idx in idx_crops:
            _out = self.backbone(torch.cat(x[start_idx: end_idx]))
            # The output is a tuple with XCiT model. See:
            # https://github.com/facebookresearch/xcit/blob/master/xcit.py#L404-L405
            if isinstance(_out, tuple):
                _out = _out[0]
            # accumulate outputs
            output = torch.cat((output, _out))
            start_idx = end_idx
        # Run the head forward on the concatenated features.
        return self.head(output)

class Loss(nn.Module):
    """The loss function.

    We subclass the `nn.Module` becuase we want to create a buffer for the
    logits center of the teacher.

    Parameters
    ----------
    out_dim : int
        The dimensionality of the final layer (we computed the softmax over).

    teacher_temp, student_temp : float
        Softmax temperature of the teacher resp. student.

    center_momentum : float
        Hyperparameter for the exponential moving average that determines
        the center logits. The higher the more the running average matters.
    """
    def __init__(
        self, out_dim, teacher_temp=0.04, student_temp=0.1, center_momentum=0.9
    ):
        super().__init__()
        self.student_temp = student_temp
        self.teacher_temp = teacher_temp
        self.center_momentum = center_momentum
        self.register_buffer("center", torch.zeros(1, out_dim))

    def forward(self, student_output, teacher_output):
        """Evaluate loss.

        Parameters
        ----------
        student_output, teacher_output : tuple
            Tuple of tensors of shape `(n_samples, out_dim)` representing
            logits. The length is equal to number of crops.
            Note that student processed all crops and that the two initial crops
            are the global ones.

        Returns
        -------
        loss : torch.Tensor
            Scalar representing the average loss.
        """
        student_temp = [s / self.student_temp for s in student_output]
        teacher_temp = [(t - self.center) / self.teacher_temp for t in teacher_output]

        student_sm = [F.log_softmax(s, dim=-1) for s in student_temp]
        teacher_sm = [F.softmax(t, dim=-1).detach() for t in teacher_temp]

        total_loss = 0
        n_loss_terms = 0

        for t_ix, t in enumerate(teacher_sm):
            for s_ix, s in enumerate(student_sm):
                if t_ix == s_ix:
                    continue

                loss = torch.sum(-t * s, dim=-1)  # (n_samples,)
                total_loss += loss.mean()  # scalar
                n_loss_terms += 1

        total_loss /= n_loss_terms
        self.update_center(teacher_output)

        return total_loss

    @torch.no_grad()
    def update_center(self, teacher_output):
        """Update center used for teacher output.

        Compute the exponential moving average.

        Parameters
        ----------
        teacher_output : tuple
            Tuple of tensors of shape `(n_samples, out_dim)` where each
            tensor represents a different crop.
        """
        batch_center = torch.cat(teacher_output).mean(
            dim=0, keepdim=True
        )  # (1, out_dim)
        self.center = self.center * self.center_momentum + batch_center * (
            1 - self.center_momentum
        )

def clip_gradients(model, clip=2.0):
    """Rescale norm of computed gradients.

    Parameters
    ----------
    model : nn.Module
        Module.

    clip : float
        Maximum norm.
    """
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            clip_coef = clip / (param_norm + 1e-6)
            if clip_coef < 1:
                p.grad.data.mul_(clip_coef)


def create_dataset(images, labels):
    dataset = []
    
    if labels is None:
        for img in images:
            sample_dict = dict()
            sample_dict['image'] = img
            dataset.append(sample_dict)
    else:
        for img, label in zip(images, labels):
            sample_dict = dict()
            sample_dict['image'] = img
            sample_dict['label'] = label
            dataset.append(sample_dict)
            
    return dataset


def get_dataloaders(augs=[]):
    
    imtrans, imvals, imtests = augs[0], augs[1], augs[2]
    

    batch_size = 8
    # cache_dir = config.DATA.CACHE_DIR
    
    # Load Data
    df_train, df_val, df_test = pd.read_csv(csv_path), \
        pd.read_csv(csv_path), pd.read_csv(csv_path)
    
    img_train = list(df_train['img_path'])
    img_val = list(df_val['img_path'])
    img_test = list(df_test['img_path'])
    
    # Create Dataset
    train_files = create_dataset(img_train, None)
    val_files = create_dataset(img_val, None)
    test_files = create_dataset(img_test, None)
    
    num_tasks = dist.get_world_size()
    global_rank = dist.get_rank()

    print("Created datasets")
    
    # Create Dataloaders
    # Train
    train_ds = data.PersistentDataset(
        data=train_files, 
        transform=imtrans, 
        cache_dir='/gpfs/scratch/wz727/temp_cache/',
    )
    sampler_train = torch.utils.data.distributed.DistributedSampler(
        train_ds,
        shuffle=True,
        num_replicas=num_tasks,
        rank=global_rank,
    )
    train_loader = data.ThreadDataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=sampler_train,
        num_workers=4,
        pin_memory=True,
    )
    
    # Validate
    val_ds = data.PersistentDataset(
        data=val_files, 
        transform=imvals, 
        cache_dir='/gpfs/scratch/ds5396/temp_cache/' 
    )
    sampler_val = torch.utils.data.distributed.DistributedSampler(
        dataset=val_ds,
        shuffle=False,
        num_replicas=num_tasks,
        rank=global_rank,
    )
    val_loader = data.ThreadDataLoader(
        dataset=val_ds,
        batch_size=batch_size,
        sampler=sampler_val,
        num_workers=4,
        pin_memory=True,
    )
    
    # Test
    test_ds = data.PersistentDataset(
        data=test_files, 
        transform=imtests, 
        cache_dir='/gpfs/scratch/ds5396/temp_cache/' 
    )
    sampler_test = torch.utils.data.distributed.DistributedSampler(
        dataset=test_ds,
        shuffle=False,
        num_replicas=num_tasks,
        rank=global_rank,
    )
    test_loader = data.ThreadDataLoader(
        dataset=test_ds,
        batch_size=batch_size,
        sampler=sampler_test,
        num_workers=4,
        pin_memory=True,
    )
    
    return train_loader, val_loader, test_loader


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print

def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        gpu = int(os.environ['LOCAL_RANK'])
    else:
        print('Not using distributed mode')
        return

    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}, gpu {}'.format(
        rank, args.dist_url, gpu), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=world_size, rank=rank, timeout=datetime.timedelta(seconds=18000))
    torch.cuda.set_device(rank)
    torch.distributed.barrier()
    setup_for_distributed(rank == 0)