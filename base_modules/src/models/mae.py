# Modified from https://github.com/cvlab-stonybrook/SelfMedMAE/blob/main/lib/models/mae3d.py and https://github.com/facebookresearch/mae

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

import numpy as np

from src.utils.pos_embed import build_sincos_position_embedding
from src.utils.patch_embedding import PatchEmbeddingBlock
from src.models.flashattentionblock import FlashAttentionBlock

from timm.models.layers import to_3tuple

from monai.networks.blocks.transformerblock import TransformerBlock
from monai.networks.layers import trunc_normal_


class MaskedAutoencoderViT(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, 
                 input_size,
                 patch_size,
                 mask_ratio,
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
                 use_bias=False,
                 use_flash_attn=False,
                 ):
        super().__init__()
        
        input_size = to_3tuple(input_size)
        patch_size = to_3tuple(patch_size)
        
        # register parameters
        self.input_size = input_size
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio
        self.spatial_dims = spatial_dims
        self.pos_embed = pos_embed
        self.norm_pix_loss = norm_pix_loss
        self.use_flash_attn = use_flash_attn
        
        self.encoder_embed_dim = encoder_embed_dim
        self.decoder_embed_dim = decoder_embed_dim
        
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
        self.decoder_cls_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding
        
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
        
        if use_flash_attn:
            # transformer encoder
            self.blocks = nn.ModuleList(
                [
                    FlashAttentionBlock(encoder_embed_dim, encoder_mlp_dim, encoder_num_heads, \
                        dropout_rate, qkv_bias=use_bias, save_attn=False)
                    for i in range(encoder_depth)
                ]
            )
            # transformer decoder
            self.decoder_blocks = nn.ModuleList(
                [
                    FlashAttentionBlock(decoder_embed_dim, decoder_mlp_dim, decoder_num_heads, \
                        dropout_rate, qkv_bias=use_bias, save_attn=False)
                    for i in range(decoder_depth)
                ]
            )
        else:
            # transformer encoder
            self.blocks = nn.ModuleList(
                [
                    TransformerBlock(encoder_embed_dim, encoder_mlp_dim, encoder_num_heads, \
                        dropout_rate, qkv_bias=use_bias, save_attn=False)
                    for i in range(encoder_depth)
                ]
            )
            # transformer decoder
            self.decoder_blocks = nn.ModuleList(
                [
                    TransformerBlock(decoder_embed_dim, decoder_mlp_dim, decoder_num_heads, \
                        dropout_rate, qkv_bias=use_bias, save_attn=False)
                    for i in range(decoder_depth)
                ]
            )
        
        self.norm = nn.LayerNorm(encoder_embed_dim)
        
        self.decoder_norm = nn.LayerNorm(decoder_embed_dim)
        self.decoder_embed = nn.Linear(encoder_embed_dim, decoder_embed_dim, bias=use_bias)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_dim * in_chans, bias=use_bias) 
        
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        
        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # build positional encoding for encoder and decoder
        if self.pos_embed == "sincos":
            with torch.no_grad():
                decoder_pos_embed = build_sincos_position_embedding(self.grid_size, 
                                                                    self.decoder_embed_dim, 
                                                                    self.spatial_dims)
                # initialize (and freeze) decoder_pos_embed by sin-cos embedding
                self.decoder_pos_embed.data.copy_(decoder_pos_embed.float())
        else:
            trunc_normal_(self.decoder_pos_embed, std=.02)

        # trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.decoder_cls_token, std=.02)
        trunc_normal_(self.mask_token, std=.02)

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
    
    def random_masking(self, x):
        # compute length for selected and masked
        N, L, D = x.shape
        len_keep = int(L * (1 - self.mask_ratio))
        noise = torch.rand(N, L, device=x.device)
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        
        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)
        
        return x_masked, mask, ids_restore, ids_keep
    
    def forward_encoder(self, x):
        # embed patches with positional embedding added
        x = self.patch_embedding(x)
        
        # masking: length -> length * mask_ratio
        x, mask, ids_restore, ids_keep = self.random_masking(x)
        
        # append cls token
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # apply Transformer blocks
        residual = None
        for blk in self.blocks:
            if self.use_flash_attn:
                x, residual = blk(x, residual)
            else:
                x = blk(x)
        
        # apply normalization
        x = self.norm(x)
        
        return x, mask, ids_restore
    
    def forward_decoder(self, x, ids_restore):
        # embed tokens
        x = self.decoder_embed(x)
        
        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token
        
        # add pos embed
        decoder_cls_token = self.decoder_cls_token.expand(x.shape[0], -1, -1)
        decoder_pos_embed = self.decoder_pos_embed.expand(x.shape[0], -1, -1)
        decoder_pos_embed = torch.cat((decoder_cls_token, decoder_pos_embed), dim=1)
        x = x + decoder_pos_embed
        
        # apply Transformer blocks
        residual = None
        for blk in self.decoder_blocks:
            if self.use_flash_attn:
                x, residual = blk(x, residual)
            else:
                x = blk(x)
        
        # apply normalization
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        # remove cls token
        x = x[:, 1:, :]

        return x
    
    def forward_loss(self, imgs, pred, mask):
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        mask = mask.view(loss.shape)

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss
    
    def forward(self, x):
        latent, mask, ids_restore = self.forward_encoder(x)
        pred = self.forward_decoder(latent, ids_restore)  # [N, L, p*p*p*1]
        loss = self.forward_loss(x, pred, mask)
        
        return loss, pred, mask