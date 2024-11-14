import collections
from typing import Optional

import torch
import torch.nn as nn

from timm.models.layers import to_2tuple, to_3tuple


def build_sincos_position_embedding(
    grid_size: Optional[int], embed_dim: int, spatial_dims: int = 3, temperature: float = 10000.0
) -> torch.nn.Parameter:
    """
    Builds a sin-cos position embedding based on the given grid size, embed dimension, spatial dimensions, and temperature.
    Reference: https://github.com/cvlab-stonybrook/SelfMedMAE/blob/68d191dfcc1c7d0145db93a6a570362de29e3b30/lib/models/mae3d.py

    Args:
        grid_size (int or Tuple[int]): The size of the grid in each spatial dimension.
        embed_dim (int): The dimension of the embedding.
        spatial_dims (int): The number of spatial dimensions (2 for 2D, 3 for 3D).
        temperature (float): The temperature for the sin-cos position embedding.

    Returns:
        pos_embed (nn.Parameter): The sin-cos position embedding as a learnable parameter.
    """

    if spatial_dims == 2:
        grid_size = to_2tuple(grid_size)
        h, w = grid_size
        grid_h = torch.arange(h, dtype=torch.float32)
        grid_w = torch.arange(w, dtype=torch.float32)

        grid_h, grid_w = torch.meshgrid(grid_h, grid_w, indexing='ij')

        assert embed_dim % 4 == 0, "Embed dimension must be divisible by 4 for 2D sin-cos position embedding"

        pos_dim = embed_dim // 4
        omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
        omega = 1.0 / (temperature**omega)
        out_h = torch.einsum("m,d->md", [grid_h.flatten(), omega])
        out_w = torch.einsum("m,d->md", [grid_w.flatten(), omega])
        pos_emb = torch.cat([torch.sin(out_h), torch.cos(out_h), torch.sin(out_w), torch.cos(out_w)], dim=1)[None, :, :]
    elif spatial_dims == 3:
        grid_size = to_3tuple(grid_size)
        h, w, d = grid_size
        grid_h = torch.arange(w, dtype=torch.float32)
        grid_w = torch.arange(h, dtype=torch.float32)
        grid_d = torch.arange(d, dtype=torch.float32)

        grid_h, grid_w, grid_d = torch.meshgrid(grid_h, grid_w, grid_d, indexing='ij')

        assert embed_dim % 6 == 0, "Embed dimension must be divisible by 6 for 3D sin-cos position embedding"

        pos_dim = embed_dim // 6
        omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
        omega = 1.0 / (temperature**omega)
        out_h = torch.einsum("m,d->md", [grid_h.flatten(), omega])
        out_w = torch.einsum("m,d->md", [grid_w.flatten(), omega])
        out_d = torch.einsum("m,d->md", [grid_d.flatten(), omega])
        pos_emb = torch.cat(
            [
                torch.sin(out_w),
                torch.cos(out_w),
                torch.sin(out_h),
                torch.cos(out_h),
                torch.sin(out_d),
                torch.cos(out_d),
            ],
            dim=1,
        )[None, :, :]
    else:
        raise NotImplementedError("Spatial Dimension Size {spatial_dims} Not Implemented!")

    pos_embed = nn.Parameter(pos_emb)
    pos_embed.requires_grad = False

    return pos_embed

def nth_root(N,k):
    """Return greatest integer x such that x**k <= N"""
    """https://stackoverflow.com/questions/15978781/how-to-find-integer-nth-roots"""
    x = int(N**(1/k))      
    while (x+1)**k <= N:
        x += 1
    while x**k > N:
        x -= 1
    return x

# --------------------------------------------------------
# Interpolate position embeddings for high-resolution
# Modified from References:
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------
def interpolate_pos_embed(
    model: torch.nn.Module, 
    checkpoint_model: collections.OrderedDict, 
    spatial_dims: int = 3,
) -> None:
    """
    Interpolates the position embedding in the given model checkpoint.

    :param model: The model to interpolate the position embedding in.
    :type model: torch.nn.Module
    :param checkpoint_model: The checkpoint model containing the position embedding.
    :type checkpoint_model: collections.OrderedDict
    :param spatial_dims: The number of spatial dimensions. Defaults to 3.
    :type spatial_dims: int
    :returns: None
    """
    
    if 'patch_embedding.position_embeddings' in checkpoint_model:
        pos_embed_checkpoint = checkpoint_model['patch_embedding.position_embeddings']
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_patches = model.patch_embedding.n_patches
        num_extra_tokens = model.patch_embedding.position_embeddings.shape[-2] - num_patches
        # height (== width) for the checkpoint position embedding
        if spatial_dims == 2 or spatial_dims == 3:
            orig_size = int(nth_root(pos_embed_checkpoint.shape[-2] - num_extra_tokens, spatial_dims))
        else:
            raise NotImplementedError(f"Spatial Dimension Size {spatial_dims} Not Implemented!")
            
        # height (== width) for the new position embedding
        new_size = int(nth_root(num_patches, spatial_dims))

        # class_token and dist_token are kept unchanged
        if orig_size != new_size:
            print("Position interpolate from origin size %d to new size %d" % (orig_size, new_size))
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
            # only the position tokens are interpolated
            pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
            if spatial_dims == 2:
                pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
                pos_tokens = torch.nn.functional.interpolate(
                    pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
                pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
            elif spatial_dims == 3:
                pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, orig_size, embedding_size).permute(0, 4, 1, 2, 3)
                pos_tokens = torch.nn.functional.interpolate(
                    pos_tokens, size=(new_size, new_size, new_size), mode='trilinear', align_corners=False)
                pos_tokens = pos_tokens.permute(0, 2, 3, 4, 1).flatten(1, 3)
            else:
                raise NotImplementedError(f"Spatial Dimension Size {spatial_dims} Not Implemented!")
            
            new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
            checkpoint_model['patch_embedding.position_embeddings'] = new_pos_embed