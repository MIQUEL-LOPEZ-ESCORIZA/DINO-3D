import torch
import torch.nn as nn

# from monai.networks.blocks.mlp import MLPBlock

#from torchvision.ops import StochasticDepth

# from flash_attn.modules.mha import MHA

# class FlashAttentionBlock(nn.Module):
#     """
#     A transformer block with FlashAttention
#     """

#     def __init__(
#         self,
#         hidden_size: int,
#         mlp_dim: int,
#         num_heads: int,
#         dropout_rate: float = 0.0,
#         qkv_bias: bool = False,
#         save_attn: bool = False,
#     ) -> None:
#         """
#         Args:
#             hidden_size (int): dimension of hidden layer.
#             mlp_dim (int): dimension of feedforward layer.
#             num_heads (int): number of attention heads.
#             dropout_rate (float, optional): fraction of the input units to drop. Defaults to 0.0.
#             qkv_bias (bool, optional): apply bias term for the qkv linear layer. Defaults to False.
#             save_attn (bool, optional): to make accessible the attention matrix. Defaults to False.

#         """

#         super().__init__()

#         if not (0 <= dropout_rate <= 1):
#             raise ValueError("dropout_rate should be between 0 and 1.")

#         if hidden_size % num_heads != 0:
#             raise ValueError("hidden_size should be divisible by num_heads.")

#         self.dropout_rate = dropout_rate

#         self.mlp = MLPBlock(hidden_size, mlp_dim, dropout_rate)
#         self.norm1 = nn.LayerNorm(hidden_size, eps=1e-6)
#         self.norm2 = nn.LayerNorm(hidden_size, eps=1e-6)
        
#         # self.dropout1 = nn.Dropout(dropout_rate)
#         # self.dropout2 = nn.Dropout(dropout_rate)
        
#         # self.drop_path1 = StochasticDepth(dropout_rate, mode="row")
#         # self.drop_path2 = StochasticDepth(dropout_rate, mode="row")
        
#         #self.mlp = FusedMLP(in_features=hidden_size, hidden_features=mlp_dim, return_residual=False)
        
#         self.attn = MHA(hidden_size, num_heads, dropout=dropout_rate, \
#             qkv_proj_bias=qkv_bias, use_flash_attn=True)

#     def forward(self, hidden_states, residual=None):
#         hidden_states = hidden_states + self.attn(self.norm1(hidden_states))
#         hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
#         return hidden_states, residual



import torch.nn as nn
import torch.nn.functional as F
from monai.networks.blocks.mlp import MLPBlock
# from monai.networks.blocks.selfattention import SABlock

from monai.utils import optional_import

einops, has_einops = optional_import("einops")


class SABlock(nn.Module):
    """
    A self-attention block, based on: "Dosovitskiy et al.,
    An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale <https://arxiv.org/abs/2010.11929>"
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        dropout_rate: float = 0.0,
    ) -> None:
        """
        Args:
            hidden_size: dimension of hidden layer.
            num_heads: number of attention heads.
            dropout_rate: faction of the input units to drop.

        """

        super().__init__()

        if not (0 <= dropout_rate <= 1):
            raise AssertionError("dropout_rate should be between 0 and 1.")

        if hidden_size % num_heads != 0:
            raise AssertionError("hidden size should be divisible by num_heads.")

        self.num_heads = num_heads
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.qkv = nn.Linear(hidden_size, hidden_size * 3, bias=False)
        self.drop_output = nn.Dropout(dropout_rate)
        self.drop_weights = nn.Dropout(dropout_rate)
        self.dropout_rate = dropout_rate
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim ** -0.5
        if has_einops:
            self.rearrange = einops.rearrange
        else:
            raise ValueError('"Requires einops.')

    def forward(self, x):
        q, k, v = self.rearrange(self.qkv(x), "b h (qkv l d) -> qkv b l h d", qkv=3, l=self.num_heads)
        # att_mat = (torch.einsum("blxd,blyd->blxy", q, k) * self.scale).softmax(dim=-1)
        # att_mat = self.drop_weights(att_mat)
        # x = torch.einsum("bhxy,bhyd->bhxd", att_mat, v)
        # x = self.rearrange(x, "b h l d -> b l (h d)")
        # with torch.backends.cuda.sdp_kernel(enable_math=False):
        #x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout_rate, scale=self.scale)
        x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout_rate)
        x = self.rearrange(x, "b h l d -> b l (h d)")
        x = self.out_proj(x)
        x = self.drop_output(x)
        return x


class FlashAttentionBlock(nn.Module):
    """
    A transformer block, based on: "Dosovitskiy et al.,
    An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale <https://arxiv.org/abs/2010.11929>"
    """


    def __init__(
        self,
        hidden_size: int,
        mlp_dim: int,
        num_heads: int,
        dropout_rate: float = 0.0,
        qkv_bias: bool = False,
        save_attn: bool = False,
    ) -> None:
        """
        Args:
            hidden_size (int): dimension of hidden layer.
            mlp_dim (int): dimension of feedforward layer.
            num_heads (int): number of attention heads.
            dropout_rate (float, optional): fraction of the input units to drop. Defaults to 0.0.
            qkv_bias (bool, optional): apply bias term for the qkv linear layer. Defaults to False.
            save_attn (bool, optional): to make accessible the attention matrix. Defaults to False.

        """

        super().__init__()

        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")

        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size should be divisible by num_heads.")

        self.mlp = MLPBlock(hidden_size, mlp_dim, dropout_rate)
        self.norm1 = nn.LayerNorm(hidden_size)
        # self.attn = SABlock(hidden_size, num_heads, dropout_rate, qkv_bias, save_attn)
        self.attn = SABlock(hidden_size, num_heads, dropout_rate)
        self.norm2 = nn.LayerNorm(hidden_size)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

