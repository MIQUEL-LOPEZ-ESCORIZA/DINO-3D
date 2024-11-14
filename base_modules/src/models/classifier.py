from typing import Optional, Any

import math

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.nn.modules import MultiheadAttention, Linear, Dropout, BatchNorm1d, TransformerEncoderLayer


class Linear_Classifier(nn.Module):
    def __init__(self, dimension, class_number):
        super().__init__()
        self.ln = nn.LayerNorm(dimension)
        self.linear = nn.Linear(dimension, class_number)
        # self.relu = nn.ReLU()
        # self.linear2 = nn.Linear(dimension // 2, class_number)

    def forward(self, x):
        #x = self.relu(self.linear1(x))
        x = self.linear(self.ln(x))
        return x
    
    
def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu
    raise ValueError("activation should be relu/gelu, not {}".format(activation))
    
    
class TransformerBatchNormEncoderLayer(nn.modules.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super(TransformerBatchNormEncoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        
        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model)

        self.norm1 = BatchNorm1d(d_model, eps=1e-5)  # normalizes each feature across batch samples and time steps
        self.norm2 = BatchNorm1d(d_model, eps=1e-5)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerBatchNormEncoderLayer, self).__setstate__(state)

    def forward(self, src: Tensor, src_mask: Optional[Tensor] = None, is_causal=False,
                src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        src2 = self.self_attn(src, src, src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)  # (seq_len, batch_size, d_model)
        src = src.permute(1, 2, 0)  # (batch_size, d_model, seq_len)
        src = self.norm1(src)
        src = src.permute(2, 0, 1)  # restore (seq_len, batch_size, d_model)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)  # (seq_len, batch_size, d_model)
        src = src.permute(1, 2, 0)  # (batch_size, d_model, seq_len)
        src = self.norm2(src)
        src = src.permute(2, 0, 1)  # restore (seq_len, batch_size, d_model)
        return src
    
    
class Attention_Classifier(nn.Module):
    def __init__(self, feat_dim, d_model, n_heads, num_layers, dim_feedforward, num_classes,
                 dropout=0., activation='gelu', norm='BatchNorm', freeze=False):
        super(Attention_Classifier, self).__init__()

        self.num_layers = num_layers
        self.d_model = d_model
        self.n_heads = n_heads

        self.project_inp = nn.Linear(feat_dim, d_model)
        
        if norm == 'LayerNorm':
            encoder_layer = TransformerEncoderLayer(self.d_model, self.n_heads, dim_feedforward, dropout*(1.0 - freeze), activation=activation)
        else:
            encoder_layer = TransformerBatchNormEncoderLayer(self.d_model, self.n_heads, dim_feedforward, dropout*(1.0 - freeze), activation=activation)

        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)

        self.act = _get_activation_fn(activation)

        self.dropout = nn.Dropout(dropout)

        self.feat_dim = feat_dim
        self.num_classes = num_classes
        
        # cls token initialization
        self.cls_token = nn.Parameter(torch.zeros(1, 1, feat_dim))
        nn.init.trunc_normal_(self.cls_token, std=.02)
        
        self.output_layer = self.build_output_module(d_model, num_classes)

    def build_output_module(self, d_model, num_classes):
        output_layer = nn.Linear(d_model, num_classes)
        return output_layer

    def forward(self, X):
        # concat cls token to input
        X = torch.cat((self.cls_token.expand(X.shape[0], -1, -1), X), dim=1)
        # permute because pytorch convention for transformers is [seq_length, batch_size, feat_dim]. padding_masks [batch_size, feat_dim]
        inp = X.permute(1, 0, 2)
        inp = self.project_inp(inp) * math.sqrt(self.d_model)  # [seq_length, batch_size, d_model] project input vectors to d_model dimensional space
        
        output = self.transformer_encoder(inp)
        output = self.act(output)  # the output transformer encoder/decoder embeddings don't include non-linearity
        output = output.permute(1, 0, 2)  # (batch_size, seq_length, d_model)
        output = self.dropout(output)

        # Output
        #if cls_token:
        output = output[:, 0, :]  # only use the first output vector (CLS token)
        # else:
        #    output = output[:, 1:, :] # remove CLS token
           
        return self.output_layer(output)