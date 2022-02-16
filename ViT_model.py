import torch 
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch import nn
from torch import Tensor
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
from torchsummary import summary
import numpy as np


class PatchEmbedding(nn.Module):
    """
    Patch embedding, position embedding and cls token.
    """
    def __init__(self, in_channels: int = 1, patch_size: int = 16 , emb_size: int=256, img_size: int= 224):
        self.patch_size = patch_size
        super().__init__()
        self.proj = nn.Sequential(
            # break down images into patches(s1xs2) and flatten them.
            # Use conv layer for best performance
            nn.Conv2d(in_channels, emb_size, kernel_size=patch_size, stride=patch_size),
            Rearrange('b e (h) (w) -> b (h w) e'),
        )
        
        #CLS TOKEN
        self.cls_token = nn.Parameter(torch.randn(1,1,emb_size))
            
        #POSTIONS
        self.positions = nn.Parameter(torch.randn((img_size // patch_size) **2 + 1, emb_size))
    
    def forward(self, x: Tensor) -> Tensor:
        b, _, _, _ = x.shape
        x = self.proj(x)
        cls_tokens = repeat(self.cls_token, '() n e -> b n e', b=b)
        #Prepend CLS token to the input
        #x is detached to give fixed random patch projection which reduces training instability
        x = torch.cat([cls_tokens, x.detach()], dim=1)
        # add position embedding
        x += self.positions
        return x


class MultiHeadAttention(nn.Module):
    """
    Attention split accross n heads.
    """
    def __init__(self, emb_size: int, num_heads: int = 8, dropout: float = 0.):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        # fuse the queries, keys and values in one matrix
        self.qkv = nn.Linear(emb_size, emb_size * 3)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)
        
    def forward(self, x : Tensor, mask: Tensor = None) -> Tensor:
        # split keys, queries and values in num_heads
        qkv = rearrange(self.qkv(x), "b n (h d qkv) -> (qkv) b h n d", h=self.num_heads, qkv=3)
        queries, keys, values = qkv[0], qkv[1], qkv[2]
        # sum up over the last axis
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys) # batch, num_heads, query_len, key_len
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)
            
        scaling = self.emb_size ** (1/2)
        att = F.softmax(energy, dim=-1) / scaling
        att = self.att_drop(att)
        # sum up over the third axis
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out

class ResidualAdd(nn.Module):
    """
    Wrapper for residual block.
    """
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x
    
class FeedForwardBlock(nn.Sequential):
    """
    MLP feed forward block.
    """
    def __init__(self, emb_size: int, expansion: int = 4, drop_p: float = 0.):
        super().__init__(
                nn.Linear(emb_size, expansion * emb_size),
                nn.GELU(),
                nn.Dropout(drop_p),
                nn.Linear(expansion *emb_size, emb_size),
                )


class TransformerEncoderBlock(nn.Sequential):
    """
    Encoder block consisting of two residual blocks, one with multihead attention and the other
    with the MLP feedforward block. Both have Norm layers and Dropout.
    """
    def __init__(self, 
                 emb_size: int = 256, 
                 drop_p: float = 0.,
                 forward_expansion: int = 4,
                 forward_drop_p: float = 0.,
                 **kwargs):
        super().__init__(
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                MultiHeadAttention(emb_size, **kwargs),
                nn.Dropout(drop_p)
                )),
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                FeedForwardBlock(
                    emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
                nn.Dropout(drop_p)
                ))
            )

class TransformerEncoder(nn.Sequential):
    """
    Full Transformer Encoder consisting of 'depth' encoder blocks.
    """
    def __init__(self, depth: int = 12, **kwargs):
        super().__init__(*[TransformerEncoderBlock(**kwargs) for _ in range(depth)])


class ClassificationHead(nn.Sequential):
    """
    MLP head for classification.
    Two classes for binary problem.
    """
    def __init__(self, emb_size: int, n_classes: int = 2):
        super().__init__(
                Reduce('b n e -> b e', reduction='mean'),
                nn.LayerNorm(emb_size),
                nn.Linear(emb_size, n_classes))


class ViT(nn.Sequential):
    """
    Full Vision Transformer model.
    """
    def __init__(self,
                 in_channels: int = 1,
                 patch_size: int = 16,
                 emb_size: int = 256,
                 img_size: int = 224,
                 depth: int = 12,
                 n_classes: int = 2,
                 **kwargs):
        super().__init__( 
                PatchEmbedding(in_channels, patch_size, emb_size, img_size),
                TransformerEncoder(depth, emb_size=emb_size, **kwargs),
                ClassificationHead(emb_size, n_classes)
                )



