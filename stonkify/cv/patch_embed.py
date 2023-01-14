from collections import OrderedDict

import torch
from torch import nn
from einops.layers.torch import Rearrange


class PatchEmbed(nn.Module):
    def __init__(self, h: int, w: int, step: int, embed_dim: int):
        super().__init__()
        if w % step != 0:
            raise ValueError('Width of image must be divisible by step (kernel width)')

        self.patch_embed = nn.Sequential(OrderedDict([
            ('patch_embed', nn.Conv2d(in_channels=1, out_channels=embed_dim, kernel_size=(h, step), stride=step)),
            ('rearrange', Rearrange('b e h w -> b (h w) e'))
        ]))

    def forward(self, x):
        out = self.patch_embed(x)
        return out
