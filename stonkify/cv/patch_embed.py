from collections import OrderedDict

import torch
from torch import nn


class PatchEmbed(nn.Module):
    def __init__(self, h: int, w: int, step: int):
        super().__init__()
        self.patch_embed = nn.Sequential(OrderedDict(
            'linear1'
        ))