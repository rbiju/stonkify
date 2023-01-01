from typing import List

import pytorch_lightning as pl
from flair.data import Sentence
from einops import rearrange

from .base import NLPEmbedder


class NLPEmbeddingModule(pl.LightningModule):
    def __init__(self, embedder: NLPEmbedder):
        super().__init__()
        self.embedder = embedder

    def forward(self, x: List[List[Sentence]]):
        out = [self.embedder.embed(sentence) for sentence in x]
        out = rearrange(out, 'b n dim -> b n dim')

        return out
