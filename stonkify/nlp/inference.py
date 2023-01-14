from datetime import datetime

import pytorch_lightning as pl
from flair.data import Sentence

from .base import NLPEmbedder


class NLPInferenceModule(pl.LightningModule):
    def __init__(self, embedder: NLPEmbedder):
        super().__init__()
        self.embedder = embedder

    def forward(self, x: list[Sentence]):
        out = self.embedder.embed(x)

        return out

    def predict_step(self, batch: tuple[tuple[str], tuple[datetime], list[list[Sentence]]],
                     batch_idx: int,
                     dataloader_idx: int = 0):
        labels, dates, titles_list = batch
        embeddings = [self.forward(titles) for titles in titles_list]

        return {"labels": labels,
                "dates": dates,
                "embeddings": embeddings}
