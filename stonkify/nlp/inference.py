from datetime import datetime

import pytorch_lightning as pl
from flair.data import Sentence

from .base import NLPEmbedder


class NLPInferenceModule(pl.LightningModule):
    def __init__(self, embedder: NLPEmbedder):
        super().__init__()
        self.embedder = embedder

    @staticmethod
    def preprocess(titles: list[str]) -> list[Sentence]:
        sentence_titles = [Sentence(title) for title in titles]

        return sentence_titles

    def forward(self, x: list[Sentence]):
        out = self.embedder.embed(x)

        return out

    def predict_step(self, batch: tuple[tuple[str], tuple[datetime], list[list[str]]],
                     batch_idx: int,
                     dataloader_idx: int = 0):
        tickers, dates, titles = batch
        sentences = [self.preprocess(titles=title) for title in titles]

        embeddings = [self.forward(sentence) for sentence in sentences]

        return {"tickers": tickers,
                "dates": dates,
                "embeddings": embeddings}
