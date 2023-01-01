from typing import List

from einops import rearrange
from flair.embeddings import DocumentEmbeddings
from flair.data import Sentence

from .base import NLPEmbedder


class FlairEmbedder(NLPEmbedder):
    def __init__(self, embedding: DocumentEmbeddings):
        super().__init__()
        self.embedding = embedding

    def embed(self, x: List[Sentence]):
        sentences: List[Sentence] = self.embedding.embed(x)
        embeddings = [sentence.get_embedding() for sentence in sentences]

        return rearrange(embeddings, 'n dim -> n dim')
