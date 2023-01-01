from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from flair.data import Sentence


class NLPEmbedder(ABC, nn.Module):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def embed(self, x: Sentence) -> torch.Tensor:
        raise NotImplementedError
