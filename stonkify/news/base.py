from abc import ABC, abstractmethod
from typing import List
from datetime import datetime

from newspaper import Article


class NewsRetriever(ABC):
    def __init__(self):
        raise NotImplementedError

    @abstractmethod
    def get_news(self, query: str, from_: datetime, to_: datetime) -> List[Article]:
        raise NotImplementedError
