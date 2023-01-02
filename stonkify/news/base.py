from abc import ABC, abstractmethod
from typing import List, Optional
from datetime import datetime

from newspaper import Article


class NewsRetriever(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def get_news(self, query: str,
                 when_: Optional[datetime] = None,
                 from_: Optional[datetime] = None,
                 to_: Optional[datetime] = None) -> List[Article]:
        raise NotImplementedError
