from typing import List, Optional
from datetime import datetime

from newspaper import Article

from stonkify.scrapers import GoogleNews
from .base import NewsRetriever


class GoogleNewsRetriever(NewsRetriever):
    def __init__(self, news_engine: GoogleNews):
        super().__init__()
        self.engine = news_engine

    def get_news(self, query: str,
                 from_: Optional[datetime or None] = None,
                 to_: Optional[datetime or None] = None) -> List[Article]:
        f = from_.strftime("%m/%d/%Y")
        t = to_.strftime("%m/%d/%Y")
        d = self.engine.search(query, from_=f, to_=t)

        articles = []
        for entry in d['entries']:
            articles.append(Article(entry['link']))

        return articles
