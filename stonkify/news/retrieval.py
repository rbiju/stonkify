from typing import List, Optional
from datetime import datetime

from newspaper import Article

from stonkify.scrapers import NewsSearch, GoogleNews, VanillaRequest
from .base import NewsRetriever


class GoogleNewsRetriever(NewsRetriever):
    def __init__(self, news_engine: NewsSearch = GoogleNews(request_modifier=VanillaRequest())):
        super().__init__()
        self.engine = news_engine

    def get_news(self, query: str,
                 from_: Optional[datetime] = None,
                 to_: Optional[datetime] = None) -> List[Article]:
        if from_ is not None:
            from_ = from_.strftime("%m/%d/%Y")
        if to_ is not None:
            to_ = to_.strftime("%m/%d/%Y")

        d = self.engine.search(query, from_=from_, to_=to_)

        articles = []
        for entry in d['entries']:
            articles.append(Article(entry['link']))

        return articles
