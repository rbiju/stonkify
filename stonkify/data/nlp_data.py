from typing import List
from dateutil.relativedelta import relativedelta
from datetime import datetime, timedelta

import yfinance as yf
from torch.utils.data import Dataset

from stonkify.news import NewsRetriever
from .helpers import ArticleDownloadHelper


def to_relativedelta(tdelta: timedelta) -> relativedelta:
    return relativedelta(seconds=int(tdelta.total_seconds()),
                         microseconds=tdelta.microseconds)


class NewsDataset(Dataset):
    def __init__(self, retriever: NewsRetriever,
                 helper: ArticleDownloadHelper,
                 start: datetime,
                 end: datetime,
                 step: timedelta,
                 ticker: str):

        self.retriever = retriever
        self.helper = helper
        self.ticker = yf.Ticker(ticker)

        self.start = start
        self.end = end
        self.step = step

    def __len__(self):
        return (self.end - self.start) // self.step

    def __getitem__(self, idx) -> List[str]:
        when: datetime = self.start + idx * to_relativedelta(self.step)

        articles = self.retriever.get_news(query=f"{self.ticker.info['longName']} financial news",
                                           when_=when)

        try:
            downloaded_articles = self.helper.download(articles)
        except ValueError:
            raise ValueError((f'Number of articles: {self.helper.num_articles} is unavailable for '
                             f'date {when} and ticker {self.ticker}'))

        return [downloaded_article.title for downloaded_article in downloaded_articles]
