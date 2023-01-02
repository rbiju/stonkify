from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

import yfinance as yf
from torch.utils.data import Dataset, DataLoader

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

    def __getitem__(self, idx) -> tuple[str, datetime, list[str]]:
        when: datetime = self.start + idx * to_relativedelta(self.step)

        articles = self.retriever.get_news(query=f"{self.ticker.info['longName']} financial news",
                                           when_=when)

        try:
            downloaded_articles = self.helper.download(articles)
        except ValueError:
            raise ValueError((f'Number of articles: {self.helper.num_articles} is unavailable for '
                              f'date {when} and ticker {self.ticker}'))

        return self.ticker.ticker, when, [downloaded_article.title for downloaded_article in downloaded_articles]


class NewsDataLoader(DataLoader):
    def __init__(self, dataset: NewsDataset, num_workers: int, batch_size=None):
        super().__init__(dataset=dataset, batch_size=batch_size, num_workers=num_workers)
