from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

import yfinance as yf
from torch.utils.data import Dataset, DataLoader

from stonkify.news import NewsRetriever
from .helpers import ArticleDownloadHelper


def to_relativedelta(tdelta: timedelta) -> relativedelta:
    return relativedelta(seconds=int(tdelta.total_seconds()),
                         microseconds=tdelta.microseconds)


def news_collate(data_list: list[tuple[str, str, list[tuple[str]]]]):
    ticker, dates, texts = list(zip(*data_list))
    date_objects = tuple([datetime.strptime(date, "%Y/%m/%d") for date in dates])

    return ticker, date_objects, list(texts)


def global_news_collate(data_list: list[tuple[str, list[tuple[str]]]]):
    dates, texts = list(zip(*data_list))
    date_objects = tuple([datetime.strptime(date, "%Y/%m/%d") for date in dates])

    return date_objects, list(texts)


class NewsDataset(Dataset):
    def __init__(self,
                 retriever: NewsRetriever,
                 helper: ArticleDownloadHelper,
                 start: datetime,
                 end: datetime,
                 step: timedelta,
                 ticker: str,
                 query: str):

        self.retriever = retriever
        self.helper = helper
        self.ticker = ticker

        self.start = start
        self.end = end
        self.step = step

        self.query = query

    def __len__(self):
        return (self.end - self.start) // self.step

    def __getitem__(self, idx) -> tuple[str, str, list[str]]:
        ticker = yf.Ticker(self.ticker)
        to: datetime = self.start + idx * to_relativedelta(self.step)

        articles = self.retriever.get_news(query=f"{ticker.info['longName']} {self.query}",
                                           to_=to)

        try:
            downloaded_articles = self.helper.download(articles)
        except ValueError:
            raise ValueError((f'Number of articles: {self.helper.num_articles} is unavailable for '
                              f'date {to} and ticker {self.ticker}'))

        return ticker.ticker, to.strftime("%Y/%m/%d"), downloaded_articles


class GlobalNewsDataset(Dataset):
    def __init__(self,
                 retriever: NewsRetriever,
                 helper: ArticleDownloadHelper,
                 start: datetime,
                 end: datetime,
                 step: timedelta,
                 query: str):

        self.retriever = retriever
        self.helper = helper

        self.start = start
        self.end = end
        self.step = step

        self.query = query

    def __len__(self):
        return (self.end - self.start) // self.step

    def __getitem__(self, idx) -> tuple[str, list[str]]:
        to: datetime = self.start + idx * to_relativedelta(self.step)

        articles = self.retriever.get_news(query=f"{self.query}",
                                           to_=to)

        try:
            downloaded_articles = self.helper.download(articles)
        except ValueError:
            raise ValueError((f'Number of articles: {self.helper.num_articles} is unavailable for '
                              f'date {to} with number of tries {self.helper.max_tries}'))

        return to.strftime("%Y/%m/%d"), downloaded_articles


class NewsDataLoader(DataLoader):
    def __init__(self, dataset: NewsDataset, batch_size: int, num_workers: int):
        super().__init__(dataset=dataset, batch_size=batch_size, num_workers=num_workers, collate_fn=news_collate)


class GlobalNewsDataLoader(DataLoader):
    def __init__(self, dataset: GlobalNewsDataset, batch_size: int, num_workers: int):
        super().__init__(dataset=dataset, batch_size=batch_size, num_workers=num_workers, collate_fn=global_news_collate)
