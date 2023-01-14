from datetime import datetime, timedelta

import yfinance as yf
from torch.utils.data import DataLoader

from stonkify.news import NewsRetriever
from .base import NewsDataset
from .helpers import ArticleDownloadHelper, to_relativedelta


class TickerNewsDataset(NewsDataset):
    def __init__(self,
                 retriever: NewsRetriever,
                 helper: ArticleDownloadHelper,
                 start: datetime,
                 end: datetime,
                 step: timedelta,
                 ticker: str,
                 query: str):
        super().__init__(retriever=retriever, helper=helper, start=start, end=end, step=step, query=query)

        self.ticker = ticker

    def construct_query(self, ticker: yf.Ticker) -> str:
        return f"{ticker.ticker} {self.query}"

    def __getitem__(self, idx) -> tuple[str, str, list[str]]:
        ticker = yf.Ticker(self.ticker)
        to: datetime = self.start + idx * to_relativedelta(self.step)

        articles = self.retriever.get_news(self.construct_query(ticker),
                                           to_=to)

        downloaded_articles = self.download_articles(articles, to)

        return ticker.ticker, to.strftime("%Y/%m/%d"), downloaded_articles


class GlobalNewsDataset(NewsDataset):
    def __init__(self,
                 retriever: NewsRetriever,
                 helper: ArticleDownloadHelper,
                 start: datetime,
                 end: datetime,
                 step: timedelta,
                 query: str):
        super().__init__(retriever=retriever, helper=helper, start=start, end=end, step=step, query=query)

    def construct_query(self) -> str:
        return self.query

    def __getitem__(self, idx) -> tuple[str, str, list[str]]:
        to: datetime = self.start + idx * to_relativedelta(self.step)

        articles = self.retriever.get_news(query=f"{self.query}",
                                           to_=to)

        downloaded_articles = self.download_articles(articles, to)

        return "GLOBAL_NEWS", to.strftime("%Y/%m/%d"), downloaded_articles


class NewsDataLoader(DataLoader):
    def __init__(self, dataset: NewsDataset, batch_size: int, num_workers: int):
        super().__init__(dataset=dataset, batch_size=batch_size, num_workers=num_workers, collate_fn=dataset.collate_fn,
                         drop_last=True)
