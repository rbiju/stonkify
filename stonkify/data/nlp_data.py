from typing import List
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

from torch.utils.data import Dataset

from newspaper.article import ArticleException
import yfinance as yf

from stonkify.news import NewsRetriever


def to_relativedelta(tdelta: timedelta) -> relativedelta:
    return relativedelta(seconds=int(tdelta.total_seconds()),
                         microseconds=tdelta.microseconds)


class NewsDataset(Dataset):
    def __init__(self, retriever: NewsRetriever,
                 num_articles: int,
                 start: datetime,
                 end: datetime,
                 span: timedelta,
                 step: timedelta,
                 ticker: str):
        self.MAX_TRIES = 50

        self.retriever = retriever
        self.ticker = yf.Ticker(ticker)
        self.num_articles = num_articles

        self.start = start
        self.span = span
        self.end = end
        self.step = step

    def __len__(self):
        return ((self.end - self.start) - self.span) // self.step

    def __getitem__(self, idx) -> List[str]:
        start: datetime = self.start + idx * to_relativedelta(self.step)
        end: datetime = start + to_relativedelta(self.span)

        articles = self.retriever.get_news(query=f"{self.ticker.info['longName']} financial news",
                                           from_=start,
                                           to_=end)

        downloaded_articles = []
        try_count = 0
        article_count = 0
        while article_count < self.num_articles and try_count < self.MAX_TRIES:
            article = articles[try_count]
            try:
                article.download()
            except ArticleException:
                try_count += 1
                continue

            article.parse()

            downloaded_articles.append(article)
            article_count += 1
            try_count += 1

        if len(downloaded_articles) < self.num_articles:
            raise ValueError(f'Number of articles: {self.num_articles} is unavailable in time range {start} - {end}')

        return [downloaded_article.title for downloaded_article in downloaded_articles]
