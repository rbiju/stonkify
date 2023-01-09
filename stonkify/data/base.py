from abc import ABC, abstractmethod
from datetime import datetime, timedelta

from newspaper.article import Article
from torch.utils.data import Dataset

from stonkify.news import NewsRetriever
from .helpers import ArticleDownloadHelper


class NewsDataset(ABC, Dataset):
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

    @staticmethod
    @abstractmethod
    def collate_fn(data_list):
        raise NotImplementedError

    @abstractmethod
    def construct_query(self, **kwargs) -> str:
        raise NotImplementedError

    def download_articles(self, articles: list[Article], date: datetime):
        try:
            downloaded_articles = self.helper.download(articles)
        except ValueError:
            raise ValueError(f'Number of articles: {self.helper.num_articles} is unavailable for date {date}')

        return downloaded_articles

    def __len__(self):
        return (self.end - self.start) // self.step

    @abstractmethod
    def __getitem__(self, idx) -> tuple[str, str, list[str]]:
        raise NotImplementedError
