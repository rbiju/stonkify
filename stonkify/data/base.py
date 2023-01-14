from abc import ABC, abstractmethod
from typing import List
from datetime import datetime, timedelta

from newspaper.article import Article
from flair.data import Sentence
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
    def collate_fn(data_list: list[tuple[str, str, tuple[list[str]]]]):
        label, dates, texts = list(zip(*data_list))
        date_objects = tuple([datetime.strptime(date, "%Y/%m/%d") for date in dates])

        text_sentences = []
        for sentence_list in texts:
            temp_sentences = [Sentence(sentence) for sentence in sentence_list]
            text_sentences.append(temp_sentences)

        return label, date_objects, text_sentences

    @abstractmethod
    def construct_query(self, **kwargs) -> str:
        raise NotImplementedError

    def download_articles(self, articles: list[Article], date: datetime) -> List[str]:
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
