from datetime import datetime, timedelta

from stonkify.news import GoogleNewsRetriever
from stonkify.data import NewsDataset, NewsDataLoader, ArticleDownloadHelper


def test_news_retrieval():
    news_retriever = GoogleNewsRetriever()
    out = news_retriever.get_news(query="Microsoft, Inc.")

    return out


def test_dataloader():
    dataset = NewsDataset(retriever=GoogleNewsRetriever(),
                          helper=ArticleDownloadHelper(max_tries=20, num_articles=10),
                          start=datetime(2022, 12, 1),
                          end=datetime(2022, 12, 30),
                          step=timedelta(weeks=1),
                          ticker='MSFT')

    dataloader = NewsDataLoader(dataset=dataset, num_workers=0)

    batch = next(iter(dataloader))

    return batch


if __name__ == "__main__":
    test_dataloader()
