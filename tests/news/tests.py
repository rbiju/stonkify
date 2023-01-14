from datetime import datetime, timedelta

from flair.embeddings import SentenceTransformerDocumentEmbeddings

from stonkify.news import GoogleNewsRetriever
from stonkify.data import TickerNewsDataset, NewsDataLoader, ArticleDownloadHelper
from stonkify.nlp import NLPInferenceModule, FlairEmbedder

test_batch = (('MSFT', 'MSFT'),
              (datetime(2022, 10, 1, 0, 0), datetime(2022, 10, 8, 0, 0)),
              [["What Is Microsoft Corporation's (NASDAQ:MSFT) Share Price Doing?",
                'Declining Stock and Solid Fundamentals: Is The Market Wrong About Microsoft Corporation (NASDAQ:MSFT)?',
                'Microsoft Stock Tests Recent Resistance (NASDAQ:MSFT)',
                'Heading Into Its Q3, Is Microsoft (MSFT) A Buy?',
                'Is Trending Stock Microsoft Corporation (MSFT) a Buy Now?',
                'Where Will Microsoft Stock Be In 5 Years? (NASDAQ:MSFT)',
                'The Zacks Analyst Blog Highlights Microsoft, Danaher, Raytheon Technologies, Stryker Corporation and Vale',
                "Microsoft Stock: It's Not Always A Buy (NASDAQ:MSFT)", 'Is Microsoft Stock Cheap? (NASDAQ:MSFT)',
                'Top Research Reports for Microsoft, Meta Platforms & Thermo Fisher Scientific'],
               ["What Is Microsoft Corporation's (NASDAQ:MSFT) Share Price Doing?",
                'Declining Stock and Solid Fundamentals: Is The Market Wrong About Microsoft Corporation (NASDAQ:MSFT)?',
                'Microsoft Stock Tests Recent Resistance (NASDAQ:MSFT)',
                'Heading Into Its Q3, Is Microsoft (MSFT) A Buy?',
                'Is Trending Stock Microsoft Corporation (MSFT) a Buy Now?',
                "The Two Sides of Microsoft: Strong Fundamentals vs. Insiders' Selling Activity",
                'Where Will Microsoft Stock Be In 5 Years? (NASDAQ:MSFT)',
                'The Zacks Analyst Blog Highlights Microsoft, Danaher, Raytheon Technologies, Stryker Corporation and Vale',
                "Microsoft Stock: It's Not Always A Buy (NASDAQ:MSFT)", 'Is Microsoft Stock Cheap? (NASDAQ:MSFT)']])


def test_news_retrieval():
    news_retriever = GoogleNewsRetriever()
    out = news_retriever.get_news(query="Microsoft, Inc.")

    return out


def test_dataloader():
    dataset = TickerNewsDataset(retriever=GoogleNewsRetriever(),
                                helper=ArticleDownloadHelper(max_tries=20, num_articles=10),
                                start=datetime(2022, 10, 1),
                                end=datetime(2022, 12, 30),
                                step=timedelta(weeks=1),
                                ticker='MSFT',
                                query="stock analysis")

    dataloader = NewsDataLoader(dataset=dataset, batch_size=2, num_workers=6)

    batch = next(iter(dataloader))

    return batch


def test_embedder():
    batch = test_batch

    embedder = NLPInferenceModule(
        embedder=FlairEmbedder(
            embedding=SentenceTransformerDocumentEmbeddings('stsb-distilroberta-base-v2')))

    embeddings = embedder.predict_step(batch, batch_idx=0)

    return embeddings


def test_pipeline():
    batch = test_dataloader()

    embedder = NLPInferenceModule(
        embedder=FlairEmbedder(
            embedding=SentenceTransformerDocumentEmbeddings('stsb-distilroberta-base-v2')))

    embeddings = embedder.predict_step(batch, batch_idx=0)

    return embeddings


if __name__ == "__main__":
    test_pipeline()
