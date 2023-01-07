from datetime import datetime, timedelta

from flair.embeddings import SentenceTransformerDocumentEmbeddings

from stonkify.news import GoogleNewsRetriever
from stonkify.data import NewsDataset, NewsDataLoader, ArticleDownloadHelper
from stonkify.nlp import NLPInferenceModule, FlairEmbedder

test_batch = (('MSFT', 'MSFT'),
              (datetime(2022, 10, 1, 0, 0), datetime(2022, 10, 8, 0, 0)),
              [['Microsoft earnings press release available on Investor Relations website',
                'Microsoft misses estimates but stock up 5% on rosy guidance',
                'Microsoft Q4 FY2022 Earnings Report Recap', 'Microsoft announces quarterly dividend increase',
                'Microsoft announces quarterly dividend increase',
                'Why Microsoft Stock Popped, Then Dropped, on Wednesday',
                'Microsoft to acquire Activision Blizzard to bring the joy and community of gaming to everyone, across every device',
                'Barclays deploys Microsoft Teams globally as its preferred collaboration platform to enable better connectivity for its employees worldwide',
                'Microsoft eases up on hiring as economic concerns hit more of the tech industry',
                'Microsoft Inspire 2022: Unlocking new partner opportunities, solutions for hybrid work and more'],
               ['Microsoft earnings press release available on Investor Relations website',
                'Microsoft misses estimates but stock up 5% on rosy guidance',
                'Microsoft Q4 FY2022 Earnings Report Recap', 'Microsoft announces quarterly dividend increase',
                'Microsoft announces quarterly dividend increase',
                'Why Microsoft Stock Popped, Then Dropped, on Wednesday',
                'Microsoft to acquire Activision Blizzard to bring the joy and community of gaming to everyone, across every device',
                'Barclays deploys Microsoft Teams globally as its preferred collaboration platform to enable better connectivity for its employees worldwide',
                'Microsoft eases up on hiring as economic concerns hit more of the tech industry',
                'Microsoft Inspire 2022: Unlocking new partner opportunities, solutions for hybrid work and more']])


def test_news_retrieval():
    news_retriever = GoogleNewsRetriever()
    out = news_retriever.get_news(query="Microsoft, Inc.")

    return out


def test_dataloader():
    dataset = NewsDataset(retriever=GoogleNewsRetriever(),
                          helper=ArticleDownloadHelper(max_tries=20, num_articles=10),
                          start=datetime(2022, 10, 1),
                          end=datetime(2022, 12, 30),
                          step=timedelta(weeks=1),
                          ticker='MSFT',
                          query="stock analysis")

    dataloader = NewsDataLoader(dataset=dataset, batch_size=2, num_workers=2)

    batch = next(iter(dataloader))

    return batch


def test_embedder():
    batch = test_batch

    embedder = NLPInferenceModule(
        embedder=FlairEmbedder(
            embedding=SentenceTransformerDocumentEmbeddings('stsb-distilroberta-base-v2')))

    embeddings = embedder.predict_step(batch, batch_idx=0)

    return embeddings


if __name__ == "__main__":
    test_embedder()
