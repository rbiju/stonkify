from typing import List

from newspaper.article import ArticleException, Article

from .constants import SKIP_PHRASES


class ArticleDownloadHelper:
    def __init__(self, max_tries: int, num_articles: int):
        self.max_tries = max_tries
        self.num_articles = num_articles

    def download(self, articles: List[Article]) -> List[str]:
        downloaded_articles = []
        try_count = 0
        article_count = 0
        while article_count < self.num_articles and try_count < self.max_tries:
            article = articles[try_count]
            try:
                article.download()
                article.parse()
                if any(substring in article.text for substring in SKIP_PHRASES):
                    try_count += 1
                    continue
            except ArticleException:
                try_count += 1
                continue

            downloaded_articles.append(article)
            article_count += 1
            try_count += 1

        if len(downloaded_articles) < self.num_articles:
            raise ValueError

        return [downloaded_article.text for downloaded_article in downloaded_articles]
