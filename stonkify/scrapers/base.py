from typing import Optional

from abc import ABC, abstractmethod
import requests
import feedparser


class RequestModifier(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def _get(self, url: str) -> requests.Response:
        raise NotImplementedError

    def get(self, url: str) -> requests.Response:
        r = self._get(url=url)

        if 'https://news.google.com/rss/unsupported' in r.url:
            raise Exception('This feed is not available')

        return r

    def parse_feed(self, url: str) -> dict[str, str]:
        r = self.get(url)
        d = feedparser.parse(r.text)

        return dict((k, d[k]) for k in ('feed', 'entries'))


class NewsSearch(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def search(self, query: str,
               helper=True,
               when: Optional[str or None] = None,
               from_: Optional[str or None] = None,
               to_: Optional[str or None] = None):
        raise NotImplementedError
