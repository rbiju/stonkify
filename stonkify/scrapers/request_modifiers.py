from typing import Dict
import requests
import feedparser
from .base import RequestModifier


class VanillaRequest(RequestModifier):
    def __init__(self):
        super().__init__()

    def _get(self, url: str) -> requests.Response:
        return requests.get(url)

    def parse_feed(self, url: str) -> dict[str, str]:
        r = self._get(url)
        d = feedparser.parse(r.text)

        if len(d['entries']) == 0:
            d = feedparser.parse(url)

        return dict((k, d[k]) for k in ('feed', 'entries'))


class ProxyRequest(RequestModifier):
    def __init__(self, proxies: Dict[str, str]):
        super().__init__()
        self.proxies = proxies

    def _get(self, url: str) -> requests.Response:
        return requests.get(url=url, proxies=self.proxies)


class ScrapingBeeRequest(RequestModifier):
    def __init__(self, api_key: str):
        super().__init__()
        self.api_key = api_key

    def _get(self, url: str) -> requests.Response:
        response = requests.get(
            url="https://app.scrapingbee.com/api/v1/",
            params={
                "api_key": self.api_key,
                "url": url,
                "render_js": "false"
            }
        )
        if response.status_code == 200:
            return response
        if response.status_code != 200:
            raise Exception("ScrapingBee status_code: " + str(response.status_code) + " " + response.text)
