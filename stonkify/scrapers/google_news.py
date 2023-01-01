from typing import Optional
from urllib.parse import quote_plus

from bs4 import BeautifulSoup
from dateparser import parse as parse_date
from requests import HTTPError

from .base import RequestModifier, NewsSearch


class GoogleNews(NewsSearch):
    def __init__(self, request_modifier: RequestModifier, lang='en', country='US'):
        super().__init__()
        self.request_modifier = request_modifier
        self.lang = lang.lower()
        self.country = country.upper()
        self.BASE_URL = 'https://news.google.com/rss'

    @staticmethod
    def __top_news_parser(text):
        """Return subarticles from the main and topic feeds"""
        try:
            bs4_html = BeautifulSoup(text, "html.parser")
            # find all li tags
            lis = bs4_html.find_all('li')
            sub_articles = []
            for li in lis:
                try:
                    sub_articles.append({"url": li.a['href'],
                                         "title": li.a.text,
                                         "publisher": li.font.text})
                except AttributeError:
                    pass
            return sub_articles
        except HTTPError:
            return text

    def __ceid(self):
        """Compile correct country-lang parameters for Google News RSS URL"""
        return '?ceid={}:{}&hl={}&gl={}'.format(self.country, self.lang, self.lang, self.country)

    def __add_sub_articles(self, entries):
        for i, val in enumerate(entries):
            if 'summary' in entries[i].keys():
                entries[i]['sub_articles'] = self.__top_news_parser(entries[i]['summary'])
            else:
                entries[i]['sub_articles'] = None
        return entries

    def __parse_feed(self, feed_url):
        return self.request_modifier.parse_feed(url=feed_url)

    @staticmethod
    def __search_helper(query):
        return quote_plus(query)

    @staticmethod
    def __from_to_helper(validate=None):
        try:
            validate = parse_date(validate).strftime('%Y-%m-%d')
            return str(validate)
        except TypeError:
            print("Date format was not specified correctly: must be in valid datetime string format. Ex:'03-14-1879' ")

    def top_news(self):
        """Return a list of all articles from the main page of Google News
        given a country and a language"""
        d = self.__parse_feed(self.BASE_URL + self.__ceid())
        d['entries'] = self.__add_sub_articles(d['entries'])
        return d

    def topic_headlines(self, topic: str):
        """Return a list of all articles from the topic page of Google News
        given a country and a language"""
        # topic = topic.upper()
        if topic.upper() in ['WORLD', 'NATION', 'BUSINESS', 'TECHNOLOGY', 'ENTERTAINMENT', 'SCIENCE', 'SPORTS',
                             'HEALTH']:
            d = self.__parse_feed(self.BASE_URL + '/headlines/section/topic/{}'.format(topic.upper()) + self.__ceid())

        else:
            d = self.__parse_feed(self.BASE_URL + '/topics/{}'.format(topic) + self.__ceid())

        d['entries'] = self.__add_sub_articles(d['entries'])
        if len(d['entries']) > 0:
            return d
        else:
            raise Exception('unsupported topic')

    def geo_headlines(self, geo: str):
        """Return a list of all articles about a specific geolocation
        given a country and a language"""
        d = self.__parse_feed(self.BASE_URL + '/headlines/section/geo/{}'.format(geo) + self.__ceid())

        d['entries'] = self.__add_sub_articles(d['entries'])
        return d

    def search(self, query: str,
               helper=True,
               when: Optional[str or None] = None,
               from_: Optional[str or None] = None,
               to_: Optional[str or None] = None):
        """
        Return a list of all articles given a full-text search parameter,
        a country and a language

        :param str query: search term
        :param bool helper: When True helps with URL quoting
        :param str when: Sets a time range for the articles that can be found
        :param str from_: Beginning date for search, must be as datetime string
        :param str to_: End date for search, must be as datetime string
        """

        if when:
            query += ' when:' + when

        if from_ and not when:
            from_ = self.__from_to_helper(validate=from_)
            query += ' after:' + from_

        if to_ and not when:
            to_ = self.__from_to_helper(validate=to_)
            query += ' before:' + to_

        if helper:
            query = self.__search_helper(query)

        search_ceid = self.__ceid()
        search_ceid = search_ceid.replace('?', '&')

        d = self.__parse_feed(self.BASE_URL + '/search?q={}'.format(query) + search_ceid)

        d['entries'] = self.__add_sub_articles(d['entries'])
        return d
