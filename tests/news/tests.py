from stonkify.news import GoogleNewsRetriever


def test_news_retrieval():
    news_retriever = GoogleNewsRetriever()
    out = news_retriever.get_news(query="Microsoft, Inc.")

    return out


if __name__ == "__main__":
    test_news_retrieval()
