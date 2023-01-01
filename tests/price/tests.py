from datetime import datetime

from stonkify.price import DeltaNormStrategy, YFinancePriceRetriever, Interval


def test_price_norm():
    norm_strat = DeltaNormStrategy(epsilon=0.01)
    price_retriever = YFinancePriceRetriever(normalizer=norm_strat)

    norm_prices = price_retriever.prices(ticker='MSFT',
                                         start=datetime(2022, 6, 30),
                                         end=datetime(2022, 12, 31),
                                         interval=Interval.WEEK)

    return norm_prices


if __name__ == "__main__":
    test_price_norm()
