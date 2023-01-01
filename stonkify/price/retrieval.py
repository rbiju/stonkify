from datetime import datetime

import pandas as pd
import numpy as np
import yfinance as yf

from .base import Interval, PriceNormStrategy, PriceRetriever


class YFinancePriceRetriever(PriceRetriever):
    def __init__(self, normalizer: PriceNormStrategy):
        super().__init__(normalizer)

    def history(self, ticker: str, start: datetime, end: datetime, interval: Interval) -> pd.DataFrame:
        ticker_obj = yf.Ticker(ticker=ticker)
        hist: pd.DataFrame = ticker_obj.history(start=start, end=end, interval=interval.value)

        return hist

    def prices(self, ticker: str, start: datetime, end: datetime, interval: Interval) -> np.array:
        df_norm: pd.DataFrame = self.normalizer.norm_prices(self.history(ticker, start, end, interval))

        return df_norm[self.normalizer.norm_column_names].to_numpy()
