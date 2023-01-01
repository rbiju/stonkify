from abc import ABC, abstractmethod
from enum import Enum
from typing import List
from datetime import datetime

import pandas as pd


class Interval(Enum):
    HOUR = '1h'
    DAY = '1d'
    WEEK = '1wk'
    MONTH = '1mo'


class HistoryColumns(Enum):
    OPEN = 'Open'
    HIGH = "High"
    LOW = "Low"
    CLOSE = "Close"
    VOLUME = "Volume"
    DIVIDENDS = "Dividends"
    SPLITS = "Stock Splits"


class PriceNormStrategy(ABC):
    def __init__(self, norm_column_names: List[str]):
        self.norm_column_names = norm_column_names
        pass

    @abstractmethod
    def norm_prices(self, history: pd.DataFrame) -> pd.DataFrame:
        """
        :param history: expects a dataframe with an "open" and "close" column, indexed by interval datetime
        :return: dataframe with a new column of name self.norm_column_name, indexed by interval datetime
        """
        raise NotImplementedError


class PriceRetriever(ABC):
    def __init__(self, normalizer: PriceNormStrategy):
        self.normalizer = normalizer
        pass

    @abstractmethod
    def history(self, ticker: str, start: datetime, end: datetime, interval: Interval) -> pd.DataFrame:
        raise NotImplementedError
