import itertools

from typing import List

import pandas as pd
import numpy as np

from .base import PriceNormStrategy, HistoryColumns


class DeltaNormStrategy(PriceNormStrategy):
    def __init__(self, epsilon: float):
        super().__init__(norm_column_names=['delta_norm'])
        self.epsilon = epsilon

    def norm_prices(self, history: pd.DataFrame) -> pd.DataFrame:
        temp: pd.Series = history[HistoryColumns.CLOSE.value].shift(periods=1)
        norm_col = (history[HistoryColumns.CLOSE.value] > temp).astype(int)
        norm_col[norm_col == 0] = -1

        close_col = np.greater(
            ((history[HistoryColumns.CLOSE.value] - temp) / history[HistoryColumns.CLOSE.value]).abs(),
            self.epsilon)

        norm_col[close_col == 0] = 0

        history[self.norm_column_names[0]] = norm_col

        return history


class CompositeNormStrategy(PriceNormStrategy):
    def __init__(self, strategies: List[PriceNormStrategy]):
        norm_column_names = [strat.norm_column_names for strat in strategies]
        norm_column_names = list(itertools.chain.from_iterable(norm_column_names))
        if len(norm_column_names) != set(norm_column_names):
            raise ValueError("Duplicate strategy provided, or multiple strategies with same column name")

        super().__init__(norm_column_names=norm_column_names)
        self.strategies = strategies

    def norm_prices(self, history: pd.DataFrame) -> pd.DataFrame:
        history_ = history
        for strat in self.strategies:
            history_ = strat.norm_prices(history_)

        return history_
