# src/my_pandas_lib/accessor.py

import pandas as pd
from ..core.returns_core import calculate_returns


class ReturnsNamespace:
    def __init__(self, df: pd.DataFrame):
        self._df = df

    def calculate_returns(
        self,
    ) -> pd.DataFrame:
        return calculate_returns(self._df)


@pd.api.extensions.register_dataframe_accessor("qf")
class QFAccessor:
    def __init__(self, pandas_obj: pd.DataFrame):
        self._obj = pandas_obj

    @property
    def returns(self) -> ReturnsNamespace:
        return ReturnsNamespace(self._obj)
