# src/my_pandas_lib/accessor.py

import pandas as pd
from ..core.returns_core import calculate_returns, calculate_wealth_index, calculate_drawdown
from ..core.summary_core import describe_returns as _describe_returns


class ReturnsNamespace:
    def __init__(self, df: pd.DataFrame):
        self._df = df

    def calculate_returns(
        self,
    ) -> pd.DataFrame:
        return calculate_returns(self._df)

    def calculate_wealth_index(self, start: float = 1.0) -> pd.DataFrame:
        """
        Calculate the wealth index for every return column.

        Parameters
        ----------
        start : float
            Starting investment value. Defaults to ``1.0`` ($1).

        Returns
        -------
        pd.DataFrame
            Copy of the DataFrame with ``'_q_<col>_wealth_index'`` columns appended.
        """
        return calculate_wealth_index(self._df, start=start)

    def calculate_drawdown(self) -> pd.DataFrame:
        """
        Calculate the drawdown series for every return column.

        Appends three columns per ticker:

        * ``_q_<col>_Wealth``   — wealth index (starting at 1).
        * ``_q_<col>_Peak``     — running maximum of the wealth index.
        * ``_q_<col>_Drawdown`` — peak minus current wealth (distance below peak).

        Returns
        -------
        pd.DataFrame
            Copy of the DataFrame with Wealth, Peak, and Drawdown columns appended.
        """
        return calculate_drawdown(self._df)


@pd.api.extensions.register_dataframe_accessor("qf")
class QFAccessor:
    def __init__(self, pandas_obj: pd.DataFrame):
        self._obj = pandas_obj

    @property
    def returns(self) -> ReturnsNamespace:
        return ReturnsNamespace(self._obj)

    def describe_returns(
        self, annualized: bool = True, pvalue: float = 0.05, risk_free_rate: float = 0.04
    ) -> pd.DataFrame:
        """
        Compute a return/risk summary for each ticker column.

        Parameters
        ----------
        annualized : bool
            Annualize metrics using the frequency inferred from the index.
        pvalue : float
            Significance level for the Jarque-Bera normality test. Defaults to ``0.05``.
        risk_free_rate : float
            Risk-free rate for Sharpe ratio calculation. Defaults to ``0.04`` (4%).

        Returns
        -------
        pd.DataFrame
            Summary with metrics as index and tickers as columns.
        """
        return _describe_returns(
            self._obj, annualized=annualized, pvalue=pvalue, risk_free_rate=risk_free_rate
        )
