# src/my_pandas_lib/finance.py

import pandas as pd


def calculate_returns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate percentage returns for every numeric price column in the DataFrame.

    Returns a new DataFrame containing only the return series, with each column
    keeping the original price column's name. The first row will be NaN
    (no prior price to compare). The input DataFrame is not mutated.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame whose numeric columns are treated as price series.

    Returns
    -------
    pd.DataFrame
        DataFrame of return series with the same column names as the input prices.
    """
    price_cols = [
        col for col in df.columns
        if df[col].dtype in ["float64", "int64"] and not col.startswith("_q_")
    ]
    return df[price_cols].pct_change()


def calculate_wealth_index(df: pd.DataFrame, start: float = 1.0) -> pd.DataFrame:
    """
    Calculate the wealth index for every numeric return column in the DataFrame.

    For each numeric column that is not already a computed column (i.e. does not
    start with '_q_'), a new column named ``'_q_<col>_wealth_index'`` is appended,
    representing the cumulative growth of ``start`` dollars invested at period 0.

    Formula: ``wealth_index = start * (1 + r).cumprod()``

    The input DataFrame is not mutated.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame whose numeric columns are treated as return series.
    start : float
        Starting value of the investment. Defaults to ``1.0`` (i.e. $1).

    Returns
    -------
    pd.DataFrame
        Copy of ``df`` with one ``'_q_<col>_wealth_index'`` column per return column.
    """
    new_cols = {
        f"_q_{col}_wealth_index": start * (1 + df[col]).cumprod()
        for col in df.columns
        if df[col].dtype in ["float64", "int64"] and not col.startswith("_q_")
    }
    return df.assign(**new_cols)


def calculate_drawdown(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the drawdown series for every numeric return column in the DataFrame.

    For each eligible column the following three columns are appended:

    * ``_q_<col>_Wealth``   — wealth index starting at 1 (cumulative growth).
    * ``_q_<col>_Peak``     — running maximum of the wealth index up to each period.
    * ``_q_<col>_Drawdown`` — ``_q_<col>_Peak - _q_<col>_Wealth`` (distance below peak).

    The input DataFrame is not mutated.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame whose numeric columns are treated as return series.

    Returns
    -------
    pd.DataFrame
        Copy of ``df`` with Wealth, Peak, and Drawdown columns appended per ticker.
    """
    new_cols = {}
    for col in df.columns:
        if df[col].dtype not in ["float64", "int64"] or col.startswith("_q_"):
            continue
        wealth = (1 + df[col]).cumprod()
        peak   = wealth.cummax()
        new_cols[f"_q_{col}_Wealth"]   = wealth
        new_cols[f"_q_{col}_Peak"]     = peak
        new_cols[f"_q_{col}_Drawdown"] = peak - wealth
    return df.assign(**new_cols)
