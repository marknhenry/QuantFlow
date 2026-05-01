# src/my_pandas_lib/finance.py

import pandas as pd


def calculate_returns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate percentage returns for every numeric price column in the DataFrame.

    For each numeric column that is not already a computed return column (i.e. does
    not start with '_q_'), a new column named '_q_<col>_return' is appended.
    The first row of each return column will be NaN (no prior price to compare).
    The input DataFrame is not mutated.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame whose numeric columns are treated as price series.

    Returns
    -------
    pd.DataFrame
        Copy of df with one '_q_<col>_ret' column added per price column.
    """
    new_cols = {
        f"_q_{col}_ret": df[col].pct_change()
        for col in df.columns
        if df[col].dtype in ["float64", "int64"] and not col.startswith("_q_")
    }
    return df.assign(**new_cols)
