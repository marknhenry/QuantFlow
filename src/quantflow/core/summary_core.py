import numpy as np
import pandas as pd
from scipy.stats import jarque_bera, norm
from .utils import infer_frequency


def is_normal(s: pd.Series, pvalue: float = 0.05) -> bool:
    """
    Test whether a return series is normally distributed using the Jarque-Bera test.

    Parameters
    ----------
    s : pd.Series
        Return series to test.
    pvalue : float
        Significance level. Returns ``True`` (normal) when the JB p-value is
        greater than or equal to this threshold, ``False`` otherwise.
        Defaults to ``0.05``.

    Returns
    -------
    bool
        ``True`` if the series passes the normality test, ``False`` if it does not.
    """
    _, p = jarque_bera(s.dropna())
    return bool(p >= pvalue)


def _is_price_data(s: pd.Series) -> bool:
    """
    Detect whether a series contains prices or returns.

    Heuristic:
    - Prices: all values > 0, no negative values (can't have negative price)
    - Returns: can be negative, typically in range [-1, 1] (including extreme months)
    """
    s_clean = s.dropna()
    if len(s_clean) == 0:
        return False
    # If any value is negative or very close to 0, it's likely returns
    if s_clean.min() < 0:
        return False
    # If most values are > 1, it's likely prices
    # If most values are in (0, 1), it's likely returns
    pct_above_one = (s_clean > 1).sum() / len(s_clean)
    return pct_above_one > 0.5


def describe_returns(
    df: pd.DataFrame,
    annualized: bool = True,
    pvalue: float = 0.05,
    risk_free_rate: float = 0.04,
) -> pd.DataFrame:
    """
    Compute a return/risk summary for every numeric column in the DataFrame.

    If the input contains prices (detected heuristically), they are converted
    to returns via ``pct_change()`` before analysis.

    Returns are computed as compound growth: ``(1 + r).prod() - 1``.
    When ``annualized=True``, the annualization factor is inferred from the
    DataFrame's time index via :func:`~quantflow.core.utils.infer_frequency`,
    so the index must be a ``DatetimeIndex`` or ``PeriodIndex``.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame of price or return series (one column per ticker).
    annualized : bool
        Annualize metrics using the frequency inferred from the index.
    pvalue : float
        Significance level for the Jarque-Bera normality test. Defaults to ``0.05``.
    risk_free_rate : float
        Risk-free rate for Sharpe ratio calculation. Defaults to ``0.04`` (4%).

    Returns
    -------
    pd.DataFrame
        Summary DataFrame with metrics as the index and tickers as columns.
    """
    cols = [
        col
        for col in df.columns
        if df[col].dtype in ["float64", "int64"] and not col.startswith("_q_")
    ]

    data = df[cols]

    # Check if input is prices or returns
    if _is_price_data(data.iloc[:, 0]):
        data = data.pct_change().dropna()

    rets = data
    factor = infer_frequency(df) if annualized else None

    compound = (1 + rets).prod()
    vol = rets.std(ddof=0)
    semi_deviation = rets[rets < 0].std(ddof=0)
    wealth = (1 + rets).cumprod().iloc[-1]

    # Calculate max drawdown
    cumulative_wealth = (1 + rets).cumprod()
    running_max = cumulative_wealth.expanding().max()
    drawdown = (cumulative_wealth - running_max) / running_max
    max_drawdown = drawdown.min()

    if annualized:
        n = rets.count()
        summary_returns = (compound ** (factor / n)) - 1
        summary_volatility = vol * (factor**0.5)
        semi_deviation_annualized = semi_deviation * (factor**0.5)
        sharpe = (summary_returns - risk_free_rate) / summary_volatility
    else:
        summary_returns = compound - 1
        summary_volatility = vol
        semi_deviation_annualized = semi_deviation
        sharpe = (summary_returns - risk_free_rate) / summary_volatility

    # Working on Value at Risk
    # Historical VaR and CVaR
    historical_var = -np.percentile(rets, pvalue * 100, axis=0)

    # Gaussian VaR and CVaR
    z = norm.ppf(pvalue)
    gaussian_var = -(rets.mean() + z * rets.std(ddof=0))

    # Cornich-Fisher VaR and CVaR
    mu = rets.mean()
    sigma = rets.std(ddof=0)
    s = rets.skew()
    k = rets.kurt()  # pandas gives excess kurtosis already

    z_cf = (
        z
        + (z**2 - 1) * s / 6
        + (z**3 - 3 * z) * (k - 3) / 24
        - (2 * z**3 - 5 * z) * (s**2) / 36
    )

    cornish_fisher_var = -(mu + z_cf * sigma)

    return pd.DataFrame(
        {
            "AnnualizedReturns": summary_returns,
            "Volatility (Ann)": summary_volatility,
            "Sharpe Ratio": sharpe,
            "Wealth Index": wealth.round(2),
            "Max Drawdown": max_drawdown
            # , "Mean": rets.mean()
            # , "Var": rets.var()
            # , "Std": rets.std()
            # , "Min": rets.min()
            # , "25%": rets.quantile(0.25)
            # , "50%": rets.quantile(0.5)
            # , "75%": rets.quantile(0.75)
            # , "Max": rets.max()
            ,
            "Skewness": rets.skew()
            # , "Kurtosis": rets.kurt()
            ,
            "Excess Kurtosis": rets.kurt() - 3,
            "Is Normal (JB)": rets.apply(lambda s: is_normal(s, pvalue=pvalue)),
            "Semi-Deviation": semi_deviation_annualized,
            "VaR Historic": historical_var,
            "CVaR Historic": rets[rets <= historical_var].mean(),
            "VaR Gaussian": gaussian_var,
            "CVaR Gaussian": rets[rets <= gaussian_var].mean(),
            "VaR Cornish-Fisher": cornish_fisher_var,
            "CVaR Cornish-Fisher": rets[rets <= cornish_fisher_var].mean(),
        }
    )
