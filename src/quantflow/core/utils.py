import pandas as pd

# Pandas freq alias prefixes → annualization factor
# Order matters: more specific prefixes before broader ones
_FREQ_MAP: list[tuple[str, int]] = [
    ("B",  252),  # business day
    ("D",  252),
    ("W",   52),
    ("ME",  12),  # pandas ≥2.2 month-end alias
    ("MS",  12),
    ("M",   12),  # older alias
    ("BM",  12),
    ("QE",   4),  # pandas ≥2.2 quarter-end alias
    ("QS",   4),
    ("Q",    4),  # older alias
    ("BQ",   4),
    ("YE",   1),  # pandas ≥2.2 year-end alias
    ("YS",   1),
    ("Y",    1),
    ("A",    1),  # older alias
    ("BA",   1),
    ("AS",   1),
]


def infer_frequency(df: pd.DataFrame) -> int:
    """
    Infer the annualization factor for a time-indexed DataFrame.

    Supports both ``DatetimeIndex`` and ``PeriodIndex``.

    Parameters
    ----------
    df : pd.DataFrame
        A DataFrame whose index is a ``DatetimeIndex`` or ``PeriodIndex``.

    Returns
    -------
    int
        Periods per year: 252 (daily), 52 (weekly), 12 (monthly),
        4 (quarterly), or 1 (annual).

    Raises
    ------
    TypeError
        If the index is not a ``DatetimeIndex`` or ``PeriodIndex``.
    ValueError
        If the frequency cannot be inferred or is not recognised.
    """
    index = df.index

    if isinstance(index, pd.PeriodIndex):
        freq_str = index.freqstr
    elif isinstance(index, pd.DatetimeIndex):
        freq_str = pd.infer_freq(index)
        if freq_str is None:
            # Real market data (e.g. from yfinance) has holiday gaps that prevent
            # pd.infer_freq from detecting the pattern. A median gap of ≤ 2 calendar
            # days indicates a daily series, so we assume 252 trading days per year.
            median_days = pd.Series(index).diff().dt.days.median()
            if len(index) >= 2 and median_days <= 2:
                return 252
            raise ValueError(
                "Could not infer frequency from the DatetimeIndex. "
                "The index may be irregular or have too few observations."
            )
    else:
        raise TypeError(
            f"Expected a DatetimeIndex or PeriodIndex, got {type(index).__name__}."
        )

    upper = freq_str.upper()
    for prefix, factor in _FREQ_MAP:
        if upper.startswith(prefix):
            return factor

    raise ValueError(
        f"Unrecognised frequency '{freq_str}'. "
        f"Supported: daily (D/B), weekly (W), monthly (M/ME), "
        f"quarterly (Q/QE), annual (A/Y/YE)."
    )
