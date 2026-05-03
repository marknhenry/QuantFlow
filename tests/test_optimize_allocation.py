import numpy as np
import pandas as pd
import pytest

import quantflow as qf
from quantflow.core import portfolio_core


def test_optimize_allocation_with_two_stocks_like_notebook_call():
    # Deterministic synthetic prices for two assets to avoid network calls.
    dates = pd.date_range("2024-01-01", periods=5, freq="D")
    prices = pd.DataFrame(
        {
            "MSFT": [100.0, 105.0, 112.0, 120.0, 130.0],
            "GOOG": [100.0, 101.0, 103.0, 104.0, 106.0],
        },
        index=dates,
    )

    def fake_get_price_data(stock_list, source="yfinance", date_start="2020-01-01", date_end=None):
        assert stock_list == ["MSFT", "GOOG"]
        return prices

    # monkeypatch.setattr(portfolio_core, "get_price_data", fake_get_price_data)

    list_of_stocks = ["MSFT", "GOOG"]
    weights, vol = qf.portfolio.optimize_allocation_for_target_return(0.2, list_of_stocks)

    assert isinstance(weights, pd.DataFrame)
    assert weights.shape == (1, 2)
    assert weights.columns.tolist() == ["MSFT", "GOOG"]
    assert weights.index.tolist() == ["weights"]
    assert np.isclose(weights.loc["weights"].sum(), 1.0, atol=1e-6)
    assert (weights.loc["weights"] >= 0).all()
    assert (weights.loc["weights"] <= 1).all()
    assert isinstance(vol, (float, np.floating))


def test_optimal_weights_returns_named_series():
    er = pd.Series([0.10, 0.20], index=["MSFT", "GOOG"])
    cov = pd.DataFrame(
        [[0.04, 0.01], [0.01, 0.09]],
        index=er.index,
        columns=er.index,
    )

    n_points = 4
    target_rs = np.linspace(er.min(), er.max(), n_points)
    weights = portfolio_core.optimal_weights(n_points, er, cov)

    assert len(weights) == n_points
    assert all(isinstance(w, pd.Series) for w in weights)
    assert all(w.index.tolist() == ["MSFT", "GOOG"] for w in weights)
    assert all(np.isclose(float(w.name), target_rs[i]) for i, w in enumerate(weights))


def test_optimize_allocation_regression_values_from_notebook_example():
    list_of_stocks = ["MSFT", "GOOG"]
    weights, vol = qf.portfolio.optimize_allocation_for_target_return(
        0.23,
        list_of_stocks,
        date_start="2020-01-01",
        date_end="2024-12-31",
    )

    assert isinstance(weights, pd.DataFrame)
    assert weights.index.tolist() == ["weights"]
    assert weights.columns.tolist() == ["MSFT", "GOOG"]

    assert np.isclose(weights.loc["weights", "MSFT"], 0.353, atol=1e-3)
    assert np.isclose(weights.loc["weights", "GOOG"], 0.647, atol=1e-3)
    assert np.isclose(vol, 0.0188, atol=1e-4)
