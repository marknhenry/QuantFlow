import pytest
import pandas as pd
import numpy as np

from quantflow.core.returns_core import calculate_drawdown
import quantflow  # noqa: F401 – registers df.qf accessor


@pytest.fixture
def rets_df():
    return pd.DataFrame({
        "AAPL": [0.10, -0.20, 0.05, 0.15, -0.05],
        "MSFT": [0.05, 0.03, -0.10, 0.08, 0.02],
    })


class TestCalculateDrawdown:
    def test_returns_dataframe(self, rets_df):
        assert isinstance(calculate_drawdown(rets_df), pd.DataFrame)

    def test_all_three_columns_created_per_ticker(self, rets_df):
        result = calculate_drawdown(rets_df)
        for col in ["AAPL", "MSFT"]:
            assert f"_q_{col}_Wealth" in result.columns
            assert f"_q_{col}_Peak" in result.columns
            assert f"_q_{col}_Drawdown" in result.columns

    def test_wealth_equals_cumprod(self, rets_df):
        result = calculate_drawdown(rets_df)
        expected = (1 + rets_df["AAPL"]).cumprod()
        pd.testing.assert_series_equal(result["_q_AAPL_Wealth"], expected, check_names=False)

    def test_peak_is_running_max_of_wealth(self, rets_df):
        result = calculate_drawdown(rets_df)
        expected = (1 + rets_df["AAPL"]).cumprod().cummax()
        pd.testing.assert_series_equal(result["_q_AAPL_Peak"], expected, check_names=False)

    def test_drawdown_equals_peak_minus_wealth(self, rets_df):
        result = calculate_drawdown(rets_df)
        expected = result["_q_AAPL_Peak"] - result["_q_AAPL_Wealth"]
        pd.testing.assert_series_equal(result["_q_AAPL_Drawdown"], expected, check_names=False)

    def test_drawdown_is_non_negative(self, rets_df):
        result = calculate_drawdown(rets_df)
        assert (result["_q_AAPL_Drawdown"] >= 0).all()
        assert (result["_q_MSFT_Drawdown"] >= 0).all()

    def test_drawdown_zero_at_peak(self, rets_df):
        result = calculate_drawdown(rets_df)
        at_peak = result["_q_AAPL_Wealth"] == result["_q_AAPL_Peak"]
        assert (result.loc[at_peak, "_q_AAPL_Drawdown"] == 0).all()

    def test_original_columns_preserved(self, rets_df):
        result = calculate_drawdown(rets_df)
        assert "AAPL" in result.columns
        assert "MSFT" in result.columns

    def test_does_not_mutate_input(self, rets_df):
        original_cols = list(rets_df.columns)
        calculate_drawdown(rets_df)
        assert list(rets_df.columns) == original_cols

    def test_skips_q_prefixed_columns(self, rets_df):
        df = rets_df.assign(_q_extra=rets_df["AAPL"])
        result = calculate_drawdown(df)
        assert not any(col.startswith("_q__q_") for col in result.columns)


class TestCalculateDrawdownAccessor:
    def test_accessor_returns_dataframe(self, rets_df):
        assert isinstance(rets_df.qf.returns.calculate_drawdown(), pd.DataFrame)

    def test_accessor_matches_core(self, rets_df):
        pd.testing.assert_frame_equal(
            rets_df.qf.returns.calculate_drawdown(),
            calculate_drawdown(rets_df),
        )
