import pytest
import pandas as pd
import numpy as np

from quantflow.core.returns_core import calculate_wealth_index
import quantflow  # noqa: F401 – registers df.qf accessor


@pytest.fixture
def rets_df():
    return pd.DataFrame({
        "AAPL": [0.10, -0.05, 0.08, 0.02],
        "MSFT": [0.05, 0.03, -0.02, 0.07],
    })


# ── Core function ─────────────────────────────────────────────────────────────

class TestCalculateWealthIndex:
    def test_returns_dataframe(self, rets_df):
        assert isinstance(calculate_wealth_index(rets_df), pd.DataFrame)

    def test_wealth_index_columns_created(self, rets_df):
        result = calculate_wealth_index(rets_df)
        assert "_q_AAPL_wealth_index" in result.columns
        assert "_q_MSFT_wealth_index" in result.columns

    def test_default_start_is_one(self, rets_df):
        result = calculate_wealth_index(rets_df)
        expected = (1 + rets_df["AAPL"]).cumprod()
        pd.testing.assert_series_equal(
            result["_q_AAPL_wealth_index"], expected, check_names=False
        )

    def test_custom_start_value(self, rets_df):
        result = calculate_wealth_index(rets_df, start=1000.0)
        expected = 1000.0 * (1 + rets_df["AAPL"]).cumprod()
        pd.testing.assert_series_equal(
            result["_q_AAPL_wealth_index"], expected, check_names=False
        )

    def test_first_value_equals_start_times_growth(self, rets_df):
        result = calculate_wealth_index(rets_df, start=100.0)
        assert np.isclose(result["_q_AAPL_wealth_index"].iloc[0], 100.0 * (1 + 0.10))

    def test_original_columns_preserved(self, rets_df):
        result = calculate_wealth_index(rets_df)
        assert "AAPL" in result.columns
        assert "MSFT" in result.columns

    def test_does_not_mutate_input(self, rets_df):
        original_cols = list(rets_df.columns)
        calculate_wealth_index(rets_df)
        assert list(rets_df.columns) == original_cols

    def test_skips_q_prefixed_columns(self, rets_df):
        df = rets_df.assign(_q_AAPL_ret=rets_df["AAPL"])
        result = calculate_wealth_index(df)
        assert not any(col.startswith("_q__q_") for col in result.columns)

    def test_custom_start_is_100_second_column(self, rets_df):
        result = calculate_wealth_index(rets_df, start=100.0)
        expected = 100.0 * (1 + rets_df["MSFT"]).cumprod()
        pd.testing.assert_series_equal(
            result["_q_MSFT_wealth_index"], expected, check_names=False
        )


# ── Accessor: df.qf.returns.calculate_wealth_index() ─────────────────────────

class TestCalculateWealthIndexAccessor:
    def test_accessor_returns_dataframe(self, rets_df):
        assert isinstance(rets_df.qf.returns.calculate_wealth_index(), pd.DataFrame)

    def test_accessor_default_matches_core(self, rets_df):
        pd.testing.assert_frame_equal(
            rets_df.qf.returns.calculate_wealth_index(),
            calculate_wealth_index(rets_df),
        )

    def test_accessor_custom_start_matches_core(self, rets_df):
        pd.testing.assert_frame_equal(
            rets_df.qf.returns.calculate_wealth_index(start=1000.0),
            calculate_wealth_index(rets_df, start=1000.0),
        )
