import pytest
import pandas as pd
import numpy as np

from quantflow.core.returns_core import calculate_returns
import quantflow  # noqa: F401 - registers df.qf accessor


@pytest.fixture
def price_df():
    return pd.DataFrame({
        "asset1": [100.0, 110.0, 105.0, 115.5],
        "asset2": [200.0, 190.0, 210.0, 205.0],
    })
    
@pytest.fixture
def ffme_df():
    return quantflow.load_data.get_ffme_returns()


class TestCalculateReturnsFunction:
    def test_returns_dataframe(self, price_df):
        assert isinstance(calculate_returns(price_df), pd.DataFrame)

    def test_output_columns_match_input_names(self, price_df):
        result = calculate_returns(price_df)
        assert list(result.columns) == ["asset1", "asset2"]

    def test_no_original_price_columns_in_output(self, price_df):
        result = calculate_returns(price_df)
        assert len(result.columns) == 2

    def test_first_row_is_nan(self, price_df):
        result = calculate_returns(price_df)
        assert np.isnan(result["asset1"].iloc[0])
        assert np.isnan(result["asset2"].iloc[0])

    def test_correct_values_col1(self, price_df):
        result = calculate_returns(price_df)
        pd.testing.assert_series_equal(result["asset1"], price_df["asset1"].pct_change())

    def test_correct_values_col2(self, price_df):
        result = calculate_returns(price_df)
        pd.testing.assert_series_equal(result["asset2"], price_df["asset2"].pct_change())

    def test_does_not_mutate_input(self, price_df):
        original_cols = list(price_df.columns)
        calculate_returns(price_df)
        assert list(price_df.columns) == original_cols

    def test_skips_q_prefixed_columns(self, price_df):
        df = price_df.assign(_q_extra=price_df["asset1"])
        result = calculate_returns(df)
        assert "_q_extra" not in result.columns


class TestCalculateReturnsAccessor:
    def test_accessor_returns_dataframe(self, price_df):
        assert isinstance(price_df.qf.returns.calculate_returns(), pd.DataFrame)

    def test_accessor_columns_match_input_names(self, price_df):
        result = price_df.qf.returns.calculate_returns()
        assert list(result.columns) == ["asset1", "asset2"]

    def test_accessor_first_row_is_nan(self, price_df):
        result = price_df.qf.returns.calculate_returns()
        assert np.isnan(result["asset1"].iloc[0])

    def test_accessor_matches_standalone(self, price_df):
        pd.testing.assert_frame_equal(
            price_df.qf.returns.calculate_returns(),
            calculate_returns(price_df),
        )
        
    def test_accessor_functions(self, ffme_df):
        """Accessor should compute returns for all columns in the real FFME dataset."""
        returns = ffme_df.qf.describe_returns()
        assert isinstance(returns, pd.DataFrame)
        assert list(returns.columns) == ["SmallCap", "LargeCap", "L"]
    
