import pytest
import pandas as pd
import numpy as np

from quantflow.core.returns_core import calculate_return
import quantflow  # noqa: F401 – registers df.qf accessor


@pytest.fixture
def price_df():
    """DataFrame with two price columns (MSFT-style names, no underscores)."""
    return pd.DataFrame({
        "asset1": [100.0, 110.0, 105.0, 115.5],
        "asset2": [200.0, 190.0, 210.0, 205.0],
    })


# ── Standalone function: calculate_return(df) ────────────────────────────────

class TestCalculateReturnFunction:
    def test_returns_dataframe(self, price_df):
        """calculate_return should return a pandas DataFrame."""
        result = calculate_return(price_df)
        assert isinstance(result, pd.DataFrame)

    def test_return_columns_created_for_all_price_cols(self, price_df):
        """A '_q_<col>_return' column should be added for every numeric price column."""
        result = calculate_return(price_df)
        assert "_q_asset1_return" in result.columns
        assert "_q_asset2_return" in result.columns

    def test_first_row_is_nan(self, price_df):
        """The first row of every return column should be NaN (no prior price)."""
        result = calculate_return(price_df)
        assert np.isnan(result["_q_asset1_return"].iloc[0])
        assert np.isnan(result["_q_asset2_return"].iloc[0])

    def test_correct_values_col1(self, price_df):
        """Returns for asset1 should equal its pct_change values."""
        result = calculate_return(price_df)
        pd.testing.assert_series_equal(
            result["_q_asset1_return"], price_df["asset1"].pct_change(), check_names=False
        )

    def test_correct_values_col2(self, price_df):
        """Returns for asset2 should equal its pct_change values."""
        result = calculate_return(price_df)
        pd.testing.assert_series_equal(
            result["_q_asset2_return"], price_df["asset2"].pct_change(), check_names=False
        )

    def test_original_columns_preserved(self, price_df):
        """All original price columns should still be present in the result."""
        result = calculate_return(price_df)
        assert "asset1" in result.columns
        assert "asset2" in result.columns

    def test_does_not_mutate_input(self, price_df):
        """calculate_return should not modify the input DataFrame in place."""
        original_cols = list(price_df.columns)
        calculate_return(price_df)
        assert list(price_df.columns) == original_cols

    def test_skips_existing_return_columns(self, price_df):
        """Columns already prefixed with '_q_' should not be processed again."""
        result = calculate_return(price_df)
        second_result = calculate_return(result)
        # No '_q__q_' double-processed columns should appear
        assert not any(col.startswith("_q__q_") for col in second_result.columns)


# ── Accessor: df.qf.returns.calculate_return() ───────────────────────────────

class TestCalculateReturnAccessor:
    def test_accessor_returns_dataframe(self, price_df):
        """df.qf.returns.calculate_return should return a pandas DataFrame."""
        result = price_df.qf.returns.calculate_return()
        assert isinstance(result, pd.DataFrame)

    def test_accessor_return_columns_created(self, price_df):
        """Accessor should produce '_q_<col>_return' columns for all price columns."""
        result = price_df.qf.returns.calculate_return()
        assert "_q_asset1_return" in result.columns
        assert "_q_asset2_return" in result.columns

    def test_accessor_first_row_is_nan(self, price_df):
        """The first row of every return column should be NaN when accessed via df.qf.returns."""
        result = price_df.qf.returns.calculate_return()
        assert np.isnan(result["_q_asset1_return"].iloc[0])
        assert np.isnan(result["_q_asset2_return"].iloc[0])

    def test_accessor_correct_values_col1(self, price_df):
        """Accessor returns for asset1 should equal its pct_change values."""
        result = price_df.qf.returns.calculate_return()
        pd.testing.assert_series_equal(
            result["_q_asset1_return"], price_df["asset1"].pct_change(), check_names=False
        )

    def test_accessor_correct_values_col2(self, price_df):
        """Accessor returns for asset2 should equal its pct_change values."""
        result = price_df.qf.returns.calculate_return()
        pd.testing.assert_series_equal(
            result["_q_asset2_return"], price_df["asset2"].pct_change(), check_names=False
        )

    def test_accessor_matches_standalone_function(self, price_df):
        """df.qf.returns.calculate_return and the standalone function should produce identical results."""
        accessor_result = price_df.qf.returns.calculate_return()
        function_result = calculate_return(price_df)
        pd.testing.assert_frame_equal(accessor_result, function_result)

