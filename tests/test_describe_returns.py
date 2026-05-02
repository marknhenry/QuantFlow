import pytest
import pandas as pd
import numpy as np
import yfinance as yf

from quantflow.core.summary_core import describe_returns, is_normal
import quantflow  # noqa: F401 – registers df.qf accessor


@pytest.fixture(scope="module")
def rets_df() -> pd.DataFrame:
    """Daily returns for AAPL and MSFT over 2024-2025 from yfinance."""
    prices = yf.download(
        ["AAPL", "MSFT"],
        start="2024-01-01",
        end="2025-12-31",
        auto_adjust=True,
        progress=False,
    )["Close"]
    rets = prices.pct_change().dropna()
    rets.index = rets.index.to_period("D")
    return rets


# ── helper is_normal ─────────────

class TestIsNormal:
    def test_normal_series_returns_true(self):
        rng = np.random.default_rng(0)
        s = pd.Series(rng.normal(0, 1, 500))
        assert is_normal(s, pvalue=0.01) == True

    def test_non_normal_series_returns_false(self):
        rng = np.random.default_rng(0)
        s = pd.Series(rng.exponential(1, 500))
        assert is_normal(s, pvalue=0.01) == False

    def test_custom_pvalue(self):
        rng = np.random.default_rng(0)
        s = pd.Series(rng.normal(0, 1, 500))
        assert is_normal(s, pvalue=0.001) == True

    def test_returns_bool(self):
        s = pd.Series(np.random.normal(0, 1, 100))
        assert isinstance(is_normal(s), bool)


# ── describe_returns core ─────────────────────────────────────────────────────

class TestDescribeReturnsFunction:
    def test_returns_dataframe(self, rets_df):
        assert isinstance(describe_returns(rets_df), pd.DataFrame)

    def test_tickers_are_index(self, rets_df):
        result = describe_returns(rets_df)
        assert "AAPL" in result.index
        assert "MSFT" in result.index

    def test_has_is_normal_column(self, rets_df):
        result = describe_returns(rets_df)
        assert "Is Normal (JB)" in result.columns

    def test_is_normal_values_are_bool(self, rets_df):
        result = describe_returns(rets_df)
        assert result["Is Normal (JB)"].dtype == bool

    def test_non_normal_data_returns_false(self):
        rng = np.random.default_rng(1)
        df = pd.DataFrame(
            {"A": rng.exponential(1, 200), "B": rng.exponential(2, 200)},
            index=pd.period_range("2000-01", periods=200, freq="M"),
        )
        result = describe_returns(df, annualized=False)
        assert result.loc["A", "Is Normal (JB)"] == False

    def test_custom_pvalue_threshold(self, rets_df):
        result_strict = describe_returns(rets_df, pvalue=0.001)
        result_loose  = describe_returns(rets_df, pvalue=0.99)
        # With pvalue=0.99 almost nothing passes; with 0.001 more may pass
        assert isinstance(result_strict["Is Normal (JB)"].iloc[0], (bool, np.bool_))
        assert isinstance(result_loose["Is Normal (JB)"].iloc[0], (bool, np.bool_))

    def test_does_not_mutate_input(self, rets_df):
        original_cols = list(rets_df.columns)
        describe_returns(rets_df)
        assert list(rets_df.columns) == original_cols

    def test_skips_q_prefixed_columns(self, rets_df):
        df = rets_df.assign(_q_extra=rets_df["AAPL"])
        result = describe_returns(df, annualized=False)
        assert "_q_extra" not in result.index


# Accessor

class TestDescribeReturnsAccessor:
    def test_accessor_returns_dataframe(self, rets_df):
        assert isinstance(rets_df.qf.describe_returns(), pd.DataFrame)

    def test_accessor_has_is_normal(self, rets_df):
        assert "Is Normal (JB)" in rets_df.qf.describe_returns().columns

    def test_accessor_matches_standalone(self, rets_df):
        pd.testing.assert_frame_equal(
            rets_df.qf.describe_returns(),
            describe_returns(rets_df),
        )

    def test_accessor_custom_pvalue(self, rets_df):
        result = rets_df.qf.describe_returns(pvalue=0.05)
        assert "Is Normal (JB)" in result.columns


# Real market data test
class TestDescribeReturnsRealData:
    def test_ffme_describe_returns(self):
        """Test describe_returns on real Fama-French market data."""
        import quantflow as qf
        ffme = qf.load_data.get_ffme_returns()
        result = ffme.qf.describe_returns(annualized=True)
        
        assert isinstance(result, pd.DataFrame)
        assert "SmallCap" in result.index
        assert "LargeCap" in result.index
        assert "Is Normal (JB)" in result.columns
        assert result.loc["SmallCap", "Wealth Index"] > 1  # Positive long-term return

    def test_hfi_var_historic_matches_expected_values(self):
        """Regression test: VaR Historic at pvalue=0.05 matches known HFI values."""
        import quantflow as qf

        hfi = qf.load_data.get_hfi_returns()
        result = hfi.qf.describe_returns(pvalue=0.05)["VaR Historic"]

        expected = pd.Series(
            {
                "Convertible Arbitrage": 0.0158,
                "CTA Global": 0.0317,
                "Distressed Securities": 0.0197,
                "Emerging Markets": 0.0425,
                "Equity Market Neutral": 0.0081,
                "Event Driven": 0.0253,
                "Fixed Income Arbitrage": 0.0079,
                "Global Macro": 0.0150,
                "Long/Short Equity": 0.0260,
                "Merger Arbitrage": 0.0105,
                "Relative Value": 0.0117,
                "Short Selling": 0.0678,
                "Funds Of Funds": 0.0205,
            },
            name="VaR Historic",
            dtype="float64",
        )

        pd.testing.assert_series_equal(result.round(4), expected)

    def test_hfi_var_gaussian_matches_expected_values(self):
        """Regression test: VaR Gaussian at pvalue=0.05 matches known HFI values."""
        import quantflow as qf

        hfi = qf.load_data.get_hfi_returns()
        result = hfi.qf.describe_returns(pvalue=0.05)["VaR Gaussian"]

        expected = pd.Series(
            {
                "Convertible Arbitrage": 0.0217,
                "CTA Global": 0.0342,
                "Distressed Securities": 0.0210,
                "Emerging Markets": 0.0472,
                "Equity Market Neutral": 0.0088,
                "Event Driven": 0.0211,
                "Fixed Income Arbitrage": 0.0146,
                "Global Macro": 0.0188,
                "Long/Short Equity": 0.0264,
                "Merger Arbitrage": 0.0104,
                "Relative Value": 0.0131,
                "Short Selling": 0.0801,
                "Funds Of Funds": 0.0213,
            },
            name="VaR Gaussian",
            dtype="float64",
        )

        pd.testing.assert_series_equal(result.round(4), expected)

    def test_hfi_var_cornish_fisher_matches_expected_values(self):
        """Regression test: VaR Cornish-Fisher at pvalue=0.05 matches known HFI values."""
        import quantflow as qf

        hfi = qf.load_data.get_hfi_returns()
        result = hfi.qf.describe_returns(pvalue=0.05)["VaR Cornish-Fisher"]

        expected = pd.Series(
            {
                "Convertible Arbitrage": 0.0261,
                "CTA Global": 0.0345,
                "Distressed Securities": 0.0261,
                "Emerging Markets": 0.0549,
                "Equity Market Neutral": 0.0112,
                "Event Driven": 0.0265,
                "Fixed Income Arbitrage": 0.0185,
                "Global Macro": 0.0144,
                "Long/Short Equity": 0.0291,
                "Merger Arbitrage": 0.0132,
                "Relative Value": 0.0168,
                "Short Selling": 0.0689,
                "Funds Of Funds": 0.0225,
            },
            name="VaR Cornish-Fisher",
            dtype="float64",
        )

        pd.testing.assert_series_equal(result.round(4), expected)
