import pytest
import pandas as pd

from quantflow.core.utils import infer_frequency


def _datetime_df(freq: str, periods: int = 10) -> pd.DataFrame:
    return pd.DataFrame(
        {"price": range(periods)},
        index=pd.date_range("2020-01-01", periods=periods, freq=freq),
    )


def _period_df(freq: str, periods: int = 10) -> pd.DataFrame:
    return pd.DataFrame(
        {"price": range(periods)},
        index=pd.period_range("2020-01", periods=periods, freq=freq),
    )


class TestInferFrequencyDatetimeIndex:
    def test_daily(self):
        assert infer_frequency(_datetime_df("D")) == 252

    def test_business_daily(self):
        assert infer_frequency(_datetime_df("B")) == 252

    def test_weekly(self):
        assert infer_frequency(_datetime_df("W")) == 52

    def test_monthly_end(self):
        assert infer_frequency(_datetime_df("ME")) == 12

    def test_monthly_start(self):
        assert infer_frequency(_datetime_df("MS")) == 12

    def test_quarterly_end(self):
        assert infer_frequency(_datetime_df("QE")) == 4

    def test_quarterly_start(self):
        assert infer_frequency(_datetime_df("QS")) == 4

    def test_annual_end(self):
        assert infer_frequency(_datetime_df("YE")) == 1

    def test_annual_start(self):
        assert infer_frequency(_datetime_df("YS")) == 1


class TestInferFrequencyPeriodIndex:
    def test_monthly_period(self):
        assert infer_frequency(_period_df("M")) == 12

    def test_quarterly_period(self):
        assert infer_frequency(_period_df("Q")) == 4

    def test_annual_period(self):
        assert infer_frequency(_period_df("Y")) == 1


class TestInferFrequencyErrors:
    def test_raises_on_non_time_index(self):
        df = pd.DataFrame({"price": [1, 2, 3]})
        with pytest.raises(TypeError, match="DatetimeIndex or PeriodIndex"):
            infer_frequency(df)

    def test_raises_on_irregular_index(self):
        df = pd.DataFrame(
            {"price": [1, 2, 3]},
            index=pd.to_datetime(["2020-01-01", "2020-01-03", "2020-01-10"]),
        )
        with pytest.raises(ValueError, match="Could not infer frequency"):
            infer_frequency(df)
