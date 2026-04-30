"""Live Alpaca API checks when ``ALPACA_API_KEY`` and ``ALPACA_SECRET_KEY`` are set."""

from __future__ import annotations

import os

import polars as pl
import pytest

pytest.importorskip("alpaca")

from ml4t.data.providers.alpaca import AlpacaProvider

pytestmark = pytest.mark.integration


@pytest.fixture
def alpaca_keys():
    """Skip unless both Alpaca credentials are present."""
    key = os.getenv("ALPACA_API_KEY")
    secret = os.getenv("ALPACA_SECRET_KEY")
    if not key or not secret:
        pytest.skip("ALPACA_API_KEY and ALPACA_SECRET_KEY not both set")
    return key, secret


class TestAlpacaLiveOhlcv:
    """Minimal real calls to validate batch and single-symbol paths."""

    def test_fetch_batch_ohlcv_stocks_5minute(self, alpaca_keys):
        """Short window + two symbols to limit payload and rate impact."""
        p = AlpacaProvider()
        df = p.fetch_batch_ohlcv(
            ["AAPL", "MSFT"],
            start="2024-01-02",
            end="2024-01-04",
            frequency="5minute",
        )
        assert isinstance(df, pl.DataFrame)
        assert not df.is_empty()
        assert set(df.columns) == {
            "timestamp",
            "symbol",
            "open",
            "high",
            "low",
            "close",
            "volume",
        }
        syms = set(df["symbol"].unique().to_list())
        assert syms <= {"AAPL", "MSFT"}
        assert len(syms) >= 1

    def test_fetch_ohlcv_single_daily(self, alpaca_keys):
        p = AlpacaProvider()
        df = p.fetch_ohlcv("AAPL", "2024-01-02", "2024-01-10", frequency="daily")
        assert isinstance(df, pl.DataFrame)
        assert not df.is_empty()
        assert df["symbol"].unique().to_list() == ["AAPL"]

    def test_batch_tolerates_whitespace_in_end_date(self, alpaca_keys):
        """Regression: ``2024-12- 31`` style typos should not raise ValueError."""
        p = AlpacaProvider()
        df = p.fetch_batch_ohlcv(
            ["AAPL"],
            start="2024-12-30",
            end="2024-12- 31",
            frequency="daily",
        )
        assert len(df) >= 1
