"""Integration tests for Alpaca provider (real API calls).

These tests verify the Alpaca provider works correctly with actual API calls.

Requirements:
    - ALPACA_API_KEY and ALPACA_API_SECRET environment variables must be set
    - Free tier uses the real-time single-exchange IEX feed; ~200 requests/min
    - API key from: https://alpaca.markets/

Test Coverage:
    - Stock daily OHLCV data (AAPL)
    - Crypto minute OHLCV data (BTC/USD)

IMPORTANT:
    These tests are excluded from the default suite (the `integration` marker is
    deselected in pyproject) and are skipped unless credentials are set. They are
    smoke/documentation coverage, not part of the green gate.
"""

import os

import polars as pl
import pytest

from ml4t.data.providers.alpaca import AlpacaDataProvider

ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_API_SECRET = os.getenv("ALPACA_API_SECRET")

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not (ALPACA_API_KEY and ALPACA_API_SECRET),
        reason="ALPACA_API_KEY/ALPACA_API_SECRET not set - get a free key at "
        "https://alpaca.markets/",
    ),
]

REQUIRED_COLS = ["timestamp", "symbol", "open", "high", "low", "close", "volume"]


@pytest.fixture
def provider():
    """Create an Alpaca provider with credentials from the environment."""
    provider = AlpacaDataProvider(api_key=ALPACA_API_KEY, api_secret=ALPACA_API_SECRET)
    yield provider
    provider.close()


class TestAlpacaProvider:
    """Test the Alpaca provider against the real Market Data API."""

    def test_provider_initialization(self):
        """Provider initializes from credentials and reports its name.

        This test makes no API calls.
        """
        provider = AlpacaDataProvider(api_key=ALPACA_API_KEY, api_secret=ALPACA_API_SECRET)
        assert provider.name == "alpaca"
        assert provider.feed == "iex"
        provider.close()

    def test_fetch_stock_daily(self, provider):
        """Fetch daily stock bars for AAPL with a real API call."""
        df = provider.fetch_ohlcv(
            symbol="AAPL",
            start="2024-01-01",
            end="2024-01-31",
            frequency="daily",
        )

        assert isinstance(df, pl.DataFrame)
        assert not df.is_empty(), "Should fetch some daily data for AAPL"
        assert all(col in df.columns for col in REQUIRED_COLS)
        assert df["timestamp"].dtype == pl.Datetime
        assert df["symbol"].dtype == pl.String
        assert df["close"].dtype == pl.Float64
        assert (df["high"] >= df["low"]).all(), "High should be >= Low"
        assert (df["symbol"] == "AAPL").all()

    def test_fetch_crypto_minute(self, provider):
        """Fetch minute crypto bars for BTC/USD with a real API call."""
        df = provider.fetch_ohlcv(
            symbol="BTC/USD",
            start="2024-01-01T00:00:00Z",
            end="2024-01-01T01:00:00Z",
            frequency="minute",
        )

        assert isinstance(df, pl.DataFrame)
        assert not df.is_empty(), "Should fetch some minute data for BTC/USD"
        assert all(col in df.columns for col in REQUIRED_COLS)
        assert (df["high"] >= df["low"]).all(), "High should be >= Low"
        # Crypto symbols keep their BASE/QUOTE form verbatim.
        assert (df["symbol"] == "BTC/USD").all()
