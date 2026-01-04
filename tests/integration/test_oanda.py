"""Integration tests for OANDA provider (real API calls).

These tests verify the OANDA provider works correctly with actual API calls.

Requirements:
    - OANDA_API_KEY environment variable must be set
    - Practice account recommended for testing
    - API key from: https://www.oanda.com/

Test Coverage:
    - Forex OHLCV data (EUR/USD, GBP/USD)
    - Multiple timeframes (daily, hourly)
    - Rate limiting behavior
    - Error handling
"""

import os
from datetime import datetime, timedelta

import polars as pl
import pytest

from ml4t.data.providers.oanda import OandaProvider

# Get API key from environment
OANDA_API_KEY = os.getenv("OANDA_API_KEY")

# Skip all tests if no API key
pytestmark = pytest.mark.skipif(
    not OANDA_API_KEY,
    reason="OANDA_API_KEY not set - get key at https://www.oanda.com/",
)


@pytest.fixture
def provider():
    """Create OANDA provider with API key (practice environment)."""
    provider = OandaProvider(
        api_key=OANDA_API_KEY,
        practice=True,  # Use practice account for testing
    )
    yield provider
    provider.close()


class TestOandaProvider:
    """Test OANDA provider with real API calls."""

    def test_provider_initialization(self):
        """Test provider can be initialized with API key."""
        provider = OandaProvider(
            api_key=OANDA_API_KEY,
            practice=True,
        )
        assert provider.name == "oanda"
        assert provider.api_key == OANDA_API_KEY
        provider.close()

    def test_fetch_ohlcv_eurusd_daily(self, provider):
        """Test fetching daily EUR/USD data with real API call."""
        # Use recent date range
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")

        # Fetch EUR/USD daily data
        df = provider.fetch_ohlcv(
            symbol="EUR_USD",
            start=start_date,
            end=end_date,
            frequency="daily",
        )

        # Verify data structure
        assert isinstance(df, pl.DataFrame)
        assert not df.is_empty(), "Should fetch some data for EUR/USD"

        # Check required columns
        required_cols = ["timestamp", "open", "high", "low", "close", "volume"]
        assert all(col in df.columns for col in required_cols)

        # Verify data types
        assert df["timestamp"].dtype == pl.Datetime
        assert df["open"].dtype == pl.Float64
        assert df["close"].dtype == pl.Float64

        # EUR/USD should be between 0.8 and 1.5 (sanity check)
        assert df["close"].min() > 0.8
        assert df["close"].max() < 1.5

        print(f"✅ Fetched {len(df)} rows of EUR/USD daily data")
        print(f"   Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        print(f"   Price range: {df['close'].min():.5f} - {df['close'].max():.5f}")

    def test_fetch_ohlcv_gbpusd_daily(self, provider):
        """Test fetching daily GBP/USD data with real API call."""
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=14)).strftime("%Y-%m-%d")

        # Fetch GBP/USD daily data
        df = provider.fetch_ohlcv(
            symbol="GBP_USD",
            start=start_date,
            end=end_date,
            frequency="daily",
        )

        # Verify data structure
        assert isinstance(df, pl.DataFrame)
        assert not df.is_empty(), "Should fetch some data for GBP/USD"

        # GBP/USD should be between 1.0 and 2.0 (sanity check)
        assert df["close"].min() > 1.0
        assert df["close"].max() < 2.0

        print(f"✅ Fetched {len(df)} rows of GBP/USD daily data")
        print(f"   Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        print(f"   Price range: {df['close'].min():.5f} - {df['close'].max():.5f}")

    @pytest.mark.skip(reason="Flaky in parallel execution - passes individually")
    def test_fetch_ohlcv_hourly(self, provider):
        """Test fetching hourly data with real API call."""
        # Use shorter date range for hourly data
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=3)).strftime("%Y-%m-%d")

        # Fetch EUR/USD hourly data
        df = provider.fetch_ohlcv(
            symbol="EUR_USD",
            start=start_date,
            end=end_date,
            frequency="hourly",
        )

        # Verify data structure
        assert isinstance(df, pl.DataFrame)
        assert not df.is_empty(), "Should fetch some hourly data for EUR/USD"

        # Should have multiple rows per day (24-hour forex market)
        # Note: Forex markets close on weekends, so 3-day range may include gaps
        # Expect at least 1 full trading day of hourly data (24 bars minimum)
        assert len(df) > 24  # At least ~1 day of hourly data (accounting for weekend gaps)

        print(f"✅ Fetched {len(df)} rows of EUR/USD hourly data")
        print(f"   Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")

    def test_invalid_symbol(self, provider):
        """Test error handling for invalid symbol."""
        from ml4t.data.core.exceptions import ProviderError

        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")

        # OANDA should raise error for invalid symbols
        with pytest.raises((ProviderError, ValueError)):
            provider.fetch_ohlcv(
                symbol="INVALID_PAIR",
                start=start_date,
                end=end_date,
                frequency="daily",
            )

        print("✅ Invalid symbol handled with exception")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
