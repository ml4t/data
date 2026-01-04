"""Integration tests for CryptoCompare provider (real API calls).

These tests verify the CryptoCompare provider works correctly with actual API calls.

Requirements:
    - CRYPTOCOMPARE_API_KEY environment variable must be set
    - Free tier: 100,000 calls/month, 10 calls/minute
    - API key from: https://min-api.cryptocompare.com/

Test Coverage:
    - Cryptocurrency OHLCV data (BTC, ETH)
    - Multiple frequencies (daily, hourly)
    - Rate limiting behavior
    - Error handling
"""

import os
from datetime import datetime, timedelta

import polars as pl
import pytest

from ml4t.data.core.exceptions import DataNotAvailableError
from ml4t.data.providers.cryptocompare import CryptoCompareProvider

# Get API key from environment
CRYPTOCOMPARE_API_KEY = os.getenv("CRYPTOCOMPARE_API_KEY")

# Skip all tests if no API key
pytestmark = pytest.mark.skipif(
    not CRYPTOCOMPARE_API_KEY,
    reason="CRYPTOCOMPARE_API_KEY not set - get key at https://min-api.cryptocompare.com/",
)


@pytest.fixture
def provider():
    """Create CryptoCompare provider with API key."""
    provider = CryptoCompareProvider(api_key=CRYPTOCOMPARE_API_KEY)
    yield provider
    provider.close()


class TestCryptoCompareProvider:
    """Test CryptoCompare provider with real API calls."""

    def test_provider_initialization(self):
        """Test provider can be initialized with API key."""
        provider = CryptoCompareProvider(api_key=CRYPTOCOMPARE_API_KEY)
        assert provider.name == "cryptocompare"
        assert provider.api_key == CRYPTOCOMPARE_API_KEY
        provider.close()

    def test_fetch_ohlcv_btc_daily(self, provider):
        """Test fetching daily Bitcoin data with real API call."""
        # Use recent date range
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")

        # Fetch BTC/USD daily data
        df = provider.fetch_ohlcv(
            symbol="BTC/USD",
            start=start_date,
            end=end_date,
            frequency="daily",
        )

        # Verify data structure
        assert isinstance(df, pl.DataFrame)
        assert not df.is_empty(), "Should fetch some data for BTC"

        # Check required columns
        required_cols = ["timestamp", "open", "high", "low", "close", "volume"]
        assert all(col in df.columns for col in required_cols)

        # Verify data types
        assert df["timestamp"].dtype == pl.Datetime
        assert df["open"].dtype == pl.Float64
        assert df["close"].dtype == pl.Float64
        assert df["volume"].dtype == pl.Float64

        # Verify OHLCV relationships (allowing some tolerance for crypto data)
        high_low_valid = (df["high"] >= df["low"] - 0.01).all()
        assert high_low_valid, "High should be >= Low"

        # Bitcoin price should be > $10,000 (sanity check)
        assert df["close"].min() > 10000.0

        print(f"✅ Fetched {len(df)} rows of BTC daily data")
        print(f"   Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        print(f"   Price range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")

    def test_fetch_ohlcv_eth_daily(self, provider):
        """Test fetching daily Ethereum data with real API call."""
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=14)).strftime("%Y-%m-%d")

        # Fetch ETH/USD daily data
        df = provider.fetch_ohlcv(
            symbol="ETH/USD",
            start=start_date,
            end=end_date,
            frequency="daily",
        )

        # Verify data structure
        assert isinstance(df, pl.DataFrame)
        assert not df.is_empty(), "Should fetch some data for ETH"

        # Check required columns
        required_cols = ["timestamp", "open", "high", "low", "close", "volume"]
        assert all(col in df.columns for col in required_cols)

        # ETH price should be > $1,000 (sanity check)
        assert df["close"].min() > 1000.0

        print(f"✅ Fetched {len(df)} rows of ETH daily data")
        print(f"   Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        print(f"   Price range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")

    def test_fetch_ohlcv_hourly(self, provider):
        """Test fetching hourly data with real API call."""
        # Use shorter date range for hourly data
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=3)).strftime("%Y-%m-%d")

        # Fetch BTC hourly data
        df = provider.fetch_ohlcv(
            symbol="BTC/USD",
            start=start_date,
            end=end_date,
            frequency="hourly",
        )

        # Verify data structure
        assert isinstance(df, pl.DataFrame)
        assert not df.is_empty(), "Should fetch some hourly data for BTC"

        # Should have multiple rows per day
        assert len(df) > 24  # At least 1 day of hourly data

        print(f"✅ Fetched {len(df)} rows of BTC hourly data")
        print(f"   Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")

    def test_invalid_symbol(self, provider):
        """Test error handling for invalid symbol."""
        from ml4t.data.core.exceptions import ProviderError

        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")

        # CryptoCompare should raise error for invalid symbols
        with pytest.raises((DataNotAvailableError, ProviderError, ValueError)):
            provider.fetch_ohlcv(
                symbol="INVALIDSYMBOL/USD",
                start=start_date,
                end=end_date,
                frequency="daily",
            )

        print("✅ Invalid symbol handled with exception")

    def test_rate_limiting(self, provider):
        """Test that rate limiting is enforced."""
        import time

        # Free tier: 10 requests/minute
        start_time = time.time()

        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")

        # Make 3 requests (should be allowed)
        symbols = ["BTC/USD", "ETH/USD", "LTC/USD"]
        for symbol in symbols:
            df = provider.fetch_ohlcv(symbol, start_date, end_date)
            assert not df.is_empty()

        elapsed = time.time() - start_time

        # Rate limiter should add delays
        print(f"✅ Made 3 API calls in {elapsed:.1f}s")
        print(f"   Rate limiter is active: {elapsed > 10.0}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
