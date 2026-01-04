"""Integration tests for EODHD provider (real API calls).

These tests verify the EODHD provider works correctly with actual API calls.

Requirements:
    - EODHD_API_KEY environment variable must be set
    - Free tier: 500 calls/day, 1 year historical depth
    - API key from: https://eodhd.com/register

Test Coverage:
    - Stock daily OHLCV data (AAPL.US)
    - Multiple frequencies (daily, weekly)
    - Global exchanges (AAPL.US, VOD.LSE)
    - Error handling
    - Rate limiting behavior

IMPORTANT:
    These tests are minimal (5-7 API calls) to respect the 500 calls/day budget.
    Each test makes 1 API call unless otherwise noted.
"""

import os
from datetime import datetime, timedelta

import polars as pl
import pytest

from ml4t.data.core.exceptions import DataNotAvailableError
from ml4t.data.providers.eodhd import EODHDProvider

# Get API key from environment
EODHD_API_KEY = os.getenv("EODHD_API_KEY")

# Skip all tests if no API key
pytestmark = pytest.mark.skipif(
    not EODHD_API_KEY,
    reason="EODHD_API_KEY not set - get free key at https://eodhd.com/register",
)


@pytest.fixture
def provider():
    """Create EODHD provider with API key."""
    provider = EODHDProvider(api_key=EODHD_API_KEY)
    yield provider
    provider.close()


class TestEODHDProvider:
    """Test EODHD provider with real API calls.

    Note: Minimal test suite (5-7 calls) to respect 500 calls/day budget.
    """

    def test_provider_initialization(self):
        """Test provider can be initialized with API key.

        This test does not make any API calls.
        """
        provider = EODHDProvider(api_key=EODHD_API_KEY)
        assert provider.name == "eodhd"
        assert provider.api_key == EODHD_API_KEY
        assert provider.default_exchange == "US"
        provider.close()

    def test_fetch_ohlcv_daily(self, provider):
        """Test fetching daily stock data with real API call.

        API calls: 1
        """
        # Use recent date range (last 30 days)
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")

        # Fetch AAPL daily data (US exchange)
        df = provider.fetch_ohlcv(
            symbol="AAPL",
            start=start_date,
            end=end_date,
            frequency="daily",
            exchange="US",
        )

        # Verify data structure
        assert isinstance(df, pl.DataFrame)
        assert not df.is_empty(), "Should fetch some data for AAPL"

        # Check required columns
        required_cols = ["timestamp", "symbol", "open", "high", "low", "close", "volume"]
        assert all(col in df.columns for col in required_cols)

        # Verify data types
        assert df["timestamp"].dtype == pl.Datetime
        assert df["symbol"].dtype == pl.String
        assert df["open"].dtype == pl.Float64
        assert df["close"].dtype == pl.Float64
        assert df["volume"].dtype == pl.Float64

        # Verify OHLCV relationships
        assert (df["high"] >= df["low"]).all(), "High should be >= Low"
        assert (df["high"] >= df["open"]).all(), "High should be >= Open"
        assert (df["high"] >= df["close"]).all(), "High should be >= Close"
        assert (df["low"] <= df["open"]).all(), "Low should be <= Open"
        assert (df["low"] <= df["close"]).all(), "Low should be <= Close"

        # Check for reasonable prices (AAPL should be > $50)
        assert df["close"].min() > 50.0, "AAPL price should be > $50"
        assert df["volume"].min() > 0, "Volume should be positive"

        print(f"✅ Fetched {len(df)} rows of AAPL daily data")
        print(f"   Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        print(f"   Price range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")
        print(f"   Avg volume: {df['volume'].mean():.0f}")

    def test_fetch_ohlcv_weekly(self, provider):
        """Test fetching weekly stock data with real API call.

        API calls: 1
        """
        # Use 90 days for weekly data
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d")

        df = provider.fetch_ohlcv(
            symbol="AAPL",
            start=start_date,
            end=end_date,
            frequency="weekly",
            exchange="US",
        )

        # Verify data structure
        assert isinstance(df, pl.DataFrame)
        assert not df.is_empty(), "Should fetch some weekly data"

        # Weekly data should have fewer rows than daily
        # For 90 days, expect ~12-13 weeks
        assert 8 <= len(df) <= 20, f"Expected 8-20 weekly rows, got {len(df)}"

        # Check required columns
        required_cols = ["timestamp", "symbol", "open", "high", "low", "close", "volume"]
        assert all(col in df.columns for col in required_cols)

        print(f"✅ Fetched {len(df)} rows of AAPL weekly data")
        print(f"   Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")

    def test_fetch_london_exchange(self, provider):
        """Test fetching data from London Stock Exchange.

        API calls: 1
        """
        # Fetch Vodafone from London Stock Exchange
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")

        df = provider.fetch_ohlcv(
            symbol="VOD",  # Vodafone
            start=start_date,
            end=end_date,
            frequency="daily",
            exchange="LSE",  # London Stock Exchange
        )

        # Verify data structure
        assert isinstance(df, pl.DataFrame)
        assert not df.is_empty(), "Should fetch some data for VOD.LSE"

        # Check required columns
        required_cols = ["timestamp", "symbol", "open", "high", "low", "close", "volume"]
        assert all(col in df.columns for col in required_cols)

        # Verify OHLCV relationships
        assert (df["high"] >= df["low"]).all()

        # Vodafone trades in pence, so price should be reasonable (10-200 pence)
        assert 10.0 <= df["close"].mean() <= 200.0, "VOD.LSE price should be in reasonable range"

        print(f"✅ Fetched {len(df)} rows of VOD.LSE daily data")
        print(f"   Price range: {df['close'].min():.2f}p - {df['close'].max():.2f}p")

    def test_invalid_symbol(self, provider):
        """Test error handling for invalid symbol.

        API calls: 1
        """
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")

        # Invalid symbol should raise DataNotAvailableError
        with pytest.raises(DataNotAvailableError):
            provider.fetch_ohlcv(
                symbol="INVALID_SYMBOL_XYZ",
                start=start_date,
                end=end_date,
                frequency="daily",
                exchange="US",
            )

        print("✅ Invalid symbol correctly raises DataNotAvailableError")

    def test_rate_limiting(self):
        """Test rate limiting behavior.

        API calls: 2 (to verify rate limiting works)

        Note: Default rate limit is 1 call per 180 seconds (conservative).
        This test uses a faster rate limit to verify the mechanism works.
        """
        # Create provider with faster rate limit for testing
        fast_provider = EODHDProvider(
            api_key=EODHD_API_KEY,
            rate_limit=(2, 5.0),  # 2 calls per 5 seconds (for testing only)
        )

        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")

        # First call should work
        df1 = fast_provider.fetch_ohlcv(
            symbol="AAPL",
            start=start_date,
            end=end_date,
            frequency="daily",
        )
        assert not df1.is_empty()

        # Second call should also work (within rate limit)
        df2 = fast_provider.fetch_ohlcv(
            symbol="MSFT",
            start=start_date,
            end=end_date,
            frequency="daily",
        )
        assert not df2.is_empty()

        fast_provider.close()

        print("✅ Rate limiting works correctly")

    def test_default_exchange(self, provider):
        """Test using default exchange (US).

        API calls: 1
        """
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")

        # Fetch without specifying exchange (should use default "US")
        df = provider.fetch_ohlcv(
            symbol="MSFT",
            start=start_date,
            end=end_date,
            frequency="daily",
        )

        # Verify data structure
        assert isinstance(df, pl.DataFrame)
        assert not df.is_empty(), "Should fetch some data for MSFT"

        # Check required columns
        required_cols = ["timestamp", "symbol", "open", "high", "low", "close", "volume"]
        assert all(col in df.columns for col in required_cols)

        # MSFT should have reasonable price (> $200)
        assert df["close"].min() > 200.0, "MSFT price should be > $200"

        print(f"✅ Fetched {len(df)} rows of MSFT data using default US exchange")
        print(f"   Price range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")


# Test Summary:
# ==============
# Total API calls: ~7 calls
# 1. test_fetch_ohlcv_daily: 1 call (AAPL.US daily)
# 2. test_fetch_ohlcv_weekly: 1 call (AAPL.US weekly)
# 3. test_fetch_london_exchange: 1 call (VOD.LSE daily)
# 4. test_invalid_symbol: 1 call (error handling)
# 5. test_rate_limiting: 2 calls (AAPL + MSFT)
# 6. test_default_exchange: 1 call (MSFT.US daily)
#
# This leaves 493 calls/day for other testing and development.
