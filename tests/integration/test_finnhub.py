"""Integration tests for Finnhub provider (real API calls).

These tests verify the Finnhub provider works correctly with actual API calls.

Requirements:
    - FINNHUB_API_KEY environment variable must be set with a VALID API key
    - Tests will skip if FINNHUB_API_KEY is not set
    - Get free API key at: https://finnhub.io/register

Pricing Tiers:
    - FREE TIER: Real-time quotes only (60 calls/min)
      ✅ test_fetch_quote - Works on free tier
      ✅ test_provider_initialization - Works on free tier
      ✅ test_invalid_frequency - Works on free tier

    - PAID TIER: Historical OHLCV data (requires $59.99+/month subscription)
      ❌ test_fetch_ohlcv_* - Requires paid subscription
      ❌ test_multiple_symbols - Requires paid subscription
      ❌ test_frequency_mapping - Requires paid subscription
      ❌ test_invalid_symbol - Requires paid subscription
      ❌ test_rate_limiting - Requires paid subscription

Test Coverage:
    - Real-time quote data (free tier)
    - Stock daily/weekly/monthly OHLCV data (paid subscription)
    - Multiple resolutions (paid subscription)
    - Rate limiting behavior (paid subscription)
    - Error handling

Note:
    OHLCV tests WILL FAIL on free tier with "You don't have access to this resource."
    This is EXPECTED BEHAVIOR. Finnhub free tier does not include historical OHLCV data.
    See: https://finnhub.io/pricing for subscription options.
"""

import os
from datetime import datetime, timedelta

import polars as pl
import pytest

from ml4t.data.providers.finnhub import FinnhubProvider

# Get API key from environment
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY")

# Skip all tests if no API key
pytestmark = pytest.mark.skipif(
    not FINNHUB_API_KEY,
    reason="FINNHUB_API_KEY not set - get free key at https://finnhub.io/register",
)


@pytest.fixture
def provider():
    """Create Finnhub provider with API key."""
    provider = FinnhubProvider(api_key=FINNHUB_API_KEY)
    yield provider
    provider.close()


class TestFinnhubProvider:
    """Test Finnhub provider with real API calls."""

    def test_provider_initialization(self):
        """Test provider can be initialized with API key."""
        provider = FinnhubProvider(api_key=FINNHUB_API_KEY)
        assert provider.name == "finnhub"
        assert provider.api_key == FINNHUB_API_KEY
        provider.close()

    @pytest.mark.paid_tier
    def test_fetch_ohlcv_daily(self, provider):
        """Test fetching daily stock data with real API call (PAID TIER REQUIRED)."""
        # Use recent date range
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")

        # Fetch AAPL daily data
        df = provider.fetch_ohlcv(
            symbol="AAPL",
            start=start_date,
            end=end_date,
            frequency="daily",
        )

        # Verify data structure
        assert isinstance(df, pl.DataFrame)
        assert not df.is_empty(), "Should fetch some data for AAPL"

        # Check required columns
        required_cols = ["timestamp", "symbol", "open", "high", "low", "close", "volume"]
        assert all(col in df.columns for col in required_cols)

        # Verify data types
        assert df["timestamp"].dtype == pl.Datetime
        assert df["open"].dtype == pl.Float64
        assert df["close"].dtype == pl.Float64
        assert df["volume"].dtype == pl.Float64

        # Verify symbol
        assert (df["symbol"] == "AAPL").all()

        # Verify OHLC relationships
        assert (df["high"] >= df["low"]).all()
        assert (df["high"] >= df["open"]).all()
        assert (df["high"] >= df["close"]).all()
        assert (df["low"] <= df["open"]).all()
        assert (df["low"] <= df["close"]).all()

        # Check for reasonable prices (AAPL should be > $50)
        assert df["close"].min() > 50.0
        assert df["volume"].min() > 0

        print(f"✅ Fetched {len(df)} rows of AAPL daily data")
        print(f"   Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        print(f"   Price range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")

    @pytest.mark.paid_tier
    def test_fetch_ohlcv_weekly(self, provider):
        """Test fetching weekly data with real API call (PAID TIER REQUIRED)."""
        # Use longer date range for weekly data
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d")

        # Fetch AAPL weekly data
        df = provider.fetch_ohlcv(
            symbol="AAPL",
            start=start_date,
            end=end_date,
            frequency="weekly",
        )

        # Verify data structure
        assert isinstance(df, pl.DataFrame)
        assert not df.is_empty(), "Should fetch some weekly data for AAPL"

        # Check required columns
        required_cols = ["timestamp", "symbol", "open", "high", "low", "close", "volume"]
        assert all(col in df.columns for col in required_cols)

        # Should have fewer rows than daily (roughly 13 weeks)
        assert len(df) <= 15

        print(f"✅ Fetched {len(df)} rows of AAPL weekly data")
        print(f"   Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        print(f"   Price range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")

    @pytest.mark.paid_tier
    def test_fetch_ohlcv_monthly(self, provider):
        """Test fetching monthly data with real API call (PAID TIER REQUIRED)."""
        # Use longer date range for monthly data
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=180)).strftime("%Y-%m-%d")

        # Fetch AAPL monthly data
        df = provider.fetch_ohlcv(
            symbol="AAPL",
            start=start_date,
            end=end_date,
            frequency="monthly",
        )

        # Verify data structure
        assert isinstance(df, pl.DataFrame)
        assert not df.is_empty(), "Should fetch some monthly data for AAPL"

        # Check required columns
        required_cols = ["timestamp", "symbol", "open", "high", "low", "close", "volume"]
        assert all(col in df.columns for col in required_cols)

        # Should have roughly 6 months of data
        assert 5 <= len(df) <= 7

        print(f"✅ Fetched {len(df)} rows of AAPL monthly data")
        print(f"   Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        print(f"   Price range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")

    @pytest.mark.paid_tier
    def test_multiple_symbols(self, provider):
        """Test fetching data for multiple symbols (PAID TIER REQUIRED)."""
        # Use short date range to minimize API calls
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")

        # Fetch data for multiple symbols
        symbols = ["AAPL", "MSFT", "GOOGL"]
        dfs = []

        for symbol in symbols:
            df = provider.fetch_ohlcv(symbol, start_date, end_date, frequency="daily")
            assert not df.is_empty()
            assert (df["symbol"] == symbol).all()
            dfs.append(df)

        print(f"✅ Fetched data for {len(dfs)} symbols")
        for df in dfs:
            symbol = df["symbol"][0]
            print(f"   {symbol}: {len(df)} rows")

    @pytest.mark.paid_tier
    def test_frequency_mapping(self, provider):
        """Test that frequency aliases work correctly (PAID TIER REQUIRED)."""
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")

        # Test various frequency aliases
        frequencies_to_test = [
            "daily",
            "1d",
            "day",
        ]

        for freq in frequencies_to_test:  # Test just a few to save API calls
            df = provider.fetch_ohlcv("AAPL", start_date, end_date, frequency=freq)
            assert not df.is_empty()
            print(f"✅ Frequency '{freq}' works: {len(df)} rows")

    @pytest.mark.paid_tier
    def test_invalid_symbol(self, provider):
        """Test error handling for invalid symbol (PAID TIER REQUIRED)."""
        from ml4t.data.core.exceptions import DataNotAvailableError

        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")

        # Finnhub returns no_data status for invalid symbols
        with pytest.raises(DataNotAvailableError):
            provider.fetch_ohlcv(
                symbol="INVALID_SYMBOL_12345",
                start=start_date,
                end=end_date,
                frequency="daily",
            )

        print("✅ Invalid symbol handled with DataNotAvailableError")

    def test_invalid_frequency(self, provider):
        """Test error handling for invalid frequency."""
        from ml4t.data.core.exceptions import DataValidationError

        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")

        with pytest.raises(DataValidationError):
            provider.fetch_ohlcv(
                symbol="AAPL",
                start=start_date,
                end=end_date,
                frequency="INVALID",
            )

        print("✅ Invalid frequency handled with DataValidationError")

    @pytest.mark.paid_tier
    def test_rate_limiting(self, provider):
        """Test that rate limiting is enforced."""
        import time

        # Free tier: 60 requests/min = 1 per second
        # Make 3 rapid requests to test rate limiter
        start_time = time.time()

        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")

        # First request
        df1 = provider.fetch_ohlcv("AAPL", start_date, end_date)
        assert not df1.is_empty()

        # Second request (should be delayed by rate limiter)
        df2 = provider.fetch_ohlcv("MSFT", start_date, end_date)
        assert not df2.is_empty()

        # Third request
        df3 = provider.fetch_ohlcv("GOOGL", start_date, end_date)
        assert not df3.is_empty()

        elapsed = time.time() - start_time

        # Rate limiter should add delays between requests
        # With 1 req/sec, min time for 3 requests is ~2 seconds
        print(f"✅ Made 3 API calls in {elapsed:.1f}s")
        print(f"   Rate limiter is active: {elapsed >= 2.0}")

        # Should take at least 2 seconds (with some tolerance for request time)
        assert elapsed >= 1.5


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
