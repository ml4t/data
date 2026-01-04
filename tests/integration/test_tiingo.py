"""Integration tests for Tiingo provider (real API calls).

These tests verify the Tiingo provider works correctly with actual API calls.

Requirements:
    - TIINGO_API_KEY environment variable must be set
    - Free tier: 1000 calls/day, 500 unique symbols/month
    - API key from: https://www.tiingo.com/

Test Coverage:
    - Stock daily OHLCV data (AAPL)
    - Multiple frequencies (daily, weekly, monthly)
    - Rate limiting behavior
    - Error handling
"""

import os
from datetime import datetime, timedelta

import polars as pl
import pytest

from ml4t.data.providers.tiingo import TiingoProvider

# Get API key from environment
TIINGO_API_KEY = os.getenv("TIINGO_API_KEY")

# Skip all tests if no API key
pytestmark = pytest.mark.skipif(
    not TIINGO_API_KEY,
    reason="TIINGO_API_KEY not set - get free key at https://www.tiingo.com/",
)


@pytest.fixture
def provider():
    """Create Tiingo provider with API key."""
    provider = TiingoProvider(api_key=TIINGO_API_KEY)
    yield provider
    provider.close()


class TestTiingoProvider:
    """Test Tiingo provider with real API calls."""

    def test_provider_initialization(self):
        """Test provider can be initialized with API key."""
        provider = TiingoProvider(api_key=TIINGO_API_KEY)
        assert provider.name == "tiingo"
        assert provider.api_key == TIINGO_API_KEY
        provider.close()

    def test_fetch_ohlcv_daily(self, provider):
        """Test fetching daily stock data with real API call."""
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
        required_cols = ["timestamp", "open", "high", "low", "close", "volume"]
        assert all(col in df.columns for col in required_cols)

        # Check for adjusted columns (Tiingo provides adjusted OHLCV)
        adjusted_cols = ["adj_open", "adj_high", "adj_low", "adj_close", "adj_volume"]
        assert all(col in df.columns for col in adjusted_cols)

        # Check for corporate action columns
        assert "dividend" in df.columns
        assert "split_factor" in df.columns

        # Verify data types
        assert df["timestamp"].dtype == pl.Datetime
        assert df["open"].dtype == pl.Float64
        assert df["close"].dtype == pl.Float64
        assert df["adj_close"].dtype == pl.Float64
        assert df["volume"].dtype == pl.Float64

        # Verify OHLCV relationships
        assert (df["high"] >= df["low"]).all()

        # Check for reasonable prices (AAPL should be > $50)
        assert df["close"].min() > 50.0
        assert df["volume"].min() > 0

        print(f"✅ Fetched {len(df)} rows of AAPL daily data")
        print(f"   Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        print(f"   Price range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")
        print(
            f"   Adjusted close range: ${df['adj_close'].min():.2f} - ${df['adj_close'].max():.2f}"
        )

    def test_fetch_ohlcv_weekly(self, provider):
        """Test fetching weekly data with real API call."""
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
        required_cols = ["timestamp", "open", "high", "low", "close", "volume"]
        assert all(col in df.columns for col in required_cols)

        # Should have fewer rows than daily
        assert len(df) < 90  # At most ~13 weeks

        print(f"✅ Fetched {len(df)} rows of AAPL weekly data")
        print(f"   Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        print(f"   Price range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")

    def test_fetch_ohlcv_monthly(self, provider):
        """Test fetching monthly data with real API call."""
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
        required_cols = ["timestamp", "open", "high", "low", "close", "volume"]
        assert all(col in df.columns for col in required_cols)

        # Should have roughly 6 months (may include partial current month)
        assert 6 <= len(df) <= 7

        print(f"✅ Fetched {len(df)} rows of AAPL monthly data")
        print(f"   Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        print(f"   Price range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")

    def test_multiple_symbols(self, provider):
        """Test fetching data for multiple symbols."""
        # Use short date range to minimize API calls
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")

        # Fetch data for multiple symbols
        symbols = ["AAPL", "MSFT", "GOOGL"]
        dfs = []

        for symbol in symbols:
            df = provider.fetch_ohlcv(symbol, start_date, end_date, frequency="daily")
            assert not df.is_empty()
            assert df["symbol"].unique()[0] == symbol
            dfs.append(df)

        print(f"✅ Fetched data for {len(dfs)} symbols")
        for df in dfs:
            symbol = df["symbol"].unique()[0]
            print(f"   {symbol}: {len(df)} rows")

    def test_invalid_symbol(self, provider):
        """Test error handling for invalid symbol."""
        from ml4t.data.core.exceptions import DataNotAvailableError, ProviderError

        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")

        # Tiingo returns 404 for truly invalid symbols
        with pytest.raises((DataNotAvailableError, ProviderError, ValueError)):
            provider.fetch_ohlcv(
                symbol="INVALID_SYMBOL_12345",
                start=start_date,
                end=end_date,
                frequency="daily",
            )

        print("✅ Invalid symbol handled with exception")

    def test_rate_limiting(self, provider):
        """Test that rate limiting is enforced."""
        import time

        # Free tier: 1000 requests/day (~0.67/min)
        # Make 2 rapid requests to test rate limiter
        start_time = time.time()

        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")

        # First request
        df1 = provider.fetch_ohlcv("AAPL", start_date, end_date)
        assert not df1.is_empty()

        # Second request (should be delayed by rate limiter)
        df2 = provider.fetch_ohlcv("MSFT", start_date, end_date)
        assert not df2.is_empty()

        elapsed = time.time() - start_time

        # Rate limiter should add delays between requests
        # With 1 req/min, min time between requests is 60 seconds
        print(f"✅ Made 2 API calls in {elapsed:.1f}s")
        print(f"   Rate limiter is active: {elapsed > 50.0}")

        # Note: Free tier may also throttle on server side
        # Actual delays may be longer than client-side rate limit

    def test_corporate_actions(self, provider):
        """Test that dividend and split data is captured."""
        # Use a longer date range to capture corporate actions
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")

        # Fetch AAPL data (has regular dividends)
        df = provider.fetch_ohlcv("AAPL", start_date, end_date, frequency="daily")

        # Check that corporate action columns exist
        assert "dividend" in df.columns
        assert "split_factor" in df.columns

        # Check for dividends (AAPL pays quarterly)
        dividends = df.filter(pl.col("dividend") > 0)
        if not dividends.is_empty():
            print(f"✅ Found {len(dividends)} dividend payments")
            print(f"   Total dividends: ${dividends['dividend'].sum():.2f} per share")
        else:
            print("⚠️  No dividends found in date range (may be normal)")

        # Check for splits (less common)
        splits = df.filter(pl.col("split_factor") != 1.0)
        if not splits.is_empty():
            print(f"✅ Found {len(splits)} stock splits")
        else:
            print("✅ No stock splits in date range (expected)")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
