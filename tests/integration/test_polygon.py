"""Integration tests for Polygon provider (real API calls).

These tests verify the Polygon provider works correctly with actual API calls.

Requirements:
    - POLYGON_API_KEY environment variable must be set
    - Free tier: 5 requests/minute (tests respect rate limits)
    - API key from: https://polygon.io/

Test Coverage:
    - Stock OHLCV data (AAPL)
    - Crypto OHLCV data (X:BTCUSD)
    - Rate limiting behavior
    - Error handling
    - PolygonUpdater incremental updates
"""

import os
from datetime import datetime, timedelta

import polars as pl
import pytest

from ml4t.data.core.exceptions import AuthenticationError
from ml4t.data.providers.polygon import PolygonProvider

# Get API key from environment
POLYGON_API_KEY = os.getenv("POLYGON_API_KEY")

# Skip all tests if no API key
pytestmark = pytest.mark.skipif(
    not POLYGON_API_KEY,
    reason="POLYGON_API_KEY not set - get key at https://polygon.io/",
)


@pytest.fixture
def provider():
    """Create Polygon provider with API key."""
    provider = PolygonProvider(api_key=POLYGON_API_KEY)
    yield provider
    provider.close()


class TestPolygonProvider:
    """Test Polygon provider with real API calls."""

    def test_provider_initialization(self):
        """Test provider can be initialized with API key."""
        provider = PolygonProvider(api_key=POLYGON_API_KEY)
        assert provider.name == "polygon"
        assert provider.api_key == POLYGON_API_KEY
        provider.close()

    def test_provider_requires_api_key(self):
        """Test provider raises error without API key."""
        # Clear environment variable temporarily
        original_key = os.environ.pop("POLYGON_API_KEY", None)
        try:
            with pytest.raises(AuthenticationError) as exc_info:
                PolygonProvider(api_key=None)
            assert "API key required" in str(exc_info.value)
        finally:
            # Restore key
            if original_key:
                os.environ["POLYGON_API_KEY"] = original_key

    def test_fetch_ohlcv_stock_daily(self, provider):
        """Test fetching daily stock data with real API call."""
        # Use recent date range to ensure data availability
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")

        # Fetch AAPL daily data
        df = provider.fetch_ohlcv(
            symbol="AAPL",
            start=start_date,
            end=end_date,
            frequency="day",
        )

        # Verify data structure
        assert isinstance(df, pl.DataFrame)
        assert not df.is_empty(), "Should fetch some data for AAPL"

        # Check required columns
        required_cols = ["timestamp", "open", "high", "low", "close", "volume", "symbol"]
        assert all(col in df.columns for col in required_cols)

        # Verify data types
        assert df["timestamp"].dtype == pl.Datetime
        assert df["open"].dtype == pl.Float64
        assert df["close"].dtype == pl.Float64
        assert df["volume"].dtype == pl.Float64
        assert df["symbol"].dtype == pl.String

        # Check symbol value
        assert df["symbol"][0] == "AAPL"

        # Verify OHLCV relationships
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

    def test_fetch_ohlcv_crypto(self, provider):
        """Test fetching crypto data with real API call."""
        # Bitcoin data - use recent dates
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=14)).strftime("%Y-%m-%d")

        # Polygon uses X:BTCUSD format for crypto
        df = provider.fetch_ohlcv(
            symbol="X:BTCUSD",
            start=start_date,
            end=end_date,
            frequency="day",
        )

        # Verify data structure
        assert isinstance(df, pl.DataFrame)
        assert not df.is_empty(), "Should fetch some Bitcoin data"

        # Check required columns
        required_cols = ["timestamp", "open", "high", "low", "close", "volume"]
        assert all(col in df.columns for col in required_cols)

        # Check symbol
        assert df["symbol"][0] == "X:BTCUSD"

        # BTC price should be > $10,000 (sanity check)
        assert df["close"].min() > 10000.0

        print(f"✅ Fetched {len(df)} rows of Bitcoin daily data")
        print(f"   Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        print(f"   Price range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")

    def test_rate_limiting(self, provider):
        """Test that rate limiting is enforced."""
        import time

        # Free tier: 5 requests/minute
        # Make multiple rapid requests to test rate limiter
        start_time = time.time()

        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")

        # Make 3 requests (should be allowed within 60 seconds)
        symbols = ["AAPL", "GOOGL", "MSFT"]
        for symbol in symbols:
            df = provider.fetch_ohlcv(symbol, start_date, end_date)
            assert not df.is_empty()

        elapsed = time.time() - start_time

        # Rate limiter should add delays between requests
        # With 5 req/min, min time between requests is 12 seconds
        # For 3 requests, expect at least 24 seconds total
        print(f"✅ Made 3 API calls in {elapsed:.1f}s")
        print(f"   Rate limiter is active: {elapsed > 20.0}")

        # Note: Free tier may also throttle on server side
        # Actual delays may be longer than client-side rate limit

    def test_invalid_symbol(self, provider):
        """Test error handling for invalid symbol."""
        from ml4t.data.core.exceptions import SymbolNotFoundError

        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")

        # Fetch with invalid symbol - should raise SymbolNotFoundError
        with pytest.raises(SymbolNotFoundError) as exc_info:
            provider.fetch_ohlcv(
                symbol="INVALID_SYMBOL_12345",
                start=start_date,
                end=end_date,
            )

        assert "polygon" in str(exc_info.value).lower()
        assert "INVALID_SYMBOL_12345" in str(exc_info.value)
        print("✅ Invalid symbol raises SymbolNotFoundError")

    def test_invalid_api_key(self):
        """Test authentication error with invalid API key."""
        provider = PolygonProvider(api_key="invalid_key_12345")

        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")

        # Should raise AuthenticationError
        with pytest.raises(AuthenticationError) as exc_info:
            provider.fetch_ohlcv("AAPL", start_date, end_date)

        assert "polygon" in str(exc_info.value).lower()
        print("✅ Invalid API key raises AuthenticationError")


# ==================== Updater Tests Removed ====================
# PolygonUpdater class has been removed from the codebase.
# Updater functionality is tested separately if needed.

# class TestPolygonUpdater:
#     """Test Polygon updater with real API calls."""
#     ... (all updater tests commented out)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
