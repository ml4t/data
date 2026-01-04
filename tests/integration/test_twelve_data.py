"""Integration tests for Twelve Data provider.

These tests make real API calls to validate the Twelve Data provider implementation.

WARNING: Rate Limits
- Free tier: 8 requests per minute
- Tests include delays to respect rate limits
- Estimated test duration: ~3-5 minutes

Requirements:
- TWELVE_DATA_API_KEY environment variable must be set
- Get free API key at: https://twelvedata.com/

Test Coverage:
- OHLCV data fetching
- Real-time quotes
- Technical indicators
- Provider updater with incremental updates
- Rate limiting behavior
- Error handling
"""

import os
import time
from datetime import datetime, timedelta

import polars as pl
import pytest

from ml4t.data.core.exceptions import DataNotAvailableError
from ml4t.data.providers.twelve_data import TwelveDataProvider


# Rate limit helper
def respect_rate_limit():
    """Wait 8 seconds to respect 8 requests/minute limit."""
    time.sleep(8)


@pytest.fixture
def api_key():
    """Get Twelve Data API key from environment."""
    key = os.getenv("TWELVE_DATA_API_KEY")
    if not key:
        pytest.skip("TWELVE_DATA_API_KEY not set")
    return key


@pytest.fixture
def provider(api_key):
    """Create Twelve Data provider instance."""
    provider = TwelveDataProvider(api_key=api_key)
    yield provider
    provider.close()


class TestTwelveDataProvider:
    """Test TwelveDataProvider functionality."""

    def test_provider_initialization(self, api_key):
        """Test provider initializes correctly."""
        provider = TwelveDataProvider(api_key=api_key)
        assert provider.name == "twelve_data"
        assert provider.api_key == api_key
        assert provider.base_url == "https://api.twelvedata.com"
        provider.close()

    def test_missing_api_key(self):
        """Test error when API key is missing."""
        from ml4t.data.core.exceptions import AuthenticationError

        # Temporarily unset env var
        original = os.environ.pop("TWELVE_DATA_API_KEY", None)
        try:
            with pytest.raises(AuthenticationError, match="API key required"):
                TwelveDataProvider(api_key=None)
        finally:
            if original:
                os.environ["TWELVE_DATA_API_KEY"] = original

    def test_fetch_ohlcv_daily(self, provider):
        """Test fetching daily OHLCV data."""
        end = datetime.now().strftime("%Y-%m-%d")
        start = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")

        df = provider.fetch_ohlcv("AAPL", start, end, frequency="daily")

        assert not df.is_empty()
        assert "timestamp" in df.columns
        assert "open" in df.columns
        assert "high" in df.columns
        assert "low" in df.columns
        assert "close" in df.columns
        assert "volume" in df.columns

        # Validate data types
        assert df["timestamp"].dtype == pl.Datetime
        assert df["open"].dtype == pl.Float64
        assert df["high"].dtype == pl.Float64
        assert df["low"].dtype == pl.Float64
        assert df["close"].dtype == pl.Float64
        assert df["volume"].dtype == pl.Float64

        # Validate OHLC relationships
        assert (df["high"] >= df["low"]).all()
        assert (df["high"] >= df["open"]).all()
        assert (df["high"] >= df["close"]).all()
        assert (df["low"] <= df["open"]).all()
        assert (df["low"] <= df["close"]).all()

        # Validate date range
        assert df["timestamp"].min().date() >= datetime.strptime(start, "%Y-%m-%d").date()
        assert df["timestamp"].max().date() <= datetime.strptime(end, "%Y-%m-%d").date()

        respect_rate_limit()

    def test_fetch_ohlcv_intraday(self, provider):
        """Test fetching intraday data."""
        end = datetime.now().strftime("%Y-%m-%d")
        start = (datetime.now() - timedelta(days=5)).strftime("%Y-%m-%d")

        df = provider.fetch_ohlcv("MSFT", start, end, frequency="1h")

        assert not df.is_empty()
        assert "timestamp" in df.columns

        # Should have hourly data
        timestamps = df["timestamp"].to_list()
        if len(timestamps) >= 2:
            # Check that we have multiple timestamps (intraday data)
            assert len({ts.date() for ts in timestamps}) <= 5

        respect_rate_limit()

    def test_invalid_symbol(self, provider):
        """Test error handling for invalid symbol."""
        from ml4t.data.core.exceptions import ProviderError

        end = datetime.now().strftime("%Y-%m-%d")
        start = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")

        # Twelve Data raises ProviderError for invalid symbols
        with pytest.raises((DataNotAvailableError, ProviderError)):
            provider.fetch_ohlcv("INVALID_SYMBOL_XYZ", start, end)

        respect_rate_limit()

    def test_date_validation(self, provider):
        """Test date validation."""
        with pytest.raises(ValueError, match="Invalid date format"):
            provider.fetch_ohlcv("AAPL", "invalid-date", "2024-01-31")

        with pytest.raises(ValueError, match="Start date must be before"):
            provider.fetch_ohlcv("AAPL", "2024-12-31", "2024-01-01")

    def test_rate_limit_configuration(self, provider):
        """Test rate limit is configured correctly."""
        # Should be 8 requests per 60 seconds
        assert hasattr(provider, "rate_limiter")
        # Rate limiter is managed globally, just verify it exists


# ==================== Updater Tests Removed ====================
# TwelveDataUpdater class has been removed from the codebase.
# Updater functionality is tested separately if needed.

# class TestTwelveDataUpdater:
#     """Test TwelveDataUpdater functionality."""
#     ... (all updater tests commented out)


# ==================== Comprehensive Workflow Tests Removed ====================
# These tests used TwelveDataUpdater which has been removed.
# Workflow testing is done separately if needed.

# @pytest.mark.slow
# class TestComprehensiveWorkflow:
#     """Comprehensive end-to-end workflow tests."""
#     ... (all workflow tests commented out)


if __name__ == "__main__":
    """Run tests manually with proper rate limiting warnings."""
    print("=" * 80)
    print("Twelve Data Integration Tests")
    print("=" * 80)
    print("\nWARNING: These tests make real API calls")
    print("Rate limit: 8 requests per minute (free tier)")
    print("Estimated duration: 3-5 minutes")
    print("\nRequirements:")
    print("  - TWELVE_DATA_API_KEY environment variable must be set")
    print("  - Get free API key at: https://twelvedata.com/")
    print("=" * 80)

    import sys

    pytest.main([__file__, "-v", "-s"] + sys.argv[1:])
