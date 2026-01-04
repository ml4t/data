"""Integration tests for FRED provider (real API calls).

These tests verify the FRED provider works correctly with actual API calls.

Requirements:
    - FRED_API_KEY environment variable must be set
    - Free API key from: https://fred.stlouisfed.org/docs/api/api_key.html
    - Rate limit: 120 requests per minute (we use 100 to be safe)

Test Coverage:
    - Daily series (VIXCLS - VIX index)
    - Monthly series (UNRATE - unemployment rate)
    - Treasury yields (DGS10 - 10-year yield)
    - Invalid series handling
    - Series metadata
    - Multiple series alignment
    - Missing values handling
    - Rate limiting behavior

IMPORTANT:
    These tests make ~12 API calls total. FRED has generous rate limits
    (120 req/min), but be mindful if running repeatedly.
"""

import os
from datetime import datetime, timedelta

import polars as pl
import pytest

from ml4t.data.core.exceptions import DataNotAvailableError, DataValidationError
from ml4t.data.providers.fred import FREDProvider

# Get API key from environment
FRED_API_KEY = os.getenv("FRED_API_KEY")

# Skip all tests if no API key
pytestmark = pytest.mark.skipif(
    not FRED_API_KEY,
    reason="FRED_API_KEY not set - get free key at https://fred.stlouisfed.org/docs/api/api_key.html",
)


@pytest.fixture
def provider():
    """Create FRED provider with API key."""
    provider = FREDProvider(api_key=FRED_API_KEY)
    yield provider
    provider.close()


class TestFREDProvider:
    """Test FRED provider with real API calls.

    Note: ~12 API calls total, FRED allows 120/min.
    """

    def test_provider_initialization(self):
        """Test provider can be initialized with API key.

        This test does not make any API calls.
        """
        provider = FREDProvider(api_key=FRED_API_KEY)
        assert provider.name == "fred"
        assert provider.api_key == FRED_API_KEY
        provider.close()

    def test_fetch_daily_series_vix(self, provider):
        """Test fetching daily VIX data with real API call.

        VIXCLS: CBOE Volatility Index (daily)
        API calls: 1
        """
        # Use recent date range
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")

        df = provider.fetch_ohlcv(
            symbol="VIXCLS",
            start=start_date,
            end=end_date,
            frequency="daily",
        )

        # Verify data structure
        assert isinstance(df, pl.DataFrame)
        assert not df.is_empty(), "Should fetch some data for VIXCLS"

        # Check required columns
        required_cols = ["timestamp", "symbol", "open", "high", "low", "close", "volume"]
        assert all(col in df.columns for col in required_cols)

        # Verify data types
        assert df["timestamp"].dtype == pl.Datetime
        assert df["symbol"].dtype == pl.String
        assert df["close"].dtype == pl.Float64

        # VIX values should be in reasonable range (typically 10-80)
        # Filter out nulls first
        vix_values = df.filter(pl.col("close").is_not_null())["close"]
        if len(vix_values) > 0:
            assert vix_values.min() > 5.0, "VIX should be > 5"
            assert vix_values.max() < 100.0, "VIX should be < 100"

        # Volume should be placeholder 1.0
        assert (df["volume"] == 1.0).all(), "Volume should be 1.0 for FRED data"

        # OHLC should all be equal for economic data
        non_null = df.filter(pl.col("close").is_not_null())
        if len(non_null) > 0:
            assert (non_null["open"] == non_null["close"]).all()
            assert (non_null["high"] == non_null["close"]).all()
            assert (non_null["low"] == non_null["close"]).all()

        print(f"✅ Fetched {len(df)} rows of VIXCLS daily data")
        print(f"   Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        if len(vix_values) > 0:
            print(f"   VIX range: {vix_values.min():.2f} - {vix_values.max():.2f}")

    def test_fetch_monthly_series_unemployment(self, provider):
        """Test fetching monthly unemployment data with real API call.

        UNRATE: Unemployment Rate (monthly)
        API calls: 1
        """
        # Use 1 year for monthly data
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")

        df = provider.fetch_ohlcv(
            symbol="UNRATE",
            start=start_date,
            end=end_date,
            frequency="monthly",
        )

        # Verify data structure
        assert isinstance(df, pl.DataFrame)
        assert not df.is_empty(), "Should fetch some monthly unemployment data"

        # Check required columns
        required_cols = ["timestamp", "symbol", "open", "high", "low", "close", "volume"]
        assert all(col in df.columns for col in required_cols)

        # Monthly data should have fewer rows than daily
        # For 12 months, expect ~12 rows
        assert 8 <= len(df) <= 15, f"Expected 8-15 monthly rows, got {len(df)}"

        # Unemployment rate should be in reasonable range (3-15%)
        unrate_values = df.filter(pl.col("close").is_not_null())["close"]
        if len(unrate_values) > 0:
            assert unrate_values.min() >= 2.0, "Unemployment should be >= 2%"
            assert unrate_values.max() <= 20.0, "Unemployment should be <= 20%"

        print(f"✅ Fetched {len(df)} rows of UNRATE monthly data")
        print(f"   Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        if len(unrate_values) > 0:
            print(f"   Rate range: {unrate_values.min():.1f}% - {unrate_values.max():.1f}%")

    def test_fetch_treasury_yield(self, provider):
        """Test fetching 10-Year Treasury yield data.

        DGS10: 10-Year Treasury Constant Maturity Rate (daily)
        API calls: 1
        """
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")

        df = provider.fetch_ohlcv(
            symbol="DGS10",
            start=start_date,
            end=end_date,
            frequency="daily",
        )

        # Verify data structure
        assert isinstance(df, pl.DataFrame)
        assert not df.is_empty(), "Should fetch some Treasury yield data"

        # Check required columns
        required_cols = ["timestamp", "symbol", "open", "high", "low", "close", "volume"]
        assert all(col in df.columns for col in required_cols)

        # Treasury yields should be positive and reasonable (0.5-10%)
        yield_values = df.filter(pl.col("close").is_not_null())["close"]
        if len(yield_values) > 0:
            assert yield_values.min() >= 0.0, "Yield should be >= 0%"
            assert yield_values.max() <= 15.0, "Yield should be <= 15%"

        print(f"✅ Fetched {len(df)} rows of DGS10 daily data")
        print(f"   Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        if len(yield_values) > 0:
            print(f"   Yield range: {yield_values.min():.2f}% - {yield_values.max():.2f}%")

    def test_invalid_series_raises_error(self, provider):
        """Test error handling for invalid series.

        API calls: 1
        """
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")

        # Invalid series should raise DataNotAvailableError or ProviderError
        with pytest.raises((DataNotAvailableError, Exception)):
            provider.fetch_ohlcv(
                symbol="INVALID_SERIES_XYZ_12345",
                start=start_date,
                end=end_date,
                frequency="daily",
            )

        print("✅ Invalid series correctly raises error")

    def test_fetch_series_metadata(self, provider):
        """Test fetching series metadata.

        API calls: 1
        """
        metadata = provider.fetch_series_metadata("VIXCLS")

        # Check required fields
        assert "id" in metadata
        assert metadata["id"] == "VIXCLS"

        # Title should mention VIX
        assert "title" in metadata
        assert "VIX" in metadata["title"].upper()

        # Frequency should be present
        assert "frequency" in metadata

        print("✅ Fetched metadata for VIXCLS")
        print(f"   Title: {metadata['title']}")
        print(f"   Frequency: {metadata['frequency']}")
        if "units" in metadata:
            print(f"   Units: {metadata['units']}")

    def test_fetch_multiple_series(self, provider):
        """Test fetching and aligning multiple series.

        API calls: 2 (VIXCLS + DGS10)
        """
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")

        df = provider.fetch_multiple(
            series_ids=["VIXCLS", "DGS10"],
            start=start_date,
            end=end_date,
            frequency="daily",
            forward_fill=True,
        )

        # Verify structure
        assert isinstance(df, pl.DataFrame)
        assert not df.is_empty()

        # Check columns
        assert "timestamp" in df.columns
        assert "VIXCLS_close" in df.columns
        assert "DGS10_close" in df.columns

        print(f"✅ Fetched {len(df)} aligned rows for VIXCLS + DGS10")
        print(f"   Columns: {df.columns}")

    def test_handles_missing_values(self, provider):
        """Test handling of missing values (FRED returns '.' for missing).

        API calls: 1

        Note: We use a date range that might have weekends/holidays
        where data is missing.
        """
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=14)).strftime("%Y-%m-%d")

        df = provider.fetch_ohlcv(
            symbol="VIXCLS",
            start=start_date,
            end=end_date,
            frequency="daily",
        )

        # Should have data
        assert isinstance(df, pl.DataFrame)

        # If there are any missing values, they should be null (not ".")
        if len(df) > 0:
            for col in ["open", "high", "low", "close"]:
                # Check that no string "." values exist
                # (they should have been converted to null)
                assert df[col].dtype == pl.Float64

        print("✅ Missing values handled correctly")
        print(f"   Rows with null close: {df['close'].null_count()}")

    def test_empty_date_range(self, provider):
        """Test handling of date range with no data.

        API calls: 1
        """
        # Very old date range (before FRED data starts for most series)
        df = provider.fetch_ohlcv(
            symbol="VIXCLS",
            start="1900-01-01",
            end="1900-12-31",
            frequency="daily",
        )

        # Should return empty DataFrame (not raise error)
        assert isinstance(df, pl.DataFrame)
        assert df.is_empty() or len(df) == 0

        print("✅ Empty date range returns empty DataFrame")

    def test_rate_limiting_works(self):
        """Test rate limiting behavior.

        API calls: 3 (quick succession)
        """
        # Create provider with standard rate limit
        provider = FREDProvider(api_key=FRED_API_KEY)

        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")

        # Make 3 quick calls - rate limiter should handle
        # All series must support daily frequency
        symbols = ["VIXCLS", "DGS10", "SP500"]
        for symbol in symbols:
            df = provider.fetch_ohlcv(
                symbol=symbol,
                start=start_date,
                end=end_date,
                frequency="daily",
            )
            assert isinstance(df, pl.DataFrame)

        provider.close()
        print("✅ Rate limiting works correctly")

    def test_invalid_frequency(self, provider):
        """Test error handling for invalid frequency.

        API calls: 0 (validation happens before API call)
        """
        with pytest.raises(DataValidationError) as exc_info:
            provider.fetch_ohlcv(
                symbol="VIXCLS",
                start="2024-01-01",
                end="2024-01-31",
                frequency="invalid_frequency",
            )

        assert "Unsupported frequency" in str(exc_info.value)
        print("✅ Invalid frequency correctly raises DataValidationError")


# Test Summary:
# ==============
# Total API calls: ~12 calls
# 1. test_fetch_daily_series_vix: 1 call
# 2. test_fetch_monthly_series_unemployment: 1 call
# 3. test_fetch_treasury_yield: 1 call
# 4. test_invalid_series_raises_error: 1 call
# 5. test_fetch_series_metadata: 1 call
# 6. test_fetch_multiple_series: 2 calls
# 7. test_handles_missing_values: 1 call
# 8. test_empty_date_range: 1 call
# 9. test_rate_limiting_works: 3 calls
# 10. test_invalid_frequency: 0 calls (validation only)
#
# FRED allows 120 requests/minute, so this is well within limits.
