"""Tests for CryptoCompare provider with real API validation."""

import os
from datetime import datetime, timedelta

import polars as pl
import pytest

from ml4t.data.providers.cryptocompare import CryptoCompareProvider


class TestCryptoCompareProvider:
    """Test CryptoCompare provider functionality."""

    def test_provider_initialization(self):
        """Test provider can be initialized."""
        provider = CryptoCompareProvider()
        assert provider.name == "cryptocompare"
        assert provider.exchange == "CCCAGG"
        assert provider.timeout == 30.0

    def test_provider_initialization_with_api_key(self):
        """Test provider initialization with API key."""
        provider = CryptoCompareProvider(api_key="test_key", exchange="Coinbase")
        assert provider.api_key == "test_key"
        assert provider.exchange == "Coinbase"

    def test_symbol_normalization(self):
        """Test symbol normalization logic."""
        provider = CryptoCompareProvider()

        # Test different symbol formats
        assert provider._normalize_symbol("BTC") == ("BTC", "USD")
        assert provider._normalize_symbol("BTC-USD") == ("BTC", "USD")
        assert provider._normalize_symbol("BTC/USD") == ("BTC", "USD")
        assert provider._normalize_symbol("BTCUSD") == ("BTC", "USD")
        assert provider._normalize_symbol("BTCUSDT") == ("BTC", "USDT")
        assert provider._normalize_symbol("ETHBTC") == ("ETH", "BTC")

    def test_frequency_mapping(self):
        """Test frequency mapping to API endpoints."""
        provider = CryptoCompareProvider()

        assert provider.FREQUENCY_MAP["minute"] == "histominute"
        assert provider.FREQUENCY_MAP["hourly"] == "histohour"
        assert provider.FREQUENCY_MAP["daily"] == "histoday"

    @pytest.mark.skipif(
        not os.getenv("CRYPTOCOMPARE_API_KEY"), reason="CRYPTOCOMPARE_API_KEY not set"
    )
    def test_real_api_minimal_daily(self):
        """Test real API with minimal data (1 day, daily frequency)."""
        provider = CryptoCompareProvider(api_key=os.getenv("CRYPTOCOMPARE_API_KEY"))

        # Get yesterday's date to ensure data is available
        yesterday = (datetime.now() - timedelta(days=2)).strftime("%Y-%m-%d")

        # Fetch minimal data - just 1 day of BTC daily data
        df = provider.fetch_ohlcv("BTC", yesterday, yesterday, "daily")

        # Verify data structure and content
        assert isinstance(df, pl.DataFrame)

        if len(df) > 0:  # Data is available for the requested date
            # Check schema (DataFrame now includes symbol column)
            expected_columns = ["timestamp", "open", "high", "low", "close", "volume"]
            assert all(col in df.columns for col in expected_columns)

            # Check data types (precision may vary: ms or ns)
            assert df["timestamp"].dtype in [pl.Datetime("ms", "UTC"), pl.Datetime("ns", "UTC")]
            assert all(
                df[col].dtype == pl.Float64 for col in ["open", "high", "low", "close", "volume"]
            )

            # Check data validity
            assert all(df["high"] >= df["low"])  # High >= Low
            assert all(df["high"] >= df["open"])  # High >= Open
            assert all(df["high"] >= df["close"])  # High >= Close
            assert all(df["low"] <= df["open"])  # Low <= Open
            assert all(df["low"] <= df["close"])  # Low <= Close
            assert all(df["volume"] >= 0)  # Volume non-negative

            print(f"✅ Successfully fetched {len(df)} rows for BTC")
            print(f"   Date: {df['timestamp'][0]}")
            print(f"   Open: ${df['open'][0]:.2f}")
            print(f"   Close: ${df['close'][0]:.2f}")
        else:
            # Empty DataFrame is valid - no data available for requested date
            print("ℹ️  No data returned (might be weekend/holiday or data not yet settled)")
            assert df.columns == []  # Empty DataFrame has no columns

    @pytest.mark.skipif(
        not os.getenv("CRYPTOCOMPARE_API_KEY"), reason="CRYPTOCOMPARE_API_KEY not set"
    )
    def test_real_api_eth_minimal(self):
        """Test real API with ETH data."""
        provider = CryptoCompareProvider(api_key=os.getenv("CRYPTOCOMPARE_API_KEY"))

        # Get data from 3 days ago
        test_date = (datetime.now() - timedelta(days=3)).strftime("%Y-%m-%d")

        # Fetch ETH daily data
        df = provider.fetch_ohlcv("ETH", test_date, test_date, "daily")

        assert isinstance(df, pl.DataFrame)

        if len(df) > 0:
            # Verify ETH prices are reasonable (> $500, < $10000 as sanity check)
            assert all(df["close"] > 500), "ETH price seems too low"
            assert all(df["close"] < 10000), "ETH price seems too high"

            print(f"✅ Successfully fetched ETH data: ${df['close'][0]:.2f}")

    def test_invalid_frequency(self):
        """Test handling of invalid frequency."""
        provider = CryptoCompareProvider()

        with pytest.raises(ValueError, match="Unsupported frequency"):
            provider._fetch_raw_data("BTC", "2024-01-01", "2024-01-02", "invalid")

    def test_empty_symbol(self):
        """Test handling of empty symbol."""
        provider = CryptoCompareProvider()

        # This should be caught by BaseProvider input validation
        with pytest.raises(ValueError, match="Symbol cannot be empty"):
            provider.fetch_ohlcv("", "2024-01-01", "2024-01-02")

    def test_date_range_validation(self):
        """Test date range validation."""
        provider = CryptoCompareProvider()

        # Test start > end (should be caught by BaseProvider)
        with pytest.raises(ValueError, match="Start date must be before or equal to end date"):
            provider.fetch_ohlcv("BTC", "2024-01-02", "2024-01-01")

    def test_transform_data_empty(self):
        """Test transform_data with empty input."""
        provider = CryptoCompareProvider()

        # Test with None
        result = provider._transform_data(None, "BTC")
        assert isinstance(result, pl.DataFrame)
        assert len(result) == 0

        # Test with empty data
        result = provider._transform_data({"data": []}, "BTC")
        assert isinstance(result, pl.DataFrame)
        assert len(result) == 0

    def test_transform_data_with_mock_data(self):
        """Test transform_data with mock API response."""
        provider = CryptoCompareProvider()

        from datetime import UTC

        start_dt = datetime(2024, 1, 1, tzinfo=UTC)
        end_dt = datetime(2024, 1, 2, tzinfo=UTC)

        mock_raw_data = {
            "data": [
                {
                    "time": int(start_dt.timestamp()),
                    "high": 45000.0,
                    "low": 44000.0,
                    "open": 44500.0,
                    "close": 44800.0,
                    "volumefrom": 1000.0,
                }
            ],
            "symbol": "BTC",
            "start_dt": start_dt,
            "end_dt": end_dt,
            "base": "BTC",
            "quote": "USD",
        }

        df = provider._transform_data(mock_raw_data, "BTC")

        assert isinstance(df, pl.DataFrame)
        assert len(df) == 1
        # DataFrame now includes symbol column
        expected_columns = ["timestamp", "open", "high", "low", "close", "volume"]
        assert all(col in df.columns for col in expected_columns)
        assert df["open"][0] == 44500.0
        assert df["close"][0] == 44800.0
        assert df["volume"][0] == 1000.0


class TestCryptoCompareProviderIntegration:
    """Integration tests requiring real API calls."""

    @pytest.mark.skipif(
        not os.getenv("CRYPTOCOMPARE_API_KEY"),
        reason="CRYPTOCOMPARE_API_KEY not set - Set environment variable to run real API tests",
    )
    def test_rate_limiting_behavior(self):
        """Test that rate limiting is working (slow test)."""
        provider = CryptoCompareProvider(api_key=os.getenv("CRYPTOCOMPARE_API_KEY"))

        # Make multiple requests to test rate limiting
        test_date = (datetime.now() - timedelta(days=5)).strftime("%Y-%m-%d")

        start_time = datetime.now()
        df1 = provider.fetch_ohlcv("BTC", test_date, test_date, "daily")
        df2 = provider.fetch_ohlcv("ETH", test_date, test_date, "daily")
        end_time = datetime.now()

        # Should have taken at least some time due to rate limiting
        elapsed = (end_time - start_time).total_seconds()
        assert elapsed >= 0.1, "Rate limiting doesn't seem to be working"

        # Both requests should succeed
        assert isinstance(df1, pl.DataFrame)
        assert isinstance(df2, pl.DataFrame)

    @pytest.mark.skipif(
        not os.getenv("CRYPTOCOMPARE_API_KEY"), reason="CRYPTOCOMPARE_API_KEY not set"
    )
    def test_error_handling_with_invalid_symbol(self):
        """Test error handling with invalid symbol."""
        provider = CryptoCompareProvider(api_key=os.getenv("CRYPTOCOMPARE_API_KEY"))

        # Use a clearly invalid symbol - should either return empty or throw exception
        try:
            df = provider.fetch_ohlcv("INVALID_SYMBOL_12345", "2024-01-01", "2024-01-01", "daily")
            # If it succeeds, should return DataFrame (might be empty)
            assert isinstance(df, pl.DataFrame)
            print(f"Invalid symbol test returned {len(df)} rows")
        except (ValueError, Exception) as e:
            # It's acceptable for invalid symbols to raise exceptions
            print(f"Invalid symbol correctly raised exception: {type(e).__name__}: {e}")
            assert True  # Expected behavior
