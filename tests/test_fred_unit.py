"""Unit tests for FRED provider (no API calls).

These tests verify the FRED provider logic using mocked responses.
No real API calls are made, so no API key is required.

Test Coverage:
    - Provider initialization (with/without API key)
    - Data transformation (including missing values)
    - Empty data handling
    - Timestamp deduplication
    - OHLCV mapping (value → all prices)
    - Error handling
"""

import os
from unittest.mock import MagicMock, patch

import polars as pl
import pytest

from ml4t.data.core.exceptions import (
    AuthenticationError,
    DataNotAvailableError,
    DataValidationError,
)


class TestFREDProviderInit:
    """Test FRED provider initialization."""

    def test_init_with_api_key_param(self):
        """Test initialization with API key as parameter."""
        from ml4t.data.providers.fred import FREDProvider

        provider = FREDProvider(api_key="test_api_key")
        assert provider.name == "fred"
        assert provider.api_key == "test_api_key"
        provider.close()

    def test_init_with_env_var(self):
        """Test initialization with API key from environment variable."""
        from ml4t.data.providers.fred import FREDProvider

        with patch.dict(os.environ, {"FRED_API_KEY": "env_api_key"}):
            provider = FREDProvider()
            assert provider.api_key == "env_api_key"
            provider.close()

    def test_init_without_api_key_raises_error(self):
        """Test initialization without API key raises AuthenticationError."""
        from ml4t.data.providers.fred import FREDProvider

        # Clear any existing env var
        with patch.dict(os.environ, {"FRED_API_KEY": ""}, clear=True):
            if "FRED_API_KEY" in os.environ:
                del os.environ["FRED_API_KEY"]

            with pytest.raises(AuthenticationError) as exc_info:
                FREDProvider()

            assert "FRED API key required" in str(exc_info.value)

    def test_provider_name(self):
        """Test provider name property."""
        from ml4t.data.providers.fred import FREDProvider

        provider = FREDProvider(api_key="test_key")
        assert provider.name == "fred"
        provider.close()

    def test_rate_limit_default(self):
        """Test default rate limit is set correctly."""
        from ml4t.data.providers.fred import FREDProvider

        provider = FREDProvider(api_key="test_key")
        # Default is 100 requests per 60 seconds
        assert provider.DEFAULT_RATE_LIMIT == (100, 60.0)
        provider.close()


class TestFREDDataTransform:
    """Test data transformation logic."""

    @pytest.fixture
    def provider(self):
        """Create a FRED provider for testing."""
        from ml4t.data.providers.fred import FREDProvider

        provider = FREDProvider(api_key="test_key")
        yield provider
        provider.close()

    def test_transform_normal_data(self, provider):
        """Test transformation of normal FRED data."""
        raw_data = [
            {"date": "2024-01-02", "value": "100.5"},
            {"date": "2024-01-03", "value": "101.2"},
            {"date": "2024-01-04", "value": "99.8"},
        ]

        df = provider._transform_data(raw_data, "TEST")

        # Check schema
        assert list(df.columns) == [
            "timestamp",
            "symbol",
            "open",
            "high",
            "low",
            "close",
            "volume",
        ]

        # Check row count
        assert len(df) == 3

        # Check values are mapped correctly (all OHLC equal)
        assert df["open"].to_list() == [100.5, 101.2, 99.8]
        assert df["high"].to_list() == [100.5, 101.2, 99.8]
        assert df["low"].to_list() == [100.5, 101.2, 99.8]
        assert df["close"].to_list() == [100.5, 101.2, 99.8]

        # Volume is placeholder 1.0
        assert df["volume"].to_list() == [1.0, 1.0, 1.0]

        # Symbol is uppercase
        assert df["symbol"].to_list() == ["TEST", "TEST", "TEST"]

    def test_transform_with_missing_values(self, provider):
        """Test transformation handles FRED's '.' missing value indicator."""
        raw_data = [
            {"date": "2024-01-02", "value": "100.5"},
            {"date": "2024-01-03", "value": "."},  # Missing value
            {"date": "2024-01-04", "value": "99.8"},
        ]

        df = provider._transform_data(raw_data, "TEST")

        # Check row count (missing values kept as null)
        assert len(df) == 3

        # Second row should have null values
        assert df["close"][0] == 100.5
        assert df["close"][1] is None
        assert df["close"][2] == 99.8

    def test_transform_empty_data(self, provider):
        """Test transformation of empty data returns empty DataFrame with schema."""
        df = provider._transform_data([], "TEST")

        assert df.is_empty()
        assert list(df.columns) == [
            "timestamp",
            "symbol",
            "open",
            "high",
            "low",
            "close",
            "volume",
        ]

    def test_transform_deduplicates_timestamps(self, provider):
        """Test that duplicate timestamps are removed."""
        raw_data = [
            {"date": "2024-01-02", "value": "100.5"},
            {"date": "2024-01-02", "value": "100.6"},  # Duplicate date
            {"date": "2024-01-03", "value": "99.8"},
        ]

        df = provider._transform_data(raw_data, "TEST")

        # Should have 2 unique dates (first occurrence kept)
        assert len(df) == 2
        timestamps = df["timestamp"].to_list()
        assert len(timestamps) == len(set(timestamps))

    def test_transform_sorts_by_timestamp(self, provider):
        """Test that data is sorted by timestamp."""
        raw_data = [
            {"date": "2024-01-04", "value": "99.8"},
            {"date": "2024-01-02", "value": "100.5"},
            {"date": "2024-01-03", "value": "101.2"},
        ]

        df = provider._transform_data(raw_data, "TEST")

        # Check timestamps are sorted
        timestamps = df["timestamp"].to_list()
        assert timestamps == sorted(timestamps)


class TestFREDValidation:
    """Test validation logic."""

    @pytest.fixture
    def provider(self):
        """Create a FRED provider for testing."""
        from ml4t.data.providers.fred import FREDProvider

        provider = FREDProvider(api_key="test_key")
        yield provider
        provider.close()

    def test_validate_response_skips_ohlc_invariants(self, provider):
        """Test that OHLC invariant validation is skipped for economic data.

        Economic data has identical OHLC values, so standard validation
        would be meaningless.
        """
        from datetime import datetime

        # Create DataFrame with identical OHLC values (normal for FRED)
        df = pl.DataFrame(
            {
                "timestamp": [datetime(2024, 1, 2), datetime(2024, 1, 3)],
                "open": [100.0, 101.0],
                "high": [100.0, 101.0],  # Same as open/low/close
                "low": [100.0, 101.0],
                "close": [100.0, 101.0],
                "volume": [1.0, 1.0],
            }
        ).with_columns(pl.col("timestamp").cast(pl.Datetime))

        # Should not raise - OHLC validation skipped
        validated = provider._validate_response(df)
        assert len(validated) == 2

    def test_validate_response_checks_required_columns(self, provider):
        """Test that required columns are validated."""
        # Missing 'close' column
        df = pl.DataFrame(
            {
                "timestamp": [pl.datetime(2024, 1, 2)],
                "open": [100.0],
                "high": [100.0],
                "low": [100.0],
                # "close" missing
                "volume": [1.0],
            }
        )

        with pytest.raises(DataValidationError) as exc_info:
            provider._validate_response(df)

        assert "Missing required column: close" in str(exc_info.value)

    def test_validate_empty_response(self, provider):
        """Test that empty DataFrame is valid."""
        df = provider._create_empty_dataframe()

        # Should not raise
        validated = provider._validate_response(df)
        assert validated.is_empty()


class TestFREDFrequencyMapping:
    """Test frequency parameter mapping."""

    @pytest.fixture
    def provider(self):
        """Create a FRED provider for testing."""
        from ml4t.data.providers.fred import FREDProvider

        provider = FREDProvider(api_key="test_key")
        yield provider
        provider.close()

    def test_frequency_map_daily(self, provider):
        """Test daily frequency mappings."""
        assert provider.FREQUENCY_MAP["daily"] == "d"
        assert provider.FREQUENCY_MAP["1d"] == "d"
        assert provider.FREQUENCY_MAP["day"] == "d"

    def test_frequency_map_weekly(self, provider):
        """Test weekly frequency mappings."""
        assert provider.FREQUENCY_MAP["weekly"] == "w"
        assert provider.FREQUENCY_MAP["1w"] == "w"
        assert provider.FREQUENCY_MAP["week"] == "w"

    def test_frequency_map_monthly(self, provider):
        """Test monthly frequency mappings."""
        assert provider.FREQUENCY_MAP["monthly"] == "m"
        assert provider.FREQUENCY_MAP["1M"] == "m"
        assert provider.FREQUENCY_MAP["month"] == "m"

    def test_frequency_map_quarterly(self, provider):
        """Test quarterly frequency mappings."""
        assert provider.FREQUENCY_MAP["quarterly"] == "q"
        assert provider.FREQUENCY_MAP["1Q"] == "q"
        assert provider.FREQUENCY_MAP["quarter"] == "q"

    def test_frequency_map_annual(self, provider):
        """Test annual frequency mappings."""
        assert provider.FREQUENCY_MAP["annual"] == "a"
        assert provider.FREQUENCY_MAP["1Y"] == "a"
        assert provider.FREQUENCY_MAP["yearly"] == "a"
        assert provider.FREQUENCY_MAP["year"] == "a"


class TestFREDFetchWithMocks:
    """Test fetch methods with mocked HTTP responses."""

    @pytest.fixture
    def provider(self):
        """Create a FRED provider for testing."""
        from ml4t.data.providers.fred import FREDProvider

        provider = FREDProvider(api_key="test_key")
        yield provider
        provider.close()

    def test_fetch_ohlcv_success(self, provider):
        """Test successful fetch_ohlcv with mocked response."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "observations": [
                {"date": "2024-01-02", "value": "15.5"},
                {"date": "2024-01-03", "value": "16.2"},
            ]
        }

        with patch.object(provider.session, "get", return_value=mock_response):
            df = provider.fetch_ohlcv("VIXCLS", "2024-01-01", "2024-01-31")

        assert len(df) == 2
        assert df["close"].to_list() == [15.5, 16.2]
        assert df["symbol"].to_list() == ["VIXCLS", "VIXCLS"]

    def test_fetch_ohlcv_with_vintage_date(self, provider):
        """Test fetch_ohlcv with vintage_date parameter."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "observations": [
                {"date": "2024-01-02", "value": "100.0"},
            ]
        }

        with patch.object(provider.session, "get", return_value=mock_response) as mock_get:
            df = provider.fetch_ohlcv("GDP", "2024-01-01", "2024-01-31", vintage_date="2024-02-15")

            # Check vintage_date was passed to API
            call_kwargs = mock_get.call_args
            params = call_kwargs[1]["params"]
            assert params["realtime_start"] == "2024-02-15"
            assert params["realtime_end"] == "2024-02-15"

        assert len(df) == 1

    def test_fetch_series_metadata_success(self, provider):
        """Test successful fetch_series_metadata with mocked response."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "seriess": [
                {
                    "id": "VIXCLS",
                    "title": "CBOE Volatility Index: VIX",
                    "frequency": "Daily, Close",
                    "units": "Index",
                    "seasonal_adjustment": "Not Seasonally Adjusted",
                }
            ]
        }

        with patch.object(provider.session, "get", return_value=mock_response):
            metadata = provider.fetch_series_metadata("VIXCLS")

        assert metadata["id"] == "VIXCLS"
        assert "VIX" in metadata["title"]
        assert metadata["frequency"] == "Daily, Close"

    def test_fetch_multiple_success(self, provider):
        """Test successful fetch_multiple with mocked responses."""
        # Mock responses for multiple series
        vix_response = MagicMock()
        vix_response.status_code = 200
        vix_response.json.return_value = {
            "observations": [
                {"date": "2024-01-02", "value": "15.5"},
                {"date": "2024-01-03", "value": "16.2"},
            ]
        }

        unrate_response = MagicMock()
        unrate_response.status_code = 200
        unrate_response.json.return_value = {
            "observations": [
                {"date": "2024-01-02", "value": "3.7"},
                {"date": "2024-01-03", "value": "3.7"},
            ]
        }

        # Return different responses for each call
        with patch.object(provider.session, "get", side_effect=[vix_response, unrate_response]):
            df = provider.fetch_multiple(["VIXCLS", "UNRATE"], "2024-01-01", "2024-01-31")

        assert "VIXCLS_close" in df.columns
        assert "UNRATE_close" in df.columns
        assert len(df) == 2

    def test_fetch_invalid_frequency_raises_error(self, provider):
        """Test that invalid frequency raises DataValidationError."""
        with pytest.raises(DataValidationError) as exc_info:
            provider._fetch_raw_data("TEST", "2024-01-01", "2024-01-31", "invalid_freq")

        assert "Unsupported frequency" in str(exc_info.value)

    def test_fetch_series_not_found(self, provider):
        """Test handling of 404 response."""
        mock_response = MagicMock()
        mock_response.status_code = 404

        with patch.object(provider.session, "get", return_value=mock_response):
            with pytest.raises(DataNotAvailableError):
                provider.fetch_ohlcv("INVALID_SERIES", "2024-01-01", "2024-01-31")


class TestFREDMultipleSeries:
    """Test fetch_multiple edge cases."""

    @pytest.fixture
    def provider(self):
        """Create a FRED provider for testing."""
        from ml4t.data.providers.fred import FREDProvider

        provider = FREDProvider(api_key="test_key")
        yield provider
        provider.close()

    def test_fetch_multiple_empty_list_raises_error(self, provider):
        """Test that empty series list raises error."""
        with pytest.raises(DataValidationError) as exc_info:
            provider.fetch_multiple([], "2024-01-01", "2024-01-31")

        assert "series_ids cannot be empty" in str(exc_info.value)

    def test_fetch_multiple_with_forward_fill(self, provider):
        """Test forward fill behavior in fetch_multiple."""
        # Create responses with different dates (simulating different frequencies)
        daily_response = MagicMock()
        daily_response.status_code = 200
        daily_response.json.return_value = {
            "observations": [
                {"date": "2024-01-02", "value": "100.0"},
                {"date": "2024-01-03", "value": "101.0"},
                {"date": "2024-01-04", "value": "102.0"},
            ]
        }

        monthly_response = MagicMock()
        monthly_response.status_code = 200
        monthly_response.json.return_value = {
            "observations": [
                {"date": "2024-01-02", "value": "3.7"},
                # No data for Jan 3, 4
            ]
        }

        with patch.object(provider.session, "get", side_effect=[daily_response, monthly_response]):
            df = provider.fetch_multiple(
                ["DAILY", "MONTHLY"],
                "2024-01-01",
                "2024-01-31",
                forward_fill=True,
            )

        # Forward fill should fill monthly value for Jan 3, 4
        assert len(df) == 3
        # Monthly value should be forward-filled
        monthly_values = df["MONTHLY_close"].to_list()
        assert monthly_values == [3.7, 3.7, 3.7]
