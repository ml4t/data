"""Unit tests for Kalshi provider (no API calls).

These tests verify the Kalshi provider logic using mocked responses.
No real API calls are made, so no API key is required.

Test Coverage:
    - Provider initialization
    - Series ticker extraction
    - Frequency mapping
    - Data transformation
    - Empty data handling
    - Timestamp deduplication
    - Error handling
    - List markets/series methods
"""

from datetime import datetime
from unittest.mock import MagicMock, patch

import polars as pl
import pytest

from ml4t.data.core.exceptions import (
    DataNotAvailableError,
    DataValidationError,
    SymbolNotFoundError,
)
from ml4t.data.providers.kalshi import KalshiProvider


class TestKalshiProviderInit:
    """Test Kalshi provider initialization."""

    def test_init_without_api_key(self):
        """Test initialization without API key (allowed for public data)."""
        provider = KalshiProvider()
        assert provider.name == "kalshi"
        assert provider.api_key is None
        provider.close()

    def test_init_with_api_key_param(self):
        """Test initialization with API key as parameter."""
        provider = KalshiProvider(api_key="test_api_key")
        assert provider.name == "kalshi"
        assert provider.api_key == "test_api_key"
        provider.close()

    def test_provider_name(self):
        """Test provider name property."""
        provider = KalshiProvider()
        assert provider.name == "kalshi"
        provider.close()

    def test_rate_limit_default(self):
        """Test default rate limit is set correctly."""
        provider = KalshiProvider()
        # Default is 10 requests per 1 second
        assert provider.DEFAULT_RATE_LIMIT == (10, 1.0)
        provider.close()

    def test_base_url(self):
        """Test base URL is correct."""
        provider = KalshiProvider()
        assert "elections.kalshi.com" in provider.BASE_URL
        assert "trade-api/v2" in provider.BASE_URL
        provider.close()


class TestKalshiSeriesExtraction:
    """Test series ticker extraction from market tickers."""

    @pytest.fixture
    def provider(self):
        """Create a Kalshi provider for testing."""
        provider = KalshiProvider()
        yield provider
        provider.close()

    def test_extract_simple_ticker(self, provider):
        """Test extraction from simple ticker (series-date)."""
        assert provider._extract_series_ticker("KXINFL-25JAN") == "KXINFL"

    def test_extract_with_strike(self, provider):
        """Test extraction from ticker with strike price."""
        assert provider._extract_series_ticker("KXSPX-25JAN03-T5950") == "KXSPX"

    def test_extract_preserves_uppercase(self, provider):
        """Test that series ticker is extracted correctly from lowercase."""
        assert provider._extract_series_ticker("kxfed-25jan") == "kxfed"

    def test_extract_invalid_ticker_raises_error(self, provider):
        """Test that invalid ticker format raises error."""
        with pytest.raises(DataValidationError) as exc_info:
            provider._extract_series_ticker("")

        assert "Invalid market ticker format" in str(exc_info.value)


class TestKalshiFrequencyMapping:
    """Test frequency parameter mapping."""

    @pytest.fixture
    def provider(self):
        """Create a Kalshi provider for testing."""
        provider = KalshiProvider()
        yield provider
        provider.close()

    def test_frequency_map_minute(self, provider):
        """Test minute frequency mappings."""
        assert provider.FREQUENCY_MAP["1m"] == 1
        assert provider.FREQUENCY_MAP["minute"] == 1

    def test_frequency_map_hourly(self, provider):
        """Test hourly frequency mappings."""
        assert provider.FREQUENCY_MAP["1h"] == 60
        assert provider.FREQUENCY_MAP["hourly"] == 60
        assert provider.FREQUENCY_MAP["hour"] == 60

    def test_frequency_map_daily(self, provider):
        """Test daily frequency mappings."""
        assert provider.FREQUENCY_MAP["1d"] == 1440
        assert provider.FREQUENCY_MAP["daily"] == 1440
        assert provider.FREQUENCY_MAP["day"] == 1440


class TestKalshiDataTransform:
    """Test data transformation logic."""

    @pytest.fixture
    def provider(self):
        """Create a Kalshi provider for testing."""
        provider = KalshiProvider()
        yield provider
        provider.close()

    def test_transform_normal_data(self, provider):
        """Test transformation of normal Kalshi candlestick data."""
        # Unix timestamps for test data
        ts1 = int(datetime(2024, 1, 2, 12, 0).timestamp())
        ts2 = int(datetime(2024, 1, 3, 12, 0).timestamp())
        ts3 = int(datetime(2024, 1, 4, 12, 0).timestamp())

        raw_data = [
            {
                "end_period_ts": ts1,
                "open": 0.45,
                "high": 0.48,
                "low": 0.42,
                "close": 0.47,
                "volume": 1250,
            },
            {
                "end_period_ts": ts2,
                "open": 0.47,
                "high": 0.52,
                "low": 0.45,
                "close": 0.51,
                "volume": 2100,
            },
            {
                "end_period_ts": ts3,
                "open": 0.51,
                "high": 0.55,
                "low": 0.49,
                "close": 0.53,
                "volume": 1800,
            },
        ]

        df = provider._transform_data(raw_data, "KXINFL-25JAN")

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

        # Check OHLCV values (probabilities 0-1)
        assert df["open"].to_list() == [0.45, 0.47, 0.51]
        assert df["high"].to_list() == [0.48, 0.52, 0.55]
        assert df["low"].to_list() == [0.42, 0.45, 0.49]
        assert df["close"].to_list() == [0.47, 0.51, 0.53]
        assert df["volume"].to_list() == [1250.0, 2100.0, 1800.0]

        # Symbol is uppercase
        assert df["symbol"].to_list() == ["KXINFL-25JAN", "KXINFL-25JAN", "KXINFL-25JAN"]

    def test_transform_empty_data(self, provider):
        """Test transformation of empty data returns empty DataFrame with schema."""
        df = provider._transform_data([], "KXINFL-25JAN")

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
        ts = int(datetime(2024, 1, 2, 12, 0).timestamp())

        raw_data = [
            {
                "end_period_ts": ts,
                "open": 0.45,
                "high": 0.48,
                "low": 0.42,
                "close": 0.47,
                "volume": 1250,
            },
            {
                "end_period_ts": ts,  # Duplicate
                "open": 0.46,
                "high": 0.49,
                "low": 0.43,
                "close": 0.48,
                "volume": 1300,
            },
            {
                "end_period_ts": ts + 86400,
                "open": 0.48,
                "high": 0.50,
                "low": 0.45,
                "close": 0.49,
                "volume": 900,
            },
        ]

        df = provider._transform_data(raw_data, "TEST")

        # Should have 2 unique timestamps
        assert len(df) == 2
        timestamps = df["timestamp"].to_list()
        assert len(timestamps) == len(set(timestamps))

    def test_transform_sorts_by_timestamp(self, provider):
        """Test that data is sorted by timestamp."""
        ts1 = int(datetime(2024, 1, 4, 12, 0).timestamp())
        ts2 = int(datetime(2024, 1, 2, 12, 0).timestamp())
        ts3 = int(datetime(2024, 1, 3, 12, 0).timestamp())

        raw_data = [
            {
                "end_period_ts": ts1,
                "open": 0.5,
                "high": 0.5,
                "low": 0.5,
                "close": 0.5,
                "volume": 100,
            },
            {
                "end_period_ts": ts2,
                "open": 0.5,
                "high": 0.5,
                "low": 0.5,
                "close": 0.5,
                "volume": 100,
            },
            {
                "end_period_ts": ts3,
                "open": 0.5,
                "high": 0.5,
                "low": 0.5,
                "close": 0.5,
                "volume": 100,
            },
        ]

        df = provider._transform_data(raw_data, "TEST")

        # Check timestamps are sorted
        timestamps = df["timestamp"].to_list()
        assert timestamps == sorted(timestamps)


class TestKalshiValidation:
    """Test validation logic."""

    @pytest.fixture
    def provider(self):
        """Create a Kalshi provider for testing."""
        provider = KalshiProvider()
        yield provider
        provider.close()

    def test_validate_response_valid_ohlc(self, provider):
        """Test OHLC invariant validation passes for valid data."""
        df = pl.DataFrame(
            {
                "timestamp": [datetime(2024, 1, 2), datetime(2024, 1, 3)],
                "open": [0.45, 0.50],
                "high": [0.48, 0.55],  # high >= open, close, low
                "low": [0.42, 0.48],  # low <= open, close, high
                "close": [0.47, 0.52],
                "volume": [1000.0, 1200.0],
            }
        ).with_columns(pl.col("timestamp").cast(pl.Datetime))

        # Should not raise
        validated = provider._validate_response(df)
        assert len(validated) == 2

    def test_validate_response_invalid_ohlc_raises_error(self, provider):
        """Test OHLC invariant validation fails for invalid data."""
        df = pl.DataFrame(
            {
                "timestamp": [datetime(2024, 1, 2)],
                "open": [0.45],
                "high": [0.40],  # Invalid: high < low
                "low": [0.48],
                "close": [0.47],
                "volume": [1000.0],
            }
        ).with_columns(pl.col("timestamp").cast(pl.Datetime))

        with pytest.raises(DataValidationError) as exc_info:
            provider._validate_response(df)

        assert "invalid OHLC" in str(exc_info.value)

    def test_validate_response_checks_required_columns(self, provider):
        """Test that required columns are validated."""
        # Missing 'close' column
        df = pl.DataFrame(
            {
                "timestamp": [pl.datetime(2024, 1, 2)],
                "open": [0.45],
                "high": [0.48],
                "low": [0.42],
                # "close" missing
                "volume": [1000.0],
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


class TestKalshiFetchWithMocks:
    """Test fetch methods with mocked HTTP responses."""

    @pytest.fixture
    def provider(self):
        """Create a Kalshi provider for testing."""
        provider = KalshiProvider()
        yield provider
        provider.close()

    def test_fetch_ohlcv_success(self, provider):
        """Test successful fetch_ohlcv with mocked response."""
        ts1 = int(datetime(2024, 1, 2, 12, 0).timestamp())
        ts2 = int(datetime(2024, 1, 3, 12, 0).timestamp())

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "candlesticks": [
                {
                    "end_period_ts": ts1,
                    "open": 0.45,
                    "high": 0.48,
                    "low": 0.42,
                    "close": 0.47,
                    "volume": 1250,
                },
                {
                    "end_period_ts": ts2,
                    "open": 0.47,
                    "high": 0.52,
                    "low": 0.45,
                    "close": 0.51,
                    "volume": 2100,
                },
            ]
        }

        with patch.object(provider.session, "get", return_value=mock_response):
            df = provider.fetch_ohlcv("KXINFL-25JAN", "2024-01-01", "2024-01-31")

        assert len(df) == 2
        assert df["close"].to_list() == [0.47, 0.51]
        assert df["symbol"].to_list() == ["KXINFL-25JAN", "KXINFL-25JAN"]

    def test_fetch_ohlcv_auto_detects_series(self, provider):
        """Test that series_ticker is auto-detected from market ticker."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"candlesticks": []}

        with patch.object(provider.session, "get", return_value=mock_response) as mock_get:
            provider.fetch_ohlcv("KXINFL-25JAN", "2024-01-01", "2024-01-31")

            # Check that endpoint includes KXINFL series
            call_args = mock_get.call_args
            endpoint = call_args[0][0]
            assert "/series/KXINFL/markets/KXINFL-25JAN/candlesticks" in endpoint

    def test_fetch_ohlcv_uses_custom_series(self, provider):
        """Test that custom series_ticker is used when provided."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"candlesticks": []}

        with patch.object(provider.session, "get", return_value=mock_response) as mock_get:
            provider.fetch_ohlcv("KXINFL-25JAN", "2024-01-01", "2024-01-31", series_ticker="CUSTOM")

            # Check that custom series is used
            call_args = mock_get.call_args
            endpoint = call_args[0][0]
            assert "/series/CUSTOM/markets/KXINFL-25JAN/candlesticks" in endpoint

    def test_fetch_invalid_frequency_raises_error(self, provider):
        """Test that invalid frequency raises DataValidationError."""
        with pytest.raises(DataValidationError) as exc_info:
            provider._fetch_raw_data("KXINFL-25JAN", "2024-01-01", "2024-01-31", "invalid_freq")

        assert "Unsupported frequency" in str(exc_info.value)

    def test_fetch_market_not_found(self, provider):
        """Test handling of 404 response."""
        mock_response = MagicMock()
        mock_response.status_code = 404

        with patch.object(provider.session, "get", return_value=mock_response):
            with pytest.raises(SymbolNotFoundError):
                provider.fetch_ohlcv("INVALID-MARKET", "2024-01-01", "2024-01-31")


class TestKalshiListMethods:
    """Test list_markets and list_series methods."""

    @pytest.fixture
    def provider(self):
        """Create a Kalshi provider for testing."""
        provider = KalshiProvider()
        yield provider
        provider.close()

    def test_list_markets_success(self, provider):
        """Test successful list_markets with mocked response."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "markets": [
                {
                    "ticker": "KXINFL-25JAN",
                    "title": "Will January 2025 CPI exceed 3%?",
                    "status": "open",
                    "yes_bid": 0.45,
                    "yes_ask": 0.47,
                    "volume": 125000,
                },
                {
                    "ticker": "KXINFL-25FEB",
                    "title": "Will February 2025 CPI exceed 3%?",
                    "status": "open",
                    "yes_bid": 0.42,
                    "yes_ask": 0.44,
                    "volume": 80000,
                },
            ]
        }

        with patch.object(provider.session, "get", return_value=mock_response):
            markets = provider.list_markets(status="open")

        assert len(markets) == 2
        assert markets[0]["ticker"] == "KXINFL-25JAN"
        assert markets[1]["ticker"] == "KXINFL-25FEB"

    def test_list_markets_with_series_filter(self, provider):
        """Test list_markets with series_ticker filter."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"markets": []}

        with patch.object(provider.session, "get", return_value=mock_response) as mock_get:
            provider.list_markets(series_ticker="KXINFL")

            # Check series_ticker was passed
            call_kwargs = mock_get.call_args[1]
            params = call_kwargs["params"]
            assert params["series_ticker"] == "KXINFL"

    def test_list_series_success(self, provider):
        """Test successful list_series with mocked response."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "series": [
                {"ticker": "KXINFL", "title": "CPI Inflation"},
                {"ticker": "KXFED", "title": "Fed Funds Rate"},
                {"ticker": "KXGDP", "title": "GDP Growth"},
            ]
        }

        with patch.object(provider.session, "get", return_value=mock_response):
            series = provider.list_series()

        assert len(series) == 3
        assert series[0]["ticker"] == "KXINFL"
        assert series[1]["ticker"] == "KXFED"
        assert series[2]["ticker"] == "KXGDP"

    def test_get_market_metadata_success(self, provider):
        """Test successful get_market_metadata with mocked response."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "market": {
                "ticker": "KXINFL-25JAN",
                "title": "Will January 2025 CPI exceed 3%?",
                "status": "open",
                "yes_bid": 0.45,
                "yes_ask": 0.47,
                "last_price": 0.46,
                "volume": 125000,
                "open_interest": 50000,
            }
        }

        with patch.object(provider.session, "get", return_value=mock_response):
            metadata = provider.get_market_metadata("KXINFL-25JAN")

        assert metadata["ticker"] == "KXINFL-25JAN"
        assert metadata["status"] == "open"
        assert metadata["volume"] == 125000

    def test_get_market_metadata_not_found(self, provider):
        """Test get_market_metadata with 404 response."""
        mock_response = MagicMock()
        mock_response.status_code = 404

        with patch.object(provider.session, "get", return_value=mock_response):
            with pytest.raises(SymbolNotFoundError):
                provider.get_market_metadata("INVALID-MARKET")


class TestKalshiFetchMultipleMarkets:
    """Test fetch_multiple_markets method."""

    @pytest.fixture
    def provider(self):
        """Create a Kalshi provider for testing."""
        provider = KalshiProvider()
        yield provider
        provider.close()

    def test_fetch_multiple_long_format(self, provider):
        """Test fetch_multiple_markets in long format (align=False)."""
        ts1 = int(datetime(2024, 1, 2, 12, 0).timestamp())
        ts2 = int(datetime(2024, 1, 3, 12, 0).timestamp())

        market1_response = MagicMock()
        market1_response.status_code = 200
        market1_response.json.return_value = {
            "candlesticks": [
                {
                    "end_period_ts": ts1,
                    "open": 0.45,
                    "high": 0.48,
                    "low": 0.42,
                    "close": 0.47,
                    "volume": 1250,
                },
            ]
        }

        market2_response = MagicMock()
        market2_response.status_code = 200
        market2_response.json.return_value = {
            "candlesticks": [
                {
                    "end_period_ts": ts2,
                    "open": 0.50,
                    "high": 0.55,
                    "low": 0.48,
                    "close": 0.52,
                    "volume": 2100,
                },
            ]
        }

        with patch.object(
            provider.session, "get", side_effect=[market1_response, market2_response]
        ):
            df = provider.fetch_multiple_markets(
                ["KXINFL-25JAN", "KXFED-25JAN"],
                "2024-01-01",
                "2024-01-31",
                align=False,
            )

        assert len(df) == 2
        assert "symbol" in df.columns
        symbols = df["symbol"].to_list()
        assert "KXINFL-25JAN" in symbols
        assert "KXFED-25JAN" in symbols

    def test_fetch_multiple_wide_format(self, provider):
        """Test fetch_multiple_markets in wide format (align=True)."""
        ts = int(datetime(2024, 1, 2, 12, 0).timestamp())

        market1_response = MagicMock()
        market1_response.status_code = 200
        market1_response.json.return_value = {
            "candlesticks": [
                {
                    "end_period_ts": ts,
                    "open": 0.45,
                    "high": 0.48,
                    "low": 0.42,
                    "close": 0.47,
                    "volume": 1250,
                },
            ]
        }

        market2_response = MagicMock()
        market2_response.status_code = 200
        market2_response.json.return_value = {
            "candlesticks": [
                {
                    "end_period_ts": ts,
                    "open": 0.50,
                    "high": 0.55,
                    "low": 0.48,
                    "close": 0.52,
                    "volume": 2100,
                },
            ]
        }

        with patch.object(
            provider.session, "get", side_effect=[market1_response, market2_response]
        ):
            df = provider.fetch_multiple_markets(
                ["KXINFL-25JAN", "KXFED-25JAN"],
                "2024-01-01",
                "2024-01-31",
                align=True,
            )

        assert "KXINFL-25JAN_close" in df.columns
        assert "KXFED-25JAN_close" in df.columns
        assert len(df) == 1

    def test_fetch_multiple_empty_list_raises_error(self, provider):
        """Test that empty ticker list raises error."""
        with pytest.raises(DataValidationError) as exc_info:
            provider.fetch_multiple_markets([], "2024-01-01", "2024-01-31")

        assert "tickers cannot be empty" in str(exc_info.value)

    def test_fetch_multiple_all_failed_raises_error(self, provider):
        """Test that error is raised when all markets fail."""
        mock_response = MagicMock()
        mock_response.status_code = 404

        with patch.object(provider.session, "get", return_value=mock_response):
            with pytest.raises(DataNotAvailableError) as exc_info:
                provider.fetch_multiple_markets(
                    ["INVALID-1", "INVALID-2"],
                    "2024-01-01",
                    "2024-01-31",
                )

        # Check it's a DataNotAvailableError with both tickers mentioned
        assert "INVALID-1" in str(exc_info.value)
        assert "INVALID-2" in str(exc_info.value)


class TestKalshiHeaders:
    """Test HTTP header handling."""

    def test_headers_without_api_key(self):
        """Test headers when no API key is set."""
        provider = KalshiProvider()
        headers = provider._get_headers()

        assert "Accept" in headers
        assert headers["Accept"] == "application/json"
        assert "Authorization" not in headers

        provider.close()

    def test_headers_with_api_key(self):
        """Test headers when API key is set."""
        provider = KalshiProvider(api_key="test_api_key")
        headers = provider._get_headers()

        assert "Accept" in headers
        assert "Authorization" in headers
        assert headers["Authorization"] == "Bearer test_api_key"

        provider.close()
