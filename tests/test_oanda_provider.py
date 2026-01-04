"""Unit tests for OANDA provider module."""

from unittest.mock import MagicMock, patch

import polars as pl
import pytest

from ml4t.data.core.exceptions import (
    AuthenticationError,
    NetworkError,
    RateLimitError,
    SymbolNotFoundError,
)
from ml4t.data.providers.oanda import OandaProvider


class TestOandaProviderInit:
    """Tests for provider initialization."""

    def test_init_with_api_key(self):
        """Test initialization with API key."""
        with patch("ml4t.data.providers.oanda.oandapyV20.API") as mock_api:
            mock_api.return_value = MagicMock()
            provider = OandaProvider(api_key="test_key")

            assert provider.name == "oanda"
            assert provider.api_key == "test_key"
            assert provider.practice is True
            assert provider.base_url == "https://api-fxpractice.oanda.com"

    def test_init_live_account(self):
        """Test initialization with live account."""
        with patch("ml4t.data.providers.oanda.oandapyV20.API") as mock_api:
            mock_api.return_value = MagicMock()
            provider = OandaProvider(api_key="test_key", practice=False)

            assert provider.practice is False
            assert provider.base_url == "https://api-fxtrade.oanda.com"

    def test_init_with_env_api_key(self):
        """Test initialization with API key from environment."""
        with patch("ml4t.data.providers.oanda.oandapyV20.API") as mock_api:
            mock_api.return_value = MagicMock()
            with patch.dict("os.environ", {"OANDA_API_KEY": "env_key"}):
                provider = OandaProvider()

                assert provider.api_key == "env_key"

    def test_init_without_api_key_raises_error(self):
        """Test initialization without API key raises AuthenticationError."""
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(AuthenticationError, match="API key required"):
                OandaProvider()

    def test_init_custom_rate_limit(self):
        """Test initialization with custom rate limit."""
        with patch("ml4t.data.providers.oanda.oandapyV20.API") as mock_api:
            mock_api.return_value = MagicMock()
            provider = OandaProvider(api_key="test_key", rate_limit=(50, 1.0))

            assert provider is not None

    def test_init_client_failure(self):
        """Test initialization failure when client raises error."""
        with patch("ml4t.data.providers.oanda.oandapyV20.API") as mock_api:
            mock_api.side_effect = Exception("Connection failed")

            with pytest.raises(AuthenticationError, match="Failed to initialize"):
                OandaProvider(api_key="test_key")


class TestNameProperty:
    """Tests for name property."""

    def test_name_returns_oanda(self):
        """Test name property returns correct value."""
        with patch("ml4t.data.providers.oanda.oandapyV20.API") as mock_api:
            mock_api.return_value = MagicMock()
            provider = OandaProvider(api_key="test_key")
            assert provider.name == "oanda"


class TestValidatePair:
    """Tests for _validate_pair method."""

    @pytest.fixture
    def provider(self):
        """Create provider instance."""
        with patch("ml4t.data.providers.oanda.oandapyV20.API") as mock_api:
            mock_api.return_value = MagicMock()
            return OandaProvider(api_key="test_key")

    def test_validate_eurusd_format(self, provider):
        """Test EURUSD format (6 chars, no separator)."""
        result = provider._validate_pair("EURUSD")
        assert result == "EUR_USD"

    def test_validate_eur_usd_format(self, provider):
        """Test EUR_USD format (underscore separator)."""
        result = provider._validate_pair("EUR_USD")
        assert result == "EUR_USD"

    def test_validate_eur_slash_usd_format(self, provider):
        """Test EUR/USD format (forward slash separator)."""
        result = provider._validate_pair("EUR/USD")
        assert result == "EUR_USD"

    def test_validate_lowercase(self, provider):
        """Test lowercase input is uppercased."""
        result = provider._validate_pair("eurusd")
        assert result == "EUR_USD"

    def test_validate_mixed_case(self, provider):
        """Test mixed case input is uppercased."""
        result = provider._validate_pair("EuRuSd")
        assert result == "EUR_USD"

    def test_validate_invalid_format_short(self, provider):
        """Test invalid short symbol raises ValueError."""
        with pytest.raises(ValueError, match="Invalid currency pair format"):
            provider._validate_pair("EUR")

    def test_validate_invalid_slash_format(self, provider):
        """Test invalid slash format raises ValueError."""
        with pytest.raises(ValueError, match="Invalid currency pair format"):
            provider._validate_pair("EU/RUSD")

    def test_validate_various_pairs(self, provider):
        """Test various valid currency pairs."""
        pairs = [
            ("GBPUSD", "GBP_USD"),
            ("USD_JPY", "USD_JPY"),
            ("AUD/CAD", "AUD_CAD"),
            ("nzdjpy", "NZD_JPY"),
        ]
        for input_pair, expected in pairs:
            result = provider._validate_pair(input_pair)
            assert result == expected


class TestMapFrequencyToGranularity:
    """Tests for _map_frequency_to_granularity method."""

    @pytest.fixture
    def provider(self):
        """Create provider instance."""
        with patch("ml4t.data.providers.oanda.oandapyV20.API") as mock_api:
            mock_api.return_value = MagicMock()
            return OandaProvider(api_key="test_key")

    def test_daily_frequency(self, provider):
        """Test daily frequency mapping."""
        assert provider._map_frequency_to_granularity("daily") == "D"
        assert provider._map_frequency_to_granularity("day") == "D"
        assert provider._map_frequency_to_granularity("1d") == "D"

    def test_hourly_frequency(self, provider):
        """Test hourly frequency mapping."""
        assert provider._map_frequency_to_granularity("hourly") == "H1"
        assert provider._map_frequency_to_granularity("hour") == "H1"
        assert provider._map_frequency_to_granularity("1h") == "H1"

    def test_minute_frequency(self, provider):
        """Test minute frequency mapping."""
        assert provider._map_frequency_to_granularity("minute") == "M1"
        assert provider._map_frequency_to_granularity("min") == "M1"
        assert provider._map_frequency_to_granularity("1m") == "M1"

    def test_multi_minute_frequencies(self, provider):
        """Test multi-minute frequency mappings."""
        assert provider._map_frequency_to_granularity("5min") == "M5"
        assert provider._map_frequency_to_granularity("5m") == "M5"
        assert provider._map_frequency_to_granularity("15min") == "M15"
        assert provider._map_frequency_to_granularity("30min") == "M30"

    def test_multi_hour_frequency(self, provider):
        """Test multi-hour frequency mapping."""
        assert provider._map_frequency_to_granularity("4hour") == "H4"
        assert provider._map_frequency_to_granularity("4h") == "H4"

    def test_weekly_frequency(self, provider):
        """Test weekly frequency mapping."""
        assert provider._map_frequency_to_granularity("weekly") == "W"
        assert provider._map_frequency_to_granularity("week") == "W"
        assert provider._map_frequency_to_granularity("1w") == "W"

    def test_oanda_native_format(self, provider):
        """Test OANDA native granularity format."""
        assert provider._map_frequency_to_granularity("M1") == "M1"
        assert provider._map_frequency_to_granularity("H4") == "H4"
        assert provider._map_frequency_to_granularity("D") == "D"

    def test_unknown_frequency_defaults_to_daily(self, provider):
        """Test unknown frequency defaults to daily."""
        result = provider._map_frequency_to_granularity("unknown")
        assert result == "D"


class TestTimeframes:
    """Tests for TIMEFRAMES constant."""

    def test_timeframes_includes_seconds(self):
        """Test TIMEFRAMES includes second intervals."""
        assert "S5" in OandaProvider.TIMEFRAMES
        assert "S10" in OandaProvider.TIMEFRAMES
        assert "S15" in OandaProvider.TIMEFRAMES
        assert "S30" in OandaProvider.TIMEFRAMES

    def test_timeframes_includes_minutes(self):
        """Test TIMEFRAMES includes minute intervals."""
        assert "M1" in OandaProvider.TIMEFRAMES
        assert "M5" in OandaProvider.TIMEFRAMES
        assert "M15" in OandaProvider.TIMEFRAMES
        assert "M30" in OandaProvider.TIMEFRAMES

    def test_timeframes_includes_hours(self):
        """Test TIMEFRAMES includes hour intervals."""
        assert "H1" in OandaProvider.TIMEFRAMES
        assert "H4" in OandaProvider.TIMEFRAMES
        assert "H12" in OandaProvider.TIMEFRAMES

    def test_timeframes_includes_day_week_month(self):
        """Test TIMEFRAMES includes D, W, M."""
        assert "D" in OandaProvider.TIMEFRAMES
        assert "W" in OandaProvider.TIMEFRAMES
        assert "M" in OandaProvider.TIMEFRAMES


class TestDefaultRateLimit:
    """Tests for DEFAULT_RATE_LIMIT constant."""

    def test_default_rate_limit_value(self):
        """Test DEFAULT_RATE_LIMIT has expected value."""
        # OANDA allows up to 100 requests per second
        assert OandaProvider.DEFAULT_RATE_LIMIT == (100, 1.0)


class TestCreateEmptyDataframe:
    """Tests for _create_empty_dataframe method."""

    @pytest.fixture
    def provider(self):
        """Create provider instance."""
        with patch("ml4t.data.providers.oanda.oandapyV20.API") as mock_api:
            mock_api.return_value = MagicMock()
            return OandaProvider(api_key="test_key")

    def test_empty_dataframe_columns(self, provider):
        """Test empty DataFrame has correct columns."""
        df = provider._create_empty_dataframe()
        expected_columns = ["timestamp", "symbol", "open", "high", "low", "close", "volume"]
        assert df.columns == expected_columns

    def test_empty_dataframe_length(self, provider):
        """Test empty DataFrame has zero rows."""
        df = provider._create_empty_dataframe()
        assert len(df) == 0

    def test_empty_dataframe_dtypes(self, provider):
        """Test empty DataFrame has correct data types."""
        df = provider._create_empty_dataframe()
        assert df.schema["timestamp"] == pl.Datetime
        assert df.schema["open"] == pl.Float64
        assert df.schema["close"] == pl.Float64
        assert df.schema["volume"] == pl.Float64


class TestTransformData:
    """Tests for _transform_data method."""

    @pytest.fixture
    def provider(self):
        """Create provider instance."""
        with patch("ml4t.data.providers.oanda.oandapyV20.API") as mock_api:
            mock_api.return_value = MagicMock()
            return OandaProvider(api_key="test_key")

    def test_transform_empty_data(self, provider):
        """Test transforming empty data returns empty DataFrame."""
        result = provider._transform_data([], "EUR_USD")
        assert len(result) == 0
        assert "timestamp" in result.columns

    def test_transform_data_success(self, provider):
        """Test successful data transformation."""
        raw_data = [
            {
                "time": "2024-01-02T00:00:00.000000000Z",
                "mid": {"o": "1.1000", "h": "1.1050", "l": "1.0950", "c": "1.1025"},
                "volume": 10000,
                "complete": True,
            },
            {
                "time": "2024-01-03T00:00:00.000000000Z",
                "mid": {"o": "1.1025", "h": "1.1100", "l": "1.1000", "c": "1.1075"},
                "volume": 11000,
                "complete": True,
            },
        ]

        df = provider._transform_data(raw_data, "EUR_USD")

        assert len(df) == 2
        assert "timestamp" in df.columns
        assert "open" in df.columns
        assert "close" in df.columns
        assert "symbol" in df.columns
        assert df["symbol"][0] == "EUR_USD"
        assert df["open"][0] == 1.1000
        assert df["close"][1] == 1.1075

    def test_transform_data_deduplication(self, provider):
        """Test duplicate timestamps are removed."""
        raw_data = [
            {
                "time": "2024-01-02T00:00:00.000000000Z",
                "mid": {"o": "1.1000", "h": "1.1050", "l": "1.0950", "c": "1.1025"},
                "volume": 10000,
            },
            {
                "time": "2024-01-02T00:00:00.000000000Z",  # Duplicate
                "mid": {"o": "1.1001", "h": "1.1051", "l": "1.0951", "c": "1.1026"},
                "volume": 10001,
            },
        ]

        df = provider._transform_data(raw_data, "EUR_USD")

        assert len(df) == 1  # Duplicates removed

    def test_transform_data_sorting(self, provider):
        """Test data is sorted by timestamp."""
        raw_data = [
            {
                "time": "2024-01-03T00:00:00.000000000Z",  # Later date first
                "mid": {"o": "1.1025", "h": "1.1100", "l": "1.1000", "c": "1.1075"},
                "volume": 11000,
            },
            {
                "time": "2024-01-02T00:00:00.000000000Z",
                "mid": {"o": "1.1000", "h": "1.1050", "l": "1.0950", "c": "1.1025"},
                "volume": 10000,
            },
        ]

        df = provider._transform_data(raw_data, "EUR_USD")

        # Should be sorted chronologically
        assert df["timestamp"][0] < df["timestamp"][1]


class TestFetchRawData:
    """Tests for _fetch_raw_data method with mocked API."""

    @pytest.fixture
    def provider(self):
        """Create provider instance with mocked client."""
        with patch("ml4t.data.providers.oanda.oandapyV20.API") as mock_api:
            mock_client = MagicMock()
            mock_api.return_value = mock_client
            provider = OandaProvider(api_key="test_key")
            provider.client = mock_client
            return provider

    def test_fetch_raw_data_auth_error(self, provider):
        """Test 401 raises AuthenticationError."""
        from oandapyV20.exceptions import V20Error

        provider.client.request.side_effect = V20Error(401, "unauthorized")

        with pytest.raises(AuthenticationError, match="Authentication failed"):
            provider._fetch_raw_data("EUR_USD", "2024-01-01", "2024-01-02", "daily")

    def test_fetch_raw_data_rate_limit(self, provider):
        """Test 429 raises RateLimitError."""
        from oandapyV20.exceptions import V20Error

        provider.client.request.side_effect = V20Error(429, "rate limit exceeded")

        with pytest.raises(RateLimitError):
            provider._fetch_raw_data("EUR_USD", "2024-01-01", "2024-01-02", "daily")

    def test_fetch_raw_data_not_found(self, provider):
        """Test 404 raises SymbolNotFoundError."""
        from oandapyV20.exceptions import V20Error

        # Error message must contain "404" for the code to detect it
        provider.client.request.side_effect = V20Error(404, "404 - instrument not found")

        with pytest.raises(SymbolNotFoundError):
            provider._fetch_raw_data("INVALID_PAIR", "2024-01-01", "2024-01-02", "daily")

    def test_fetch_raw_data_invalid_instrument(self, provider):
        """Test invalid instrument error raises SymbolNotFoundError."""
        from oandapyV20.exceptions import V20Error

        provider.client.request.side_effect = V20Error(
            400, "invalid value specified for 'instrument'"
        )

        with pytest.raises(SymbolNotFoundError):
            provider._fetch_raw_data("BAD_PAIR", "2024-01-01", "2024-01-02", "daily")

    def test_fetch_raw_data_network_error(self, provider):
        """Test generic V20Error raises NetworkError."""
        from oandapyV20.exceptions import V20Error

        provider.client.request.side_effect = V20Error(500, "internal server error")

        with pytest.raises(NetworkError, match="API error"):
            provider._fetch_raw_data("EUR_USD", "2024-01-01", "2024-01-02", "daily")

    def test_fetch_raw_data_generic_exception(self, provider):
        """Test generic exception raises NetworkError."""
        provider.client.request.side_effect = Exception("Connection timeout")

        with pytest.raises(NetworkError, match="Failed to fetch data"):
            provider._fetch_raw_data("EUR_USD", "2024-01-01", "2024-01-02", "daily")


class TestIntegration:
    """Integration tests for OandaProvider."""

    def test_full_workflow_with_mocks(self):
        """Test complete fetch workflow with mocked responses."""
        with patch("ml4t.data.providers.oanda.oandapyV20.API") as mock_api:
            mock_client = MagicMock()
            mock_api.return_value = mock_client

            # Mock response
            mock_request = MagicMock()
            mock_request.response = {
                "candles": [
                    {
                        "time": "2024-01-02T00:00:00.000000000Z",
                        "mid": {"o": "1.1000", "h": "1.1050", "l": "1.0950", "c": "1.1025"},
                        "volume": 10000,
                        "complete": True,
                    }
                ]
            }

            with patch(
                "ml4t.data.providers.oanda.instruments.InstrumentsCandles",
                return_value=mock_request,
            ):
                provider = OandaProvider(api_key="test_key")

                raw_data = provider._fetch_raw_data("EUR_USD", "2024-01-01", "2024-01-02", "daily")
                df = provider._transform_data(raw_data, "EUR_USD")

                assert len(df) == 1
                assert df["symbol"][0] == "EUR_USD"
