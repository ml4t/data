"""Tests for Finnhub provider module."""

from unittest.mock import MagicMock, patch

import pytest

from ml4t.data.core.exceptions import (
    AuthenticationError,
    DataNotAvailableError,
    DataValidationError,
    NetworkError,
    ProviderError,
    RateLimitError,
    SymbolNotFoundError,
)
from ml4t.data.providers.finnhub import FinnhubProvider


class TestFinnhubProviderInit:
    """Tests for provider initialization."""

    def test_init_with_api_key(self):
        """Test initialization with API key."""
        provider = FinnhubProvider(api_key="test_key")

        assert provider.name == "finnhub"
        assert provider.api_key == "test_key"
        assert provider.base_url == "https://finnhub.io/api/v1"

    def test_init_with_env_api_key(self):
        """Test initialization with API key from environment."""
        with patch.dict("os.environ", {"FINNHUB_API_KEY": "env_key"}):
            provider = FinnhubProvider()

            assert provider.api_key == "env_key"

    def test_init_without_api_key_raises_error(self):
        """Test initialization without API key raises AuthenticationError."""
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(AuthenticationError, match="API key required"):
                FinnhubProvider()

    def test_init_custom_rate_limit(self):
        """Test initialization with custom rate limit."""
        provider = FinnhubProvider(api_key="test_key", rate_limit=(100, 60.0))

        assert provider is not None


class TestNameProperty:
    """Tests for name property."""

    def test_name_returns_finnhub(self):
        """Test name property returns correct value."""
        provider = FinnhubProvider(api_key="test_key")
        assert provider.name == "finnhub"


class TestFetchRawData:
    """Tests for _fetch_raw_data method."""

    @pytest.fixture
    def provider(self):
        """Create provider instance."""
        return FinnhubProvider(api_key="test_key")

    def test_fetch_raw_data_success(self, provider):
        """Test successful raw data fetch."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "s": "ok",
            "c": [171.0, 172.0],
            "h": [172.0, 173.0],
            "l": [169.0, 170.0],
            "o": [170.0, 171.0],
            "t": [1704067200, 1704153600],
            "v": [1000000, 1100000],
        }

        with patch.object(provider.session, "get", return_value=mock_response):
            data = provider._fetch_raw_data("AAPL", "2024-01-01", "2024-01-02", "daily")

        assert data["s"] == "ok"
        assert len(data["c"]) == 2

    def test_fetch_raw_data_invalid_frequency(self, provider):
        """Test invalid frequency raises DataValidationError."""
        with pytest.raises(DataValidationError, match="Unsupported frequency"):
            provider._fetch_raw_data("AAPL", "2024-01-01", "2024-01-02", "invalid")

    def test_fetch_raw_data_invalid_date_format(self, provider):
        """Test invalid date format raises DataValidationError."""
        with pytest.raises(DataValidationError, match="Invalid date format"):
            provider._fetch_raw_data("AAPL", "not-a-date", "2024-01-02", "daily")

    def test_fetch_raw_data_auth_error(self, provider):
        """Test 401 raises AuthenticationError."""
        mock_response = MagicMock()
        mock_response.status_code = 401

        with patch.object(provider.session, "get", return_value=mock_response):
            with pytest.raises(AuthenticationError, match="Invalid API key"):
                provider._fetch_raw_data("AAPL", "2024-01-01", "2024-01-02", "daily")

    def test_fetch_raw_data_forbidden(self, provider):
        """Test 403 raises AuthenticationError."""
        mock_response = MagicMock()
        mock_response.status_code = 403

        with patch.object(provider.session, "get", return_value=mock_response):
            with pytest.raises(AuthenticationError):
                provider._fetch_raw_data("AAPL", "2024-01-01", "2024-01-02", "daily")

    def test_fetch_raw_data_rate_limit(self, provider):
        """Test 429 raises RateLimitError."""
        mock_response = MagicMock()
        mock_response.status_code = 429

        with patch.object(provider.session, "get", return_value=mock_response):
            with pytest.raises(RateLimitError):
                provider._fetch_raw_data("AAPL", "2024-01-01", "2024-01-02", "daily")

    def test_fetch_raw_data_not_found(self, provider):
        """Test 404 raises DataNotAvailableError."""
        mock_response = MagicMock()
        mock_response.status_code = 404

        with patch.object(provider.session, "get", return_value=mock_response):
            with pytest.raises(DataNotAvailableError):
                provider._fetch_raw_data("INVALID", "2024-01-01", "2024-01-02", "daily")

    def test_fetch_raw_data_other_http_error(self, provider):
        """Test other HTTP errors raise NetworkError."""
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"

        with patch.object(provider.session, "get", return_value=mock_response):
            with pytest.raises(NetworkError, match="HTTP 500"):
                provider._fetch_raw_data("AAPL", "2024-01-01", "2024-01-02", "daily")

    def test_fetch_raw_data_json_parse_error(self, provider):
        """Test JSON parse error raises NetworkError."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.side_effect = ValueError("Invalid JSON")

        with patch.object(provider.session, "get", return_value=mock_response):
            with pytest.raises(NetworkError, match="Failed to parse JSON"):
                provider._fetch_raw_data("AAPL", "2024-01-01", "2024-01-02", "daily")

    def test_fetch_raw_data_no_data_status(self, provider):
        """Test 'no_data' status raises SymbolNotFoundError."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"s": "no_data"}

        with patch.object(provider.session, "get", return_value=mock_response):
            with pytest.raises(SymbolNotFoundError):
                provider._fetch_raw_data("UNKNOWN", "2024-01-01", "2024-01-02", "daily")

    def test_fetch_raw_data_error_status(self, provider):
        """Test 'error' status raises ProviderError."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"s": "error", "error": "Some API error"}

        with patch.object(provider.session, "get", return_value=mock_response):
            with pytest.raises(ProviderError, match="API error"):
                provider._fetch_raw_data("AAPL", "2024-01-01", "2024-01-02", "daily")

    def test_fetch_raw_data_missing_data_arrays(self, provider):
        """Test missing data arrays raise SymbolNotFoundError."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"s": "ok", "c": None, "t": None}

        with patch.object(provider.session, "get", return_value=mock_response):
            with pytest.raises(SymbolNotFoundError):
                provider._fetch_raw_data("AAPL", "2024-01-01", "2024-01-02", "daily")


class TestTransformData:
    """Tests for _transform_data method."""

    @pytest.fixture
    def provider(self):
        """Create provider instance."""
        return FinnhubProvider(api_key="test_key")

    def test_transform_data_success(self, provider):
        """Test successful data transformation."""
        raw_data = {
            "s": "ok",
            "c": [171.0, 172.0],
            "h": [172.0, 173.0],
            "l": [169.0, 170.0],
            "o": [170.0, 171.0],
            "t": [1704067200, 1704153600],
            "v": [1000000, 1100000],
        }

        df = provider._transform_data(raw_data, "AAPL")

        assert len(df) == 2
        assert "timestamp" in df.columns
        assert "open" in df.columns
        assert "high" in df.columns
        assert "low" in df.columns
        assert "close" in df.columns
        assert "volume" in df.columns
        assert "symbol" in df.columns
        assert df["symbol"][0] == "AAPL"

    def test_transform_data_uppercase_symbol(self, provider):
        """Test symbol is uppercased."""
        raw_data = {
            "c": [171.0],
            "h": [172.0],
            "l": [169.0],
            "o": [170.0],
            "t": [1704067200],
            "v": [1000000],
        }

        df = provider._transform_data(raw_data, "aapl")

        assert df["symbol"][0] == "AAPL"

    def test_transform_data_correct_column_order(self, provider):
        """Test columns are in correct order."""
        raw_data = {
            "c": [171.0],
            "h": [172.0],
            "l": [169.0],
            "o": [170.0],
            "t": [1704067200],
            "v": [1000000],
        }

        df = provider._transform_data(raw_data, "AAPL")

        expected_order = ["timestamp", "symbol", "open", "high", "low", "close", "volume"]
        assert df.columns == expected_order

    def test_transform_data_sorts_by_timestamp(self, provider):
        """Test data is sorted by timestamp."""
        raw_data = {
            "c": [172.0, 171.0],
            "h": [173.0, 172.0],
            "l": [170.0, 169.0],
            "o": [171.0, 170.0],
            "t": [1704153600, 1704067200],  # Out of order
            "v": [1100000, 1000000],
        }

        df = provider._transform_data(raw_data, "AAPL")

        assert df["timestamp"][0] < df["timestamp"][1]


class TestFrequencyMapping:
    """Tests for frequency mapping."""

    @pytest.fixture
    def provider(self):
        """Create provider instance."""
        return FinnhubProvider(api_key="test_key")

    def _create_success_response(self):
        """Create a mock success response."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "s": "ok",
            "c": [171.0],
            "h": [172.0],
            "l": [169.0],
            "o": [170.0],
            "t": [1704067200],
            "v": [1000000],
        }
        return mock_response

    def test_daily_frequency(self, provider):
        """Test daily frequency mapping."""
        with patch.object(provider.session, "get", return_value=self._create_success_response()):
            data = provider._fetch_raw_data("AAPL", "2024-01-01", "2024-01-02", "daily")
            assert data["s"] == "ok"

    def test_1d_frequency(self, provider):
        """Test 1d frequency alias."""
        with patch.object(provider.session, "get", return_value=self._create_success_response()):
            data = provider._fetch_raw_data("AAPL", "2024-01-01", "2024-01-02", "1d")
            assert data["s"] == "ok"

    def test_hourly_frequency(self, provider):
        """Test hourly frequency mapping."""
        with patch.object(provider.session, "get", return_value=self._create_success_response()):
            data = provider._fetch_raw_data("AAPL", "2024-01-01", "2024-01-02", "hourly")
            assert data["s"] == "ok"

    def test_weekly_frequency(self, provider):
        """Test weekly frequency mapping."""
        with patch.object(provider.session, "get", return_value=self._create_success_response()):
            data = provider._fetch_raw_data("AAPL", "2024-01-01", "2024-01-15", "weekly")
            assert data["s"] == "ok"

    def test_monthly_frequency(self, provider):
        """Test monthly frequency mapping."""
        with patch.object(provider.session, "get", return_value=self._create_success_response()):
            data = provider._fetch_raw_data("AAPL", "2024-01-01", "2024-03-01", "monthly")
            assert data["s"] == "ok"

    def test_minute_frequency(self, provider):
        """Test minute frequency mapping."""
        with patch.object(provider.session, "get", return_value=self._create_success_response()):
            data = provider._fetch_raw_data("AAPL", "2024-01-01", "2024-01-01", "1min")
            assert data["s"] == "ok"


class TestDefaultRateLimit:
    """Tests for DEFAULT_RATE_LIMIT constant."""

    def test_default_rate_limit_value(self):
        """Test DEFAULT_RATE_LIMIT has expected value."""
        # Free tier: 1 request per second
        assert FinnhubProvider.DEFAULT_RATE_LIMIT == (1, 1.0)


class TestResolutionMap:
    """Tests for RESOLUTION_MAP constant."""

    def test_resolution_map_has_daily(self):
        """Test RESOLUTION_MAP contains daily variants."""
        assert "daily" in FinnhubProvider.RESOLUTION_MAP
        assert "1d" in FinnhubProvider.RESOLUTION_MAP
        assert "day" in FinnhubProvider.RESOLUTION_MAP
        assert FinnhubProvider.RESOLUTION_MAP["daily"] == "D"

    def test_resolution_map_has_weekly(self):
        """Test RESOLUTION_MAP contains weekly variants."""
        assert "weekly" in FinnhubProvider.RESOLUTION_MAP
        assert "1w" in FinnhubProvider.RESOLUTION_MAP
        assert FinnhubProvider.RESOLUTION_MAP["weekly"] == "W"

    def test_resolution_map_has_monthly(self):
        """Test RESOLUTION_MAP contains monthly variants."""
        assert "monthly" in FinnhubProvider.RESOLUTION_MAP
        assert "1M" in FinnhubProvider.RESOLUTION_MAP
        assert FinnhubProvider.RESOLUTION_MAP["monthly"] == "M"

    def test_resolution_map_has_minute(self):
        """Test RESOLUTION_MAP contains minute variants."""
        assert "1min" in FinnhubProvider.RESOLUTION_MAP
        assert "5min" in FinnhubProvider.RESOLUTION_MAP
        assert "15min" in FinnhubProvider.RESOLUTION_MAP
        assert "30min" in FinnhubProvider.RESOLUTION_MAP

    def test_resolution_map_has_hourly(self):
        """Test RESOLUTION_MAP contains hourly variants."""
        assert "hourly" in FinnhubProvider.RESOLUTION_MAP
        assert "1h" in FinnhubProvider.RESOLUTION_MAP
        assert "60min" in FinnhubProvider.RESOLUTION_MAP


class TestIntegration:
    """Integration tests for FinnhubProvider."""

    def test_full_fetch_workflow_with_mocks(self):
        """Test complete fetch workflow with mocked responses."""
        provider = FinnhubProvider(api_key="test_key")

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "s": "ok",
            "c": [171.0, 172.0],
            "h": [172.0, 173.0],
            "l": [169.0, 170.0],
            "o": [170.0, 171.0],
            "t": [1704067200, 1704153600],
            "v": [1000000, 1100000],
        }

        with patch.object(provider.session, "get", return_value=mock_response):
            raw_data = provider._fetch_raw_data("AAPL", "2024-01-01", "2024-01-02", "daily")
            df = provider._transform_data(raw_data, "AAPL")

        assert len(df) == 2
        assert "timestamp" in df.columns
        assert "close" in df.columns
        assert "symbol" in df.columns
        assert df["symbol"][0] == "AAPL"

    def test_api_parameters(self):
        """Test API parameters are correctly set."""
        provider = FinnhubProvider(api_key="test_key")

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"s": "no_data"}

        with patch.object(provider.session, "get", return_value=mock_response) as mock_get:
            try:
                provider._fetch_raw_data("AAPL", "2024-01-01", "2024-01-02", "daily")
            except SymbolNotFoundError:
                pass  # Expected for no_data

            # Check that API was called
            call_args = mock_get.call_args
            params = call_args.kwargs.get("params", call_args[1].get("params", {}))
            assert params.get("symbol") == "AAPL"
            assert params.get("resolution") == "D"
            assert params.get("token") == "test_key"
