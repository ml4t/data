"""Tests for Twelve Data provider module."""

from unittest.mock import MagicMock, patch

import pytest

from ml4t.data.core.exceptions import (
    AuthenticationError,
    NetworkError,
    ProviderError,
    RateLimitError,
    SymbolNotFoundError,
)
from ml4t.data.providers.twelve_data import TwelveDataProvider


class TestTwelveDataProviderInit:
    """Tests for provider initialization."""

    def test_init_with_api_key(self):
        """Test initialization with API key."""
        provider = TwelveDataProvider(api_key="test_key")

        assert provider.name == "twelve_data"
        assert provider.api_key == "test_key"
        assert provider.base_url == "https://api.twelvedata.com"

    def test_init_with_env_api_key(self):
        """Test initialization with API key from environment."""
        with patch.dict("os.environ", {"TWELVE_DATA_API_KEY": "env_key"}):
            provider = TwelveDataProvider()

            assert provider.api_key == "env_key"

    def test_init_without_api_key_raises_error(self):
        """Test initialization without API key raises AuthenticationError."""
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(AuthenticationError, match="API key required"):
                TwelveDataProvider()

    def test_init_custom_rate_limit(self):
        """Test initialization with custom rate limit."""
        provider = TwelveDataProvider(api_key="test_key", rate_limit=(100, 60.0))

        assert provider is not None


class TestNameProperty:
    """Tests for name property."""

    def test_name_returns_twelve_data(self):
        """Test name property returns correct value."""
        provider = TwelveDataProvider(api_key="test_key")
        assert provider.name == "twelve_data"


class TestFetchRawData:
    """Tests for _fetch_raw_data method."""

    @pytest.fixture
    def provider(self):
        """Create provider instance."""
        return TwelveDataProvider(api_key="test_key")

    def test_fetch_raw_data_success(self, provider):
        """Test successful raw data fetch."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "meta": {"symbol": "AAPL"},
            "values": [
                {
                    "datetime": "2024-01-02",
                    "open": "170.0",
                    "high": "172.0",
                    "low": "169.0",
                    "close": "171.0",
                    "volume": "1000000",
                },
            ],
        }

        with patch.object(provider.session, "get", return_value=mock_response):
            data = provider._fetch_raw_data("AAPL", "2024-01-01", "2024-01-02", "daily")

        assert "values" in data
        assert len(data["values"]) == 1

    def test_fetch_raw_data_rate_limit(self, provider):
        """Test 429 raises RateLimitError."""
        mock_response = MagicMock()
        mock_response.status_code = 429

        with patch.object(provider.session, "get", return_value=mock_response):
            with pytest.raises(RateLimitError):
                provider._fetch_raw_data("AAPL", "2024-01-01", "2024-01-02", "daily")

    def test_fetch_raw_data_auth_error(self, provider):
        """Test 401 raises NetworkError (provider doesn't distinguish auth errors)."""
        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.text = "Unauthorized"

        with patch.object(provider.session, "get", return_value=mock_response):
            with pytest.raises(NetworkError, match="HTTP 401"):
                provider._fetch_raw_data("AAPL", "2024-01-01", "2024-01-02", "daily")

    def test_fetch_raw_data_http_error(self, provider):
        """Test other HTTP errors raise NetworkError."""
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"

        with patch.object(provider.session, "get", return_value=mock_response):
            with pytest.raises(NetworkError):
                provider._fetch_raw_data("AAPL", "2024-01-01", "2024-01-02", "daily")

    def test_fetch_raw_data_api_error(self, provider):
        """Test API error status raises ProviderError."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "status": "error",
            "message": "API error message",
        }

        with patch.object(provider.session, "get", return_value=mock_response):
            with pytest.raises(ProviderError):
                provider._fetch_raw_data("AAPL", "2024-01-01", "2024-01-02", "daily")

    def test_fetch_raw_data_symbol_not_found(self, provider):
        """Test empty values raises SymbolNotFoundError."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"values": []}

        with patch.object(provider.session, "get", return_value=mock_response):
            with pytest.raises(SymbolNotFoundError):
                provider._fetch_raw_data("INVALID", "2024-01-01", "2024-01-02", "daily")

    def test_fetch_raw_data_json_parse_error(self, provider):
        """Test JSON parse error raises ValueError (not caught by provider)."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.side_effect = ValueError("Invalid JSON")

        with patch.object(provider.session, "get", return_value=mock_response):
            with pytest.raises(ValueError, match="Invalid JSON"):
                provider._fetch_raw_data("AAPL", "2024-01-01", "2024-01-02", "daily")


class TestTransformData:
    """Tests for _transform_data method."""

    @pytest.fixture
    def provider(self):
        """Create provider instance."""
        return TwelveDataProvider(api_key="test_key")

    def test_transform_data_success(self, provider):
        """Test successful data transformation."""
        raw_data = {
            "values": [
                {
                    "datetime": "2024-01-02",
                    "open": "170.0",
                    "high": "172.0",
                    "low": "169.0",
                    "close": "171.0",
                    "volume": "1000000",
                },
                {
                    "datetime": "2024-01-03",
                    "open": "171.0",
                    "high": "173.0",
                    "low": "170.0",
                    "close": "172.0",
                    "volume": "1100000",
                },
            ]
        }

        df = provider._transform_data(raw_data, "AAPL")

        assert len(df) == 2
        assert "timestamp" in df.columns
        assert "open" in df.columns
        assert "close" in df.columns
        assert "symbol" in df.columns
        assert df["symbol"][0] == "AAPL"

    def test_transform_data_uppercase_symbol(self, provider):
        """Test symbol is uppercased."""
        raw_data = {
            "values": [
                {
                    "datetime": "2024-01-02",
                    "open": "170.0",
                    "high": "172.0",
                    "low": "169.0",
                    "close": "171.0",
                    "volume": "1000000",
                },
            ]
        }

        df = provider._transform_data(raw_data, "aapl")

        assert df["symbol"][0] == "AAPL"


class TestDefaultRateLimit:
    """Tests for DEFAULT_RATE_LIMIT constant."""

    def test_default_rate_limit_value(self):
        """Test DEFAULT_RATE_LIMIT has expected value."""
        # Free tier: 8 requests per minute
        assert TwelveDataProvider.DEFAULT_RATE_LIMIT == (8, 60.0)


class TestIntegration:
    """Integration tests for TwelveDataProvider."""

    def test_full_fetch_workflow_with_mocks(self):
        """Test complete fetch workflow with mocked responses."""
        provider = TwelveDataProvider(api_key="test_key")

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "meta": {"symbol": "AAPL"},
            "values": [
                {
                    "datetime": "2024-01-02",
                    "open": "170.0",
                    "high": "172.0",
                    "low": "169.0",
                    "close": "171.0",
                    "volume": "1000000",
                },
            ],
        }

        with patch.object(provider.session, "get", return_value=mock_response):
            raw_data = provider._fetch_raw_data("AAPL", "2024-01-01", "2024-01-02", "daily")
            df = provider._transform_data(raw_data, "AAPL")

        assert len(df) == 1
        assert df["symbol"][0] == "AAPL"
