"""Tests for Massive/Polygon provider module."""

from unittest.mock import MagicMock, patch

import polars as pl
import pytest

from ml4t.data.core.exceptions import (
    AuthenticationError,
    DataValidationError,
    NetworkError,
    ProviderError,
    RateLimitError,
    SymbolNotFoundError,
)
from ml4t.data.providers.polygon import MassiveProvider, PolygonProvider


class TestMassiveProviderInit:
    """Tests for canonical Massive provider initialization."""

    def test_init_with_api_key(self):
        """Test initialization with API key."""
        provider = MassiveProvider(api_key="test_key")

        assert provider.name == "massive"
        assert provider.api_key == "test_key"
        assert provider.base_url == "https://api.massive.com"

    def test_init_with_massive_env_api_key(self):
        """Test initialization with MASSIVE_API_KEY."""
        with patch.dict("os.environ", {"MASSIVE_API_KEY": "massive_env_key"}, clear=True):
            provider = MassiveProvider()

            assert provider.api_key == "massive_env_key"

    def test_massive_env_preferred_over_polygon_env(self):
        """Test MASSIVE_API_KEY takes precedence over legacy POLYGON_API_KEY."""
        with patch.dict(
            "os.environ",
            {"MASSIVE_API_KEY": "massive_env_key", "POLYGON_API_KEY": "polygon_env_key"},
            clear=True,
        ):
            provider = MassiveProvider()

            assert provider.api_key == "massive_env_key"

    def test_legacy_polygon_env_api_key_remains_valid(self):
        """Test existing POLYGON_API_KEY values remain valid."""
        with patch.dict("os.environ", {"POLYGON_API_KEY": "polygon_env_key"}, clear=True):
            provider = MassiveProvider()

            assert provider.api_key == "polygon_env_key"

    def test_custom_base_url(self):
        """Test custom base URL override."""
        provider = MassiveProvider(api_key="test_key", base_url="https://api.polygon.io/")

        assert provider.base_url == "https://api.polygon.io"


class TestPolygonProviderInit:
    """Tests for provider initialization."""

    def test_init_with_api_key(self):
        """Test initialization with API key."""
        with pytest.warns(DeprecationWarning, match="PolygonProvider is deprecated"):
            provider = PolygonProvider(api_key="test_key")

        assert provider.name == "polygon"
        assert provider.api_key == "test_key"
        assert provider.base_url == "https://api.massive.com"

    def test_init_with_env_api_key(self):
        """Test initialization with API key from environment."""
        with patch.dict("os.environ", {"POLYGON_API_KEY": "env_key"}, clear=True):
            with pytest.warns(DeprecationWarning, match="PolygonProvider is deprecated"):
                provider = PolygonProvider()

            assert provider.api_key == "env_key"

    def test_init_without_api_key_raises_error(self):
        """Test initialization without API key raises AuthenticationError."""
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(AuthenticationError, match="API key required"):
                PolygonProvider()

    def test_init_custom_rate_limit(self):
        """Test initialization with custom rate limit."""
        with pytest.warns(DeprecationWarning, match="PolygonProvider is deprecated"):
            provider = PolygonProvider(api_key="test_key", rate_limit=(100, 60.0))

        # Provider should be initialized successfully
        assert provider is not None


class TestNameProperty:
    """Tests for name property."""

    def test_name_returns_polygon(self):
        """Test name property returns correct value."""
        with pytest.warns(DeprecationWarning, match="PolygonProvider is deprecated"):
            provider = PolygonProvider(api_key="test_key")
        assert provider.name == "polygon"


class TestFetchRawData:
    """Tests for _fetch_raw_data method."""

    @pytest.fixture
    def provider(self):
        """Create provider instance."""
        return MassiveProvider(api_key="test_key")

    def test_fetch_raw_data_success(self, provider):
        """Test successful raw data fetch."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "status": "OK",
            "results": [
                {"t": 1704067200000, "o": 170.0, "h": 172.0, "l": 169.0, "c": 171.0, "v": 1000000},
            ],
        }

        with patch.object(provider.session, "get", return_value=mock_response):
            data = provider._fetch_raw_data("AAPL", "2024-01-01", "2024-01-02", "daily")

        assert data["status"] == "OK"
        assert len(data["results"]) == 1

    def test_fetch_raw_data_invalid_frequency(self, provider):
        """Test invalid frequency raises DataValidationError."""
        with pytest.raises(DataValidationError, match="Unsupported frequency"):
            provider._fetch_raw_data("AAPL", "2024-01-01", "2024-01-02", "invalid")

    def test_fetch_raw_data_auth_error(self, provider):
        """Test 401 raises AuthenticationError."""
        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.text = "Unauthorized"

        with patch.object(provider.session, "get", return_value=mock_response):
            with pytest.raises(AuthenticationError, match="Invalid API key"):
                provider._fetch_raw_data("AAPL", "2024-01-01", "2024-01-02", "daily")

    def test_fetch_raw_data_rate_limit(self, provider):
        """Test 429 raises RateLimitError."""
        mock_response = MagicMock()
        mock_response.status_code = 429

        with patch.object(provider.session, "get", return_value=mock_response):
            with pytest.raises(RateLimitError):
                provider._fetch_raw_data("AAPL", "2024-01-01", "2024-01-02", "daily")

    def test_fetch_raw_data_other_http_error(self, provider):
        """Test other HTTP errors raise NetworkError."""
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"

        with patch.object(provider.session, "get", return_value=mock_response):
            with pytest.raises(NetworkError, match="API error"):
                provider._fetch_raw_data("AAPL", "2024-01-01", "2024-01-02", "daily")

    def test_fetch_raw_data_json_parse_error(self, provider):
        """Test JSON parse error raises NetworkError."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.side_effect = ValueError("Invalid JSON")

        with patch.object(provider.session, "get", return_value=mock_response):
            with pytest.raises(NetworkError, match="Failed to parse JSON"):
                provider._fetch_raw_data("AAPL", "2024-01-01", "2024-01-02", "daily")

    def test_fetch_raw_data_api_error(self, provider):
        """Test API-level error raises ProviderError."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "status": "ERROR",
            "error": "Something went wrong",
        }

        with patch.object(provider.session, "get", return_value=mock_response):
            with pytest.raises(ProviderError, match="API error"):
                provider._fetch_raw_data("AAPL", "2024-01-01", "2024-01-02", "daily")

    def test_fetch_raw_data_symbol_not_found(self, provider):
        """Test symbol not found error."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "status": "ERROR",
            "error": "NOT_FOUND: Symbol INVALID does not exist",
        }

        with patch.object(provider.session, "get", return_value=mock_response):
            with pytest.raises(SymbolNotFoundError):
                provider._fetch_raw_data("INVALID", "2024-01-01", "2024-01-02", "daily")

    def test_fetch_raw_data_request_exception(self, provider):
        """Test request exception raises NetworkError."""
        with patch.object(provider.session, "get", side_effect=Exception("Connection failed")):
            with pytest.raises(NetworkError, match="Request failed"):
                provider._fetch_raw_data("AAPL", "2024-01-01", "2024-01-02", "daily")


class TestTransformData:
    """Tests for _transform_data method."""

    @pytest.fixture
    def provider(self):
        """Create provider instance."""
        return MassiveProvider(api_key="test_key")

    def test_transform_data_success(self, provider):
        """Test successful data transformation."""
        raw_data = {
            "results": [
                {"t": 1704067200000, "o": 170.0, "h": 172.0, "l": 169.0, "c": 171.0, "v": 1000000},
                {"t": 1704153600000, "o": 171.0, "h": 173.0, "l": 170.0, "c": 172.0, "v": 1100000},
            ],
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

    def test_transform_data_empty_results(self, provider):
        """Test empty results raises SymbolNotFoundError."""
        raw_data = {"results": []}

        with pytest.raises(SymbolNotFoundError):
            provider._transform_data(raw_data, "AAPL")

    def test_transform_data_no_results_key(self, provider):
        """Test missing results key raises SymbolNotFoundError."""
        raw_data = {"status": "OK"}

        with pytest.raises(SymbolNotFoundError):
            provider._transform_data(raw_data, "AAPL")

    def test_transform_data_casts_to_float(self, provider):
        """Test OHLCV columns are cast to float."""
        raw_data = {
            "results": [
                {
                    "t": 1704067200000,
                    "o": "170",
                    "h": "172",
                    "l": "169",
                    "c": "171",
                    "v": "1000000",
                },
            ],
        }

        df = provider._transform_data(raw_data, "AAPL")

        assert df["open"].dtype == pl.Float64
        assert df["high"].dtype == pl.Float64
        assert df["low"].dtype == pl.Float64
        assert df["close"].dtype == pl.Float64
        assert df["volume"].dtype == pl.Float64

    def test_transform_data_sorts_by_timestamp(self, provider):
        """Test data is sorted by timestamp."""
        raw_data = {
            "results": [
                {"t": 1704153600000, "o": 171.0, "h": 173.0, "l": 170.0, "c": 172.0, "v": 1100000},
                {"t": 1704067200000, "o": 170.0, "h": 172.0, "l": 169.0, "c": 171.0, "v": 1000000},
            ],
        }

        df = provider._transform_data(raw_data, "AAPL")

        # First row should have earlier timestamp
        assert df["timestamp"][0] < df["timestamp"][1]

    def test_transform_data_uppercase_symbol(self, provider):
        """Test symbol is uppercased."""
        raw_data = {
            "results": [
                {"t": 1704067200000, "o": 170.0, "h": 172.0, "l": 169.0, "c": 171.0, "v": 1000000},
            ],
        }

        df = provider._transform_data(raw_data, "aapl")

        assert df["symbol"][0] == "AAPL"


class TestFrequencyMapping:
    """Tests for frequency mapping."""

    @pytest.fixture
    def provider(self):
        """Create provider instance."""
        return MassiveProvider(api_key="test_key")

    def _create_success_response(self):
        """Create a mock success response."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "status": "OK",
            "results": [
                {"t": 1704067200000, "o": 170.0, "h": 172.0, "l": 169.0, "c": 171.0, "v": 1000000},
            ],
        }
        return mock_response

    def test_daily_frequency(self, provider):
        """Test daily frequency mapping."""
        with patch.object(provider.session, "get", return_value=self._create_success_response()):
            data = provider._fetch_raw_data("AAPL", "2024-01-01", "2024-01-02", "daily")
            assert "results" in data

    def test_day_frequency(self, provider):
        """Test day frequency alias."""
        with patch.object(provider.session, "get", return_value=self._create_success_response()):
            data = provider._fetch_raw_data("AAPL", "2024-01-01", "2024-01-02", "day")
            assert "results" in data

    def test_1d_frequency(self, provider):
        """Test 1d frequency alias."""
        with patch.object(provider.session, "get", return_value=self._create_success_response()):
            data = provider._fetch_raw_data("AAPL", "2024-01-01", "2024-01-02", "1d")
            assert "results" in data

    def test_hourly_frequency(self, provider):
        """Test hourly frequency mapping."""
        with patch.object(provider.session, "get", return_value=self._create_success_response()):
            data = provider._fetch_raw_data("AAPL", "2024-01-01", "2024-01-02", "hourly")
            assert "results" in data

    def test_weekly_frequency(self, provider):
        """Test weekly frequency mapping."""
        with patch.object(provider.session, "get", return_value=self._create_success_response()):
            data = provider._fetch_raw_data("AAPL", "2024-01-01", "2024-01-15", "weekly")
            assert "results" in data

    def test_monthly_frequency(self, provider):
        """Test monthly frequency mapping."""
        with patch.object(provider.session, "get", return_value=self._create_success_response()):
            data = provider._fetch_raw_data("AAPL", "2024-01-01", "2024-03-01", "monthly")
            assert "results" in data

    def test_minute_frequency(self, provider):
        """Test minute frequency mapping."""
        with patch.object(provider.session, "get", return_value=self._create_success_response()):
            data = provider._fetch_raw_data("AAPL", "2024-01-01", "2024-01-01", "minute")
            assert "results" in data


class TestDefaultRateLimit:
    """Tests for DEFAULT_RATE_LIMIT constant."""

    def test_default_rate_limit_value(self):
        """Test DEFAULT_RATE_LIMIT has expected value."""
        # Free tier: 5 requests per minute
        assert MassiveProvider.DEFAULT_RATE_LIMIT == (5, 60.0)


class TestIntegration:
    """Integration-style tests for MassiveProvider with mocked responses."""

    def test_full_fetch_workflow_with_mocks(self):
        """Test complete fetch workflow with mocked responses."""
        provider = MassiveProvider(api_key="test_key")

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "status": "OK",
            "results": [
                {"t": 1704067200000, "o": 170.0, "h": 172.0, "l": 169.0, "c": 171.0, "v": 1000000},
                {"t": 1704153600000, "o": 171.0, "h": 173.0, "l": 170.0, "c": 172.0, "v": 1100000},
            ],
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
        provider = MassiveProvider(api_key="test_key")

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "status": "OK",
            "results": [],
        }

        with patch.object(provider.session, "get", return_value=mock_response) as mock_get:
            try:
                provider._fetch_raw_data("AAPL", "2024-01-01", "2024-01-02", "daily")
            except SymbolNotFoundError:
                pass  # Expected for empty results

            # Check that API was called with correct parameters
            call_args = mock_get.call_args
            params = call_args.kwargs.get("params", call_args[1].get("params", {}))
            assert params.get("adjusted") == "true"
            assert params.get("sort") == "asc"
            assert params.get("limit") == 50000
            assert params.get("apiKey") == "test_key"


class TestMassiveAssetClassRouting:
    """Tests for Massive endpoint routing across asset classes."""

    @pytest.fixture
    def provider(self):
        """Create provider instance."""
        return MassiveProvider(api_key="test_key")

    def _create_success_response(self):
        """Create a mock success response."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "status": "OK",
            "results": [
                {"t": 1704067200000, "o": 170.0, "h": 172.0, "l": 169.0, "c": 171.0, "v": 1000000},
            ],
        }
        return mock_response

    @pytest.mark.parametrize(
        ("symbol", "expected_asset_class", "expected_path"),
        [
            ("AAPL", "stocks", "/v2/aggs/ticker/AAPL/range/1/day/2024-01-01/2024-01-02"),
            (
                "O:SPY240119C00480000",
                "options",
                "/v2/aggs/ticker/O:SPY240119C00480000/range/1/day/2024-01-01/2024-01-02",
            ),
            ("X:BTCUSD", "crypto", "/v2/aggs/ticker/X:BTCUSD/range/1/day/2024-01-01/2024-01-02"),
            ("C:EURUSD", "forex", "/v2/aggs/ticker/C:EURUSD/range/1/day/2024-01-01/2024-01-02"),
            ("F:ESM6", "futures", "/futures/v1/aggs/ESM6"),
        ],
    )
    def test_infers_asset_class_and_routes_endpoint(
        self,
        provider,
        symbol,
        expected_asset_class,
        expected_path,
    ):
        """Test endpoint routing for all supported asset classes."""
        assert provider._infer_asset_class(symbol) == expected_asset_class

        with patch.object(
            provider.session, "get", return_value=self._create_success_response()
        ) as get:
            provider._fetch_raw_data(symbol, "2024-01-01", "2024-01-02", "daily")

        url = get.call_args.args[0]
        assert url == f"https://api.massive.com{expected_path}"

    def test_explicit_futures_asset_class_routes_unprefixed_symbol(self, provider):
        """Test unprefixed futures symbols can be routed explicitly."""
        with patch.object(
            provider.session, "get", return_value=self._create_success_response()
        ) as get:
            provider._fetch_raw_data(
                "ESM6",
                "2024-01-01",
                "2024-01-02",
                "daily",
                asset_class="futures",
            )

        assert get.call_args.args[0] == "https://api.massive.com/futures/v1/aggs/ESM6"
        params = get.call_args.kwargs["params"]
        assert params["multiplier"] == 1
        assert params["timespan"] == "day"
        assert params["from"] == "2024-01-01"
        assert params["to"] == "2024-01-02"
