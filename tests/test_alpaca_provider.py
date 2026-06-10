"""Tests for Alpaca data provider module."""

from unittest.mock import AsyncMock, MagicMock, patch

import polars as pl
import pytest

from ml4t.data.core.exceptions import (
    AuthenticationError,
    DataNotAvailableError,
    DataValidationError,
    NetworkError,
    RateLimitError,
)
from ml4t.data.providers.alpaca import AlpacaDataProvider
from ml4t.data.providers.protocols import OHLCVProvider

STOCK_BARS = [
    {"t": "2024-01-03T05:00:00Z", "o": 2.0, "h": 3.0, "l": 1.5, "c": 2.5, "v": 2000},
    {"t": "2024-01-02T05:00:00Z", "o": 1.0, "h": 2.0, "l": 0.5, "c": 1.5, "v": 1000},
]

# Crypto trades 24/7; 2024-01-06 is a Saturday, included to confirm no
# weekday-only assumption filters weekend bars out.
CRYPTO_BARS = [
    {"t": "2024-01-06T00:00:00Z", "o": 44.0, "h": 46.0, "l": 43.0, "c": 45.0, "v": 12.0},
    {"t": "2024-01-05T00:00:00Z", "o": 42.0, "h": 44.0, "l": 41.0, "c": 43.0, "v": 10.0},
]


class TestAlpacaProviderInit:
    """Tests for provider construction and authentication."""

    def test_init_with_explicit_keys(self):
        """Explicit api_key/api_secret are stored on the provider."""
        provider = AlpacaDataProvider(api_key="k", api_secret="s")

        assert provider.api_key == "k"
        assert provider.api_secret == "s"
        assert provider.feed == "iex"

    def test_init_with_env_keys(self):
        """Credentials are read from ALPACA_* environment variables."""
        with patch.dict(
            "os.environ",
            {"ALPACA_API_KEY": "k", "ALPACA_API_SECRET": "s"},
        ):
            provider = AlpacaDataProvider()

            assert provider.api_key == "k"
            assert provider.api_secret == "s"

    def test_init_with_apca_env_fallback(self):
        """Alpaca SDK/CLI env names are honored as a fallback."""
        with patch.dict(
            "os.environ",
            {"APCA_API_KEY_ID": "k", "APCA_API_SECRET_KEY": "s"},
            clear=True,
        ):
            provider = AlpacaDataProvider()

            assert provider.api_key == "k"
            assert provider.api_secret == "s"

    def test_init_missing_key_raises(self):
        """A missing API key raises AuthenticationError."""
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(AuthenticationError):
                AlpacaDataProvider(api_secret="s")

    def test_init_missing_secret_raises(self):
        """A missing API secret raises AuthenticationError."""
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(AuthenticationError):
                AlpacaDataProvider(api_key="k")

    def test_default_feed_is_iex(self):
        """Feed defaults to IEX (free tier)."""
        provider = AlpacaDataProvider(api_key="k", api_secret="s")

        assert provider.feed == "iex"

    def test_feed_sip_honored(self):
        """An explicit SIP feed is honored."""
        provider = AlpacaDataProvider(api_key="k", api_secret="s", feed="sip")

        assert provider.feed == "sip"


class TestAuthHeaders:
    """Tests for auth header wiring on both sessions."""

    def test_auth_headers_on_sync_session(self):
        """Both auth headers are present on the sync httpx.Client."""
        provider = AlpacaDataProvider(api_key="k", api_secret="s")

        assert provider.session.headers["APCA-API-KEY-ID"] == "k"
        assert provider.session.headers["APCA-API-SECRET-KEY"] == "s"

    @pytest.mark.asyncio
    async def test_auth_headers_on_async_session(self):
        """Both auth headers are present on the async httpx.AsyncClient."""
        provider = AlpacaDataProvider(api_key="k", api_secret="s")

        await provider.init_async_session()
        try:
            assert provider.async_session is not None
            assert provider.async_session.headers["APCA-API-KEY-ID"] == "k"
            assert provider.async_session.headers["APCA-API-SECRET-KEY"] == "s"
        finally:
            await provider.close_async_session()


class TestNameProperty:
    """Tests for the name property."""

    def test_name_property(self):
        """Name property returns 'alpaca'."""
        provider = AlpacaDataProvider(api_key="k", api_secret="s")

        assert provider.name == "alpaca"


class TestFrequencyMapping:
    """Tests for frequency mapping to Alpaca timeframes."""

    @pytest.fixture
    def provider(self):
        """Create a provider instance."""
        return AlpacaDataProvider(api_key="k", api_secret="s")

    @pytest.mark.parametrize(
        ("frequency", "expected"),
        [
            ("daily", "1Day"),
            ("day", "1Day"),
            ("1d", "1Day"),
            ("1day", "1Day"),
            ("hourly", "1Hour"),
            ("hour", "1Hour"),
            ("1h", "1Hour"),
            ("1hour", "1Hour"),
            ("minute", "1Min"),
            ("1m", "1Min"),
            ("1minute", "1Min"),
        ],
    )
    def test_frequency_maps_to_timeframe(self, provider, frequency, expected):
        """Canonical keys and aliases map to the right Alpaca timeframe."""
        assert provider._map_frequency(frequency) == expected

    def test_frequency_mapping_is_case_insensitive(self, provider):
        """Frequency lookup is case-insensitive."""
        assert provider._map_frequency("DAILY") == "1Day"

    def test_unsupported_frequency_raises(self, provider):
        """Unsupported frequency raises DataValidationError listing supported keys."""
        with pytest.raises(DataValidationError) as exc_info:
            provider._map_frequency("yearly")

        message = str(exc_info.value)
        assert "daily" in message
        assert "minute" in message


class TestCapabilities:
    """Tests for the capabilities declaration."""

    def test_capabilities(self):
        """Capabilities report crypto, intraday, api-key, and a rate limit."""
        provider = AlpacaDataProvider(api_key="k", api_secret="s")
        caps = provider.capabilities()

        assert caps.supports_crypto is True
        assert caps.supports_intraday is True
        assert caps.requires_api_key is True
        assert caps.rate_limit == AlpacaDataProvider.DEFAULT_RATE_LIMIT


class TestProtocolConformance:
    """Tests for OHLCVProvider protocol conformance."""

    def test_is_ohlcv_provider(self):
        """An instance satisfies the OHLCVProvider protocol."""
        provider = AlpacaDataProvider(api_key="k", api_secret="s")

        assert isinstance(provider, OHLCVProvider)


class TestDefaultRateLimit:
    """Tests for the default rate-limit class variable."""

    def test_default_rate_limit(self):
        """DEFAULT_RATE_LIMIT reflects the documented Basic-plan figure."""
        assert AlpacaDataProvider.DEFAULT_RATE_LIMIT == (200, 60.0)


class TestFetchRawDataStock:
    """Tests for single-page stock bar fetching and error mapping."""

    @pytest.fixture
    def provider(self):
        """Create a provider instance."""
        return AlpacaDataProvider(api_key="k", api_secret="s")

    def test_fetch_raw_data_success(self, provider):
        """A 200 response returns the parsed bars structure."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"bars": STOCK_BARS, "next_page_token": None}

        with (
            patch.object(provider.session, "get", return_value=mock_response),
            patch.object(provider.rate_limiter, "acquire"),
        ):
            data = provider._fetch_raw_data("AAPL", "2024-01-01", "2024-01-05", "daily")

        assert data["bars"] == STOCK_BARS
        assert data["next_page_token"] is None

    def test_feed_param_sent_for_stock(self):
        """The configured feed is forwarded in the request params."""
        provider = AlpacaDataProvider(api_key="k", api_secret="s", feed="sip")
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"bars": [], "next_page_token": None}

        with (
            patch.object(provider.session, "get", return_value=mock_response) as mock_get,
            patch.object(provider.rate_limiter, "acquire"),
        ):
            provider._fetch_raw_data("AAPL", "2024-01-01", "2024-01-05", "daily")

        assert mock_get.call_args.kwargs["params"]["feed"] == "sip"

    def test_fetch_429_raises_rate_limit(self, provider):
        """A 429 maps to RateLimitError."""
        mock_response = MagicMock()
        mock_response.status_code = 429

        with (
            patch.object(provider.session, "get", return_value=mock_response),
            patch.object(provider.rate_limiter, "acquire"),
        ):
            with pytest.raises(RateLimitError):
                provider._fetch_raw_data("AAPL", "2024-01-01", "2024-01-05", "daily")

    def test_fetch_401_raises_auth(self, provider):
        """A 401 maps to AuthenticationError."""
        mock_response = MagicMock()
        mock_response.status_code = 401

        with (
            patch.object(provider.session, "get", return_value=mock_response),
            patch.object(provider.rate_limiter, "acquire"),
        ):
            with pytest.raises(AuthenticationError):
                provider._fetch_raw_data("AAPL", "2024-01-01", "2024-01-05", "daily")

    def test_fetch_404_raises_data_not_available(self, provider):
        """A 404 maps to DataNotAvailableError."""
        mock_response = MagicMock()
        mock_response.status_code = 404

        with (
            patch.object(provider.session, "get", return_value=mock_response),
            patch.object(provider.rate_limiter, "acquire"),
        ):
            with pytest.raises(DataNotAvailableError):
                provider._fetch_raw_data("AAPL", "2024-01-01", "2024-01-05", "daily")

    def test_fetch_500_raises_network(self, provider):
        """A 500 maps to NetworkError."""
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"

        with (
            patch.object(provider.session, "get", return_value=mock_response),
            patch.object(provider.rate_limiter, "acquire"),
        ):
            with pytest.raises(NetworkError):
                provider._fetch_raw_data("AAPL", "2024-01-01", "2024-01-05", "daily")

    def test_json_parse_error_raises_network(self, provider):
        """A JSON parse failure maps to NetworkError."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.side_effect = ValueError("Invalid JSON")

        with (
            patch.object(provider.session, "get", return_value=mock_response),
            patch.object(provider.rate_limiter, "acquire"),
        ):
            with pytest.raises(NetworkError, match="Failed to parse JSON"):
                provider._fetch_raw_data("AAPL", "2024-01-01", "2024-01-05", "daily")

    @pytest.mark.asyncio
    async def test_fetch_raw_data_async_stock(self, provider):
        """The async path returns the same parsed structure as the sync path."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"bars": STOCK_BARS, "next_page_token": None}

        with (
            patch.object(provider, "_aget", new=AsyncMock(return_value=mock_response)),
            patch.object(provider.rate_limiter, "acquire"),
        ):
            data = await provider._fetch_raw_data_async("AAPL", "2024-01-01", "2024-01-05", "daily")

        assert data["bars"] == STOCK_BARS


class TestTransformDataStock:
    """Tests for transforming stock bars into the standard schema."""

    @pytest.fixture
    def provider(self):
        """Create a provider instance."""
        return AlpacaDataProvider(api_key="k", api_secret="s")

    def test_transform_data_stock(self, provider):
        """Bars transform to the canonical OHLCV schema, sorted, with invariants."""
        raw = {"bars": STOCK_BARS, "next_page_token": None}

        df = provider._transform_data(raw, "AAPL")

        assert df.columns == [
            "timestamp",
            "symbol",
            "open",
            "high",
            "low",
            "close",
            "volume",
        ]
        assert df.schema["timestamp"] == pl.Datetime
        for col in ["open", "high", "low", "close", "volume"]:
            assert df.schema[col] == pl.Float64
        assert df["symbol"].to_list() == ["AAPL", "AAPL"]
        # Rows sorted ascending by timestamp.
        timestamps = df["timestamp"].to_list()
        assert timestamps == sorted(timestamps)
        # OHLC invariants hold for every row.
        assert (df["high"] >= df["low"]).all()
        assert (df["high"] >= df["open"]).all()
        assert (df["high"] >= df["close"]).all()

    def test_transform_data_stock_dict_shape(self, provider):
        """A multi-symbol dict-keyed bars payload is also accepted."""
        raw = {"bars": {"AAPL": STOCK_BARS}, "next_page_token": None}

        df = provider._transform_data(raw, "AAPL")

        assert df["symbol"].to_list() == ["AAPL", "AAPL"]
        assert df.height == 2

    def test_empty_bars_returns_empty_dataframe(self, provider):
        """An empty bars list yields the canonical empty DataFrame."""
        raw = {"bars": [], "next_page_token": None}

        df = provider._transform_data(raw, "AAPL")
        expected = provider._create_empty_dataframe()

        assert df.columns == expected.columns
        assert df.schema == expected.schema
        assert df.height == 0


class TestAssetRouting:
    """Tests for routing symbols to the stock vs crypto endpoint."""

    @pytest.fixture
    def provider(self):
        """Create a provider instance."""
        return AlpacaDataProvider(api_key="k", api_secret="s")

    def test_resolve_asset_class_by_slash(self, provider):
        """A slash in the symbol resolves to crypto; otherwise stock."""
        assert provider._resolve_asset_class("BTC/USD", None) == "crypto"
        assert provider._resolve_asset_class("AAPL", None) == "stock"

    def test_resolve_asset_class_override(self, provider):
        """An explicit asset_class kwarg wins over the slash heuristic."""
        assert provider._resolve_asset_class("AAPL", "crypto") == "crypto"
        assert provider._resolve_asset_class("BTC/USD", "stock") == "stock"

    def test_routes_crypto_by_slash(self, provider):
        """A BTC/USD symbol hits the crypto bars endpoint; AAPL hits the stock one."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"bars": {}, "next_page_token": None}

        with (
            patch.object(provider.session, "get", return_value=mock_response) as mock_get,
            patch.object(provider.rate_limiter, "acquire"),
        ):
            provider.fetch_ohlcv("BTC/USD", "2024-01-01", "2024-01-07", "daily")
            crypto_url = mock_get.call_args.args[0]

            provider.fetch_ohlcv("AAPL", "2024-01-01", "2024-01-07", "daily")
            stock_url = mock_get.call_args.args[0]

        assert "/v1beta3/crypto/" in crypto_url
        assert crypto_url.endswith("/bars")
        assert "/v2/stocks/" in stock_url

    def test_asset_class_override(self, provider):
        """An explicit asset_class forces crypto routing regardless of the slash."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"bars": {}, "next_page_token": None}

        with (
            patch.object(provider.session, "get", return_value=mock_response) as mock_get,
            patch.object(provider.rate_limiter, "acquire"),
        ):
            provider.fetch_ohlcv("X", "2024-01-01", "2024-01-07", "daily", asset_class="crypto")
            url = mock_get.call_args.args[0]

        assert "/v1beta3/crypto/" in url

    def test_feed_not_sent_for_crypto(self, provider):
        """Crypto request params must not include the stock-only feed parameter."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"bars": {}, "next_page_token": None}

        with (
            patch.object(provider.session, "get", return_value=mock_response) as mock_get,
            patch.object(provider.rate_limiter, "acquire"),
        ):
            provider.fetch_ohlcv("BTC/USD", "2024-01-01", "2024-01-07", "daily")
            params = mock_get.call_args.kwargs["params"]

        assert "feed" not in params
        assert params["symbols"] == "BTC/USD"


class TestFetchRawDataCrypto:
    """Tests for single-page crypto bar fetching."""

    @pytest.fixture
    def provider(self):
        """Create a provider instance."""
        return AlpacaDataProvider(api_key="k", api_secret="s")

    def test_fetch_raw_data_crypto_success(self, provider):
        """A 200 crypto response returns the parsed symbol-keyed bars structure."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "bars": {"BTC/USD": CRYPTO_BARS},
            "next_page_token": None,
        }

        with (
            patch.object(provider.session, "get", return_value=mock_response) as mock_get,
            patch.object(provider.rate_limiter, "acquire"),
        ):
            data = provider._fetch_raw_data(
                "BTC/USD", "2024-01-01", "2024-01-07", "daily", asset_class="crypto"
            )

        assert data["bars"]["BTC/USD"] == CRYPTO_BARS
        # Symbol is forwarded verbatim, slash preserved, via the symbols param.
        assert mock_get.call_args.kwargs["params"]["symbols"] == "BTC/USD"
        assert "/v1beta3/crypto/" in mock_get.call_args.args[0]

    @pytest.mark.asyncio
    async def test_crypto_async(self, provider):
        """The async crypto path returns the same parsed structure as the sync path."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "bars": {"BTC/USD": CRYPTO_BARS},
            "next_page_token": None,
        }

        with (
            patch.object(provider, "_aget", new=AsyncMock(return_value=mock_response)) as mock_aget,
            patch.object(provider.rate_limiter, "acquire"),
        ):
            data = await provider._fetch_raw_data_async(
                "BTC/USD", "2024-01-01", "2024-01-07", "daily", asset_class="crypto"
            )

        assert data["bars"]["BTC/USD"] == CRYPTO_BARS
        assert "/v1beta3/crypto/" in mock_aget.call_args.args[0]
        assert "feed" not in mock_aget.call_args.kwargs["params"]


class TestTransformDataCrypto:
    """Tests for transforming crypto bars into the standard schema."""

    @pytest.fixture
    def provider(self):
        """Create a provider instance."""
        return AlpacaDataProvider(api_key="k", api_secret="s")

    def test_transform_data_crypto(self, provider):
        """Crypto bars transform with the slash-preserved symbol and weekend bars kept."""
        raw = {"bars": {"BTC/USD": CRYPTO_BARS}, "next_page_token": None}

        df = provider._transform_data(raw, "BTC/USD", asset_class="crypto")

        assert df.columns == [
            "timestamp",
            "symbol",
            "open",
            "high",
            "low",
            "close",
            "volume",
        ]
        assert df.schema["timestamp"] == pl.Datetime
        for col in ["open", "high", "low", "close", "volume"]:
            assert df.schema[col] == pl.Float64
        # Slash preserved, not uppercased-stripped.
        assert df["symbol"].to_list() == ["BTC/USD", "BTC/USD"]
        # Every bar is retained, including the Saturday timestamp.
        assert df.height == 2
        weekdays = df["timestamp"].dt.weekday().to_list()
        # Polars weekday: Saturday == 6. The weekend bar must survive transform.
        assert 6 in weekdays
        timestamps = df["timestamp"].to_list()
        assert timestamps == sorted(timestamps)
