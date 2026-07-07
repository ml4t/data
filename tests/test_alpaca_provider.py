"""Tests for Alpaca data provider module."""

import time
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import polars as pl
import pytest

from ml4t.data.core.exceptions import (
    AuthenticationError,
    CircuitBreakerOpenError,
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


def _page_response(bars, token=None, status=200, headers=None):
    """Build a mock bars response with a given status, token, and headers."""
    response = MagicMock()
    response.status_code = status
    response.text = ""
    response.headers = headers or {}
    response.json.return_value = {"bars": bars, "next_page_token": token}
    return response


@pytest.fixture
def provider():
    """Create a provider instance."""
    return AlpacaDataProvider(api_key="k", api_secret="s")


@pytest.fixture(autouse=True)
def _no_retry_sleep():
    """Null out tenacity's sleep so retry-path tests never wait for real."""
    with (
        patch.object(AlpacaDataProvider._get_page.retry, "sleep", MagicMock()),
        patch.object(AlpacaDataProvider._get_page_async.retry, "sleep", AsyncMock()),
    ):
        yield


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

    @pytest.mark.parametrize("feed", ["iex", "sip", "otc", "boats"])
    def test_supported_feeds_accepted(self, feed):
        """Every feed Alpaca's bars endpoint accepts is honored."""
        provider = AlpacaDataProvider(api_key="k", api_secret="s", feed=feed)

        assert provider.feed == feed

    def test_feed_normalized_case_insensitively(self):
        """A mixed-case feed is normalized to the lowercase API value."""
        provider = AlpacaDataProvider(api_key="k", api_secret="s", feed="SIP")

        assert provider.feed == "sip"

    def test_invalid_feed_raises(self):
        """A typo'd feed raises DataValidationError instead of a remote error."""
        with pytest.raises(DataValidationError, match="feed") as exc_info:
            AlpacaDataProvider(api_key="k", api_secret="s", feed="sup")

        assert exc_info.value.field == "feed"
        assert exc_info.value.value == "sup"


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
            ("5m", "5Min"),
            ("5minute", "5Min"),
            ("15m", "15Min"),
            ("15minute", "15Min"),
            ("30m", "30Min"),
            ("30minute", "30Min"),
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

        # The fetcher returns the merged-across-pages shape carrying only bars;
        # the per-page next_page_token is consumed internally, not surfaced.
        assert data["bars"] == STOCK_BARS

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


class TestPagination:
    """Tests for following next_page_token across multiple pages."""

    def test_follows_next_page_token(self, provider):
        """Two stock pages are merged and page 2 is requested with the token."""
        page1 = _page_response([STOCK_BARS[0]], "abc")
        page2 = _page_response([STOCK_BARS[1]], None)

        with (
            patch.object(provider.session, "get", side_effect=[page1, page2]) as mock_get,
            patch.object(provider.rate_limiter, "acquire"),
        ):
            data = provider._fetch_raw_data("AAPL", "2024-01-01", "2024-01-07", "daily")

        assert data["bars"] == [STOCK_BARS[0], STOCK_BARS[1]]
        assert mock_get.call_count == 2
        # The second request carries the page token returned by page 1.
        assert mock_get.call_args_list[1].kwargs["params"]["page_token"] == "abc"
        # The first request does not send a page token.
        assert "page_token" not in mock_get.call_args_list[0].kwargs["params"]

    def test_single_page_no_extra_request(self, provider):
        """A null token on page 1 yields exactly one request."""
        page1 = _page_response(STOCK_BARS, None)

        with (
            patch.object(provider.session, "get", side_effect=[page1]) as mock_get,
            patch.object(provider.rate_limiter, "acquire"),
        ):
            data = provider._fetch_raw_data("AAPL", "2024-01-01", "2024-01-07", "daily")

        assert data["bars"] == STOCK_BARS
        assert mock_get.call_count == 1

    def test_pagination_crypto(self, provider):
        """Two crypto pages merge per-symbol bar lists across pages."""
        page1 = _page_response({"BTC/USD": [CRYPTO_BARS[0]]}, "abc")
        page2 = _page_response({"BTC/USD": [CRYPTO_BARS[1]]}, None)

        with (
            patch.object(provider.session, "get", side_effect=[page1, page2]) as mock_get,
            patch.object(provider.rate_limiter, "acquire"),
        ):
            data = provider._fetch_raw_data(
                "BTC/USD", "2024-01-01", "2024-01-07", "daily", asset_class="crypto"
            )

        assert data["bars"]["BTC/USD"] == [CRYPTO_BARS[0], CRYPTO_BARS[1]]
        assert mock_get.call_count == 2
        assert mock_get.call_args_list[1].kwargs["params"]["page_token"] == "abc"

    def test_rate_limit_acquired_once_per_page(self, provider):
        """acquire is called exactly once per page, not doubled per request."""
        page1 = _page_response([STOCK_BARS[0]], "abc")
        page2 = _page_response([STOCK_BARS[1]], None)

        with (
            patch.object(provider.session, "get", side_effect=[page1, page2]),
            patch.object(provider.rate_limiter, "acquire") as mock_acquire,
        ):
            provider._fetch_raw_data("AAPL", "2024-01-01", "2024-01-07", "daily")

        assert mock_acquire.call_count == 2

    @pytest.mark.asyncio
    async def test_follows_next_page_token_async(self, provider):
        """The async path follows the token and merges both pages."""
        page1 = _page_response([STOCK_BARS[0]], "abc")
        page2 = _page_response([STOCK_BARS[1]], None)

        with (
            patch.object(provider, "_aget", new=AsyncMock(side_effect=[page1, page2])) as mock_aget,
            patch.object(provider.rate_limiter, "acquire"),
        ):
            data = await provider._fetch_raw_data_async("AAPL", "2024-01-01", "2024-01-07", "daily")

        assert data["bars"] == [STOCK_BARS[0], STOCK_BARS[1]]
        assert mock_aget.call_count == 2
        assert mock_aget.call_args_list[1].kwargs["params"]["page_token"] == "abc"


class TestAlpacaRegistration:
    """Tests that wire Alpaca into the registry, config, and exports."""

    def test_registry_resolves_alpaca(self):
        """The ProviderManager registry maps 'alpaca' to AlpacaDataProvider."""
        from ml4t.data.managers.provider_manager import ProviderManager

        classes = ProviderManager._get_provider_classes()

        assert "alpaca" in classes
        assert classes["alpaca"] is AlpacaDataProvider

    def test_alpaca_in_keyed_providers(self):
        """Alpaca is registered as an API-key-requiring provider."""
        from ml4t.data.managers.provider_manager import ProviderManager

        assert "alpaca" in ProviderManager.KEYED_PROVIDERS

    def test_alpaca_exported(self):
        """AlpacaDataProvider is importable from the package and in __all__."""
        import ml4t.data.providers as providers
        from ml4t.data.providers import AlpacaDataProvider as Exported

        assert Exported is AlpacaDataProvider
        assert "AlpacaDataProvider" in providers.__all__

    @staticmethod
    def _env_without_alpaca():
        """Return a copy of the environment with all Alpaca credentials removed."""
        import os

        return {k: v for k, v in os.environ.items() if not k.startswith(("ALPACA_", "APCA_"))}

    def test_env_autodetect_alpaca(self):
        """Both ALPACA_* env credentials together mark alpaca available."""
        import os

        from ml4t.data.managers.provider_manager import ProviderManager

        env = self._env_without_alpaca()
        env["ALPACA_API_KEY"] = "k"
        env["ALPACA_API_SECRET"] = "s"
        with patch.dict(os.environ, env, clear=True):
            manager = ProviderManager(config={"providers": {}})
            assert "alpaca" in manager.available_providers

    def test_env_autodetect_requires_both_credentials(self):
        """A key without a secret must not mark alpaca available."""
        import os

        from ml4t.data.managers.provider_manager import ProviderManager

        env = self._env_without_alpaca()
        env["ALPACA_API_KEY"] = "k"
        with patch.dict(os.environ, env, clear=True):
            manager = ProviderManager(config={"providers": {}})
            assert "alpaca" not in manager.available_providers

    def test_env_autodetect_apca_aliases(self):
        """The Alpaca SDK's APCA_* env names also satisfy auto-detection."""
        import os

        from ml4t.data.managers.provider_manager import ProviderManager

        env = self._env_without_alpaca()
        env["APCA_API_KEY_ID"] = "k"
        env["APCA_API_SECRET_KEY"] = "s"
        with patch.dict(os.environ, env, clear=True):
            manager = ProviderManager(config={"providers": {}})
            assert "alpaca" in manager.available_providers

    def test_provider_type_enum_accepts_alpaca(self):
        """ProviderType accepts 'alpaca' and ProviderConfig validates with it."""
        from ml4t.data.config.models import ProviderConfig, ProviderType

        assert ProviderType("alpaca") == ProviderType.ALPACA

        config = ProviderConfig(name="x", type="alpaca")
        assert config.type == ProviderType.ALPACA

    def test_config_manager_injects_both_credentials(self):
        """Both Alpaca env credentials land in the resolved provider config."""
        import os

        from ml4t.data.managers.config_manager import ConfigManager

        env = self._env_without_alpaca()
        env["ALPACA_API_KEY"] = "k"
        env["ALPACA_API_SECRET"] = "s"
        with patch.dict(os.environ, env, clear=True):
            manager = ConfigManager()
            provider_config = manager.get_provider_config("alpaca")

        assert provider_config.get("api_key") == "k"
        assert provider_config.get("api_secret") == "s"

    def test_config_manager_accepts_apca_aliases(self):
        """APCA_* env names inject credentials, and ALPACA_* names win over them."""
        import os

        from ml4t.data.managers.config_manager import ConfigManager

        env = self._env_without_alpaca()
        env["APCA_API_KEY_ID"] = "apca-key"
        env["APCA_API_SECRET_KEY"] = "apca-secret"
        with patch.dict(os.environ, env, clear=True):
            provider_config = ConfigManager().get_provider_config("alpaca")
        assert provider_config.get("api_key") == "apca-key"
        assert provider_config.get("api_secret") == "apca-secret"

        env["ALPACA_API_KEY"] = "alpaca-key"
        with patch.dict(os.environ, env, clear=True):
            provider_config = ConfigManager().get_provider_config("alpaca")
        assert provider_config.get("api_key") == "alpaca-key"

    def test_alpaca_config_key_without_secret_not_available(self):
        """A config with api_key but no api_secret leaves alpaca unavailable.

        A key without a secret is not constructable, so availability must not
        be reported; get_provider then fails with the not-available error
        instead of a late construction failure.
        """
        import os

        from ml4t.data.managers.provider_manager import ProviderManager

        env = self._env_without_alpaca()
        with patch.dict(os.environ, env, clear=True):
            manager = ProviderManager(config={"providers": {"alpaca": {"api_key": "k"}}})

            assert "alpaca" not in manager.available_providers
            with pytest.raises(ValueError, match="not available"):
                manager.get_provider("alpaca")

    def test_alpaca_config_key_with_env_secret_available(self):
        """A configured key plus an env secret is constructable, so available."""
        import os

        from ml4t.data.managers.provider_manager import ProviderManager

        env = self._env_without_alpaca()
        env["ALPACA_API_SECRET"] = "s"
        with patch.dict(os.environ, env, clear=True):
            manager = ProviderManager(config={"providers": {"alpaca": {"api_key": "k"}}})

            assert "alpaca" in manager.available_providers

    def test_provider_info_redacts_secrets(self):
        """get_provider_info must strip api_key and api_secret from the config."""
        import os

        from ml4t.data.managers.provider_manager import ProviderManager

        env = self._env_without_alpaca()
        with patch.dict(os.environ, env, clear=True):
            manager = ProviderManager(
                config={"providers": {"alpaca": {"api_key": "k", "api_secret": "s", "feed": "iex"}}}
            )
            info = manager.get_provider_info("alpaca")

        assert info["has_api_key"] is True
        assert "api_key" not in info["config"]
        assert "api_secret" not in info["config"]
        assert info["config"]["feed"] == "iex"

    def test_validator_warns_on_missing_secret(self):
        """The config validator warns when an alpaca provider lacks a secret."""
        from ml4t.data.config.models import DataConfig, ProviderConfig
        from ml4t.data.config.validator import ConfigValidator

        config = DataConfig(providers=[ProviderConfig(name="alpaca", type="alpaca", api_key="k")])
        validator = ConfigValidator(config)
        validator.validate()

        assert any("api_secret" in warning for warning in validator.warnings)


class TestValidateInputs:
    """Tests for the date/datetime input validation override."""

    def test_accepts_date_bounds(self, provider):
        """Plain YYYY-MM-DD bounds validate."""
        provider._validate_inputs("AAPL", "2024-01-01", "2024-01-31", "daily")

    def test_accepts_rfc3339_bounds(self, provider):
        """RFC-3339 datetime bounds with a Z offset validate."""
        provider._validate_inputs(
            "BTC/USD", "2024-01-01T00:00:00Z", "2024-01-01T01:00:00Z", "minute"
        )

    def test_accepts_mixed_date_and_datetime_bounds(self, provider):
        """A naive date start and an aware datetime end stay comparable."""
        provider._validate_inputs("AAPL", "2024-01-01", "2024-01-02T01:00:00Z", "hourly")

    def test_rejects_malformed_bound(self, provider):
        """A non-ISO bound raises ValueError mentioning both accepted forms."""
        with pytest.raises(ValueError, match="YYYY-MM-DD or RFC-3339"):
            provider._validate_inputs("AAPL", "01/02/2024", "2024-01-31", "daily")

    def test_rejects_start_after_end(self, provider):
        """A start bound after the end bound raises ValueError."""
        with pytest.raises(ValueError, match="before or equal"):
            provider._validate_inputs("AAPL", "2024-02-01", "2024-01-01", "daily")

    def test_rejects_empty_symbol(self, provider):
        """An empty symbol raises ValueError."""
        with pytest.raises(ValueError, match="Symbol cannot be empty"):
            provider._validate_inputs(" ", "2024-01-01", "2024-01-31", "daily")

    def test_fetch_ohlcv_accepts_datetime_bounds(self, provider):
        """The public sync path accepts RFC-3339 bounds end to end."""
        response = _page_response({"BTC/USD": CRYPTO_BARS})

        with (
            patch.object(provider.session, "get", return_value=response) as mock_get,
            patch.object(provider.rate_limiter, "acquire"),
        ):
            df = provider.fetch_ohlcv(
                "BTC/USD", "2024-01-01T00:00:00Z", "2024-01-06T01:00:00Z", "minute"
            )

        assert df.height == 2
        assert mock_get.call_args.kwargs["params"]["start"] == "2024-01-01T00:00:00Z"


class TestAssetClassValidation:
    """Tests for explicit asset_class validation and normalization."""

    def test_invalid_asset_class_raises(self, provider):
        """A typo'd asset_class raises DataValidationError, not a silent 404."""
        with pytest.raises(DataValidationError, match="asset_class"):
            provider._resolve_asset_class("AAPL", "equity")

    def test_asset_class_is_case_insensitive(self, provider):
        """Mixed-case asset_class values normalize to lowercase."""
        assert provider._resolve_asset_class("AAPL", "Crypto") == "crypto"
        assert provider._resolve_asset_class("BTC/USD", "STOCK") == "stock"


class TestSymbolNormalization:
    """Tests for the canonical uppercase-symbol contract."""

    def test_lowercase_crypto_symbol_uppercased_in_request(self, provider):
        """A lowercase BASE/QUOTE symbol is uppercased into the symbols param."""
        _, params = provider._bars_request("btc/usd", "2024-01-01", "2024-01-07", "daily", "crypto")

        assert params["symbols"] == "BTC/USD"

    def test_lowercase_crypto_symbol_uppercased_in_output(self, provider):
        """The symbol column is uppercase even for lowercase crypto input."""
        raw = {"bars": {"BTC/USD": CRYPTO_BARS}, "next_page_token": None}

        df = provider._transform_data(raw, "btc/usd", asset_class="crypto")

        assert df["symbol"].to_list() == ["BTC/USD", "BTC/USD"]

    def test_lowercase_stock_symbol_uppercased_in_output(self, provider):
        """The symbol column is uppercase for lowercase stock input."""
        raw = {"bars": STOCK_BARS, "next_page_token": None}

        df = provider._transform_data(raw, "aapl")

        assert df["symbol"].to_list() == ["AAPL", "AAPL"]


class TestAdjustmentParam:
    """Tests for the stock price-adjustment parameter."""

    def test_default_adjustment_is_raw(self, provider):
        """The default sends Alpaca's own unadjusted default explicitly."""
        _, params = provider._bars_request("AAPL", "2024-01-01", "2024-01-07", "daily", "stock")

        assert params["adjustment"] == "raw"

    def test_custom_adjustment_forwarded(self):
        """A custom adjustment is forwarded in the stock request params."""
        provider = AlpacaDataProvider(api_key="k", api_secret="s", adjustment="all")

        _, params = provider._bars_request("AAPL", "2024-01-01", "2024-01-07", "daily", "stock")

        assert params["adjustment"] == "all"

    def test_adjustment_not_sent_for_crypto(self, provider):
        """Crypto has no adjustment concept, so the param must not be sent."""
        _, params = provider._bars_request("BTC/USD", "2024-01-01", "2024-01-07", "daily", "crypto")

        assert "adjustment" not in params


class TestRetryAfterHeaders:
    """Tests for deriving retry_after from 429 rate-limit headers."""

    def test_retry_after_header_honored(self, provider):
        """An explicit Retry-After header becomes the retry_after value."""
        response = _page_response([], status=429, headers={"Retry-After": "7"})

        with pytest.raises(RateLimitError) as exc_info:
            provider._check_response_status(response, "AAPL")

        assert exc_info.value.retry_after == 7.0

    def test_rate_limit_reset_header_honored(self, provider):
        """X-RateLimit-Reset (epoch seconds) maps to a relative delay."""
        reset_at = time.time() + 30
        response = _page_response([], status=429, headers={"X-RateLimit-Reset": str(reset_at)})

        with pytest.raises(RateLimitError) as exc_info:
            provider._check_response_status(response, "AAPL")

        assert 0 < exc_info.value.retry_after <= 30.0

    def test_missing_headers_fall_back(self, provider):
        """Without usable headers the delay falls back to one rate-limit period."""
        response = _page_response([], status=429)

        with pytest.raises(RateLimitError) as exc_info:
            provider._check_response_status(response, "AAPL")

        assert exc_info.value.retry_after == 60.0


class TestRetryWiring:
    """Tests for the per-page retry policy."""

    def test_persistent_500_retries_three_times(self, provider):
        """A persistent 500 is retried 3 times, then NetworkError is raised."""
        response = _page_response([], status=500)

        with (
            patch.object(provider.session, "get", return_value=response) as mock_get,
            patch.object(provider.rate_limiter, "acquire") as mock_acquire,
        ):
            with pytest.raises(NetworkError):
                provider._fetch_raw_data("AAPL", "2024-01-01", "2024-01-05", "daily")

        assert mock_get.call_count == 3
        # Each attempt is a real request, so each acquires its own token.
        assert mock_acquire.call_count == 3

    def test_401_fails_fast_without_retry(self, provider):
        """An auth failure is not retried."""
        response = _page_response([], status=401)

        with (
            patch.object(provider.session, "get", return_value=response) as mock_get,
            patch.object(provider.rate_limiter, "acquire"),
        ):
            with pytest.raises(AuthenticationError):
                provider._fetch_raw_data("AAPL", "2024-01-01", "2024-01-05", "daily")

        assert mock_get.call_count == 1

    def test_page_retry_preserves_earlier_pages(self, provider):
        """A transient failure on page 2 retries only page 2, keeping page 1."""
        page1 = _page_response([STOCK_BARS[0]], "abc")
        flaky = _page_response([], status=500)
        page2 = _page_response([STOCK_BARS[1]], None)

        with (
            patch.object(provider.session, "get", side_effect=[page1, flaky, page2]) as mock_get,
            patch.object(provider.rate_limiter, "acquire"),
        ):
            data = provider._fetch_raw_data("AAPL", "2024-01-01", "2024-01-07", "daily")

        assert data["bars"] == [STOCK_BARS[0], STOCK_BARS[1]]
        # Page 1 was fetched once; only page 2 was retried.
        assert mock_get.call_count == 3
        assert mock_get.call_args_list[0].kwargs["params"] is not None
        assert mock_get.call_args_list[1].kwargs["params"]["page_token"] == "abc"
        assert mock_get.call_args_list[2].kwargs["params"]["page_token"] == "abc"


class TestErrorFallbacks:
    """Tests for transport-failure and transform-failure fallbacks."""

    def test_sync_transport_error_wrapped(self, provider):
        """A transport-level failure wraps into NetworkError with its cause."""
        with (
            patch.object(
                provider.session, "get", side_effect=httpx.ConnectError("boom")
            ) as mock_get,
            patch.object(provider.rate_limiter, "acquire"),
        ):
            with pytest.raises(NetworkError, match="Request failed") as exc_info:
                provider._fetch_raw_data("AAPL", "2024-01-01", "2024-01-05", "daily")

        assert isinstance(exc_info.value.__cause__, httpx.ConnectError)
        # NetworkError is transient, so the transport failure is retried.
        assert mock_get.call_count == 3

    @pytest.mark.asyncio
    async def test_async_transport_error_wrapped(self, provider):
        """The async transport failure wraps into NetworkError the same way."""
        with (
            patch.object(
                provider, "_aget", new=AsyncMock(side_effect=httpx.ConnectError("boom"))
            ) as mock_aget,
            patch.object(provider.rate_limiter, "acquire"),
        ):
            with pytest.raises(NetworkError, match="Request failed") as exc_info:
                await provider._fetch_raw_data_async("AAPL", "2024-01-01", "2024-01-05", "daily")

        assert isinstance(exc_info.value.__cause__, httpx.ConnectError)
        assert mock_aget.call_count == 3

    @pytest.mark.asyncio
    async def test_async_429_maps_to_rate_limit(self, provider):
        """An async 429 maps to RateLimitError after the per-page retries."""
        response = _page_response([], status=429, headers={"Retry-After": "0"})

        with (
            patch.object(provider, "_aget", new=AsyncMock(return_value=response)) as mock_aget,
            patch.object(provider.rate_limiter, "acquire"),
        ):
            with pytest.raises(RateLimitError):
                await provider._fetch_raw_data_async("AAPL", "2024-01-01", "2024-01-05", "daily")

        assert mock_aget.call_count == 3

    def test_malformed_bars_raise_data_validation(self, provider):
        """Untransformable bars raise DataValidationError naming the symbol."""
        raw = {
            "bars": [{"t": "not-a-timestamp", "o": 1, "h": 1, "l": 1, "c": 1, "v": 1}],
            "next_page_token": None,
        }

        with pytest.raises(DataValidationError, match="AAPL"):
            provider._transform_data(raw, "AAPL")


class TestFetchOhlcvAsync:
    """Tests for the public async fetch path."""

    @pytest.mark.asyncio
    async def test_fetch_ohlcv_async_stock(self, provider):
        """The async public path returns the canonical schema for stocks."""
        response = _page_response(STOCK_BARS)

        with (
            patch.object(provider, "_aget", new=AsyncMock(return_value=response)),
            patch.object(provider.rate_limiter, "acquire"),
        ):
            df = await provider.fetch_ohlcv_async("AAPL", "2024-01-01", "2024-01-05")

        assert df.columns == ["timestamp", "symbol", "open", "high", "low", "close", "volume"]
        assert df.height == 2
        assert df["symbol"].to_list() == ["AAPL", "AAPL"]
        timestamps = df["timestamp"].to_list()
        assert timestamps == sorted(timestamps)

    @pytest.mark.asyncio
    async def test_fetch_ohlcv_async_crypto(self, provider):
        """The async public path handles the crypto dict shape and casing."""
        response = _page_response({"BTC/USD": CRYPTO_BARS})

        with (
            patch.object(provider, "_aget", new=AsyncMock(return_value=response)),
            patch.object(provider.rate_limiter, "acquire"),
        ):
            df = await provider.fetch_ohlcv_async(
                "BTC/USD", "2024-01-01T00:00:00Z", "2024-01-07T00:00:00Z", "daily"
            )

        assert df.height == 2
        assert df["symbol"].to_list() == ["BTC/USD", "BTC/USD"]

    @pytest.mark.asyncio
    async def test_fetch_ohlcv_async_validates_before_fetching(self, provider):
        """Invalid bounds raise ValueError before any request goes out."""
        with patch.object(provider, "_aget", new=AsyncMock()) as mock_aget:
            with pytest.raises(ValueError, match="YYYY-MM-DD or RFC-3339"):
                await provider.fetch_ohlcv_async("AAPL", "bad", "2024-01-05")

        mock_aget.assert_not_called()

    @pytest.mark.asyncio
    async def test_fetch_ohlcv_async_open_breaker_blocks_fetch(self, provider):
        """An OPEN breaker refuses the async fetch before any request."""
        provider.init_circuit_breaker()
        provider.circuit_breaker.state = "OPEN"
        provider.circuit_breaker.last_failure_time = time.time()

        with patch.object(provider, "_aget", new=AsyncMock()) as mock_aget:
            with pytest.raises(CircuitBreakerOpenError):
                await provider.fetch_ohlcv_async("AAPL", "2024-01-01", "2024-01-05")

        mock_aget.assert_not_called()

    @pytest.mark.asyncio
    async def test_fetch_ohlcv_async_failure_counts_toward_breaker(self, provider):
        """An async fetch failure increments the breaker's failure count."""
        response = _page_response([], status=404)

        with (
            patch.object(provider, "_aget", new=AsyncMock(return_value=response)),
            patch.object(provider.rate_limiter, "acquire"),
        ):
            with pytest.raises(DataNotAvailableError):
                await provider.fetch_ohlcv_async("AAPL", "2024-01-01", "2024-01-05")

        assert provider.circuit_breaker.failure_count == 1


class TestMalformedRateLimitHeaders:
    """Tests for unparsable rate-limit header values."""

    def test_malformed_rate_limit_headers_fall_back(self, provider):
        """Unparsable header values fall through to the default delay."""
        response = _page_response(
            [],
            status=429,
            headers={"Retry-After": "soon", "X-RateLimit-Reset": "later"},
        )

        with pytest.raises(RateLimitError) as exc_info:
            provider._check_response_status(response, "AAPL")

        assert exc_info.value.retry_after == 60.0


class TestAsyncBreakerWiring:
    """Tests for the async circuit-breaker mixin plumbing."""

    @pytest.mark.asyncio
    async def test_with_circuit_breaker_async_lazy_init(self, provider):
        """The async wrapper initializes the breaker when none exists yet."""
        del provider.circuit_breaker

        async def _ok():
            return 42

        result = await provider._with_circuit_breaker_async(_ok)

        assert result == 42
        assert provider.circuit_breaker.state == "CLOSED"


class TestRegistryCaching:
    """Tests for the lazy provider-class registry."""

    def test_provider_classes_cached(self):
        """Repeated lookups return the same cached registry mapping."""
        from ml4t.data.managers.provider_manager import ProviderManager

        first = ProviderManager._get_provider_classes()
        second = ProviderManager._get_provider_classes()

        assert first is second
        assert "alpaca" in first
