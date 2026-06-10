"""Tests for Alpaca data provider module."""

from unittest.mock import patch

import pytest

from ml4t.data.core.exceptions import AuthenticationError, DataValidationError
from ml4t.data.providers.alpaca import AlpacaDataProvider
from ml4t.data.providers.protocols import OHLCVProvider


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
