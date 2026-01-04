"""Tests for CoinGecko provider module."""

from datetime import datetime
from unittest.mock import MagicMock, patch

import httpx
import polars as pl
import pytest

from ml4t.data.core.exceptions import (
    DataNotAvailableError,
    NetworkError,
    RateLimitError,
    SymbolNotFoundError,
)
from ml4t.data.providers.coingecko import CoinGeckoProvider


class TestCoinGeckoProviderInit:
    """Tests for provider initialization."""

    def test_default_init(self):
        """Test default initialization."""
        with patch.dict("os.environ", {}, clear=True):
            provider = CoinGeckoProvider()

            assert provider.name == "coingecko"
            assert provider.api_key is None
            assert provider.use_pro is False
            assert provider.base_url == "https://api.coingecko.com/api/v3"

    def test_init_with_api_key(self):
        """Test initialization with API key."""
        provider = CoinGeckoProvider(api_key="test_key")

        assert provider.api_key == "test_key"
        assert provider.use_pro is False

    def test_init_with_env_api_key(self):
        """Test initialization with API key from environment."""
        with patch.dict("os.environ", {"COINGECKO_API_KEY": "env_key"}):
            provider = CoinGeckoProvider()

            assert provider.api_key == "env_key"

    def test_init_pro_tier(self):
        """Test initialization with Pro tier."""
        provider = CoinGeckoProvider(use_pro=True)

        assert provider.use_pro is True
        assert provider.base_url == "https://pro-api.coingecko.com/api/v3"

    def test_init_custom_rate_limit(self):
        """Test initialization with custom rate limit."""
        provider = CoinGeckoProvider(rate_limit=(100, 60.0))

        # Provider should be initialized successfully
        assert provider is not None


class TestNameProperty:
    """Tests for name property."""

    def test_name_returns_coingecko(self):
        """Test name property returns correct value."""
        provider = CoinGeckoProvider()
        assert provider.name == "coingecko"


class TestSymbolToId:
    """Tests for symbol_to_id method."""

    @pytest.fixture
    def provider(self):
        """Create provider instance."""
        return CoinGeckoProvider()

    def test_btc_to_bitcoin(self, provider):
        """Test BTC converts to bitcoin."""
        assert provider.symbol_to_id("BTC") == "bitcoin"

    def test_eth_to_ethereum(self, provider):
        """Test ETH converts to ethereum."""
        assert provider.symbol_to_id("ETH") == "ethereum"

    def test_lowercase_btc(self, provider):
        """Test lowercase btc still works."""
        assert provider.symbol_to_id("btc") == "bitcoin"

    def test_already_lowercase_id(self, provider):
        """Test already lowercase ID passes through."""
        assert provider.symbol_to_id("bitcoin") == "bitcoin"

    def test_unknown_symbol_lowercase(self, provider):
        """Test unknown symbol is lowercased."""
        assert provider.symbol_to_id("NEWCOIN") == "newcoin"

    def test_sol_to_solana(self, provider):
        """Test SOL converts to solana."""
        assert provider.symbol_to_id("SOL") == "solana"

    def test_doge_to_dogecoin(self, provider):
        """Test DOGE converts to dogecoin."""
        assert provider.symbol_to_id("DOGE") == "dogecoin"


class TestSymbolMap:
    """Tests for SYMBOL_TO_ID_MAP constant."""

    def test_map_has_major_coins(self):
        """Test map contains major cryptocurrency symbols."""
        assert "BTC" in CoinGeckoProvider.SYMBOL_TO_ID_MAP
        assert "ETH" in CoinGeckoProvider.SYMBOL_TO_ID_MAP
        assert "SOL" in CoinGeckoProvider.SYMBOL_TO_ID_MAP
        assert "DOGE" in CoinGeckoProvider.SYMBOL_TO_ID_MAP

    def test_map_values_are_lowercase(self):
        """Test all map values are lowercase (CoinGecko IDs)."""
        for coin_id in CoinGeckoProvider.SYMBOL_TO_ID_MAP.values():
            assert coin_id.islower()


class TestRoundToValidDays:
    """Tests for _round_to_valid_days method."""

    @pytest.fixture
    def provider(self):
        """Create provider instance."""
        return CoinGeckoProvider()

    def test_round_1_day(self, provider):
        """Test 1 day returns 1."""
        assert provider._round_to_valid_days(1) == 1

    def test_round_3_days(self, provider):
        """Test 3 days rounds up to 7."""
        assert provider._round_to_valid_days(3) == 7

    def test_round_7_days(self, provider):
        """Test 7 days returns 7."""
        assert provider._round_to_valid_days(7) == 7

    def test_round_10_days(self, provider):
        """Test 10 days rounds up to 14."""
        assert provider._round_to_valid_days(10) == 14

    def test_round_20_days(self, provider):
        """Test 20 days rounds up to 30."""
        assert provider._round_to_valid_days(20) == 30

    def test_round_60_days(self, provider):
        """Test 60 days rounds up to 90."""
        assert provider._round_to_valid_days(60) == 90

    def test_round_100_days(self, provider):
        """Test 100 days rounds up to 180."""
        assert provider._round_to_valid_days(100) == 180

    def test_round_200_days(self, provider):
        """Test 200 days rounds up to 365."""
        assert provider._round_to_valid_days(200) == 365

    def test_round_365_days(self, provider):
        """Test 365 days returns 365."""
        assert provider._round_to_valid_days(365) == 365

    def test_round_over_365_returns_max(self, provider):
        """Test over 365 days returns 'max'."""
        assert provider._round_to_valid_days(400) == "max"
        assert provider._round_to_valid_days(1000) == "max"


class TestCreateEmptyDataframe:
    """Tests for _create_empty_dataframe method."""

    def test_empty_dataframe_schema(self):
        """Test empty DataFrame has correct schema."""
        provider = CoinGeckoProvider()
        df = provider._create_empty_dataframe()

        assert df.is_empty()
        assert "timestamp" in df.columns
        assert "open" in df.columns
        assert "high" in df.columns
        assert "low" in df.columns
        assert "close" in df.columns
        assert "volume" in df.columns
        assert "symbol" in df.columns


class TestFetchOhlc:
    """Tests for _fetch_ohlc method."""

    @pytest.fixture
    def provider(self):
        """Create provider instance."""
        return CoinGeckoProvider()

    def test_fetch_ohlc_success(self, provider):
        """Test successful OHLC fetch."""
        mock_response = MagicMock()
        mock_response.json.return_value = [
            [1704067200000, 42000.0, 42500.0, 41800.0, 42300.0],
            [1704153600000, 42300.0, 42800.0, 42000.0, 42600.0],
        ]

        with patch.object(provider.session, "get", return_value=mock_response):
            df = provider._fetch_ohlc("bitcoin", 7)

        assert len(df) == 2
        assert "timestamp" in df.columns
        assert "open" in df.columns
        assert "close" in df.columns
        assert "volume" in df.columns
        assert df["volume"][0] == 0.0  # Volume is placeholder

    def test_fetch_ohlc_empty_response(self, provider):
        """Test empty OHLC response."""
        mock_response = MagicMock()
        mock_response.json.return_value = []

        with patch.object(provider.session, "get", return_value=mock_response):
            df = provider._fetch_ohlc("bitcoin", 7)

        assert df.is_empty()

    def test_fetch_ohlc_rate_limit_error(self, provider):
        """Test rate limit (429) raises RateLimitError."""
        mock_response = MagicMock()
        mock_response.status_code = 429
        error = httpx.HTTPStatusError("Rate limited", request=MagicMock(), response=mock_response)

        with patch.object(provider.session, "get", side_effect=error):
            with pytest.raises(RateLimitError):
                provider._fetch_ohlc("bitcoin", 7)

    def test_fetch_ohlc_not_found_error(self, provider):
        """Test 404 raises SymbolNotFoundError."""
        mock_response = MagicMock()
        mock_response.status_code = 404
        error = httpx.HTTPStatusError("Not found", request=MagicMock(), response=mock_response)

        with patch.object(provider.session, "get", side_effect=error):
            with pytest.raises(SymbolNotFoundError):
                provider._fetch_ohlc("invalid_coin", 7)

    def test_fetch_ohlc_other_http_error(self, provider):
        """Test other HTTP errors raise DataNotAvailableError."""
        mock_response = MagicMock()
        mock_response.status_code = 500
        error = httpx.HTTPStatusError("Server error", request=MagicMock(), response=mock_response)

        with patch.object(provider.session, "get", side_effect=error):
            with pytest.raises(DataNotAvailableError):
                provider._fetch_ohlc("bitcoin", 7)

    def test_fetch_ohlc_network_error(self, provider):
        """Test network error raises NetworkError."""
        error = httpx.RequestError("Connection failed", request=MagicMock())

        with patch.object(provider.session, "get", side_effect=error):
            with pytest.raises(NetworkError):
                provider._fetch_ohlc("bitcoin", 7)


class TestFetchAndTransformData:
    """Tests for _fetch_and_transform_data method."""

    @pytest.fixture
    def provider(self):
        """Create provider instance."""
        return CoinGeckoProvider()

    def test_fetch_transform_success(self, provider):
        """Test successful fetch and transform."""
        mock_ohlc_df = pl.DataFrame(
            {
                "timestamp": [
                    datetime(2024, 1, 1, 12, 0),
                    datetime(2024, 1, 2, 12, 0),
                ],
                "open": [42000.0, 42300.0],
                "high": [42500.0, 42800.0],
                "low": [41800.0, 42000.0],
                "close": [42300.0, 42600.0],
                "volume": [0.0, 0.0],
            }
        )

        with patch.object(provider, "_fetch_ohlc", return_value=mock_ohlc_df):
            df = provider._fetch_and_transform_data("BTC", "2024-01-01", "2024-01-02", "daily")

        assert len(df) == 2
        assert "symbol" in df.columns
        assert df["symbol"][0] == "BTC"

    def test_fetch_transform_empty_result(self, provider):
        """Test empty result returns empty DataFrame."""
        empty_df = provider._create_empty_dataframe().drop("symbol")

        with patch.object(provider, "_fetch_ohlc", return_value=empty_df):
            df = provider._fetch_and_transform_data("BTC", "2024-01-01", "2024-01-02", "daily")

        assert df.is_empty()

    def test_fetch_transform_filters_date_range(self, provider):
        """Test data is filtered to requested date range."""
        mock_ohlc_df = pl.DataFrame(
            {
                "timestamp": [
                    datetime(2023, 12, 31, 12, 0),  # Before range
                    datetime(2024, 1, 1, 12, 0),  # In range
                    datetime(2024, 1, 2, 12, 0),  # In range
                    datetime(2024, 1, 3, 12, 0),  # After range
                ],
                "open": [41000.0, 42000.0, 42300.0, 42600.0],
                "high": [41500.0, 42500.0, 42800.0, 43000.0],
                "low": [40800.0, 41800.0, 42000.0, 42400.0],
                "close": [41300.0, 42300.0, 42600.0, 42900.0],
                "volume": [0.0, 0.0, 0.0, 0.0],
            }
        )

        with patch.object(provider, "_fetch_ohlc", return_value=mock_ohlc_df):
            df = provider._fetch_and_transform_data("BTC", "2024-01-01", "2024-01-02", "daily")

        assert len(df) == 2  # Only dates in range


class TestGetCoinList:
    """Tests for get_coin_list method."""

    @pytest.fixture
    def provider(self):
        """Create provider instance."""
        return CoinGeckoProvider()

    def test_get_coin_list_success(self, provider):
        """Test successful coin list fetch."""
        mock_response = MagicMock()
        mock_response.json.return_value = [
            {"id": "bitcoin", "symbol": "btc", "name": "Bitcoin"},
            {"id": "ethereum", "symbol": "eth", "name": "Ethereum"},
        ]

        with patch.object(provider.session, "get", return_value=mock_response):
            df = provider.get_coin_list()

        assert len(df) == 2
        assert "id" in df.columns
        assert "symbol" in df.columns
        assert "name" in df.columns

    def test_get_coin_list_empty(self, provider):
        """Test empty coin list."""
        mock_response = MagicMock()
        mock_response.json.return_value = []

        with patch.object(provider.session, "get", return_value=mock_response):
            df = provider.get_coin_list()

        assert df.is_empty()

    def test_get_coin_list_rate_limit_error(self, provider):
        """Test rate limit raises RateLimitError."""
        mock_response = MagicMock()
        mock_response.status_code = 429
        error = httpx.HTTPStatusError("Rate limited", request=MagicMock(), response=mock_response)

        with patch.object(provider.session, "get", side_effect=error):
            with pytest.raises(RateLimitError):
                provider.get_coin_list()

    def test_get_coin_list_http_error(self, provider):
        """Test HTTP error raises DataNotAvailableError."""
        mock_response = MagicMock()
        mock_response.status_code = 500
        error = httpx.HTTPStatusError("Server error", request=MagicMock(), response=mock_response)

        with patch.object(provider.session, "get", side_effect=error):
            with pytest.raises(DataNotAvailableError):
                provider.get_coin_list()

    def test_get_coin_list_network_error(self, provider):
        """Test network error raises NetworkError."""
        error = httpx.RequestError("Connection failed", request=MagicMock())

        with patch.object(provider.session, "get", side_effect=error):
            with pytest.raises(NetworkError):
                provider.get_coin_list()


class TestGetPrice:
    """Tests for get_price method."""

    @pytest.fixture
    def provider(self):
        """Create provider instance."""
        return CoinGeckoProvider()

    def test_get_price_success(self, provider):
        """Test successful price fetch."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "bitcoin": {"usd": 42000.0},
            "ethereum": {"usd": 2200.0},
        }

        with patch.object(provider.session, "get", return_value=mock_response):
            df = provider.get_price(["bitcoin", "ethereum"])

        assert len(df) == 2
        assert "coin_id" in df.columns
        assert "currency" in df.columns
        assert "price" in df.columns

    def test_get_price_multiple_currencies(self, provider):
        """Test price fetch with multiple currencies."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "bitcoin": {"usd": 42000.0, "eur": 38000.0},
        }

        with patch.object(provider.session, "get", return_value=mock_response):
            df = provider.get_price(["bitcoin"], vs_currencies=["usd", "eur"])

        assert len(df) == 2  # One row per currency

    def test_get_price_empty(self, provider):
        """Test empty price response."""
        mock_response = MagicMock()
        mock_response.json.return_value = {}

        with patch.object(provider.session, "get", return_value=mock_response):
            df = provider.get_price(["bitcoin"])

        assert df.is_empty()

    def test_get_price_rate_limit_error(self, provider):
        """Test rate limit raises RateLimitError."""
        mock_response = MagicMock()
        mock_response.status_code = 429
        error = httpx.HTTPStatusError("Rate limited", request=MagicMock(), response=mock_response)

        with patch.object(provider.session, "get", side_effect=error):
            with pytest.raises(RateLimitError):
                provider.get_price(["bitcoin"])

    def test_get_price_http_error(self, provider):
        """Test HTTP error raises DataNotAvailableError."""
        mock_response = MagicMock()
        mock_response.status_code = 500
        error = httpx.HTTPStatusError("Server error", request=MagicMock(), response=mock_response)

        with patch.object(provider.session, "get", side_effect=error):
            with pytest.raises(DataNotAvailableError):
                provider.get_price(["bitcoin"])

    def test_get_price_network_error(self, provider):
        """Test network error raises NetworkError."""
        error = httpx.RequestError("Connection failed", request=MagicMock())

        with patch.object(provider.session, "get", side_effect=error):
            with pytest.raises(NetworkError):
                provider.get_price(["bitcoin"])


class TestIntegration:
    """Integration tests for CoinGeckoProvider."""

    def test_full_fetch_workflow_with_mocks(self):
        """Test complete fetch workflow with mocked responses."""
        provider = CoinGeckoProvider()

        mock_ohlc_df = pl.DataFrame(
            {
                "timestamp": [datetime(2024, 1, 1, 12, 0)],
                "open": [42000.0],
                "high": [42500.0],
                "low": [41800.0],
                "close": [42300.0],
                "volume": [0.0],
            }
        )

        with patch.object(provider, "_fetch_ohlc", return_value=mock_ohlc_df):
            df = provider.fetch_ohlcv("BTC", "2024-01-01", "2024-01-01", "daily")

        assert not df.is_empty()
        assert "timestamp" in df.columns
        assert "close" in df.columns
        assert "symbol" in df.columns

    def test_fetch_with_symbol_conversion(self):
        """Test fetch converts symbol to coin ID."""
        provider = CoinGeckoProvider()

        mock_ohlc_df = pl.DataFrame(
            {
                "timestamp": [datetime(2024, 1, 1, 12, 0)],
                "open": [42000.0],
                "high": [42500.0],
                "low": [41800.0],
                "close": [42300.0],
                "volume": [0.0],
            }
        )

        with patch.object(provider, "_fetch_ohlc", return_value=mock_ohlc_df) as mock_fetch:
            provider.fetch_ohlcv("BTC", "2024-01-01", "2024-01-01", "daily")

        # Verify bitcoin (not BTC) was passed to _fetch_ohlc
        call_args = mock_fetch.call_args
        assert call_args[0][0] == "bitcoin"
