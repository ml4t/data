"""Unit tests for Polymarket provider (mock-based).

These tests verify the Polymarket provider logic without making real API calls.
"""

from datetime import datetime
from unittest.mock import MagicMock, patch

import polars as pl
import pytest

from ml4t.data.core.exceptions import (
    DataValidationError,
    NetworkError,
    RateLimitError,
    SymbolNotFoundError,
)
from ml4t.data.providers.polymarket import PolymarketProvider


class TestPolymarketProviderInit:
    """Test provider initialization."""

    def test_default_initialization(self):
        """Test provider initializes with defaults."""
        provider = PolymarketProvider()
        assert provider.name == "polymarket"
        provider.close()

    def test_custom_rate_limit(self):
        """Test provider with custom rate limit."""
        provider = PolymarketProvider(rate_limit=(10, 60.0))
        assert provider.name == "polymarket"
        provider.close()

    def test_market_cache_initialized(self):
        """Test that market cache is initialized."""
        provider = PolymarketProvider()
        assert hasattr(provider, "_market_cache")
        assert provider._market_cache == {}
        provider.close()


class TestSymbolIdentification:
    """Test symbol type identification."""

    @pytest.fixture
    def provider(self):
        """Create provider for tests."""
        provider = PolymarketProvider()
        yield provider
        provider.close()

    def test_is_token_id_valid(self, provider):
        """Test valid token ID detection."""
        # Token IDs are long numeric strings (>15 digits)
        assert provider._is_token_id("12345678901234567890")
        assert provider._is_token_id("98765432109876543210")

    def test_is_token_id_invalid(self, provider):
        """Test non-token ID detection."""
        assert not provider._is_token_id("1234567890")  # Too short
        assert not provider._is_token_id("abc123")  # Not numeric
        assert not provider._is_token_id("0xabcd1234")  # Hex
        assert not provider._is_token_id("will-btc-reach-100k")  # Slug

    def test_is_condition_id_valid(self, provider):
        """Test valid condition ID detection."""
        assert provider._is_condition_id("0xabcd1234")
        assert provider._is_condition_id("0x123456789abcdef")

    def test_is_condition_id_invalid(self, provider):
        """Test non-condition ID detection."""
        assert not provider._is_condition_id("12345678901234567890")
        assert not provider._is_condition_id("will-btc-reach-100k")
        assert not provider._is_condition_id("abcd1234")  # No 0x prefix


class TestIntervalMapping:
    """Test frequency/interval mapping."""

    @pytest.fixture
    def provider(self):
        """Create provider for tests."""
        provider = PolymarketProvider()
        yield provider
        provider.close()

    def test_interval_map_daily(self, provider):
        """Test daily interval mapping."""
        assert provider.INTERVAL_MAP["daily"] == "1d"
        assert provider.INTERVAL_MAP["day"] == "1d"
        assert provider.INTERVAL_MAP["1d"] == "1d"

    def test_interval_map_hourly(self, provider):
        """Test hourly interval mapping."""
        assert provider.INTERVAL_MAP["hourly"] == "1h"
        assert provider.INTERVAL_MAP["hour"] == "1h"
        assert provider.INTERVAL_MAP["1h"] == "1h"

    def test_interval_map_minute(self, provider):
        """Test minute interval mapping."""
        assert provider.INTERVAL_MAP["minute"] == "1m"
        assert provider.INTERVAL_MAP["1m"] == "1m"

    def test_interval_map_weekly(self, provider):
        """Test weekly interval mapping."""
        assert provider.INTERVAL_MAP["weekly"] == "1w"
        assert provider.INTERVAL_MAP["week"] == "1w"
        assert provider.INTERVAL_MAP["1w"] == "1w"


class TestSymbolResolution:
    """Test symbol resolution logic."""

    @pytest.fixture
    def provider(self):
        """Create provider for tests."""
        provider = PolymarketProvider()
        yield provider
        provider.close()

    def test_resolve_token_id_returns_as_is(self, provider):
        """Test that token ID is returned without API call."""
        token_id = "12345678901234567890"
        result = provider.resolve_symbol(token_id)
        assert result == token_id

    @patch.object(PolymarketProvider, "_get_market_by_slug")
    def test_resolve_slug_to_yes_token(self, mock_get_market, provider):
        """Test resolving slug to YES token ID."""
        mock_get_market.return_value = {
            "id": "0xtest",
            "slug": "test-market",
            "tokens": [
                {"token_id": "yes_token_123456789", "outcome": "Yes", "price": 0.65},
                {"token_id": "no_token_987654321", "outcome": "No", "price": 0.35},
            ],
        }

        result = provider.resolve_symbol("test-market", outcome="yes")
        assert result == "yes_token_123456789"

    @patch.object(PolymarketProvider, "_get_market_by_slug")
    def test_resolve_slug_to_no_token(self, mock_get_market, provider):
        """Test resolving slug to NO token ID."""
        mock_get_market.return_value = {
            "id": "0xtest",
            "slug": "test-market",
            "tokens": [
                {"token_id": "yes_token_123456789", "outcome": "Yes", "price": 0.65},
                {"token_id": "no_token_987654321", "outcome": "No", "price": 0.35},
            ],
        }

        result = provider.resolve_symbol("test-market", outcome="no")
        assert result == "no_token_987654321"

    @patch.object(PolymarketProvider, "_get_market_by_condition")
    def test_resolve_condition_id_to_token(self, mock_get_market, provider):
        """Test resolving condition ID to token ID."""
        mock_get_market.return_value = {
            "id": "0xabcd1234",
            "slug": "test-market",
            "tokens": [
                {"token_id": "yes_token_123456789", "outcome": "Yes", "price": 0.65},
                {"token_id": "no_token_987654321", "outcome": "No", "price": 0.35},
            ],
        }

        result = provider.resolve_symbol("0xabcd1234", outcome="yes")
        assert result == "yes_token_123456789"

    def test_resolve_invalid_outcome_raises_error(self, provider):
        """Test that invalid outcome raises error."""
        with pytest.raises(DataValidationError) as exc_info:
            provider.resolve_symbol("test-market", outcome="maybe")

        assert "Invalid outcome" in str(exc_info.value)


class TestOHLCAggregation:
    """Test OHLC aggregation from price-only data."""

    @pytest.fixture
    def provider(self):
        """Create provider for tests."""
        provider = PolymarketProvider()
        yield provider
        provider.close()

    def test_aggregate_single_price_point(self, provider):
        """Test aggregation with single price point."""
        price_data = [{"t": 1704067200, "p": 0.45}]

        df = provider._aggregate_to_ohlc(price_data, "TEST", "daily")

        assert len(df) == 1
        assert df["open"][0] == 0.45
        assert df["high"][0] == 0.45
        assert df["low"][0] == 0.45
        assert df["close"][0] == 0.45

    def test_aggregate_multiple_price_points(self, provider):
        """Test aggregation with multiple price points in same day."""
        # All in same day (Jan 1, 2024)
        price_data = [
            {"t": 1704067200, "p": 0.45},  # 00:00
            {"t": 1704070800, "p": 0.50},  # 01:00
            {"t": 1704074400, "p": 0.42},  # 02:00
            {"t": 1704078000, "p": 0.48},  # 03:00
        ]

        df = provider._aggregate_to_ohlc(price_data, "TEST", "daily")

        assert len(df) == 1
        assert df["open"][0] == 0.45  # First price
        assert df["high"][0] == 0.50  # Max price
        assert df["low"][0] == 0.42  # Min price
        assert df["close"][0] == 0.48  # Last price

    def test_aggregate_empty_data(self, provider):
        """Test aggregation with empty data."""
        df = provider._aggregate_to_ohlc([], "TEST", "daily")

        assert df.is_empty()
        assert "timestamp" in df.columns
        assert "open" in df.columns

    def test_aggregate_preserves_symbol(self, provider):
        """Test that symbol is preserved in output."""
        price_data = [{"t": 1704067200, "p": 0.45}]

        df = provider._aggregate_to_ohlc(price_data, "my-test-market:YES", "daily")

        assert "symbol" in df.columns
        assert df["symbol"][0] == "MY-TEST-MARKET:YES"

    def test_aggregate_hourly(self, provider):
        """Test hourly aggregation."""
        # Two hours of data
        price_data = [
            {"t": 1704067200, "p": 0.45},  # 00:00
            {"t": 1704069000, "p": 0.47},  # 00:30
            {"t": 1704070800, "p": 0.50},  # 01:00
            {"t": 1704072600, "p": 0.48},  # 01:30
        ]

        df = provider._aggregate_to_ohlc(price_data, "TEST", "hourly")

        # Should have 2 hourly bars
        assert len(df) == 2


class TestFetchOHLCV:
    """Test fetch_ohlcv method."""

    @pytest.fixture
    def provider(self):
        """Create provider for tests."""
        provider = PolymarketProvider()
        yield provider
        provider.close()

    @patch.object(PolymarketProvider, "_fetch_price_history")
    def test_fetch_ohlcv_with_token_id(self, mock_fetch, provider):
        """Test fetching with direct token ID."""
        mock_fetch.return_value = [
            {"t": 1704067200, "p": 0.45},
            {"t": 1704153600, "p": 0.48},
        ]

        df = provider.fetch_ohlcv(
            "12345678901234567890", "2024-01-01", "2024-01-02", frequency="daily"
        )

        assert not df.is_empty()
        assert "timestamp" in df.columns
        mock_fetch.assert_called_once()

    @patch.object(PolymarketProvider, "resolve_symbol")
    @patch.object(PolymarketProvider, "_fetch_price_history")
    def test_fetch_ohlcv_with_slug(self, mock_fetch, mock_resolve, provider):
        """Test fetching with slug resolves to token ID."""
        mock_resolve.return_value = "resolved_token_123456"
        mock_fetch.return_value = [
            {"t": 1704067200, "p": 0.45},
        ]

        df = provider.fetch_ohlcv(
            "test-market-slug", "2024-01-01", "2024-01-02", frequency="daily", outcome="yes"
        )

        mock_resolve.assert_called_once_with("test-market-slug", "yes")
        assert not df.is_empty()

    @patch.object(PolymarketProvider, "_fetch_price_history")
    def test_fetch_ohlcv_invalid_frequency(self, mock_fetch, provider):
        """Test that invalid frequency raises error."""
        with pytest.raises(DataValidationError) as exc_info:
            provider.fetch_ohlcv(
                "12345678901234567890", "2024-01-01", "2024-01-02", frequency="invalid"
            )

        assert "Unsupported frequency" in str(exc_info.value)

    @patch.object(PolymarketProvider, "_fetch_price_history")
    def test_fetch_ohlcv_empty_response(self, mock_fetch, provider):
        """Test handling empty API response."""
        mock_fetch.return_value = []

        df = provider.fetch_ohlcv(
            "12345678901234567890", "2024-01-01", "2024-01-02", frequency="daily"
        )

        assert df.is_empty()


class TestFetchBothOutcomes:
    """Test fetch_both_outcomes method."""

    @pytest.fixture
    def provider(self):
        """Create provider for tests."""
        provider = PolymarketProvider()
        yield provider
        provider.close()

    @patch.object(PolymarketProvider, "fetch_ohlcv")
    def test_fetch_both_outcomes_success(self, mock_fetch, provider):
        """Test fetching both YES and NO outcomes."""
        # Return different data for each outcome
        mock_fetch.side_effect = [
            pl.DataFrame(
                {
                    "timestamp": [datetime(2024, 1, 1)],
                    "symbol": ["TEST:YES"],
                    "open": [0.65],
                    "high": [0.70],
                    "low": [0.60],
                    "close": [0.68],
                    "volume": [100.0],
                }
            ),
            pl.DataFrame(
                {
                    "timestamp": [datetime(2024, 1, 1)],
                    "symbol": ["TEST:NO"],
                    "open": [0.35],
                    "high": [0.40],
                    "low": [0.30],
                    "close": [0.32],
                    "volume": [100.0],
                }
            ),
        ]

        df = provider.fetch_both_outcomes("test-market", "2024-01-01", "2024-01-02")

        assert len(df) == 2
        assert "TEST:YES" in df["symbol"].to_list()
        assert "TEST:NO" in df["symbol"].to_list()

    def test_fetch_both_outcomes_rejects_token_id(self, provider):
        """Test that token_id is rejected for both outcomes."""
        with pytest.raises(DataValidationError) as exc_info:
            provider.fetch_both_outcomes("12345678901234567890", "2024-01-01", "2024-01-02")

        assert "Cannot fetch both outcomes from token_id" in str(exc_info.value)


class TestListMarkets:
    """Test list_markets method."""

    @pytest.fixture
    def provider(self):
        """Create provider for tests."""
        provider = PolymarketProvider()
        yield provider
        provider.close()

    def test_list_markets_success(self, provider):
        """Test successful market listing."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = [
            {
                "id": "0x123",
                "question": "Will Bitcoin reach $100K?",
                "slug": "will-bitcoin-reach-100k",
                "tokens": [],
            },
            {
                "id": "0x456",
                "question": "Will ETH reach $5K?",
                "slug": "will-eth-reach-5k",
                "tokens": [],
            },
        ]
        provider.session = MagicMock()
        provider.session.get.return_value = mock_response

        markets = provider.list_markets(active=True, limit=10)

        assert len(markets) == 2
        assert markets[0]["slug"] == "will-bitcoin-reach-100k"

    def test_list_markets_with_category_filter(self, provider):
        """Test market listing with category filter."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = [
            {"id": "0x123", "slug": "btc-100k", "category": "Crypto"},
            {"id": "0x456", "slug": "election-2024", "category": "Politics"},
        ]
        provider.session = MagicMock()
        provider.session.get.return_value = mock_response

        markets = provider.list_markets(category="Crypto")

        assert len(markets) == 1
        assert markets[0]["category"] == "Crypto"

    def test_list_markets_rate_limit_error(self, provider):
        """Test rate limit error handling."""
        mock_response = MagicMock()
        mock_response.status_code = 429
        provider.session = MagicMock()
        provider.session.get.return_value = mock_response

        with pytest.raises(RateLimitError):
            provider.list_markets()


class TestSearchMarkets:
    """Test search_markets method."""

    @pytest.fixture
    def provider(self):
        """Create provider for tests."""
        provider = PolymarketProvider()
        yield provider
        provider.close()

    @patch.object(PolymarketProvider, "list_markets")
    def test_search_markets_by_question(self, mock_list, provider):
        """Test searching markets by question text."""
        mock_list.return_value = [
            {"question": "Will Bitcoin reach $100K?", "slug": "btc-100k"},
            {"question": "Will ETH reach $5K?", "slug": "eth-5k"},
            {"question": "Will Dogecoin reach $1?", "slug": "doge-1"},
        ]

        results = provider.search_markets("bitcoin")

        assert len(results) == 1
        assert "bitcoin" in results[0]["question"].lower()

    @patch.object(PolymarketProvider, "list_markets")
    def test_search_markets_by_slug(self, mock_list, provider):
        """Test searching markets by slug."""
        mock_list.return_value = [
            {"question": "Question 1", "slug": "bitcoin-100k"},
            {"question": "Question 2", "slug": "eth-5k"},
        ]

        results = provider.search_markets("bitcoin")

        assert len(results) == 1
        assert "bitcoin" in results[0]["slug"]

    @patch.object(PolymarketProvider, "list_markets")
    def test_search_markets_limit(self, mock_list, provider):
        """Test search respects limit."""
        mock_list.return_value = [
            {"question": f"Bitcoin market {i}", "slug": f"btc-{i}"} for i in range(50)
        ]

        results = provider.search_markets("bitcoin", limit=5)

        assert len(results) == 5


class TestGetMarketMetadata:
    """Test get_market_metadata method."""

    @pytest.fixture
    def provider(self):
        """Create provider for tests."""
        provider = PolymarketProvider()
        yield provider
        provider.close()

    @patch.object(PolymarketProvider, "_get_market_by_slug")
    def test_get_metadata_by_slug(self, mock_get, provider):
        """Test getting metadata by slug."""
        mock_get.return_value = {
            "id": "0x123",
            "question": "Will BTC reach $100K?",
            "slug": "btc-100k",
            "volume": 5000000,
        }

        meta = provider.get_market_metadata("btc-100k")

        assert meta["question"] == "Will BTC reach $100K?"
        mock_get.assert_called_once_with("btc-100k")

    @patch.object(PolymarketProvider, "_get_market_by_condition")
    def test_get_metadata_by_condition_id(self, mock_get, provider):
        """Test getting metadata by condition ID."""
        mock_get.return_value = {
            "id": "0x123abc",
            "question": "Will BTC reach $100K?",
        }

        meta = provider.get_market_metadata("0x123abc")

        assert meta["id"] == "0x123abc"
        mock_get.assert_called_once_with("0x123abc")

    def test_get_metadata_rejects_token_id(self, provider):
        """Test that token_id is rejected."""
        with pytest.raises(DataValidationError) as exc_info:
            provider.get_market_metadata("12345678901234567890")

        assert "Cannot get metadata from token_id" in str(exc_info.value)


class TestGetTokenPrices:
    """Test get_token_prices method."""

    @pytest.fixture
    def provider(self):
        """Create provider for tests."""
        provider = PolymarketProvider()
        yield provider
        provider.close()

    @patch.object(PolymarketProvider, "get_market_metadata")
    def test_get_token_prices(self, mock_meta, provider):
        """Test getting current token prices."""
        mock_meta.return_value = {
            "tokens": [
                {"token_id": "123", "outcome": "Yes", "price": 0.65},
                {"token_id": "456", "outcome": "No", "price": 0.35},
            ]
        }

        prices = provider.get_token_prices("test-market")

        assert prices["yes"] == 0.65
        assert prices["no"] == 0.35


class TestValidation:
    """Test response validation."""

    @pytest.fixture
    def provider(self):
        """Create provider for tests."""
        provider = PolymarketProvider()
        yield provider
        provider.close()

    def test_validate_empty_dataframe(self, provider):
        """Test validation of empty DataFrame."""
        df = provider._create_empty_dataframe()
        result = provider._validate_response(df)
        assert result.is_empty()

    def test_validate_valid_ohlc(self, provider):
        """Test validation of valid OHLC data."""
        df = pl.DataFrame(
            {
                "timestamp": [datetime(2024, 1, 1)],
                "symbol": ["TEST"],
                "open": [0.45],
                "high": [0.50],
                "low": [0.40],
                "close": [0.48],
                "volume": [100.0],
            }
        )

        result = provider._validate_response(df)
        assert len(result) == 1

    def test_validate_identical_ohlc_accepted(self, provider):
        """Test that identical OHLC (price-only) is accepted."""
        df = pl.DataFrame(
            {
                "timestamp": [datetime(2024, 1, 1)],
                "symbol": ["TEST"],
                "open": [0.45],
                "high": [0.45],
                "low": [0.45],
                "close": [0.45],
                "volume": [1.0],
            }
        )

        result = provider._validate_response(df)
        assert len(result) == 1

    def test_validate_invalid_ohlc_raises_error(self, provider):
        """Test that invalid OHLC raises error."""
        df = pl.DataFrame(
            {
                "timestamp": [datetime(2024, 1, 1)],
                "symbol": ["TEST"],
                "open": [0.45],
                "high": [0.40],  # Invalid: high < low
                "low": [0.50],
                "close": [0.48],
                "volume": [100.0],
            }
        )

        with pytest.raises(DataValidationError) as exc_info:
            provider._validate_response(df)

        assert "invalid OHLC relationships" in str(exc_info.value)

    def test_validate_missing_column_raises_error(self, provider):
        """Test that missing column raises error."""
        df = pl.DataFrame(
            {
                "timestamp": [datetime(2024, 1, 1)],
                "symbol": ["TEST"],
                "open": [0.45],
                # Missing high, low, close, volume
            }
        )

        with pytest.raises(DataValidationError) as exc_info:
            provider._validate_response(df)

        assert "Missing required column" in str(exc_info.value)


class TestCaching:
    """Test market cache behavior."""

    @pytest.fixture
    def provider(self):
        """Create provider for tests."""
        provider = PolymarketProvider()
        yield provider
        provider.close()

    def test_market_cache_by_slug(self, provider):
        """Test that market data is cached by slug."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = [{"id": "0x123", "slug": "test-market"}]
        provider.session = MagicMock()
        provider.session.get.return_value = mock_response

        # First call
        provider._get_market_by_slug("test-market")
        # Second call (should use cache)
        provider._get_market_by_slug("test-market")

        # Should only call API once
        assert provider.session.get.call_count == 1

    def test_cache_cleared_on_close(self, provider):
        """Test that cache is cleared on close."""
        provider._market_cache["test"] = {"data": "value"}
        provider.close()
        assert provider._market_cache == {}


class TestErrorHandling:
    """Test error handling scenarios."""

    @pytest.fixture
    def provider(self):
        """Create provider for tests."""
        provider = PolymarketProvider()
        yield provider
        provider.close()

    def test_network_error_on_api_failure(self, provider):
        """Test NetworkError on API failure."""
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"
        provider.session = MagicMock()
        provider.session.get.return_value = mock_response

        with pytest.raises(NetworkError):
            provider.list_markets()

    def test_symbol_not_found_on_404(self, provider):
        """Test SymbolNotFoundError on 404."""
        mock_response = MagicMock()
        mock_response.status_code = 404
        provider.session = MagicMock()
        provider.session.get.return_value = mock_response

        with pytest.raises(SymbolNotFoundError):
            provider._get_market_by_condition("0xnonexistent")

    def test_symbol_not_found_empty_result(self, provider):
        """Test SymbolNotFoundError on empty result."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = []  # Empty result
        provider.session = MagicMock()
        provider.session.get.return_value = mock_response

        with pytest.raises(SymbolNotFoundError):
            provider._get_market_by_slug("nonexistent-market")
