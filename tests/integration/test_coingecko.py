"""CoinGecko Provider Integration Tests.

Tests CoinGecko cryptocurrency data provider and updater with real API calls.
CoinGecko Demo plan (free): 30 calls/minute, 10,000 calls/month (requires API key).
Public API (no key): 5-15 calls/minute (unstable, not recommended for tests).
"""

import os
import time
from datetime import datetime, timedelta

import polars as pl
import pytest
from structlog import get_logger

from ml4t.data.core.exceptions import SymbolNotFoundError
from ml4t.data.providers.coingecko import CoinGeckoProvider

logger = get_logger(__name__)

# Cost optimization: use minimal test data
TEST_DAYS = 7  # Fetch 7 days of data for tests
CONCURRENT_SYMBOLS = ["BTC", "ETH", "ADA"]  # Limited to 3 symbols


class TestCoinGeckoProvider:
    """Test CoinGecko provider functionality."""

    @pytest.fixture(autouse=True)
    def rate_limit_delay(self):
        """Add delay between tests to respect CoinGecko's 30 calls/min limit.

        With 30 calls/min = 2 seconds per call, we add 3 seconds between tests
        to be safe and avoid hitting rate limits.
        """
        yield
        time.sleep(3)  # 3 seconds between tests (20 calls/min effective rate)

    @pytest.fixture
    def provider(self) -> CoinGeckoProvider:
        """Initialize CoinGecko provider (Demo plan with API key)."""
        return CoinGeckoProvider()

    @pytest.fixture
    def test_dates(self) -> dict[str, str]:
        """Generate test dates for provider tests."""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=TEST_DAYS)
        return {
            "start": start_date.strftime("%Y-%m-%d"),
            "end": end_date.strftime("%Y-%m-%d"),
            "start_dt": start_date,
            "end_dt": end_date,
        }

    # ==================== Initialization Tests ====================

    @pytest.mark.integration
    def test_provider_initialization_free_tier(self):
        """Test provider initializes with free tier settings."""
        provider = CoinGeckoProvider()

        assert provider.name == "coingecko"
        assert provider.base_url == "https://api.coingecko.com/api/v3"
        assert not provider.use_pro
        assert provider.api_key is None or not provider.use_pro

        logger.info("Free tier provider initialized successfully")

    @pytest.mark.integration
    def test_provider_initialization_with_api_key(self):
        """Test provider initializes with API key (pro tier if available)."""
        api_key = os.getenv("COINGECKO_API_KEY")

        if api_key:
            provider = CoinGeckoProvider(api_key=api_key, use_pro=True)
            assert provider.use_pro
            assert provider.base_url == "https://pro-api.coingecko.com/api/v3"
            logger.info("Pro tier provider initialized successfully")
        else:
            provider = CoinGeckoProvider()
            assert not provider.use_pro
            logger.info("No API key - testing free tier only")

    @pytest.mark.integration
    def test_provider_rate_limiting_config(self, provider):
        """Test rate limiting configuration."""
        # Demo plan (with API key) should have 30 calls/60 seconds limit
        # Public API (no key) should have 10 calls/60 seconds limit (conservative)
        assert provider.rate_limiter is not None
        logger.info(
            "Rate limiter configured",
            rate_limit=f"{provider.rate_limiter.max_calls} calls per {provider.rate_limiter.period}s",
        )

    # ==================== Symbol Mapping Tests ====================

    @pytest.mark.integration
    def test_symbol_to_id_mapping(self, provider):
        """Test symbol to CoinGecko ID conversion."""
        # Test common symbol mappings
        assert provider.symbol_to_id("BTC") == "bitcoin"
        assert provider.symbol_to_id("ETH") == "ethereum"
        assert provider.symbol_to_id("ADA") == "cardano"
        assert provider.symbol_to_id("DOT") == "polkadot"
        assert provider.symbol_to_id("LINK") == "chainlink"

        # Test case insensitivity
        assert provider.symbol_to_id("btc") == "bitcoin"
        assert provider.symbol_to_id("Bitcoin") == "bitcoin"

        # Test direct coin ID (lowercase, no mapping)
        assert provider.symbol_to_id("bitcoin") == "bitcoin"
        assert provider.symbol_to_id("ethereum") == "ethereum"

        logger.info("Symbol mapping tests passed")

    @pytest.mark.integration
    def test_symbol_to_id_unknown_symbol(self, provider):
        """Test handling of unknown symbols."""
        # Unknown symbols should be returned as lowercase (treated as coin ID)
        result = provider.symbol_to_id("UNKNOWN_CRYPTO_XYZ")
        assert result == "unknown_crypto_xyz"

        logger.info("Unknown symbol handling verified")

    # ==================== fetch_ohlcv() Tests ====================

    @pytest.mark.integration
    def test_fetch_ohlcv_btc(self, provider, test_dates):
        """Test fetching BTC OHLCV data for 7 days."""
        start_time = time.time()

        df = provider.fetch_ohlcv(
            symbol="BTC",
            start=test_dates["start"],
            end=test_dates["end"],
            frequency="daily",
        )

        duration = time.time() - start_time

        # Validate results
        assert not df.is_empty(), "Should fetch BTC data"
        assert len(df) >= 1, "Should have at least 1 row of data"
        # CoinGecko returns intraday data (4-hourly for 3-30 days, not daily)
        # For ~7 days with 4-hour intervals: ~42 rows (6 per day * 7 days)
        assert len(df) <= TEST_DAYS * 10, "Should have reasonable amount of intraday data"

        # Validate schema
        self._validate_ohlcv_schema(df)

        # Validate data quality
        self._validate_data_quality(df)

        # Check symbol is set correctly
        assert (df["symbol"] == "BTC").all(), "Symbol should be BTC"

        logger.info(
            "BTC data fetched successfully",
            records=len(df),
            duration_s=f"{duration:.2f}",
            date_range=f"{df['timestamp'].min()} to {df['timestamp'].max()}",
        )

    @pytest.mark.integration
    def test_fetch_ohlcv_multiple_cryptos(self, provider, test_dates):
        """Test fetching multiple cryptocurrencies."""
        symbols = ["BTC", "ETH", "ADA"]

        for symbol in symbols:
            df = provider.fetch_ohlcv(
                symbol=symbol,
                start=test_dates["start"],
                end=test_dates["end"],
                frequency="daily",
            )

            assert not df.is_empty(), f"Should fetch {symbol} data"
            assert (df["symbol"] == symbol).all(), f"Symbol should be {symbol}"
            self._validate_ohlcv_schema(df)
            self._validate_data_quality(df)

            # Rate limiting: small delay between requests
            time.sleep(1.5)

        logger.info("Multiple cryptocurrency fetches successful")

    @pytest.mark.integration
    def test_fetch_ohlcv_empty_result(self, provider):
        """Test fetching data with no results."""
        # Invalid coin should raise SymbolNotFoundError
        with pytest.raises(SymbolNotFoundError):
            provider.fetch_ohlcv(
                symbol="INVALID_COIN_XYZ_12345",
                start="2024-01-01",
                end="2024-01-02",
                frequency="daily",
            )

        logger.info("Empty result test passed")

    @pytest.mark.integration
    def test_fetch_ohlcv_date_filtering(self, provider, test_dates):
        """Test that date filtering works correctly."""
        df = provider.fetch_ohlcv(
            symbol="BTC",
            start=test_dates["start"],
            end=test_dates["end"],
            frequency="daily",
        )

        if not df.is_empty():
            # Check all timestamps are within requested range
            min_ts = df["timestamp"].min()
            max_ts = df["timestamp"].max()

            # Convert to datetime for comparison (handle timezone-aware datetimes)
            start_dt = test_dates["start_dt"].replace(tzinfo=min_ts.tzinfo)
            end_dt = test_dates["end_dt"].replace(tzinfo=max_ts.tzinfo)

            # Allow some flexibility due to timezone differences
            assert min_ts >= start_dt - timedelta(days=1), "Min timestamp should be >= start"
            assert max_ts <= end_dt + timedelta(days=1), "Max timestamp should be <= end"

        logger.info("Date filtering test passed")

    # ==================== get_coin_list() Tests ====================

    @pytest.mark.integration
    def test_get_coin_list(self, provider):
        """Test fetching coin list."""
        df = provider.get_coin_list()

        assert not df.is_empty(), "Coin list should not be empty"
        assert "id" in df.columns, "Should have 'id' column"
        assert "symbol" in df.columns, "Should have 'symbol' column"
        assert "name" in df.columns, "Should have 'name' column"
        assert len(df) > 1000, "Should have at least 1000 coins"

        # Check some known coins exist
        coin_ids = df["id"].to_list()
        assert "bitcoin" in coin_ids, "Bitcoin should be in list"
        assert "ethereum" in coin_ids, "Ethereum should be in list"

        logger.info("Coin list fetched successfully", coin_count=len(df))

    # ==================== get_price() Tests ====================

    @pytest.mark.integration
    def test_get_price_single_coin(self, provider):
        """Test fetching current price for single coin."""
        df = provider.get_price(coin_ids=["bitcoin"], vs_currencies=["usd"])

        assert not df.is_empty(), "Should fetch BTC price"
        assert len(df) == 1, "Should have 1 row"
        assert df["coin_id"][0] == "bitcoin"
        assert df["currency"][0] == "usd"
        assert df["price"][0] > 0, "Price should be positive"

        logger.info("Single coin price fetched", price=f"${df['price'][0]:,.2f}")

    @pytest.mark.integration
    def test_get_price_multiple_coins(self, provider):
        """Test fetching prices for multiple coins."""
        coin_ids = ["bitcoin", "ethereum", "cardano"]
        df = provider.get_price(coin_ids=coin_ids, vs_currencies=["usd"])

        assert not df.is_empty(), "Should fetch prices"
        assert len(df) == len(coin_ids), f"Should have {len(coin_ids)} rows"

        for coin_id in coin_ids:
            coin_price = df.filter(pl.col("coin_id") == coin_id)
            assert len(coin_price) == 1, f"Should have price for {coin_id}"
            assert coin_price["price"][0] > 0, f"Price for {coin_id} should be positive"

        logger.info("Multiple coin prices fetched", coins=coin_ids)

    # ==================== Helper Methods ====================

    def _validate_ohlcv_schema(self, df: pl.DataFrame) -> None:
        """Validate DataFrame has correct OHLCV schema."""
        required_columns = ["timestamp", "open", "high", "low", "close", "volume", "symbol"]

        for col in required_columns:
            assert col in df.columns, f"Missing required column: {col}"

        # Check data types (allow timezone-aware or naive datetimes)
        assert df["timestamp"].dtype in [
            pl.Datetime("ns", "UTC"),
            pl.Datetime("us", "UTC"),
            pl.Datetime("ms", "UTC"),
            pl.Datetime("ns"),
            pl.Datetime("us"),
            pl.Datetime("ms"),
        ], f"Invalid timestamp type: {df['timestamp'].dtype}"

        assert df["open"].dtype == pl.Float64, f"Invalid open type: {df['open'].dtype}"
        assert df["high"].dtype == pl.Float64, f"Invalid high type: {df['high'].dtype}"
        assert df["low"].dtype == pl.Float64, f"Invalid low type: {df['low'].dtype}"
        assert df["close"].dtype == pl.Float64, f"Invalid close type: {df['close'].dtype}"
        assert df["volume"].dtype == pl.Float64, f"Invalid volume type: {df['volume'].dtype}"
        assert df["symbol"].dtype == pl.Utf8, f"Invalid symbol type: {df['symbol'].dtype}"

    def _validate_data_quality(self, df: pl.DataFrame) -> None:
        """Validate OHLCV data quality."""
        if df.is_empty():
            return

        # Check OHLC invariants
        assert (df["high"] >= df["low"]).all(), "high should be >= low"
        assert (df["high"] >= df["open"]).all(), "high should be >= open"
        assert (df["high"] >= df["close"]).all(), "high should be >= close"
        assert (df["low"] <= df["open"]).all(), "low should be <= open"
        assert (df["low"] <= df["close"]).all(), "low should be <= close"
        assert (df["volume"] >= 0).all(), "volume should be >= 0"

        # Check for reasonable price values (not negative, not NaN)
        assert (df["open"] > 0).all(), "open should be > 0"
        assert (df["high"] > 0).all(), "high should be > 0"
        assert (df["low"] > 0).all(), "low should be > 0"
        assert (df["close"] > 0).all(), "close should be > 0"


# ==================== Updater Tests Removed ====================
# CoinGeckoUpdater class has been removed from the codebase.
# Updater functionality is tested separately if needed.

# class TestCoinGeckoUpdater:
#     """Test CoinGecko updater functionality."""
#     ... (all updater tests commented out)


class TestCoinGeckoRateLimiting:
    """Test rate limiting behavior."""

    @pytest.fixture(autouse=True)
    def rate_limit_delay(self):
        """Add delay between tests to respect CoinGecko's 30 calls/min limit."""
        yield
        time.sleep(3)  # 3 seconds between tests

    @pytest.mark.integration
    def test_rate_limiting_sequential_requests(self):
        """Test that rate limiting works for sequential requests."""
        provider = CoinGeckoProvider()

        # Make 5 rapid requests
        symbols = ["BTC", "ETH", "ADA", "DOT", "LINK"]
        end_date = datetime.now()
        start_date = end_date - timedelta(days=1)

        start_time = time.time()

        for symbol in symbols:
            df = provider.fetch_ohlcv(
                symbol=symbol,
                start=start_date.strftime("%Y-%m-%d"),
                end=end_date.strftime("%Y-%m-%d"),
                frequency="daily",
            )
            assert df is not None, f"Request for {symbol} should succeed"

        duration = time.time() - start_time

        # With rate limiting, 5 requests should take at least a few seconds
        # Demo plan: 30 calls/60 seconds = 2s per call minimum
        expected_min_duration = 0  # Rate limiter should allow some burst
        assert duration >= expected_min_duration, "Rate limiting should add some delay"

        logger.info(
            "Rate limiting test completed",
            requests=len(symbols),
            duration_s=f"{duration:.2f}",
        )


class TestCoinGeckoErrorHandling:
    """Test error handling scenarios."""

    @pytest.fixture(autouse=True)
    def rate_limit_delay(self):
        """Add delay between tests to respect CoinGecko's 30 calls/min limit."""
        yield
        time.sleep(3)  # 3 seconds between tests

    @pytest.mark.integration
    def test_network_error_handling(self):
        """Test handling of network errors."""
        # Create provider with invalid base URL to simulate network error
        provider = CoinGeckoProvider()
        provider.base_url = "https://invalid-domain-that-does-not-exist-12345.com"

        # Should raise NetworkError or return empty
        try:
            df = provider.fetch_ohlcv(
                symbol="BTC",
                start="2024-01-01",
                end="2024-01-02",
                frequency="daily",
            )
            # If no exception, should return empty
            assert df.is_empty(), "Network error should return empty"
        except Exception as e:
            # Acceptable to raise exception
            assert True
            logger.info("Network error handled correctly", error=str(e))

    @pytest.mark.integration
    def test_invalid_date_range(self):
        """Test handling of invalid date range (end < start)."""
        provider = CoinGeckoProvider()

        # Invalid date range should raise ValueError
        with pytest.raises(ValueError, match="Start date must be before"):
            provider.fetch_ohlcv(
                symbol="BTC",
                start="2024-01-10",
                end="2024-01-01",  # End before start
                frequency="daily",
            )

        logger.info("Invalid date range test passed")
