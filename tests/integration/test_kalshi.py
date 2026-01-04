"""Integration tests for Kalshi provider (real API calls).

These tests verify the Kalshi provider works correctly with actual API calls.

Requirements:
    - No API key required for public data
    - Rate limit: ~10 requests per second
    - Internet connection required

Test Coverage:
    - List series (available event templates)
    - List markets (open prediction markets)
    - Fetch daily OHLCV (native candlestick data)
    - Fetch hourly OHLCV
    - Market metadata
    - Invalid market handling
    - Multiple markets alignment

IMPORTANT:
    These tests make ~10 API calls total. Kalshi has conservative rate limits,
    so be mindful if running repeatedly.
"""

import time
from datetime import datetime, timedelta

import polars as pl
import pytest

from ml4t.data.core.exceptions import SymbolNotFoundError
from ml4t.data.providers.kalshi import KalshiProvider


@pytest.fixture
def provider():
    """Create Kalshi provider (no API key needed for public data)."""
    provider = KalshiProvider()
    yield provider
    provider.close()


class TestKalshiProvider:
    """Test Kalshi provider with real API calls.

    Note: ~10 API calls total.
    """

    def test_provider_initialization(self):
        """Test provider can be initialized without API key.

        This test does not make any API calls.
        """
        provider = KalshiProvider()
        assert provider.name == "kalshi"
        assert provider.api_key is None
        provider.close()

    def test_list_series(self, provider):
        """Test listing available series.

        API calls: 1
        """
        series = provider.list_series()

        # Should return a list
        assert isinstance(series, list)

        # Kalshi should have some series available
        # Note: this could be empty if API changes, so we just check structure
        if len(series) > 0:
            # Each series should have a ticker
            assert "ticker" in series[0]
            print(f"✅ Listed {len(series)} series")
            # Show first few
            for s in series[:5]:
                print(f"   {s.get('ticker', 'N/A')}: {s.get('title', 'N/A')[:50]}")
        else:
            print("⚠️ No series returned (API may have changed)")

    def test_list_markets(self, provider):
        """Test listing available markets.

        API calls: 1
        """
        # List open markets
        markets = provider.list_markets(status="open", limit=10)

        # Should return a list
        assert isinstance(markets, list)

        # Should have some open markets
        if len(markets) > 0:
            # Each market should have key fields
            first_market = markets[0]
            assert "ticker" in first_market
            assert "status" in first_market

            print(f"✅ Listed {len(markets)} open markets")
            # Show first few
            for m in markets[:5]:
                ticker = m.get("ticker", "N/A")
                title = m.get("title", "N/A")[:50]
                status = m.get("status", "N/A")
                print(f"   {ticker}: {title} ({status})")
        else:
            print("⚠️ No open markets returned")

    def test_list_markets_with_series_filter(self, provider):
        """Test listing markets filtered by series.

        API calls: 1
        """
        # First get some series to use for filtering
        series = provider.list_series()

        if not series:
            pytest.skip("No series available to filter by")

        # Use first series ticker for filter
        series_ticker = series[0].get("ticker", "")
        if not series_ticker:
            pytest.skip("No series ticker available")

        # List markets for that series
        markets = provider.list_markets(series_ticker=series_ticker, limit=10)

        assert isinstance(markets, list)
        print(f"✅ Listed {len(markets)} markets for series {series_ticker}")

    def test_get_market_metadata(self, provider):
        """Test getting market metadata.

        API calls: 2 (list + get)
        """
        # First get a market ticker
        markets = provider.list_markets(status="open", limit=1)

        if not markets:
            pytest.skip("No open markets available")

        ticker = markets[0].get("ticker", "")
        if not ticker:
            pytest.skip("No market ticker available")

        # Get metadata for that market
        metadata = provider.get_market_metadata(ticker)

        # Should have key fields
        assert "ticker" in metadata
        assert metadata["ticker"] == ticker

        print(f"✅ Got metadata for {ticker}")
        print(f"   Title: {metadata.get('title', 'N/A')[:60]}")
        print(f"   Status: {metadata.get('status', 'N/A')}")
        print(f"   Volume: {metadata.get('volume', 'N/A')}")

    def test_fetch_candlesticks_daily(self, provider):
        """Test fetching daily OHLCV candlestick data.

        API calls: 2 (list + fetch)
        """
        # First find a market with some history
        markets = provider.list_markets(status="open", limit=50)

        if not markets:
            pytest.skip("No open markets available")

        # Find market with volume (likely to have data)
        market_ticker = None
        series_ticker = None
        for m in markets:
            if m.get("volume", 0) > 100:
                market_ticker = m.get("ticker")
                # Extract series from event_ticker or ticker
                series_ticker = m.get("event_ticker") or m.get("series_ticker")
                if not series_ticker and market_ticker:
                    series_ticker = market_ticker.split("-")[0]
                break

        if not market_ticker:
            # Just use first market
            market_ticker = markets[0].get("ticker")
            series_ticker = market_ticker.split("-")[0] if market_ticker else None

        if not market_ticker:
            pytest.skip("No market ticker available")

        # Fetch last 30 days of data
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")

        try:
            df = provider.fetch_ohlcv(
                symbol=market_ticker,
                start=start_date,
                end=end_date,
                frequency="daily",
                series_ticker=series_ticker,
            )

            # Verify structure
            assert isinstance(df, pl.DataFrame)

            # Check columns
            required_cols = ["timestamp", "symbol", "open", "high", "low", "close", "volume"]
            assert all(col in df.columns for col in required_cols)

            print(f"✅ Fetched {len(df)} daily candlesticks for {market_ticker}")
            if not df.is_empty():
                print(f"   Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
                print(f"   Close range: {df['close'].min():.2f} - {df['close'].max():.2f}")
                print(f"   Volume range: {df['volume'].min():.0f} - {df['volume'].max():.0f}")
            else:
                print("   (Empty result - market may be new)")

        except SymbolNotFoundError:
            # Market may have been delisted between list and fetch
            print(f"⚠️ Market {market_ticker} not found (may have been delisted)")

    def test_fetch_candlesticks_hourly(self, provider):
        """Test fetching hourly OHLCV candlestick data.

        API calls: 2 (list + fetch)
        """
        # Find a market with volume
        markets = provider.list_markets(status="open", limit=50)

        if not markets:
            pytest.skip("No open markets available")

        # Find market with volume
        market_ticker = None
        series_ticker = None
        for m in markets:
            if m.get("volume", 0) > 1000:  # Higher volume for hourly
                market_ticker = m.get("ticker")
                series_ticker = m.get("event_ticker") or m.get("series_ticker")
                if not series_ticker and market_ticker:
                    series_ticker = market_ticker.split("-")[0]
                break

        if not market_ticker:
            market_ticker = markets[0].get("ticker")
            series_ticker = market_ticker.split("-")[0] if market_ticker else None

        if not market_ticker:
            pytest.skip("No market ticker available")

        # Fetch last 7 days of hourly data
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")

        try:
            df = provider.fetch_ohlcv(
                symbol=market_ticker,
                start=start_date,
                end=end_date,
                frequency="hourly",
                series_ticker=series_ticker,
            )

            # Verify structure
            assert isinstance(df, pl.DataFrame)

            # Check columns
            required_cols = ["timestamp", "symbol", "open", "high", "low", "close", "volume"]
            assert all(col in df.columns for col in required_cols)

            print(f"✅ Fetched {len(df)} hourly candlesticks for {market_ticker}")
            if not df.is_empty():
                print(f"   Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")

        except SymbolNotFoundError:
            print(f"⚠️ Market {market_ticker} not found (may have been delisted)")

    def test_invalid_market_raises_error(self, provider):
        """Test error handling for invalid market.

        API calls: 1

        Note: May raise RateLimitError if API limits are hit during test run.
        Both SymbolNotFoundError and RateLimitError are acceptable outcomes.
        """
        from ml4t.data.core.exceptions import RateLimitError

        with pytest.raises((SymbolNotFoundError, RateLimitError)) as exc_info:
            provider.fetch_ohlcv(
                symbol="INVALID-MARKET-XYZ-12345",
                start="2024-01-01",
                end="2024-12-31",
                frequency="daily",
            )

        if isinstance(exc_info.value, RateLimitError):
            print("⚠️ Rate limited - test inconclusive but API is responding")
        else:
            print("✅ Invalid market correctly raises SymbolNotFoundError")

    def test_price_range_validation(self, provider):
        """Test that prices are in valid probability range (0-1).

        API calls: 2 (list + fetch)
        """
        # Find a market with volume
        markets = provider.list_markets(status="open", limit=50)

        if not markets:
            pytest.skip("No open markets available")

        # Find market with volume
        market_ticker = None
        series_ticker = None
        for m in markets:
            if m.get("volume", 0) > 100:
                market_ticker = m.get("ticker")
                series_ticker = m.get("event_ticker") or m.get("series_ticker")
                if not series_ticker and market_ticker:
                    series_ticker = market_ticker.split("-")[0]
                break

        if not market_ticker:
            pytest.skip("No market with volume available")

        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")

        try:
            df = provider.fetch_ohlcv(
                symbol=market_ticker,
                start=start_date,
                end=end_date,
                frequency="daily",
                series_ticker=series_ticker,
            )

            if not df.is_empty():
                # All prices should be in 0-1 range (probabilities)
                for col in ["open", "high", "low", "close"]:
                    values = df.filter(pl.col(col).is_not_null())[col]
                    if len(values) > 0:
                        assert values.min() >= 0.0, f"{col} should be >= 0"
                        assert values.max() <= 1.0, f"{col} should be <= 1"

                print(f"✅ Prices in valid 0-1 probability range for {market_ticker}")
            else:
                print(f"⚠️ No data to validate for {market_ticker}")

        except SymbolNotFoundError:
            print(f"⚠️ Market {market_ticker} not found")

    def test_ohlc_invariants(self, provider):
        """Test that OHLC invariants hold (high >= low, etc.).

        API calls: 2 (list + fetch)
        """
        # Find a market with volume
        markets = provider.list_markets(status="open", limit=50)

        if not markets:
            pytest.skip("No open markets available")

        market_ticker = None
        series_ticker = None
        for m in markets:
            if m.get("volume", 0) > 100:
                market_ticker = m.get("ticker")
                series_ticker = m.get("event_ticker") or m.get("series_ticker")
                if not series_ticker and market_ticker:
                    series_ticker = market_ticker.split("-")[0]
                break

        if not market_ticker:
            pytest.skip("No market with volume available")

        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")

        try:
            df = provider.fetch_ohlcv(
                symbol=market_ticker,
                start=start_date,
                end=end_date,
                frequency="daily",
                series_ticker=series_ticker,
            )

            if not df.is_empty():
                # OHLC invariants
                assert (df["high"] >= df["low"]).all(), "high should be >= low"
                assert (df["high"] >= df["open"]).all(), "high should be >= open"
                assert (df["high"] >= df["close"]).all(), "high should be >= close"
                assert (df["low"] <= df["open"]).all(), "low should be <= open"
                assert (df["low"] <= df["close"]).all(), "low should be <= close"

                print(f"✅ OHLC invariants hold for {market_ticker}")
            else:
                print(f"⚠️ No data to validate for {market_ticker}")

        except SymbolNotFoundError:
            print(f"⚠️ Market {market_ticker} not found")


class TestKalshiRateLimiting:
    """Test rate limiting behavior."""

    def test_multiple_requests(self):
        """Test that multiple requests work with rate limiting.

        API calls: 3
        """
        provider = KalshiProvider()

        # Make several quick requests
        start_time = time.time()
        for _ in range(3):
            series = provider.list_series()
            assert isinstance(series, list)

        elapsed = time.time() - start_time
        provider.close()

        print(f"✅ 3 requests completed in {elapsed:.2f}s")


# Test Summary:
# ==============
# Total API calls: ~12 calls
# 1. test_list_series: 1 call
# 2. test_list_markets: 1 call
# 3. test_list_markets_with_series_filter: 1 call
# 4. test_get_market_metadata: 2 calls
# 5. test_fetch_candlesticks_daily: 2 calls
# 6. test_fetch_candlesticks_hourly: 2 calls
# 7. test_invalid_market_raises_error: 1 call
# 8. test_price_range_validation: 2 calls
# 9. test_ohlc_invariants: 2 calls
# 10. test_multiple_requests: 3 calls
#
# Kalshi rate limit is ~10 req/sec, so these tests should run fine.
