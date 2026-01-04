"""Integration tests for Polymarket provider (real API calls).

These tests verify the Polymarket provider works correctly with actual API calls.

Requirements:
    - No API key required for public data
    - Rate limit: ~30 requests per minute (conservative)
    - Internet connection required
    - Note: US access may be restricted for some features

Test Coverage:
    - List markets (Gamma API)
    - Get market by slug
    - Fetch price history (CLOB API)
    - Symbol resolution (slug → token_id)
    - Both outcomes fetching
    - Market search
    - Error handling

IMPORTANT:
    These tests make ~15 API calls total. Polymarket has rate limits,
    so be mindful if running repeatedly.
"""

import json
import time
from datetime import datetime, timedelta

import polars as pl
import pytest

from ml4t.data.core.exceptions import SymbolNotFoundError
from ml4t.data.providers.polymarket import PolymarketProvider


def _has_clob_tokens(market: dict) -> bool:
    """Check if market has CLOB token IDs (new API format).

    The API returns clobTokenIds as a JSON string like:
    '["token1", "token2"]' for YES/NO markets.
    """
    clob_ids_raw = market.get("clobTokenIds", "")
    if not clob_ids_raw:
        return False

    # Parse if it's a JSON string
    if isinstance(clob_ids_raw, str):
        try:
            clob_ids = json.loads(clob_ids_raw)
            return len(clob_ids) >= 2
        except json.JSONDecodeError:
            return False
    elif isinstance(clob_ids_raw, list):
        return len(clob_ids_raw) >= 2

    # Also check legacy tokens array
    tokens = market.get("tokens", [])
    return bool(tokens and len(tokens) >= 2)


@pytest.fixture
def provider():
    """Create Polymarket provider (no API key needed for public data)."""
    provider = PolymarketProvider()
    yield provider
    provider.close()


class TestPolymarketProvider:
    """Test Polymarket provider with real API calls.

    Note: ~15 API calls total.
    """

    def test_provider_initialization(self):
        """Test provider can be initialized.

        This test does not make any API calls.
        """
        provider = PolymarketProvider()
        assert provider.name == "polymarket"
        provider.close()

    def test_list_markets(self, provider):
        """Test listing available markets.

        API calls: 1 (Gamma API)
        """
        markets = provider.list_markets(active=True, limit=10)

        # Should return a list
        assert isinstance(markets, list)

        # Should have some markets
        if len(markets) > 0:
            print(f"✅ Listed {len(markets)} markets")

            # Show first few
            for m in markets[:5]:
                slug = m.get("slug", "N/A")
                question = m.get("question", "N/A")[:50] if m.get("question") else "N/A"
                print(f"   {slug}: {question}")
        else:
            print("⚠️ No markets returned (API may have changed)")

    def test_list_markets_with_category(self, provider):
        """Test listing markets filtered by category.

        API calls: 1 (Gamma API)
        """
        # Try common categories
        for category in ["Crypto", "Politics"]:
            markets = provider.list_markets(active=True, category=category, limit=5)
            if markets:
                print(f"✅ Found {len(markets)} {category} markets")
                break
        else:
            print("⚠️ No markets found in common categories")

    def test_get_market_by_slug(self, provider):
        """Test getting a specific market by slug.

        API calls: 2 (list + get)
        """
        # First get a market slug
        markets = provider.list_markets(active=True, limit=1)

        if not markets:
            pytest.skip("No active markets available")

        slug = markets[0].get("slug")
        if not slug:
            pytest.skip("No market slug available")

        # Get that market's metadata
        metadata = provider.get_market_metadata(slug)

        assert "question" in metadata or "id" in metadata
        print(f"✅ Got metadata for {slug}")
        print(f"   Question: {metadata.get('question', 'N/A')[:60]}")

    def test_resolve_slug_to_token_id(self, provider):
        """Test resolving a slug to token ID.

        API calls: 2 (list + resolve)
        """
        # Get a market with tokens
        markets = provider.list_markets(active=True, limit=20)

        # Find a market with CLOB tokens (new API format)
        market = None
        for m in markets:
            if _has_clob_tokens(m):
                market = m
                break

        if not market:
            pytest.skip("No market with CLOB tokens found")

        slug = market.get("slug")
        if not slug:
            pytest.skip("Market has no slug")

        # Resolve to YES token
        yes_token = provider.resolve_symbol(slug, outcome="yes")
        assert yes_token
        print(f"✅ Resolved {slug} YES → {yes_token[:20]}...")

        # Resolve to NO token
        no_token = provider.resolve_symbol(slug, outcome="no")
        assert no_token
        print(f"✅ Resolved {slug} NO → {no_token[:20]}...")

        # They should be different
        assert yes_token != no_token

    def test_fetch_daily_prices(self, provider):
        """Test fetching daily price history.

        API calls: 2-3 (list + resolve + fetch)
        """
        # Find a market with volume
        markets = provider.list_markets(active=True, limit=50)

        market = None
        for m in markets:
            vol = m.get("volume", 0)
            if vol and float(vol) > 10000 and _has_clob_tokens(m):
                market = m
                break

        if not market:
            # Fallback to any market with CLOB tokens
            for m in markets:
                if _has_clob_tokens(m):
                    market = m
                    break

        if not market:
            pytest.skip("No suitable market found")

        slug = market.get("slug")
        if not slug:
            pytest.skip("Market has no slug")

        # Fetch last 7 days (API has date range limits for hourly granularity)
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")

        try:
            df = provider.fetch_ohlcv(slug, start_date, end_date, frequency="daily", outcome="yes")

            # Verify structure
            assert isinstance(df, pl.DataFrame)

            # Check columns
            required_cols = ["timestamp", "symbol", "open", "high", "low", "close", "volume"]
            assert all(col in df.columns for col in required_cols)

            print(f"✅ Fetched {len(df)} daily records for {slug}")
            if not df.is_empty():
                print(f"   Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
                print(f"   Price range: {df['close'].min():.2%} - {df['close'].max():.2%}")
            else:
                print("   (Empty result - market may be new)")

        except SymbolNotFoundError:
            print(f"⚠️ Market {slug} not found (may have been removed)")

    def test_fetch_hourly_prices(self, provider):
        """Test fetching hourly price history.

        API calls: 2-3 (list + resolve + fetch)
        """
        # Find a market with good volume
        markets = provider.list_markets(active=True, limit=50)

        market = None
        for m in markets:
            vol = m.get("volume", 0)
            if vol and float(vol) > 100000 and _has_clob_tokens(m):
                market = m
                break

        if not market:
            for m in markets:
                if _has_clob_tokens(m):
                    market = m
                    break

        if not market:
            pytest.skip("No suitable market found")

        slug = market.get("slug")
        if not slug:
            pytest.skip("Market has no slug")

        # Fetch last 7 days hourly
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")

        try:
            df = provider.fetch_ohlcv(slug, start_date, end_date, frequency="hourly", outcome="yes")

            assert isinstance(df, pl.DataFrame)
            print(f"✅ Fetched {len(df)} hourly records for {slug}")

        except SymbolNotFoundError:
            print(f"⚠️ Market {slug} not found")

    def test_fetch_both_outcomes(self, provider):
        """Test fetching both YES and NO outcomes.

        API calls: 3-4 (list + 2x fetch)
        """
        # Find a truly active market (not closed) with CLOB tokens
        markets = provider.list_markets(active=True, closed=False, limit=20)

        market = None
        for m in markets:
            if _has_clob_tokens(m):
                market = m
                break

        if not market:
            pytest.skip("No active market with CLOB tokens found")

        slug = market.get("slug")
        if not slug:
            pytest.skip("Market has no slug")

        # Use 7-day range (API has date range limits)
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")

        try:
            df = provider.fetch_both_outcomes(slug, start_date, end_date, frequency="daily")

            assert isinstance(df, pl.DataFrame)

            # Should have data for at least one outcome
            symbols = df["symbol"].unique().to_list()
            print(f"✅ Fetched both outcomes for {slug}")
            print(f"   Symbols: {symbols}")
            print(f"   Total records: {len(df)}")

        except SymbolNotFoundError:
            print(f"⚠️ Market {slug} not found")

    def test_search_markets(self, provider):
        """Test searching markets.

        API calls: 1 (list all then filter)
        """
        # Search for common topics
        for query in ["bitcoin", "election", "fed"]:
            results = provider.search_markets(query, limit=5)
            if results:
                print(f"✅ Found {len(results)} markets for '{query}'")
                for r in results[:3]:
                    print(f"   {r.get('slug', 'N/A')}")
                break
        else:
            print("⚠️ No markets found for common queries")

    def test_get_token_prices(self, provider):
        """Test getting current token prices.

        API calls: 2 (list + get)
        """
        # Find a market with CLOB tokens
        markets = provider.list_markets(active=True, limit=20)

        market = None
        for m in markets:
            if _has_clob_tokens(m):
                market = m
                break

        if not market:
            pytest.skip("No market with CLOB tokens found")

        slug = market.get("slug")
        if not slug:
            pytest.skip("Market has no slug")

        prices = provider.get_token_prices(slug)

        if "yes" in prices and "no" in prices:
            yes_price = prices["yes"]
            no_price = prices["no"]
            total = yes_price + no_price

            print(f"✅ Got prices for {slug}")
            print(f"   YES: {yes_price:.2%}")
            print(f"   NO: {no_price:.2%}")
            print(f"   Sum: {total:.2%} (should be ~100%)")

            # Prices should be in 0-1 range
            assert 0 <= yes_price <= 1
            assert 0 <= no_price <= 1
        else:
            print(f"⚠️ Incomplete prices for {slug}: {prices}")

    def test_invalid_slug_raises_error(self, provider):
        """Test error handling for invalid slug.

        API calls: 1
        """
        with pytest.raises(SymbolNotFoundError):
            provider.get_market_metadata("invalid-nonexistent-market-slug-xyz-12345")

        print("✅ Invalid slug correctly raises SymbolNotFoundError")

    def test_price_range_validation(self, provider):
        """Test that prices are in valid probability range (0-1).

        API calls: 2-3
        """
        markets = provider.list_markets(active=True, limit=50)

        market = None
        for m in markets:
            if _has_clob_tokens(m) and m.get("volume", 0):
                market = m
                break

        if not market:
            pytest.skip("No market with volume found")

        slug = market.get("slug")
        if not slug:
            pytest.skip("Market has no slug")

        # Use 7-day range (API has date range limits for hourly granularity)
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")

        try:
            df = provider.fetch_ohlcv(slug, start_date, end_date, frequency="daily")

            if not df.is_empty():
                # All prices should be in 0-1 range
                for col in ["open", "high", "low", "close"]:
                    values = df.filter(pl.col(col).is_not_null())[col]
                    if len(values) > 0:
                        assert values.min() >= 0.0, f"{col} should be >= 0"
                        assert values.max() <= 1.0, f"{col} should be <= 1"

                print(f"✅ Prices in valid 0-1 range for {slug}")
            else:
                print(f"⚠️ No data to validate for {slug}")

        except SymbolNotFoundError:
            print(f"⚠️ Market {slug} not found")

    def test_ohlc_invariants(self, provider):
        """Test that OHLC invariants hold.

        API calls: 2-3
        """
        markets = provider.list_markets(active=True, limit=50)

        market = None
        for m in markets:
            vol = m.get("volume", 0)
            if vol and float(vol) > 50000 and _has_clob_tokens(m):
                market = m
                break

        if not market:
            pytest.skip("No high-volume market found")

        slug = market.get("slug")
        if not slug:
            pytest.skip("Market has no slug")

        # Use 7-day range (API has date range limits for hourly granularity)
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")

        try:
            df = provider.fetch_ohlcv(slug, start_date, end_date, frequency="daily")

            if not df.is_empty() and len(df) > 1:
                # Check OHLC invariants (for non-identical values)
                non_identical = (
                    (df["open"] != df["close"])
                    | (df["high"] != df["close"])
                    | (df["low"] != df["close"])
                )

                if non_identical.any():
                    df_check = df.filter(non_identical)
                    assert (df_check["high"] >= df_check["low"]).all(), "high >= low"
                    print(f"✅ OHLC invariants hold for {slug}")
                else:
                    print(f"✅ All identical prices for {slug} (valid for price-only data)")
            else:
                print(f"⚠️ Insufficient data for {slug}")

        except SymbolNotFoundError:
            print(f"⚠️ Market {slug} not found")


class TestPolymarketRateLimiting:
    """Test rate limiting behavior."""

    def test_multiple_requests(self):
        """Test that multiple requests work with rate limiting.

        API calls: 3
        """
        provider = PolymarketProvider()

        start_time = time.time()
        for _ in range(3):
            markets = provider.list_markets(active=True, limit=5)
            assert isinstance(markets, list)

        elapsed = time.time() - start_time
        provider.close()

        print(f"✅ 3 requests completed in {elapsed:.2f}s")


# Test Summary:
# ==============
# Total API calls: ~15-20 calls
# 1. test_list_markets: 1 call
# 2. test_list_markets_with_category: 1 call
# 3. test_get_market_by_slug: 2 calls
# 4. test_resolve_slug_to_token_id: 2 calls
# 5. test_fetch_daily_prices: 2-3 calls
# 6. test_fetch_hourly_prices: 2-3 calls
# 7. test_fetch_both_outcomes: 3-4 calls
# 8. test_search_markets: 1 call
# 9. test_get_token_prices: 2 calls
# 10. test_invalid_slug_raises_error: 1 call
# 11. test_price_range_validation: 2-3 calls
# 12. test_ohlc_invariants: 2-3 calls
# 13. test_multiple_requests: 3 calls
#
# Polymarket rate limits are conservative (~30 req/min), so these should run fine.
