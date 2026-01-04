"""Tests for universe symbol lists and retrieval."""

import pytest

from ml4t.data.universe import Universe


class TestUniverseConstants:
    """Test that pre-defined universes are properly defined."""

    def test_sp500_exists(self):
        """SP500 universe exists and has symbols."""
        assert hasattr(Universe, "SP500")
        assert isinstance(Universe.SP500, list)
        assert len(Universe.SP500) > 0

    def test_sp500_has_correct_count(self):
        """SP500 contains 503 symbols (includes share classes)."""
        # S&P 500 has 503 constituents (some companies have multiple share classes)
        assert len(Universe.SP500) == 503

    def test_sp500_contains_major_symbols(self):
        """SP500 contains well-known major stocks."""
        major_stocks = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "BRK.B", "JPM"]
        for symbol in major_stocks:
            assert symbol in Universe.SP500, f"{symbol} should be in SP500"

    def test_nasdaq100_exists(self):
        """NASDAQ100 universe exists and has symbols."""
        assert hasattr(Universe, "NASDAQ100")
        assert isinstance(Universe.NASDAQ100, list)
        assert len(Universe.NASDAQ100) > 0

    def test_nasdaq100_has_correct_count(self):
        """NASDAQ100 contains exactly 100 symbols."""
        assert len(Universe.NASDAQ100) == 100

    def test_nasdaq100_contains_major_tech(self):
        """NASDAQ100 contains major tech stocks."""
        major_tech = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA"]
        for symbol in major_tech:
            assert symbol in Universe.NASDAQ100, f"{symbol} should be in NASDAQ100"

    def test_crypto_top_100_exists(self):
        """CRYPTO_TOP_100 universe exists and has symbols."""
        assert hasattr(Universe, "CRYPTO_TOP_100")
        assert isinstance(Universe.CRYPTO_TOP_100, list)
        assert len(Universe.CRYPTO_TOP_100) > 0

    def test_crypto_top_100_has_correct_count(self):
        """CRYPTO_TOP_100 contains exactly 100 symbols."""
        assert len(Universe.CRYPTO_TOP_100) == 100

    def test_crypto_top_100_contains_major_coins(self):
        """CRYPTO_TOP_100 contains major cryptocurrencies."""
        major_crypto = ["BTC", "ETH", "SOL", "ADA", "AVAX", "DOT", "LINK"]
        for symbol in major_crypto:
            assert symbol in Universe.CRYPTO_TOP_100, f"{symbol} should be in CRYPTO_TOP_100"

    def test_forex_majors_exists(self):
        """FOREX_MAJORS universe exists and has symbols."""
        assert hasattr(Universe, "FOREX_MAJORS")
        assert isinstance(Universe.FOREX_MAJORS, list)
        assert len(Universe.FOREX_MAJORS) > 0

    def test_forex_majors_has_correct_count(self):
        """FOREX_MAJORS contains 28 major pairs."""
        assert len(Universe.FOREX_MAJORS) == 28

    def test_forex_majors_contains_major_pairs(self):
        """FOREX_MAJORS contains major currency pairs."""
        major_pairs = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "USDCHF"]
        for pair in major_pairs:
            assert pair in Universe.FOREX_MAJORS, f"{pair} should be in FOREX_MAJORS"

    def test_all_symbols_are_strings(self):
        """All symbols in all universes are strings."""
        for universe_name in ["SP500", "NASDAQ100", "CRYPTO_TOP_100", "FOREX_MAJORS"]:
            universe = getattr(Universe, universe_name)
            for symbol in universe:
                assert isinstance(symbol, str), f"{symbol} in {universe_name} is not a string"

    def test_no_duplicates_in_universes(self):
        """No universe contains duplicate symbols."""
        for universe_name in ["SP500", "NASDAQ100", "CRYPTO_TOP_100", "FOREX_MAJORS"]:
            universe = getattr(Universe, universe_name)
            assert len(universe) == len(set(universe)), f"{universe_name} contains duplicates"


class TestUniverseGet:
    """Test Universe.get() method for retrieving universes by name."""

    def test_get_sp500_uppercase(self):
        """Can retrieve SP500 with uppercase name."""
        symbols = Universe.get("SP500")
        assert symbols == Universe.SP500
        assert len(symbols) == 503

    def test_get_sp500_lowercase(self):
        """Can retrieve SP500 with lowercase name (case-insensitive)."""
        symbols = Universe.get("sp500")
        assert symbols == Universe.SP500
        assert len(symbols) == 503

    def test_get_sp500_mixed_case(self):
        """Can retrieve SP500 with mixed case name."""
        symbols = Universe.get("Sp500")
        assert symbols == Universe.SP500

    def test_get_nasdaq100_uppercase(self):
        """Can retrieve NASDAQ100 with uppercase name."""
        symbols = Universe.get("NASDAQ100")
        assert symbols == Universe.NASDAQ100
        assert len(symbols) == 100

    def test_get_nasdaq100_lowercase(self):
        """Can retrieve NASDAQ100 with lowercase name."""
        symbols = Universe.get("nasdaq100")
        assert symbols == Universe.NASDAQ100

    def test_get_crypto_top_100_with_underscores(self):
        """Can retrieve CRYPTO_TOP_100 with underscores."""
        symbols = Universe.get("CRYPTO_TOP_100")
        assert symbols == Universe.CRYPTO_TOP_100
        assert len(symbols) == 100

    def test_get_crypto_top_100_with_hyphens(self):
        """Can retrieve CRYPTO_TOP_100 with hyphens (normalized to underscores)."""
        symbols = Universe.get("crypto-top-100")
        assert symbols == Universe.CRYPTO_TOP_100

    def test_get_crypto_top_100_with_spaces(self):
        """Can retrieve CRYPTO_TOP_100 with spaces (normalized to underscores)."""
        symbols = Universe.get("crypto top 100")
        assert symbols == Universe.CRYPTO_TOP_100

    def test_get_crypto_top_100_without_separators(self):
        """Can retrieve CRYPTO_TOP_100 without separators."""
        symbols = Universe.get("cryptotop100")
        assert symbols == Universe.CRYPTO_TOP_100

    def test_get_forex_majors(self):
        """Can retrieve FOREX_MAJORS."""
        symbols = Universe.get("FOREX_MAJORS")
        assert symbols == Universe.FOREX_MAJORS
        assert len(symbols) == 28

    def test_get_forex_majors_lowercase(self):
        """Can retrieve FOREX_MAJORS with lowercase."""
        symbols = Universe.get("forex_majors")
        assert symbols == Universe.FOREX_MAJORS

    def test_get_returns_copy(self):
        """get() returns a copy, not the original list."""
        symbols1 = Universe.get("sp500")
        symbols2 = Universe.get("sp500")

        # They should be equal but not the same object
        assert symbols1 == symbols2
        assert symbols1 is not symbols2

        # Modifying one should not affect the other
        symbols1.append("TEST")
        assert "TEST" not in symbols2
        assert "TEST" not in Universe.SP500

    def test_get_invalid_universe_raises(self):
        """get() raises ValueError for invalid universe name."""
        with pytest.raises(ValueError, match="Unknown universe 'invalid'"):
            Universe.get("invalid")

    def test_get_invalid_universe_shows_available(self):
        """Error message shows available universes."""
        with pytest.raises(ValueError, match="Available universes:"):
            Universe.get("nonexistent")

        # Should mention at least some of the available universes
        with pytest.raises(ValueError, match="SP500"):
            Universe.get("nonexistent")


class TestUniverseListUniverses:
    """Test Universe.list_universes() method."""

    def test_list_universes_returns_list(self):
        """list_universes() returns a list."""
        universes = Universe.list_universes()
        assert isinstance(universes, list)

    def test_list_universes_contains_all_builtin(self):
        """list_universes() contains all built-in universes."""
        universes = Universe.list_universes()
        assert "SP500" in universes
        assert "NASDAQ100" in universes
        assert "CRYPTO_TOP_100" in universes
        assert "FOREX_MAJORS" in universes

    def test_list_universes_is_sorted(self):
        """list_universes() returns sorted list."""
        universes = Universe.list_universes()
        assert universes == sorted(universes)

    def test_list_universes_minimum_count(self):
        """list_universes() returns at least 4 universes."""
        universes = Universe.list_universes()
        assert len(universes) >= 4


class TestUniverseCustomUniverses:
    """Test adding and removing custom universes."""

    def test_add_custom_universe(self):
        """Can add a custom universe."""
        custom_symbols = ["AAPL", "MSFT", "GOOGL"]
        Universe.add_custom("my_portfolio", custom_symbols)

        # Should be retrievable
        symbols = Universe.get("my_portfolio")
        assert symbols == custom_symbols

        # Should appear in list
        assert "MY_PORTFOLIO" in Universe.list_universes()

        # Cleanup
        Universe.remove_custom("my_portfolio")

    def test_add_custom_universe_normalizes_name(self):
        """Custom universe name is normalized to uppercase."""
        Universe.add_custom("test-universe", ["AAPL"])

        # Should be retrievable with different casing
        assert Universe.get("test_universe") == ["AAPL"]
        assert Universe.get("TEST_UNIVERSE") == ["AAPL"]

        # Cleanup
        Universe.remove_custom("test-universe")

    def test_add_custom_universe_stores_copy(self):
        """add_custom() stores a copy of the symbol list."""
        original = ["AAPL", "MSFT"]
        Universe.add_custom("test", original)

        # Modifying original should not affect stored universe
        original.append("GOOGL")
        assert Universe.get("test") == ["AAPL", "MSFT"]

        # Cleanup
        Universe.remove_custom("test")

    def test_add_custom_duplicate_raises(self):
        """Cannot add custom universe with existing name."""
        Universe.add_custom("test", ["AAPL"])

        with pytest.raises(ValueError, match="already exists"):
            Universe.add_custom("test", ["MSFT"])

        # Cleanup
        Universe.remove_custom("test")

    def test_add_custom_cannot_override_builtin(self):
        """Cannot add custom universe with built-in name."""
        with pytest.raises(ValueError, match="already exists"):
            Universe.add_custom("SP500", ["AAPL"])

        with pytest.raises(ValueError, match="already exists"):
            Universe.add_custom("nasdaq100", ["MSFT"])

    def test_remove_custom_universe(self):
        """Can remove a custom universe."""
        Universe.add_custom("temp", ["AAPL"])
        assert "TEMP" in Universe.list_universes()

        Universe.remove_custom("temp")
        assert "TEMP" not in Universe.list_universes()

        with pytest.raises(ValueError, match="Unknown universe"):
            Universe.get("temp")

    def test_remove_custom_nonexistent_raises(self):
        """Cannot remove non-existent universe."""
        with pytest.raises(ValueError, match="does not exist"):
            Universe.remove_custom("nonexistent")

    def test_remove_custom_cannot_remove_builtin(self):
        """Cannot remove built-in universes."""
        with pytest.raises(ValueError, match="Cannot remove built-in"):
            Universe.remove_custom("SP500")

        with pytest.raises(ValueError, match="Cannot remove built-in"):
            Universe.remove_custom("NASDAQ100")

        with pytest.raises(ValueError, match="Cannot remove built-in"):
            Universe.remove_custom("crypto_top_100")

        with pytest.raises(ValueError, match="Cannot remove built-in"):
            Universe.remove_custom("forex_majors")


class TestUniverseEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_custom_universe(self):
        """Can create custom universe with empty symbol list."""
        Universe.add_custom("empty", [])
        assert Universe.get("empty") == []

        # Cleanup
        Universe.remove_custom("empty")

    def test_custom_universe_with_duplicates(self):
        """Custom universe can have duplicates (not automatically deduplicated)."""
        Universe.add_custom("dupes", ["AAPL", "AAPL", "MSFT"])
        symbols = Universe.get("dupes")
        assert len(symbols) == 3  # Includes duplicate

        # Cleanup
        Universe.remove_custom("dupes")

    def test_get_with_extra_whitespace(self):
        """get() handles extra whitespace in universe name."""
        symbols = Universe.get("  sp500  ")
        assert symbols == Universe.SP500

    def test_forex_pairs_are_six_chars(self):
        """All forex pairs are exactly 6 characters (XXXYYY format)."""
        for pair in Universe.FOREX_MAJORS:
            assert len(pair) == 6, f"Forex pair {pair} should be 6 characters"
            assert pair.isupper(), f"Forex pair {pair} should be uppercase"
            assert pair.isalpha(), f"Forex pair {pair} should be alphabetic"

    def test_crypto_symbols_no_quote_currency(self):
        """Crypto symbols are base currency only (no -USD or -USDT suffix)."""
        for symbol in Universe.CRYPTO_TOP_100:
            # Should not contain hyphens (quote currency separator)
            # This is the main check - symbols should be base currency only
            assert "-" not in symbol, f"Crypto symbol {symbol} should not contain hyphen"

            # Additional sanity check: should not be empty or whitespace-only
            assert symbol.strip() == symbol, f"Crypto symbol {symbol} should not have whitespace"
            assert len(symbol) > 0, "Crypto symbol should not be empty"
