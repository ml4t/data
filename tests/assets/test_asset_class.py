"""Tests for asset_class module."""

from __future__ import annotations

from ml4t.data.assets.asset_class import (
    AssetClass,
    AssetInfo,
    MarketHours,
    get_asset_class_from_symbol,
    normalize_crypto_symbol,
    parse_crypto_symbol,
)


class TestAssetClass:
    """Tests for AssetClass enum."""

    def test_all_asset_classes_defined(self):
        """Test all expected asset classes are defined."""
        assert AssetClass.EQUITY.value == "equity"
        assert AssetClass.CRYPTO.value == "crypto"
        assert AssetClass.FOREX.value == "forex"
        assert AssetClass.COMMODITY.value == "commodity"
        assert AssetClass.FIXED_INCOME.value == "fixed_income"
        assert AssetClass.INDEX.value == "index"
        assert AssetClass.ETF.value == "etf"
        assert AssetClass.OPTION.value == "option"
        assert AssetClass.FUTURE.value == "future"

    def test_asset_class_is_str(self):
        """Test AssetClass values are strings."""
        for ac in AssetClass:
            assert isinstance(ac.value, str)


class TestAssetInfo:
    """Tests for AssetInfo dataclass."""

    def test_create_equity_asset(self):
        """Test creating equity asset."""
        info = AssetInfo(
            symbol="AAPL",
            asset_class=AssetClass.EQUITY,
            name="Apple Inc.",
            exchange="NASDAQ",
            sector="Technology",
        )
        assert info.symbol == "AAPL"
        assert info.asset_class == AssetClass.EQUITY
        assert info.sector == "Technology"

    def test_crypto_pair_parsing_slash(self):
        """Test crypto pair parsing with slash separator."""
        info = AssetInfo(
            symbol="BTC/USDT",
            asset_class=AssetClass.CRYPTO,
        )
        assert info.base_currency == "BTC"
        assert info.quote_currency == "USDT"

    def test_crypto_pair_parsing_dash(self):
        """Test crypto pair parsing with dash separator."""
        info = AssetInfo(
            symbol="ETH-USD",
            asset_class=AssetClass.CRYPTO,
        )
        assert info.base_currency == "ETH"
        assert info.quote_currency == "USD"

    def test_stablecoin_detection_usdt(self):
        """Test stablecoin detection for USDT pairs."""
        info = AssetInfo(
            symbol="BTC/USDT",
            asset_class=AssetClass.CRYPTO,
        )
        assert info.is_stablecoin is True

    def test_stablecoin_detection_usdc(self):
        """Test stablecoin detection for USDC pairs."""
        info = AssetInfo(
            symbol="ETH/USDC",
            asset_class=AssetClass.CRYPTO,
        )
        assert info.is_stablecoin is True

    def test_non_stablecoin_pair(self):
        """Test non-stablecoin pair."""
        info = AssetInfo(
            symbol="ETH/BTC",
            asset_class=AssetClass.CRYPTO,
        )
        assert info.is_stablecoin is False

    def test_is_derivative_property(self):
        """Test is_derivative property."""
        option = AssetInfo(symbol="AAPL_C100", asset_class=AssetClass.OPTION)
        future = AssetInfo(symbol="ES", asset_class=AssetClass.FUTURE)
        equity = AssetInfo(symbol="AAPL", asset_class=AssetClass.EQUITY)

        assert option.is_derivative is True
        assert future.is_derivative is True
        assert equity.is_derivative is False

    def test_is_crypto_pair_property(self):
        """Test is_crypto_pair property."""
        pair = AssetInfo(symbol="BTC/USDT", asset_class=AssetClass.CRYPTO)
        single = AssetInfo(symbol="BTC", asset_class=AssetClass.CRYPTO)

        assert pair.is_crypto_pair is True
        assert single.is_crypto_pair is False

    def test_requires_24_7_calendar_property(self):
        """Test requires_24_7_calendar property."""
        crypto = AssetInfo(symbol="BTC", asset_class=AssetClass.CRYPTO)
        forex = AssetInfo(symbol="EURUSD", asset_class=AssetClass.FOREX)
        equity = AssetInfo(symbol="AAPL", asset_class=AssetClass.EQUITY)

        assert crypto.requires_24_7_calendar is True
        assert forex.requires_24_7_calendar is True
        assert equity.requires_24_7_calendar is False


class TestMarketHours:
    """Tests for MarketHours class."""

    def test_get_schedule_nyse(self):
        """Test getting NYSE schedule."""
        schedule = MarketHours.get_schedule(AssetClass.EQUITY, "NYSE")
        assert schedule["open"] == "09:30"
        assert schedule["close"] == "16:00"
        assert schedule["timezone"] == "America/New_York"

    def test_get_schedule_crypto(self):
        """Test getting crypto schedule (24/7)."""
        schedule = MarketHours.get_schedule(AssetClass.CRYPTO)
        assert schedule["open"] == "00:00"
        assert schedule["close"] == "23:59"
        assert 6 in schedule["days"]  # Sunday

    def test_get_schedule_forex(self):
        """Test getting forex schedule."""
        schedule = MarketHours.get_schedule(AssetClass.FOREX)
        assert schedule is not None
        assert schedule["timezone"] == "America/New_York"

    def test_get_schedule_unknown_exchange(self):
        """Test getting schedule for unknown exchange returns default."""
        schedule = MarketHours.get_schedule(AssetClass.EQUITY, "UNKNOWN_EXCHANGE")
        assert schedule is not None  # Returns first available

    def test_get_schedule_no_schedules_returns_none(self):
        """Test asset class with no schedules returns None."""
        schedule = MarketHours.get_schedule(AssetClass.OPTION)
        assert schedule is None


class TestParseCryptoSymbol:
    """Tests for parse_crypto_symbol function."""

    def test_parse_slash_separator(self):
        """Test parsing with slash separator."""
        base, quote = parse_crypto_symbol("BTC/USDT")
        assert base == "BTC"
        assert quote == "USDT"

    def test_parse_dash_separator(self):
        """Test parsing with dash separator."""
        base, quote = parse_crypto_symbol("ETH-USD")
        assert base == "ETH"
        assert quote == "USD"

    def test_parse_no_separator_usdt(self):
        """Test parsing without separator (USDT suffix)."""
        base, quote = parse_crypto_symbol("BTCUSDT")
        assert base == "BTC"
        assert quote == "USDT"

    def test_parse_no_separator_btc(self):
        """Test parsing without separator (BTC suffix)."""
        base, quote = parse_crypto_symbol("ETHBTC")
        assert base == "ETH"
        assert quote == "BTC"

    def test_parse_single_currency(self):
        """Test parsing single currency (no pair)."""
        base, quote = parse_crypto_symbol("BTC")
        assert base == "BTC"
        assert quote == ""

    def test_parse_unknown_format(self):
        """Test parsing unknown format."""
        base, quote = parse_crypto_symbol("UNKNOWN")
        assert base == "UNKNOWN"
        assert quote == ""


class TestNormalizeCryptoSymbol:
    """Tests for normalize_crypto_symbol function."""

    def test_normalize_slash_to_slash(self):
        """Test normalizing with slash separator."""
        result = normalize_crypto_symbol("BTC/USDT", separator="/")
        assert result == "BTC/USDT"

    def test_normalize_dash_to_slash(self):
        """Test normalizing dash to slash."""
        result = normalize_crypto_symbol("BTC-USDT", separator="/")
        assert result == "BTC/USDT"

    def test_normalize_no_separator_to_slash(self):
        """Test normalizing no separator to slash."""
        result = normalize_crypto_symbol("BTCUSDT", separator="/")
        assert result == "BTC/USDT"

    def test_normalize_to_dash(self):
        """Test normalizing to dash separator."""
        result = normalize_crypto_symbol("BTC/USDT", separator="-")
        assert result == "BTC-USDT"

    def test_normalize_single_currency(self):
        """Test normalizing single currency."""
        result = normalize_crypto_symbol("BTC", separator="/")
        assert result == "BTC"


class TestGetAssetClassFromSymbol:
    """Tests for get_asset_class_from_symbol function."""

    def test_detect_crypto_slash(self):
        """Test detecting crypto with slash."""
        assert get_asset_class_from_symbol("BTC/USDT") == AssetClass.CRYPTO

    def test_detect_crypto_dash(self):
        """Test detecting crypto with dash."""
        assert get_asset_class_from_symbol("ETH-USD") == AssetClass.CRYPTO

    def test_detect_crypto_suffix(self):
        """Test detecting crypto with suffix."""
        assert get_asset_class_from_symbol("BTCUSDT") == AssetClass.CRYPTO
        assert get_asset_class_from_symbol("ETHBTC") == AssetClass.CRYPTO

    def test_detect_forex(self):
        """Test detecting forex pairs."""
        assert get_asset_class_from_symbol("EURUSD") == AssetClass.FOREX
        assert get_asset_class_from_symbol("GBPJPY") == AssetClass.FOREX
        assert get_asset_class_from_symbol("AUDUSD") == AssetClass.FOREX

    def test_detect_equity_default(self):
        """Test default to equity."""
        assert get_asset_class_from_symbol("AAPL") == AssetClass.EQUITY
        assert get_asset_class_from_symbol("MSFT") == AssetClass.EQUITY
        assert get_asset_class_from_symbol("GOOGL") == AssetClass.EQUITY
