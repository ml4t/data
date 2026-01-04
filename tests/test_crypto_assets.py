"""Tests for crypto and asset class features."""

from datetime import UTC, datetime, timedelta

import polars as pl

from ml4t.data.assets import AssetClass, AssetInfo, AssetSchema, AssetValidator
from ml4t.data.assets.asset_class import (
    get_asset_class_from_symbol,
    normalize_crypto_symbol,
    parse_crypto_symbol,
)
from ml4t.data.calendar.crypto import CryptoCalendar


class TestCryptoCalendar:
    """Test crypto market calendar."""

    def test_always_open(self):
        """Test that crypto markets are always open."""
        calendar = CryptoCalendar()

        # Test various times
        times = [
            datetime(2024, 1, 1, 0, 0, tzinfo=UTC),  # New Year
            datetime(2024, 7, 4, 12, 0, tzinfo=UTC),  # July 4th
            datetime(2024, 12, 25, 18, 0, tzinfo=UTC),  # Christmas
            datetime(2024, 3, 15, 3, 30, tzinfo=UTC),  # Random time
        ]

        for timestamp in times:
            assert calendar.is_open(timestamp) is True

    def test_sessions(self):
        """Test session generation for crypto."""
        calendar = CryptoCalendar()

        start = datetime(2024, 1, 1, tzinfo=UTC)
        end = datetime(2024, 1, 7, tzinfo=UTC)

        sessions = calendar.get_sessions(start, end)

        # Should have 6 daily sessions
        assert len(sessions) == 6

        # Each session should be 24 hours
        for session_start, session_end in sessions[:-1]:  # Exclude last partial
            duration = session_end - session_start
            assert duration == timedelta(days=1)

    def test_expected_sessions_count(self):
        """Test expected data point calculation."""
        calendar = CryptoCalendar()

        start = datetime(2024, 1, 1, tzinfo=UTC)
        end = datetime(2024, 1, 8, tzinfo=UTC)  # 7 days

        # Daily
        assert calendar.get_expected_sessions_count(start, end, "daily") == 7

        # Hourly
        assert calendar.get_expected_sessions_count(start, end, "hourly") == 7 * 24

        # 5-minute
        assert calendar.get_expected_sessions_count(start, end, "5minute") == 7 * 24 * 12

    def test_gap_detection(self):
        """Test gap detection in crypto data."""
        calendar = CryptoCalendar()

        # Create data with a gap
        timestamps = [datetime(2024, 1, 1, i, 0, tzinfo=UTC) for i in range(10)]
        # Skip hours 10-14 (5 hour gap)
        timestamps.extend([datetime(2024, 1, 1, i, 0, tzinfo=UTC) for i in range(15, 24)])

        df = pl.DataFrame(
            {
                "timestamp": timestamps,
                "close": [100.0] * len(timestamps),
            }
        )

        gaps = calendar.find_gaps(df, frequency="hourly")

        assert len(gaps) == 1
        gap_start, gap_end = gaps[0]
        assert gap_start.hour == 9
        assert gap_end.hour == 15


class TestAssetClass:
    """Test asset class definitions."""

    def test_asset_info_crypto(self):
        """Test AssetInfo for crypto assets."""
        # Test with slash separator
        info = AssetInfo(
            symbol="BTC/USDT",
            asset_class=AssetClass.CRYPTO,
            exchange="binance",
        )

        assert info.base_currency == "BTC"
        assert info.quote_currency == "USDT"
        assert info.is_stablecoin is True
        assert info.is_crypto_pair is True
        assert info.requires_24_7_calendar is True

    def test_asset_info_equity(self):
        """Test AssetInfo for equity assets."""
        info = AssetInfo(
            symbol="AAPL",
            asset_class=AssetClass.EQUITY,
            exchange="NASDAQ",
            sector="Technology",
        )

        assert info.is_derivative is False
        assert info.requires_24_7_calendar is False

    def test_parse_crypto_symbol(self):
        """Test crypto symbol parsing."""
        # Test different formats
        assert parse_crypto_symbol("BTC/USDT") == ("BTC", "USDT")
        assert parse_crypto_symbol("BTC-USDT") == ("BTC", "USDT")
        assert parse_crypto_symbol("BTCUSDT") == ("BTC", "USDT")
        assert parse_crypto_symbol("ETHBTC") == ("ETH", "BTC")

    def test_normalize_crypto_symbol(self):
        """Test crypto symbol normalization."""
        assert normalize_crypto_symbol("BTCUSDT") == "BTC/USDT"
        assert normalize_crypto_symbol("BTC-USDT") == "BTC/USDT"
        assert normalize_crypto_symbol("BTC/USDT", "-") == "BTC-USDT"

    def test_asset_class_inference(self):
        """Test inferring asset class from symbol."""
        assert get_asset_class_from_symbol("BTC/USDT") == AssetClass.CRYPTO
        assert get_asset_class_from_symbol("BTCUSDT") == AssetClass.CRYPTO
        assert get_asset_class_from_symbol("EURUSD") == AssetClass.FOREX
        assert get_asset_class_from_symbol("AAPL") == AssetClass.EQUITY


class TestAssetSchema:
    """Test asset-specific schemas."""

    def test_crypto_schema(self):
        """Test crypto asset schema."""
        schema = AssetSchema.get_schema(AssetClass.CRYPTO)

        assert "timestamp" in schema["required"]
        assert "volume_quote" in schema["optional"]
        assert schema["types"]["trades_count"] == pl.Int64

    def test_validate_dataframe(self):
        """Test DataFrame validation against schema."""
        # Create valid crypto DataFrame
        df = pl.DataFrame(
            {
                "timestamp": [datetime.now()],
                "open": [100.0],
                "high": [101.0],
                "low": [99.0],
                "close": [100.5],
                "volume": [1000.0],
            }
        )

        is_valid, issues = AssetSchema.validate_dataframe(df, AssetClass.CRYPTO)
        assert is_valid is True
        assert len(issues) == 0

        # Test with missing required column
        df_invalid = df.drop("volume")
        is_valid, issues = AssetSchema.validate_dataframe(df_invalid, AssetClass.CRYPTO)
        assert is_valid is False
        assert "Missing required column: volume" in issues

    def test_normalize_dataframe(self):
        """Test DataFrame normalization."""
        # Create DataFrame with wrong types
        df = pl.DataFrame(
            {
                "timestamp": [datetime.now()],
                "open": ["100.0"],  # String instead of float
                "high": [101],  # Int instead of float
                "low": [99.0],
                "close": [100.5],
            }
        )

        normalized = AssetSchema.normalize_dataframe(df, AssetClass.CRYPTO)

        # Should have volume column added
        assert "volume" in normalized.columns

        # Types should be corrected
        assert normalized["open"].dtype in [pl.Float64, pl.Float32]


class TestAssetValidator:
    """Test asset-specific validation."""

    def test_crypto_validation(self):
        """Test crypto-specific validation rules."""
        info = AssetInfo(
            symbol="BTC/USDT",
            asset_class=AssetClass.CRYPTO,
        )
        validator = AssetValidator(info)

        # Create crypto data with issues
        df = pl.DataFrame(
            {
                "timestamp": [datetime(2024, 1, i, tzinfo=UTC) for i in range(1, 6)],
                "open": [40000.0, 41000.0, 42000.0, 43000.0, 44000.0],
                "high": [40500.0, 41500.0, 42500.0, 43500.0, 44500.0],
                "low": [39500.0, 40500.0, 41500.0, 42500.0, 43500.0],
                "close": [40200.0, 41200.0, 42200.0, 80000.0, 44200.0],  # Spike
                "volume": [1000.0, 1100.0, 0.0, 1300.0, 1400.0],  # Zero volume
            }
        )

        issues = validator.validate(df, frequency="daily")

        # Should detect issues
        assert "volume" in issues or "price_movements" in issues

    def test_stablecoin_validation(self):
        """Test stablecoin-specific validation."""
        info = AssetInfo(
            symbol="USDT/USD",
            asset_class=AssetClass.CRYPTO,
        )
        validator = AssetValidator(info)

        # Create stablecoin data with depeg
        df = pl.DataFrame(
            {
                "timestamp": [datetime(2024, 1, i) for i in range(1, 6)],
                "open": [1.0, 1.0, 0.85, 1.0, 1.0],  # Depeg event
                "high": [1.01, 1.01, 0.90, 1.01, 1.01],
                "low": [0.99, 0.99, 0.80, 0.99, 0.99],
                "close": [1.0, 1.0, 0.85, 1.0, 1.0],
                "volume": [1000000.0] * 5,
            }
        )

        issues = validator.validate(df)

        # Should detect depeg
        assert "stablecoin" in issues
        assert any("depegged" in issue for issue in issues["stablecoin"])

    def test_equity_validation(self):
        """Test equity-specific validation."""
        info = AssetInfo(
            symbol="AAPL",
            asset_class=AssetClass.EQUITY,
            exchange="NASDAQ",
        )
        validator = AssetValidator(info)

        # Create equity data with weekend trading
        df = pl.DataFrame(
            {
                "timestamp": [
                    datetime(2024, 1, 6, 10, 0),  # Saturday
                    datetime(2024, 1, 7, 14, 0),  # Sunday
                ],
                "open": [150.0, 151.0],
                "high": [152.0, 153.0],
                "low": [149.0, 150.0],
                "close": [151.0, 152.0],
                "volume": [1000000.0, 1100000.0],
            }
        )

        issues = validator.validate(df)

        # Should detect weekend data
        assert "equity_specific" in issues
        assert any("weekend" in issue for issue in issues["equity_specific"])
