"""Tests for asset validation module."""

from __future__ import annotations

from datetime import datetime

import polars as pl
import pytest

from ml4t.data.assets.asset_class import AssetClass, AssetInfo
from ml4t.data.assets.validation import AssetValidator


class TestAssetValidator:
    """Tests for AssetValidator class."""

    @pytest.fixture
    def equity_validator(self):
        """Create equity asset validator."""
        info = AssetInfo(
            symbol="AAPL",
            asset_class=AssetClass.EQUITY,
            exchange="NASDAQ",
        )
        return AssetValidator(info)

    @pytest.fixture
    def crypto_validator(self):
        """Create crypto asset validator."""
        info = AssetInfo(
            symbol="BTC/USDT",
            asset_class=AssetClass.CRYPTO,
            exchange="binance",
        )
        return AssetValidator(info)

    @pytest.fixture
    def forex_validator(self):
        """Create forex asset validator."""
        info = AssetInfo(
            symbol="EURUSD",
            asset_class=AssetClass.FOREX,
        )
        return AssetValidator(info)

    @pytest.fixture
    def stablecoin_validator(self):
        """Create stablecoin asset validator."""
        info = AssetInfo(
            symbol="USDT/USD",
            asset_class=AssetClass.CRYPTO,
            exchange="binance",
        )
        # Manually set stablecoin flag
        info.is_stablecoin = True
        return AssetValidator(info)


class TestAssetValidatorInit:
    """Tests for AssetValidator initialization."""

    def test_init_equity(self):
        """Test initializing equity validator."""
        info = AssetInfo(symbol="AAPL", asset_class=AssetClass.EQUITY)
        validator = AssetValidator(info)

        assert validator.asset_info == info
        assert validator.asset_class == AssetClass.EQUITY
        assert validator.calendar is None
        assert validator.session_validator is None

    def test_init_crypto_creates_calendar(self):
        """Test initializing crypto validator creates calendar."""
        info = AssetInfo(
            symbol="BTC/USDT",
            asset_class=AssetClass.CRYPTO,
            exchange="binance",
        )
        validator = AssetValidator(info)

        assert validator.calendar is not None
        assert validator.session_validator is not None


class TestValidateCommon:
    """Tests for common validation rules."""

    def test_validate_empty_dataframe(self):
        """Test validating empty DataFrame."""
        info = AssetInfo(symbol="AAPL", asset_class=AssetClass.EQUITY)
        validator = AssetValidator(info)

        df = pl.DataFrame()
        results = validator.validate(df)

        assert "data_quality" in results
        assert any("empty" in issue.lower() for issue in results["data_quality"])

    def test_validate_null_values_in_required_columns(self):
        """Test detecting null values in required columns."""
        info = AssetInfo(symbol="AAPL", asset_class=AssetClass.EQUITY)
        validator = AssetValidator(info)

        df = pl.DataFrame(
            {
                "timestamp": [datetime(2023, 1, 1), datetime(2023, 1, 2)],
                "open": [100.0, None],
                "high": [101.0, 102.0],
                "low": [99.0, 100.0],
                "close": [100.5, 101.5],
                "volume": [1000, 2000],
            }
        )

        results = validator.validate(df)

        assert "data_quality" in results
        assert any("null" in issue.lower() for issue in results["data_quality"])

    def test_validate_ohlc_relationships_high_low(self):
        """Test OHLC validation: high >= low."""
        info = AssetInfo(symbol="AAPL", asset_class=AssetClass.EQUITY)
        validator = AssetValidator(info)

        df = pl.DataFrame(
            {
                "timestamp": [datetime(2023, 1, 1)],
                "open": [100.0],
                "high": [99.0],  # Invalid: high < low
                "low": [101.0],
                "close": [100.5],
                "volume": [1000],
            }
        )

        results = validator.validate(df)

        assert "data_quality" in results
        assert any("high < low" in issue for issue in results["data_quality"])

    def test_validate_ohlc_relationships_high_open_close(self):
        """Test OHLC validation: high >= open, close."""
        info = AssetInfo(symbol="AAPL", asset_class=AssetClass.EQUITY)
        validator = AssetValidator(info)

        df = pl.DataFrame(
            {
                "timestamp": [datetime(2023, 1, 1)],
                "open": [105.0],  # Open > High (invalid)
                "high": [101.0],
                "low": [99.0],
                "close": [100.5],
                "volume": [1000],
            }
        )

        results = validator.validate(df)

        assert "data_quality" in results
        assert any("high < open" in issue for issue in results["data_quality"])

    def test_validate_ohlc_relationships_low_open_close(self):
        """Test OHLC validation: low <= open, close."""
        info = AssetInfo(symbol="AAPL", asset_class=AssetClass.EQUITY)
        validator = AssetValidator(info)

        df = pl.DataFrame(
            {
                "timestamp": [datetime(2023, 1, 1)],
                "open": [100.0],
                "high": [101.0],
                "low": [102.0],  # Low > Open (invalid)
                "close": [100.5],
                "volume": [1000],
            }
        )

        results = validator.validate(df)

        assert "data_quality" in results
        assert any("low > open" in issue for issue in results["data_quality"])

    def test_validate_duplicate_timestamps(self):
        """Test detecting duplicate timestamps."""
        info = AssetInfo(symbol="AAPL", asset_class=AssetClass.EQUITY)
        validator = AssetValidator(info)

        df = pl.DataFrame(
            {
                "timestamp": [
                    datetime(2023, 1, 1),
                    datetime(2023, 1, 1),  # Duplicate
                ],
                "open": [100.0, 100.5],
                "high": [101.0, 101.5],
                "low": [99.0, 99.5],
                "close": [100.5, 101.0],
                "volume": [1000, 2000],
            }
        )

        results = validator.validate(df)

        assert "data_quality" in results
        assert any("duplicate" in issue.lower() for issue in results["data_quality"])

    def test_validate_chronological_order(self):
        """Test detecting non-chronological timestamps."""
        info = AssetInfo(symbol="AAPL", asset_class=AssetClass.EQUITY)
        validator = AssetValidator(info)

        df = pl.DataFrame(
            {
                "timestamp": [
                    datetime(2023, 1, 2),
                    datetime(2023, 1, 1),  # Out of order
                ],
                "open": [100.0, 100.5],
                "high": [101.0, 101.5],
                "low": [99.0, 99.5],
                "close": [100.5, 101.0],
                "volume": [1000, 2000],
            }
        )

        results = validator.validate(df)

        assert "data_quality" in results
        assert any("chronological" in issue.lower() for issue in results["data_quality"])

    def test_validate_valid_data_no_issues(self):
        """Test valid data returns empty results."""
        info = AssetInfo(symbol="AAPL", asset_class=AssetClass.EQUITY)
        validator = AssetValidator(info)

        df = pl.DataFrame(
            {
                "timestamp": [datetime(2023, 1, 2), datetime(2023, 1, 3)],
                "open": [100.0, 100.5],
                "high": [101.0, 101.5],
                "low": [99.0, 99.5],
                "close": [100.5, 101.0],
                "volume": [1000.0, 2000.0],
            }
        )

        results = validator.validate(df)

        # Should have no issues for valid weekday data
        assert "data_quality" not in results or len(results.get("data_quality", [])) == 0


class TestValidateEquity:
    """Tests for equity-specific validation."""

    def test_validate_weekend_data(self):
        """Test detecting weekend data for equities."""
        info = AssetInfo(symbol="AAPL", asset_class=AssetClass.EQUITY)
        validator = AssetValidator(info)

        # January 7, 2023 is a Saturday
        df = pl.DataFrame(
            {
                "timestamp": [datetime(2023, 1, 7)],
                "open": [100.0],
                "high": [101.0],
                "low": [99.0],
                "close": [100.5],
                "volume": [1000.0],
            }
        )

        results = validator.validate(df)

        assert "equity_specific" in results
        assert any("weekend" in issue.lower() for issue in results["equity_specific"])

    def test_validate_negative_prices(self):
        """Test detecting non-positive prices."""
        info = AssetInfo(symbol="AAPL", asset_class=AssetClass.EQUITY)
        validator = AssetValidator(info)

        df = pl.DataFrame(
            {
                "timestamp": [datetime(2023, 1, 2)],
                "open": [100.0],
                "high": [101.0],
                "low": [99.0],
                "close": [0.0],  # Non-positive
                "volume": [1000.0],
            }
        )

        results = validator.validate(df)

        assert "equity_specific" in results
        assert any("non-positive" in issue.lower() for issue in results["equity_specific"])

    def test_validate_extremely_high_prices(self):
        """Test detecting extremely high prices."""
        info = AssetInfo(symbol="AAPL", asset_class=AssetClass.EQUITY)
        validator = AssetValidator(info)

        df = pl.DataFrame(
            {
                "timestamp": [datetime(2023, 1, 2)],
                "open": [100.0],
                "high": [101.0],
                "low": [99.0],
                "close": [2000000.0],  # > $1M
                "volume": [1000.0],
            }
        )

        results = validator.validate(df)

        assert "equity_specific" in results
        assert any("high prices" in issue.lower() for issue in results["equity_specific"])


class TestValidateForex:
    """Tests for forex-specific validation."""

    def test_validate_saturday_data(self):
        """Test detecting Saturday data for forex."""
        info = AssetInfo(symbol="EURUSD", asset_class=AssetClass.FOREX)
        validator = AssetValidator(info)

        # January 7, 2023 is a Saturday
        df = pl.DataFrame(
            {
                "timestamp": [datetime(2023, 1, 7)],
                "open": [1.0500],
                "high": [1.0600],
                "low": [1.0400],
                "close": [1.0550],
            }
        )

        results = validator.validate(df)

        assert "forex_specific" in results
        assert any("saturday" in issue.lower() for issue in results["forex_specific"])

    def test_validate_low_exchange_rate(self):
        """Test detecting unusually low exchange rates."""
        info = AssetInfo(symbol="EURUSD", asset_class=AssetClass.FOREX)
        validator = AssetValidator(info)

        df = pl.DataFrame(
            {
                "timestamp": [datetime(2023, 1, 2)],
                "open": [0.00001],  # Too low
                "high": [0.00002],
                "low": [0.000005],
                "close": [0.00001],
            }
        )

        results = validator.validate(df)

        assert "forex_specific" in results
        assert any("low exchange rates" in issue.lower() for issue in results["forex_specific"])

    def test_validate_high_exchange_rate(self):
        """Test detecting unusually high exchange rates."""
        info = AssetInfo(symbol="EURUSD", asset_class=AssetClass.FOREX)
        validator = AssetValidator(info)

        df = pl.DataFrame(
            {
                "timestamp": [datetime(2023, 1, 2)],
                "open": [15000.0],  # Too high
                "high": [16000.0],
                "low": [14000.0],
                "close": [15500.0],
            }
        )

        results = validator.validate(df)

        assert "forex_specific" in results
        assert any("high exchange rates" in issue.lower() for issue in results["forex_specific"])

    def test_validate_negative_spread(self):
        """Test detecting negative bid-ask spread."""
        info = AssetInfo(symbol="EURUSD", asset_class=AssetClass.FOREX)
        validator = AssetValidator(info)

        df = pl.DataFrame(
            {
                "timestamp": [datetime(2023, 1, 2)],
                "open": [1.0500],
                "high": [1.0600],
                "low": [1.0400],
                "close": [1.0550],
                "bid": [1.0560],  # Bid > Ask (invalid)
                "ask": [1.0540],
            }
        )

        results = validator.validate(df)

        assert "forex_specific" in results
        assert any("negative spreads" in issue.lower() for issue in results["forex_specific"])


class TestValidateCrypto:
    """Tests for crypto-specific validation."""

    def test_validate_extreme_price_movement(self):
        """Test detecting extreme price movements."""
        info = AssetInfo(
            symbol="BTC/USDT",
            asset_class=AssetClass.CRYPTO,
            exchange="binance",
        )
        validator = AssetValidator(info)

        df = pl.DataFrame(
            {
                "timestamp": [datetime(2023, 1, 1), datetime(2023, 1, 2)],
                "open": [100.0, 200.0],
                "high": [101.0, 201.0],
                "low": [99.0, 199.0],
                "close": [100.0, 200.0],  # 100% move
                "volume": [1000.0, 2000.0],
            }
        )

        results = validator.validate(df)

        assert "price_movements" in results
        assert any("extreme" in issue.lower() for issue in results["price_movements"])


class TestValidateStablecoin:
    """Tests for stablecoin-specific validation."""

    def test_validate_depeg_event(self):
        """Test detecting depeg events."""
        info = AssetInfo(
            symbol="USDT/USD",
            asset_class=AssetClass.CRYPTO,
            exchange="binance",
        )
        info.is_stablecoin = True
        validator = AssetValidator(info)

        df = pl.DataFrame(
            {
                "timestamp": [datetime(2023, 1, 1)],
                "open": [1.0],
                "high": [1.0],
                "low": [1.0],
                "close": [0.85],  # Depegged (>10% from $1)
                "volume": [1000.0],
            }
        )

        results = validator._validate_stablecoin(df)

        assert len(results) > 0
        assert any("depegged" in issue.lower() for issue in results)

    def test_validate_stablecoin_high_volatility(self):
        """Test detecting high volatility in stablecoin."""
        info = AssetInfo(
            symbol="USDT/USD",
            asset_class=AssetClass.CRYPTO,
            exchange="binance",
        )
        info.is_stablecoin = True
        validator = AssetValidator(info)

        # Create data with high volatility
        closes = [1.0 + (0.1 * (i % 2)) for i in range(20)]  # Alternating 1.0 and 1.1
        df = pl.DataFrame(
            {
                "timestamp": [datetime(2023, 1, i + 1) for i in range(20)],
                "open": [1.0] * 20,
                "high": [1.0] * 20,
                "low": [1.0] * 20,
                "close": closes,
                "volume": [1000.0] * 20,
            }
        )

        results = validator._validate_stablecoin(df)

        # Should detect high volatility
        assert len(results) > 0

    def test_validate_stablecoin_empty_data(self):
        """Test stablecoin validation with empty data."""
        info = AssetInfo(
            symbol="USDT/USD",
            asset_class=AssetClass.CRYPTO,
            exchange="binance",
        )
        info.is_stablecoin = True
        validator = AssetValidator(info)

        df = pl.DataFrame({"close": []})
        results = validator._validate_stablecoin(df)
        assert results == []

    def test_validate_stablecoin_no_close_column(self):
        """Test stablecoin validation without close column."""
        info = AssetInfo(
            symbol="USDT/USD",
            asset_class=AssetClass.CRYPTO,
            exchange="binance",
        )
        info.is_stablecoin = True
        validator = AssetValidator(info)

        df = pl.DataFrame({"other": [1, 2, 3]})
        results = validator._validate_stablecoin(df)
        assert results == []
