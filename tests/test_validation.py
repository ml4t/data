"""Tests for validation module."""

from datetime import datetime, timedelta

import polars as pl

from ml4t.data.validation import OHLCVValidator, ValidationReport, ValidationResult
from ml4t.data.validation.base import Severity, ValidationIssue
from ml4t.data.validation.cross_validation import CrossValidator
from ml4t.data.validation.rules import (
    AssetClass,
    ValidationRuleConfig,
    ValidationRulePresets,
    ValidationRuleSet,
)


def create_sample_ohlcv_data(rows: int = 100, pattern: str = "normal") -> pl.DataFrame:
    """Create sample OHLCV data for testing."""
    import random

    random.seed(42)

    base_time = datetime(2024, 1, 1)
    timestamps = [base_time + timedelta(days=i) for i in range(rows)]

    data = {
        "timestamp": timestamps,
        "open": [],
        "high": [],
        "low": [],
        "close": [],
        "volume": [],
    }

    if pattern == "normal":
        # Generate normal OHLCV data
        price = 100.0
        for _ in range(rows):
            daily_return = random.gauss(0, 0.01)
            open_price = price * (1 + daily_return)
            close_price = open_price * (1 + random.gauss(0, 0.005))
            high_price = max(open_price, close_price) * (1 + abs(random.gauss(0, 0.002)))
            low_price = min(open_price, close_price) * (1 - abs(random.gauss(0, 0.002)))

            data["open"].append(open_price)
            data["high"].append(high_price)
            data["low"].append(low_price)
            data["close"].append(close_price)
            data["volume"].append(random.randint(1000000, 10000000))

            price = close_price

    elif pattern == "invalid":
        # Generate data with various issues
        for i in range(rows):
            if i < 10:
                # Negative prices
                data["open"].append(-100.0)
                data["high"].append(-90.0)
                data["low"].append(-110.0)
                data["close"].append(-95.0)
                data["volume"].append(-1000)
            elif i < 20:
                # High < Low
                data["open"].append(100.0)
                data["high"].append(90.0)
                data["low"].append(110.0)
                data["close"].append(100.0)
                data["volume"].append(1000000)
            elif i < 30:
                # Nulls
                data["open"].append(None)
                data["high"].append(None)
                data["low"].append(None)
                data["close"].append(None)
                data["volume"].append(None)
            else:
                # Normal data
                data["open"].append(100.0)
                data["high"].append(105.0)
                data["low"].append(95.0)
                data["close"].append(102.0)
                data["volume"].append(1000000)

    elif pattern == "stale":
        # Generate stale price data
        for i in range(rows):
            if i < 50:
                # First half: normal
                data["open"].append(100.0 + i * 0.1)
                data["high"].append(105.0 + i * 0.1)
                data["low"].append(95.0 + i * 0.1)
                data["close"].append(102.0 + i * 0.1)
                data["volume"].append(1000000)
            else:
                # Second half: stale (same price)
                data["open"].append(150.0)
                data["high"].append(150.0)
                data["low"].append(150.0)
                data["close"].append(150.0)
                data["volume"].append(100)

    elif pattern == "extreme":
        # Generate extreme returns
        for i in range(rows):
            if i == 50:
                # Extreme spike
                data["open"].append(100.0)
                data["high"].append(500.0)
                data["low"].append(100.0)
                data["close"].append(400.0)
                data["volume"].append(100000000)
            else:
                data["open"].append(100.0)
                data["high"].append(105.0)
                data["low"].append(95.0)
                data["close"].append(102.0)
                data["volume"].append(1000000)

    return pl.DataFrame(data)


class TestOHLCVValidator:
    """Tests for OHLCV validator."""

    def test_valid_data(self):
        """Test validator with valid data."""
        df = create_sample_ohlcv_data(100, "normal")
        validator = OHLCVValidator()

        result = validator.validate(df)

        assert result.passed
        assert result.error_count == 0
        assert result.critical_count == 0

    def test_invalid_data(self):
        """Test validator with invalid data."""
        df = create_sample_ohlcv_data(100, "invalid")
        validator = OHLCVValidator()

        result = validator.validate(df)

        assert not result.passed
        assert result.error_count > 0
        assert result.critical_count > 0  # Negative prices are critical

    def test_null_detection(self):
        """Test null value detection."""
        df = create_sample_ohlcv_data(100, "normal")
        # Add some nulls
        df = df.with_columns(
            pl.when(pl.col("timestamp").dt.day() == 15)
            .then(None)
            .otherwise(pl.col("close"))
            .alias("close")
        )

        validator = OHLCVValidator(check_nulls=True)
        result = validator.validate(df)

        assert not result.passed
        null_issues = [i for i in result.issues if i.check == "null_values"]
        assert len(null_issues) > 0

    def test_price_consistency(self):
        """Test price consistency checks."""
        df = pl.DataFrame(
            {
                "timestamp": [datetime(2024, 1, i) for i in range(1, 6)],
                "open": [100.0, 100.0, 100.0, 100.0, 100.0],
                "high": [90.0, 105.0, 105.0, 105.0, 105.0],  # First row: high < low
                "low": [95.0, 95.0, 95.0, 95.0, 95.0],
                "close": [100.0, 100.0, 100.0, 100.0, 100.0],
                "volume": [1000000] * 5,
            }
        )

        validator = OHLCVValidator(check_price_consistency=True)
        result = validator.validate(df)

        assert not result.passed
        consistency_issues = [i for i in result.issues if i.check == "price_consistency"]
        assert len(consistency_issues) > 0

    def test_stale_price_detection(self):
        """Test stale price detection."""
        df = create_sample_ohlcv_data(100, "stale")

        validator = OHLCVValidator(check_price_staleness=True, staleness_threshold=5)
        result = validator.validate(df)

        staleness_issues = [i for i in result.issues if i.check == "price_staleness"]
        assert len(staleness_issues) > 0

    def test_extreme_returns(self):
        """Test extreme return detection."""
        df = create_sample_ohlcv_data(100, "extreme")

        validator = OHLCVValidator(check_extreme_returns=True, max_return_threshold=0.5)
        result = validator.validate(df)

        extreme_issues = [i for i in result.issues if i.check == "extreme_returns"]
        assert len(extreme_issues) > 0


class TestCrossValidator:
    """Tests for cross-validation."""

    def test_price_continuity(self):
        """Test price continuity check."""
        df = create_sample_ohlcv_data(100, "normal")
        # Create a gap
        df = df.with_columns(
            pl.when(pl.col("timestamp").dt.day() == 15)
            .then(pl.col("open") * 1.2)  # 20% gap
            .otherwise(pl.col("open"))
            .alias("open")
        )

        validator = CrossValidator(check_price_continuity=True, price_gap_threshold=0.1)
        result = validator.validate(df)

        gap_issues = [i for i in result.issues if i.check == "price_continuity"]
        assert len(gap_issues) > 0

    def test_volume_spikes(self):
        """Test volume spike detection."""
        df = create_sample_ohlcv_data(100, "normal")
        # Create a volume spike at a later position (after rolling window can establish)
        df = df.with_columns(
            pl.when(pl.col("timestamp").dt.day() == 25)
            .then(pl.col("volume") * 20)  # 20x spike
            .otherwise(pl.col("volume"))
            .alias("volume")
        )

        validator = CrossValidator(check_volume_spikes=True, volume_spike_threshold=10.0)
        result = validator.validate(df)

        spike_issues = [i for i in result.issues if i.check == "volume_spikes"]
        assert len(spike_issues) > 0

    def test_weekend_trading(self):
        """Test weekend trading detection for non-crypto."""
        # Create data with weekend dates
        base_time = datetime(2024, 1, 6)  # Saturday
        timestamps = [base_time + timedelta(days=i) for i in range(10)]

        df = pl.DataFrame(
            {
                "timestamp": timestamps,
                "open": [100.0] * 10,
                "high": [105.0] * 10,
                "low": [95.0] * 10,
                "close": [102.0] * 10,
                "volume": [1000000] * 10,
            }
        )

        validator = CrossValidator(check_weekend_trading=True, is_crypto=False)
        result = validator.validate(df)

        weekend_issues = [i for i in result.issues if i.check == "weekend_trading"]
        assert len(weekend_issues) > 0

    def test_crypto_no_weekend_check(self):
        """Test that crypto doesn't flag weekend trading."""
        # Create data with weekend dates
        base_time = datetime(2024, 1, 6)  # Saturday
        timestamps = [base_time + timedelta(days=i) for i in range(10)]

        df = pl.DataFrame(
            {
                "timestamp": timestamps,
                "open": [100.0] * 10,
                "high": [105.0] * 10,
                "low": [95.0] * 10,
                "close": [102.0] * 10,
                "volume": [1000000] * 10,
            }
        )

        validator = CrossValidator(
            check_weekend_trading=True,
            is_crypto=True,  # Crypto flag
        )
        result = validator.validate(df)

        weekend_issues = [i for i in result.issues if i.check == "weekend_trading"]
        assert len(weekend_issues) == 0


class TestValidationReport:
    """Tests for validation reporting."""

    def test_report_creation(self):
        """Test creating a validation report."""
        report = ValidationReport(symbol="AAPL", provider="yahoo")

        # Add some results
        result1 = ValidationResult(passed=True)
        result2 = ValidationResult(passed=False)
        result2.add_issue(
            ValidationIssue(severity=Severity.ERROR, check="test_check", message="Test issue")
        )

        report.add_result(result1, "OHLCVValidator")
        report.add_result(result2, "CrossValidator")

        assert report.total_issues == 1
        assert report.error_count == 1
        assert not report.passed

    def test_report_serialization(self, tmp_path):
        """Test report serialization."""
        report = ValidationReport(symbol="AAPL", provider="yahoo")

        result = ValidationResult(passed=False)
        result.add_issue(
            ValidationIssue(
                severity=Severity.WARNING,
                check="test_check",
                message="Test warning",
                details={"key": "value"},
                row_count=5,
            )
        )

        report.add_result(result, "TestValidator")

        # Save and load
        report_path = tmp_path / "report.json"
        report.save(report_path)

        loaded_report = ValidationReport.load(report_path)

        assert loaded_report.symbol == "AAPL"
        assert loaded_report.provider == "yahoo"
        assert loaded_report.total_issues == 1
        assert loaded_report.warning_count == 1

    def test_report_dataframe(self):
        """Test converting report to DataFrame."""
        report = ValidationReport(symbol="AAPL", provider="yahoo")

        result = ValidationResult(passed=False)
        result.add_issue(
            ValidationIssue(
                severity=Severity.ERROR, check="check1", message="Error 1", row_count=10
            )
        )
        result.add_issue(
            ValidationIssue(
                severity=Severity.WARNING, check="check2", message="Warning 1", row_count=5
            )
        )

        report.add_result(result, "TestValidator")

        df = report.to_dataframe()

        assert len(df) == 2
        assert "validator" in df.columns
        assert "severity" in df.columns
        assert df["row_count"].sum() == 15


class TestValidationRules:
    """Tests for validation rules."""

    def test_rule_config(self):
        """Test validation rule configuration."""
        config = ValidationRuleConfig(
            asset_class=AssetClass.CRYPTO, max_return_threshold=0.75, check_weekend_trading=False
        )

        assert config.asset_class == "crypto"
        assert config.max_return_threshold == 0.75
        assert not config.check_weekend_trading

    def test_rule_presets(self):
        """Test validation rule presets."""
        equity_rules = ValidationRulePresets.equity_rules()
        assert equity_rules.check_weekend_trading
        assert equity_rules.max_return_threshold == 0.2

        crypto_rules = ValidationRulePresets.crypto_rules()
        assert not crypto_rules.check_weekend_trading
        assert crypto_rules.max_return_threshold == 0.5

        strict_rules = ValidationRulePresets.strict_rules()
        assert strict_rules.max_return_threshold == 0.1

    def test_rule_set(self, tmp_path):
        """Test validation rule set."""
        rule_set = ValidationRuleSet(name="test_rules", description="Test rule set")

        # Add default rule
        rule_set.default_rule = ValidationRulePresets.equity_rules()

        # Add specific rules
        rule_set.add_rule("BTC*", ValidationRulePresets.crypto_rules())
        rule_set.add_rule("EUR*", ValidationRulePresets.forex_rules())

        # Test rule retrieval
        assert rule_set.get_rules("AAPL").asset_class == "equity"
        assert rule_set.get_rules("BTCUSD").asset_class == "crypto"
        assert rule_set.get_rules("EURUSD").asset_class == "forex"

        # Test save/load
        rule_path = tmp_path / "rules.yaml"
        rule_set.save(rule_path)

        loaded_set = ValidationRuleSet.load(rule_path)
        assert loaded_set.name == "test_rules"
        assert loaded_set.get_rules("BTCUSD").asset_class == "crypto"
