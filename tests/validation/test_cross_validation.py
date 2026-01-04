"""Comprehensive tests for CrossValidator.

Tests cover:
- Price continuity checks
- Volume spike detection
- Weekend trading detection
- Market hours validation
- Cross-reference validation
"""

from datetime import UTC, datetime, timedelta

import polars as pl

from ml4t.data.validation.base import Severity
from ml4t.data.validation.cross_validation import CrossValidator


def create_valid_ohlcv_df(
    n_rows: int = 30,
    start_date: datetime | None = None,
    base_price: float = 100.0,
) -> pl.DataFrame:
    """Create a valid OHLCV DataFrame for testing."""
    if start_date is None:
        # Start on Monday at 14:30 UTC (9:30 ET)
        start_date = datetime(2024, 1, 1, 14, 30, tzinfo=UTC)

    timestamps = []
    current_date = start_date
    for _ in range(n_rows):
        # Skip weekends
        while current_date.weekday() >= 5:
            current_date += timedelta(days=1)
        timestamps.append(current_date)
        current_date += timedelta(days=1)

    return pl.DataFrame(
        {
            "timestamp": timestamps,
            "open": [base_price + i * 0.5 for i in range(n_rows)],
            "high": [base_price + i * 0.5 + 1.0 for i in range(n_rows)],
            "low": [base_price + i * 0.5 - 0.5 for i in range(n_rows)],
            "close": [base_price + i * 0.5 + 0.3 for i in range(n_rows)],
            "volume": [1000000.0 + i * 10000 for i in range(n_rows)],
        }
    )


class TestCrossValidatorInitialization:
    """Test validator initialization and configuration."""

    def test_default_initialization(self):
        """All checks enabled by default."""
        validator = CrossValidator()
        assert validator.check_price_continuity is True
        assert validator.check_volume_spikes is True
        assert validator.check_weekend_trading is True
        assert validator.check_market_hours is True
        assert validator.is_crypto is False

    def test_default_thresholds(self):
        """Default thresholds set correctly."""
        validator = CrossValidator()
        assert validator.volume_spike_threshold == 10.0
        assert validator.price_gap_threshold == 0.1

    def test_custom_thresholds(self):
        """Custom thresholds are stored correctly."""
        validator = CrossValidator(
            volume_spike_threshold=5.0,
            price_gap_threshold=0.05,
        )
        assert validator.volume_spike_threshold == 5.0
        assert validator.price_gap_threshold == 0.05

    def test_crypto_mode(self):
        """Crypto mode disables weekend/market hours checks."""
        validator = CrossValidator(is_crypto=True)
        assert validator.is_crypto is True

    def test_disable_specific_checks(self):
        """Individual checks can be disabled."""
        validator = CrossValidator(
            check_price_continuity=False,
            check_volume_spikes=False,
        )
        assert validator.check_price_continuity is False
        assert validator.check_volume_spikes is False

    def test_name_property(self):
        """Validator name is correct."""
        validator = CrossValidator()
        assert validator.name() == "CrossValidator"


class TestPriceContinuityCheck:
    """Test price gap detection between sessions."""

    def test_continuous_prices_pass(self):
        """Continuous prices without gaps pass."""
        validator = CrossValidator(price_gap_threshold=0.1)
        df = create_valid_ohlcv_df()
        result = validator.validate(df)

        gap_issues = [i for i in result.issues if i.check == "price_continuity"]
        assert len(gap_issues) == 0

    def test_price_gap_detected(self):
        """Significant price gap detected as warning."""
        validator = CrossValidator(price_gap_threshold=0.05)
        # Create data with a 15% price gap
        df = pl.DataFrame(
            {
                "timestamp": [
                    datetime(2024, 1, 1, 14, 30, tzinfo=UTC),
                    datetime(2024, 1, 2, 14, 30, tzinfo=UTC),
                    datetime(2024, 1, 3, 14, 30, tzinfo=UTC),
                ],
                "open": [100.0, 115.0, 116.0],  # 15% gap from close to next open
                "high": [101.0, 116.0, 117.0],
                "low": [99.0, 114.0, 115.0],
                "close": [100.0, 115.0, 116.0],
                "volume": [1000000.0] * 3,
            }
        )
        result = validator.validate(df)

        gap_issues = [i for i in result.issues if i.check == "price_continuity"]
        assert len(gap_issues) == 1
        assert gap_issues[0].severity == Severity.WARNING

    def test_single_row_skipped(self):
        """Single row DataFrame skipped."""
        validator = CrossValidator()
        df = pl.DataFrame(
            {
                "timestamp": [datetime(2024, 1, 1, 14, 30, tzinfo=UTC)],
                "open": [100.0],
                "high": [101.0],
                "low": [99.0],
                "close": [100.0],
                "volume": [1000000.0],
            }
        )
        result = validator.validate(df)

        gap_issues = [i for i in result.issues if i.check == "price_continuity"]
        assert len(gap_issues) == 0

    def test_custom_gap_threshold(self):
        """Custom gap threshold respected."""
        # With 20% threshold, 15% gap should not trigger
        validator = CrossValidator(price_gap_threshold=0.20)
        df = pl.DataFrame(
            {
                "timestamp": [
                    datetime(2024, 1, 1, 14, 30, tzinfo=UTC),
                    datetime(2024, 1, 2, 14, 30, tzinfo=UTC),
                ],
                "open": [100.0, 115.0],  # 15% gap
                "high": [101.0, 116.0],
                "low": [99.0, 114.0],
                "close": [100.0, 115.0],
                "volume": [1000000.0] * 2,
            }
        )
        result = validator.validate(df)

        gap_issues = [i for i in result.issues if i.check == "price_continuity"]
        assert len(gap_issues) == 0

    def test_continuity_check_disabled(self):
        """Price continuity check can be disabled."""
        validator = CrossValidator(check_price_continuity=False)
        df = pl.DataFrame(
            {
                "timestamp": [
                    datetime(2024, 1, 1, 14, 30, tzinfo=UTC),
                    datetime(2024, 1, 2, 14, 30, tzinfo=UTC),
                ],
                "open": [100.0, 200.0],  # 100% gap
                "high": [101.0, 201.0],
                "low": [99.0, 199.0],
                "close": [100.0, 200.0],
                "volume": [1000000.0] * 2,
            }
        )
        result = validator.validate(df)

        gap_issues = [i for i in result.issues if i.check == "price_continuity"]
        assert len(gap_issues) == 0


class TestVolumeSpikeCheck:
    """Test volume spike detection."""

    def test_normal_volume_passes(self):
        """Normal volume variation passes."""
        validator = CrossValidator(volume_spike_threshold=10.0)
        df = create_valid_ohlcv_df(n_rows=30)
        result = validator.validate(df)

        spike_issues = [i for i in result.issues if i.check == "volume_spikes"]
        assert len(spike_issues) == 0

    def test_volume_spike_detected(self):
        """Volume spike detected as info."""
        validator = CrossValidator(volume_spike_threshold=5.0)
        # Create data with a large volume spike
        volumes = [1000000.0] * 25 + [20000000.0] + [1000000.0] * 4  # 20x spike
        df = pl.DataFrame(
            {
                "timestamp": [datetime(2024, 1, i, 14, 30, tzinfo=UTC) for i in range(1, 31)],
                "open": [100.0] * 30,
                "high": [101.0] * 30,
                "low": [99.0] * 30,
                "close": [100.0] * 30,
                "volume": volumes,
            }
        )
        result = validator.validate(df)

        spike_issues = [i for i in result.issues if i.check == "volume_spikes"]
        assert len(spike_issues) == 1
        assert spike_issues[0].severity == Severity.INFO

    def test_short_dataframe_skipped(self):
        """DataFrame with <20 rows skipped."""
        validator = CrossValidator()
        df = create_valid_ohlcv_df(n_rows=15)
        result = validator.validate(df)

        spike_issues = [i for i in result.issues if i.check == "volume_spikes"]
        assert len(spike_issues) == 0

    def test_custom_spike_threshold(self):
        """Custom spike threshold respected."""
        # With very high threshold, spikes should not trigger
        validator = CrossValidator(volume_spike_threshold=100.0)
        volumes = [1000000.0] * 25 + [20000000.0] + [1000000.0] * 4  # 20x spike
        df = pl.DataFrame(
            {
                "timestamp": [datetime(2024, 1, i, 14, 30, tzinfo=UTC) for i in range(1, 31)],
                "open": [100.0] * 30,
                "high": [101.0] * 30,
                "low": [99.0] * 30,
                "close": [100.0] * 30,
                "volume": volumes,
            }
        )
        result = validator.validate(df)

        spike_issues = [i for i in result.issues if i.check == "volume_spikes"]
        assert len(spike_issues) == 0

    def test_volume_check_disabled(self):
        """Volume spike check can be disabled."""
        validator = CrossValidator(check_volume_spikes=False)
        volumes = [1000000.0] * 25 + [100000000.0] + [1000000.0] * 4  # 100x spike
        df = pl.DataFrame(
            {
                "timestamp": [datetime(2024, 1, i, 14, 30, tzinfo=UTC) for i in range(1, 31)],
                "open": [100.0] * 30,
                "high": [101.0] * 30,
                "low": [99.0] * 30,
                "close": [100.0] * 30,
                "volume": volumes,
            }
        )
        result = validator.validate(df)

        spike_issues = [i for i in result.issues if i.check == "volume_spikes"]
        assert len(spike_issues) == 0


class TestWeekendTradingCheck:
    """Test weekend trading detection."""

    def test_weekday_only_passes(self):
        """Data with only weekday trading passes."""
        validator = CrossValidator()
        df = create_valid_ohlcv_df()  # Helper creates weekdays only
        result = validator.validate(df)

        weekend_issues = [i for i in result.issues if i.check == "weekend_trading"]
        assert len(weekend_issues) == 0

    def test_weekend_trading_detected(self):
        """Weekend trading detected as warning."""
        validator = CrossValidator()
        # Create data with Saturday trading
        df = pl.DataFrame(
            {
                "timestamp": [
                    datetime(2024, 1, 6, 14, 30, tzinfo=UTC),  # Saturday
                ],
                "open": [100.0],
                "high": [101.0],
                "low": [99.0],
                "close": [100.0],
                "volume": [1000000.0],
            }
        )
        result = validator.validate(df)

        weekend_issues = [i for i in result.issues if i.check == "weekend_trading"]
        assert len(weekend_issues) == 1
        assert weekend_issues[0].severity == Severity.WARNING

    def test_sunday_detected(self):
        """Sunday trading detected."""
        validator = CrossValidator()
        df = pl.DataFrame(
            {
                "timestamp": [
                    datetime(2024, 1, 7, 14, 30, tzinfo=UTC),  # Sunday
                ],
                "open": [100.0],
                "high": [101.0],
                "low": [99.0],
                "close": [100.0],
                "volume": [1000000.0],
            }
        )
        result = validator.validate(df)

        weekend_issues = [i for i in result.issues if i.check == "weekend_trading"]
        assert len(weekend_issues) == 1

    def test_crypto_skips_weekend_check(self):
        """Crypto mode skips weekend trading check."""
        validator = CrossValidator(is_crypto=True)
        df = pl.DataFrame(
            {
                "timestamp": [
                    datetime(2024, 1, 6, 14, 30, tzinfo=UTC),  # Saturday
                    datetime(2024, 1, 7, 14, 30, tzinfo=UTC),  # Sunday
                ],
                "open": [100.0, 101.0],
                "high": [101.0, 102.0],
                "low": [99.0, 100.0],
                "close": [100.0, 101.0],
                "volume": [1000000.0, 1000000.0],
            }
        )
        result = validator.validate(df)

        weekend_issues = [i for i in result.issues if i.check == "weekend_trading"]
        assert len(weekend_issues) == 0

    def test_weekend_check_disabled(self):
        """Weekend check can be disabled."""
        validator = CrossValidator(check_weekend_trading=False)
        df = pl.DataFrame(
            {
                "timestamp": [
                    datetime(2024, 1, 6, 14, 30, tzinfo=UTC),  # Saturday
                ],
                "open": [100.0],
                "high": [101.0],
                "low": [99.0],
                "close": [100.0],
                "volume": [1000000.0],
            }
        )
        result = validator.validate(df)

        weekend_issues = [i for i in result.issues if i.check == "weekend_trading"]
        assert len(weekend_issues) == 0


class TestMarketHoursCheck:
    """Test market hours validation."""

    def test_market_hours_trading_passes(self):
        """Trading during market hours passes."""
        validator = CrossValidator()
        # 14:30 UTC = 9:30 AM ET (market open)
        df = pl.DataFrame(
            {
                "timestamp": [
                    datetime(2024, 1, 2, 15, 30, tzinfo=UTC),  # Within market hours
                ],
                "open": [100.0],
                "high": [101.0],
                "low": [99.0],
                "close": [100.0],
                "volume": [1000000.0],
            }
        )
        result = validator.validate(df)

        hours_issues = [i for i in result.issues if i.check == "market_hours"]
        assert len(hours_issues) == 0

    def test_early_trading_detected(self):
        """Pre-market trading detected as info."""
        validator = CrossValidator()
        # 10:00 UTC = 5:00 AM ET (pre-market)
        df = pl.DataFrame(
            {
                "timestamp": [
                    datetime(2024, 1, 2, 10, 0, tzinfo=UTC),  # Early trading
                ],
                "open": [100.0],
                "high": [101.0],
                "low": [99.0],
                "close": [100.0],
                "volume": [1000000.0],
            }
        )
        result = validator.validate(df)

        hours_issues = [i for i in result.issues if i.check == "market_hours"]
        assert len(hours_issues) == 1
        assert hours_issues[0].severity == Severity.INFO

    def test_crypto_skips_market_hours_check(self):
        """Crypto mode skips market hours check."""
        validator = CrossValidator(is_crypto=True)
        df = pl.DataFrame(
            {
                "timestamp": [
                    datetime(2024, 1, 2, 3, 0, tzinfo=UTC),  # 3 AM UTC
                ],
                "open": [100.0],
                "high": [101.0],
                "low": [99.0],
                "close": [100.0],
                "volume": [1000000.0],
            }
        )
        result = validator.validate(df)

        hours_issues = [i for i in result.issues if i.check == "market_hours"]
        assert len(hours_issues) == 0

    def test_market_hours_check_disabled(self):
        """Market hours check can be disabled."""
        validator = CrossValidator(check_market_hours=False)
        df = pl.DataFrame(
            {
                "timestamp": [
                    datetime(2024, 1, 2, 3, 0, tzinfo=UTC),  # 3 AM UTC
                ],
                "open": [100.0],
                "high": [101.0],
                "low": [99.0],
                "close": [100.0],
                "volume": [1000000.0],
            }
        )
        result = validator.validate(df)

        hours_issues = [i for i in result.issues if i.check == "market_hours"]
        assert len(hours_issues) == 0


class TestCrossReferenceValidation:
    """Test cross-reference validation between data sources."""

    def test_matching_data_passes(self):
        """Matching reference data passes."""
        validator = CrossValidator()
        ts = datetime(2024, 1, 2, 14, 30, tzinfo=UTC)

        df = pl.DataFrame(
            {
                "timestamp": [ts],
                "open": [100.0],
                "high": [101.0],
                "low": [99.0],
                "close": [100.0],
                "volume": [1000000.0],
            }
        )
        ref_df = pl.DataFrame(
            {
                "timestamp": [ts],
                "open": [100.0],
                "high": [101.0],
                "low": [99.0],
                "close": [100.0],  # Same price
                "volume": [1000000.0],
            }
        )

        result = validator.validate(df, reference_df=ref_df)

        cross_issues = [i for i in result.issues if i.check == "cross_reference"]
        assert len(cross_issues) == 0

    def test_price_difference_detected(self):
        """Price difference from reference detected."""
        validator = CrossValidator()
        ts = datetime(2024, 1, 2, 14, 30, tzinfo=UTC)

        df = pl.DataFrame(
            {
                "timestamp": [ts],
                "open": [100.0],
                "high": [101.0],
                "low": [99.0],
                "close": [100.0],
                "volume": [1000000.0],
            }
        )
        ref_df = pl.DataFrame(
            {
                "timestamp": [ts],
                "open": [100.0],
                "high": [101.0],
                "low": [99.0],
                "close": [105.0],  # 5% difference
                "volume": [1000000.0],
            }
        )

        result = validator.validate(df, reference_df=ref_df)

        cross_issues = [i for i in result.issues if i.check == "cross_reference"]
        assert len(cross_issues) == 1
        assert cross_issues[0].severity == Severity.WARNING

    def test_no_overlapping_timestamps(self):
        """No overlapping timestamps detected."""
        validator = CrossValidator()

        df = pl.DataFrame(
            {
                "timestamp": [datetime(2024, 1, 2, 14, 30, tzinfo=UTC)],
                "open": [100.0],
                "high": [101.0],
                "low": [99.0],
                "close": [100.0],
                "volume": [1000000.0],
            }
        )
        ref_df = pl.DataFrame(
            {
                "timestamp": [datetime(2024, 1, 3, 14, 30, tzinfo=UTC)],  # Different day
                "open": [100.0],
                "high": [101.0],
                "low": [99.0],
                "close": [100.0],
                "volume": [1000000.0],
            }
        )

        result = validator.validate(df, reference_df=ref_df)

        cross_issues = [i for i in result.issues if i.check == "cross_reference"]
        assert len(cross_issues) == 1
        assert "No overlapping" in cross_issues[0].message

    def test_no_reference_df(self):
        """Validation without reference DataFrame."""
        validator = CrossValidator()
        df = create_valid_ohlcv_df()

        result = validator.validate(df)  # No reference_df

        cross_issues = [i for i in result.issues if i.check == "cross_reference"]
        assert len(cross_issues) == 0

    def test_small_difference_passes(self):
        """Difference <1% passes cross-reference check."""
        validator = CrossValidator()
        ts = datetime(2024, 1, 2, 14, 30, tzinfo=UTC)

        df = pl.DataFrame(
            {
                "timestamp": [ts],
                "open": [100.0],
                "high": [101.0],
                "low": [99.0],
                "close": [100.0],
                "volume": [1000000.0],
            }
        )
        ref_df = pl.DataFrame(
            {
                "timestamp": [ts],
                "open": [100.0],
                "high": [101.0],
                "low": [99.0],
                "close": [100.5],  # 0.5% difference (< 1% threshold)
                "volume": [1000000.0],
            }
        )

        result = validator.validate(df, reference_df=ref_df)

        cross_issues = [i for i in result.issues if i.check == "cross_reference"]
        assert len(cross_issues) == 0


class TestValidationResult:
    """Test ValidationResult metadata."""

    def test_result_includes_row_count(self):
        """Validation result includes row count."""
        validator = CrossValidator()
        df = create_valid_ohlcv_df(n_rows=25)
        result = validator.validate(df)

        assert result.metadata["row_count"] == 25

    def test_result_includes_reference_count(self):
        """Result includes reference row count when provided."""
        validator = CrossValidator()
        df = create_valid_ohlcv_df(n_rows=10)
        ref_df = create_valid_ohlcv_df(n_rows=15)

        result = validator.validate(df, reference_df=ref_df)

        assert result.metadata["row_count"] == 10
        assert result.metadata["reference_row_count"] == 15

    def test_result_includes_duration(self):
        """Validation result includes duration."""
        validator = CrossValidator()
        df = create_valid_ohlcv_df()
        result = validator.validate(df)

        assert result.duration_seconds is not None
        assert result.duration_seconds >= 0


class TestAllChecksDisabled:
    """Test behavior with all checks disabled."""

    def test_no_issues_with_all_disabled(self):
        """With all checks disabled, no issues found."""
        validator = CrossValidator(
            check_price_continuity=False,
            check_volume_spikes=False,
            check_weekend_trading=False,
            check_market_hours=False,
        )

        # Create data with weekend trading and volume spike
        df = pl.DataFrame(
            {
                "timestamp": [
                    datetime(2024, 1, 6, 3, 0, tzinfo=UTC),  # Saturday, 3 AM
                ],
                "open": [100.0],
                "high": [101.0],
                "low": [99.0],
                "close": [100.0],
                "volume": [100000000.0],  # Large volume
            }
        )

        result = validator.validate(df)

        # Should have no issues
        assert len(result.issues) == 0
