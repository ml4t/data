"""Comprehensive tests for OHLCVValidator.

Tests cover:
- All 8 configurable validation checks
- Edge cases and boundary conditions
- Severity levels and issue reporting
- Configuration options
"""

from datetime import UTC, datetime, timedelta

import polars as pl

from ml4t.data.validation.base import Severity
from ml4t.data.validation.ohlcv import OHLCVValidator


def create_valid_ohlcv_df(
    n_rows: int = 10,
    start_date: datetime | None = None,
    base_price: float = 100.0,
) -> pl.DataFrame:
    """Create a valid OHLCV DataFrame for testing."""
    if start_date is None:
        start_date = datetime(2024, 1, 1, tzinfo=UTC)

    timestamps = [start_date + timedelta(days=i) for i in range(n_rows)]

    return pl.DataFrame(
        {
            "timestamp": timestamps,
            "open": [base_price + i * 0.1 for i in range(n_rows)],
            "high": [base_price + i * 0.1 + 0.5 for i in range(n_rows)],
            "low": [base_price + i * 0.1 - 0.3 for i in range(n_rows)],
            "close": [base_price + i * 0.1 + 0.2 for i in range(n_rows)],
            "volume": [1000000.0 + i * 1000 for i in range(n_rows)],
        }
    )


class TestOHLCVValidatorInitialization:
    """Test validator initialization and configuration."""

    def test_default_initialization(self):
        """All checks enabled by default."""
        validator = OHLCVValidator()
        assert validator.check_nulls is True
        assert validator.check_price_consistency is True
        assert validator.check_negative_prices is True
        assert validator.check_negative_volume is True
        assert validator.check_duplicate_timestamps is True
        assert validator.check_chronological_order is True
        assert validator.check_price_staleness is True
        assert validator.check_extreme_returns is True

    def test_custom_thresholds(self):
        """Custom thresholds are stored correctly."""
        validator = OHLCVValidator(
            max_return_threshold=0.75,
            staleness_threshold=10,
        )
        assert validator.max_return_threshold == 0.75
        assert validator.staleness_threshold == 10

    def test_disable_specific_checks(self):
        """Individual checks can be disabled."""
        validator = OHLCVValidator(
            check_nulls=False,
            check_price_consistency=False,
        )
        assert validator.check_nulls is False
        assert validator.check_price_consistency is False
        assert validator.check_negative_prices is True  # Still enabled

    def test_name_property(self):
        """Validator name is correct."""
        validator = OHLCVValidator()
        assert validator.name() == "OHLCVValidator"


class TestRequiredColumnsCheck:
    """Test required columns validation."""

    def test_valid_dataframe_passes(self):
        """DataFrame with all required columns passes."""
        validator = OHLCVValidator()
        df = create_valid_ohlcv_df()
        result = validator.validate(df)
        # Should have no CRITICAL issues for missing columns
        critical_issues = [i for i in result.issues if i.severity == Severity.CRITICAL]
        column_issues = [i for i in critical_issues if i.check == "required_columns"]
        assert len(column_issues) == 0

    def test_missing_single_column(self):
        """Missing one required column raises CRITICAL issue."""
        validator = OHLCVValidator()
        df = create_valid_ohlcv_df().drop("close")
        result = validator.validate(df)

        assert len(result.issues) == 1
        issue = result.issues[0]
        assert issue.severity == Severity.CRITICAL
        assert issue.check == "required_columns"
        assert "close" in issue.message

    def test_missing_multiple_columns(self):
        """Missing multiple columns all reported."""
        validator = OHLCVValidator()
        df = create_valid_ohlcv_df().drop(["open", "high", "low"])
        result = validator.validate(df)

        assert len(result.issues) == 1
        issue = result.issues[0]
        assert issue.severity == Severity.CRITICAL
        missing = issue.details["missing"]
        assert set(missing) == {"open", "high", "low"}

    def test_missing_column_stops_further_validation(self):
        """Validation stops early if required columns missing."""
        validator = OHLCVValidator()
        # Create df with nulls but missing timestamp column
        df = pl.DataFrame(
            {
                "open": [1.0, None, 3.0],
                "high": [2.0, 2.0, 4.0],
                "low": [0.5, 1.0, 2.0],
                "close": [1.5, 1.5, 3.5],
                "volume": [100.0, 200.0, 300.0],
            }
        )
        result = validator.validate(df)

        # Should only have the missing column issue, not null check issues
        assert len(result.issues) == 1
        assert result.issues[0].check == "required_columns"


class TestNullValueCheck:
    """Test null value detection."""

    def test_no_nulls_passes(self):
        """DataFrame without nulls passes."""
        validator = OHLCVValidator()
        df = create_valid_ohlcv_df()
        result = validator.validate(df)

        null_issues = [i for i in result.issues if i.check == "null_values"]
        assert len(null_issues) == 0

    def test_single_null_detected(self):
        """Single null value detected."""
        validator = OHLCVValidator()
        df = create_valid_ohlcv_df()
        df = df.with_columns(
            pl.when(pl.col("close") == df["close"][0])
            .then(None)
            .otherwise(pl.col("close"))
            .alias("close")
        )
        result = validator.validate(df)

        null_issues = [i for i in result.issues if i.check == "null_values"]
        assert len(null_issues) == 1
        assert null_issues[0].details["column"] == "close"
        assert null_issues[0].details["null_count"] == 1

    def test_nulls_in_multiple_columns(self):
        """Nulls in multiple columns each reported separately."""
        validator = OHLCVValidator()
        df = pl.DataFrame(
            {
                "timestamp": [datetime(2024, 1, i, tzinfo=UTC) for i in range(1, 6)],
                "open": [100.0, None, 102.0, 103.0, 104.0],
                "high": [101.0, 102.0, None, 104.0, 105.0],
                "low": [99.0, 100.0, 101.0, 102.0, 103.0],
                "close": [100.5, 101.5, 102.5, 103.5, 104.5],
                "volume": [1000.0, 2000.0, 3000.0, 4000.0, 5000.0],
            }
        )
        result = validator.validate(df)

        null_issues = [i for i in result.issues if i.check == "null_values"]
        assert len(null_issues) == 2  # open and high
        columns = {i.details["column"] for i in null_issues}
        assert columns == {"open", "high"}

    def test_null_check_disabled(self):
        """Null check can be disabled."""
        validator = OHLCVValidator(check_nulls=False)
        df = pl.DataFrame(
            {
                "timestamp": [datetime(2024, 1, 1, tzinfo=UTC)],
                "open": [None],
                "high": [101.0],
                "low": [99.0],
                "close": [100.0],
                "volume": [1000.0],
            }
        )
        result = validator.validate(df)

        null_issues = [i for i in result.issues if i.check == "null_values"]
        assert len(null_issues) == 0

    def test_sample_rows_limited_to_10(self):
        """Sample rows limited to first 10."""
        validator = OHLCVValidator()
        # Create df with 15 null values
        df = pl.DataFrame(
            {
                "timestamp": [datetime(2024, 1, i, tzinfo=UTC) for i in range(1, 21)],
                "open": [None if i < 16 else 100.0 for i in range(20)],
                "high": [101.0] * 20,
                "low": [99.0] * 20,
                "close": [100.0] * 20,
                "volume": [1000.0] * 20,
            }
        )
        result = validator.validate(df)

        null_issues = [i for i in result.issues if i.check == "null_values"]
        assert len(null_issues) == 1
        assert len(null_issues[0].sample_rows) == 10  # Limited to 10


class TestPriceConsistencyCheck:
    """Test OHLC price relationship validation."""

    def test_valid_prices_pass(self):
        """Valid OHLC relationships pass."""
        validator = OHLCVValidator()
        df = create_valid_ohlcv_df()
        result = validator.validate(df)

        consistency_issues = [i for i in result.issues if i.check == "price_consistency"]
        assert len(consistency_issues) == 0

    def test_high_less_than_low_detected(self):
        """High < Low detected as error."""
        validator = OHLCVValidator()
        df = pl.DataFrame(
            {
                "timestamp": [datetime(2024, 1, 1, tzinfo=UTC)],
                "open": [100.0],
                "high": [98.0],  # Less than low
                "low": [99.0],
                "close": [97.5],
                "volume": [1000.0],
            }
        )
        result = validator.validate(df)

        consistency_issues = [i for i in result.issues if i.check == "price_consistency"]
        assert len(consistency_issues) >= 1
        high_low_issues = [i for i in consistency_issues if "high < low" in i.message]
        assert len(high_low_issues) == 1
        assert high_low_issues[0].severity == Severity.ERROR

    def test_high_less_than_open_detected(self):
        """High < Open detected as error."""
        validator = OHLCVValidator()
        df = pl.DataFrame(
            {
                "timestamp": [datetime(2024, 1, 1, tzinfo=UTC)],
                "open": [105.0],  # Higher than high
                "high": [100.0],
                "low": [99.0],
                "close": [99.5],
                "volume": [1000.0],
            }
        )
        result = validator.validate(df)

        consistency_issues = [i for i in result.issues if i.check == "price_consistency"]
        high_open_issues = [i for i in consistency_issues if "high < open" in i.message.lower()]
        assert len(high_open_issues) == 1

    def test_high_less_than_close_detected(self):
        """High < Close detected as error."""
        validator = OHLCVValidator()
        df = pl.DataFrame(
            {
                "timestamp": [datetime(2024, 1, 1, tzinfo=UTC)],
                "open": [99.0],
                "high": [100.0],
                "low": [98.0],
                "close": [105.0],  # Higher than high
                "volume": [1000.0],
            }
        )
        result = validator.validate(df)

        consistency_issues = [i for i in result.issues if i.check == "price_consistency"]
        high_close_issues = [i for i in consistency_issues if "high < close" in i.message.lower()]
        assert len(high_close_issues) == 1

    def test_low_greater_than_open_detected(self):
        """Low > Open detected as error."""
        validator = OHLCVValidator()
        df = pl.DataFrame(
            {
                "timestamp": [datetime(2024, 1, 1, tzinfo=UTC)],
                "open": [95.0],  # Lower than low
                "high": [100.0],
                "low": [98.0],
                "close": [99.0],
                "volume": [1000.0],
            }
        )
        result = validator.validate(df)

        consistency_issues = [i for i in result.issues if i.check == "price_consistency"]
        low_open_issues = [i for i in consistency_issues if "low > open" in i.message.lower()]
        assert len(low_open_issues) == 1

    def test_low_greater_than_close_detected(self):
        """Low > Close detected as error."""
        validator = OHLCVValidator()
        df = pl.DataFrame(
            {
                "timestamp": [datetime(2024, 1, 1, tzinfo=UTC)],
                "open": [99.0],
                "high": [100.0],
                "low": [98.0],
                "close": [95.0],  # Lower than low
                "volume": [1000.0],
            }
        )
        result = validator.validate(df)

        consistency_issues = [i for i in result.issues if i.check == "price_consistency"]
        low_close_issues = [i for i in consistency_issues if "low > close" in i.message.lower()]
        assert len(low_close_issues) == 1

    def test_price_consistency_disabled(self):
        """Price consistency check can be disabled."""
        validator = OHLCVValidator(check_price_consistency=False)
        df = pl.DataFrame(
            {
                "timestamp": [datetime(2024, 1, 1, tzinfo=UTC)],
                "open": [100.0],
                "high": [90.0],  # Invalid
                "low": [95.0],
                "close": [92.0],
                "volume": [1000.0],
            }
        )
        result = validator.validate(df)

        consistency_issues = [i for i in result.issues if i.check == "price_consistency"]
        assert len(consistency_issues) == 0


class TestNegativePriceCheck:
    """Test negative price detection."""

    def test_positive_prices_pass(self):
        """Positive prices pass validation."""
        validator = OHLCVValidator()
        df = create_valid_ohlcv_df()
        result = validator.validate(df)

        neg_issues = [i for i in result.issues if i.check == "negative_prices"]
        assert len(neg_issues) == 0

    def test_negative_open_detected(self):
        """Negative open price detected."""
        validator = OHLCVValidator()
        df = pl.DataFrame(
            {
                "timestamp": [datetime(2024, 1, 1, tzinfo=UTC)],
                "open": [-100.0],
                "high": [101.0],
                "low": [99.0],
                "close": [100.0],
                "volume": [1000.0],
            }
        )
        result = validator.validate(df)

        neg_issues = [i for i in result.issues if i.check == "negative_prices"]
        assert len(neg_issues) >= 1
        assert neg_issues[0].severity == Severity.CRITICAL
        assert neg_issues[0].details["column"] == "open"

    def test_negative_prices_in_multiple_columns(self):
        """Negative prices in multiple columns each reported."""
        validator = OHLCVValidator()
        df = pl.DataFrame(
            {
                "timestamp": [datetime(2024, 1, 1, tzinfo=UTC)],
                "open": [-100.0],
                "high": [-90.0],
                "low": [-110.0],
                "close": [-95.0],
                "volume": [1000.0],
            }
        )
        result = validator.validate(df)

        neg_issues = [i for i in result.issues if i.check == "negative_prices"]
        assert len(neg_issues) == 4  # All OHLC columns
        columns = {i.details["column"] for i in neg_issues}
        assert columns == {"open", "high", "low", "close"}

    def test_zero_price_passes(self):
        """Zero price passes (only negative detected)."""
        validator = OHLCVValidator()
        df = pl.DataFrame(
            {
                "timestamp": [datetime(2024, 1, 1, tzinfo=UTC)],
                "open": [0.0],
                "high": [1.0],
                "low": [0.0],
                "close": [0.5],
                "volume": [1000.0],
            }
        )
        result = validator.validate(df)

        neg_issues = [i for i in result.issues if i.check == "negative_prices"]
        assert len(neg_issues) == 0

    def test_negative_prices_check_disabled(self):
        """Negative price check can be disabled."""
        validator = OHLCVValidator(check_negative_prices=False)
        df = pl.DataFrame(
            {
                "timestamp": [datetime(2024, 1, 1, tzinfo=UTC)],
                "open": [-100.0],
                "high": [101.0],
                "low": [99.0],
                "close": [100.0],
                "volume": [1000.0],
            }
        )
        result = validator.validate(df)

        neg_issues = [i for i in result.issues if i.check == "negative_prices"]
        assert len(neg_issues) == 0


class TestNegativeVolumeCheck:
    """Test negative volume detection."""

    def test_positive_volume_passes(self):
        """Positive volume passes validation."""
        validator = OHLCVValidator()
        df = create_valid_ohlcv_df()
        result = validator.validate(df)

        vol_issues = [i for i in result.issues if i.check == "negative_volume"]
        assert len(vol_issues) == 0

    def test_negative_volume_detected(self):
        """Negative volume detected as error."""
        validator = OHLCVValidator()
        df = pl.DataFrame(
            {
                "timestamp": [datetime(2024, 1, 1, tzinfo=UTC)],
                "open": [100.0],
                "high": [101.0],
                "low": [99.0],
                "close": [100.0],
                "volume": [-1000.0],
            }
        )
        result = validator.validate(df)

        vol_issues = [i for i in result.issues if i.check == "negative_volume"]
        assert len(vol_issues) == 1
        assert vol_issues[0].severity == Severity.ERROR

    def test_zero_volume_passes(self):
        """Zero volume passes (only negative detected)."""
        validator = OHLCVValidator()
        df = pl.DataFrame(
            {
                "timestamp": [datetime(2024, 1, 1, tzinfo=UTC)],
                "open": [100.0],
                "high": [101.0],
                "low": [99.0],
                "close": [100.0],
                "volume": [0.0],
            }
        )
        result = validator.validate(df)

        vol_issues = [i for i in result.issues if i.check == "negative_volume"]
        assert len(vol_issues) == 0

    def test_negative_volume_check_disabled(self):
        """Negative volume check can be disabled."""
        validator = OHLCVValidator(check_negative_volume=False)
        df = pl.DataFrame(
            {
                "timestamp": [datetime(2024, 1, 1, tzinfo=UTC)],
                "open": [100.0],
                "high": [101.0],
                "low": [99.0],
                "close": [100.0],
                "volume": [-1000.0],
            }
        )
        result = validator.validate(df)

        vol_issues = [i for i in result.issues if i.check == "negative_volume"]
        assert len(vol_issues) == 0


class TestDuplicateTimestampsCheck:
    """Test duplicate timestamp detection."""

    def test_unique_timestamps_pass(self):
        """Unique timestamps pass validation."""
        validator = OHLCVValidator()
        df = create_valid_ohlcv_df()
        result = validator.validate(df)

        dup_issues = [i for i in result.issues if i.check == "duplicate_timestamps"]
        assert len(dup_issues) == 0

    def test_duplicate_timestamp_detected(self):
        """Duplicate timestamps detected as error."""
        validator = OHLCVValidator()
        ts = datetime(2024, 1, 1, tzinfo=UTC)
        df = pl.DataFrame(
            {
                "timestamp": [ts, ts, ts + timedelta(days=1)],  # First two are duplicates
                "open": [100.0, 101.0, 102.0],
                "high": [101.0, 102.0, 103.0],
                "low": [99.0, 100.0, 101.0],
                "close": [100.5, 101.5, 102.5],
                "volume": [1000.0, 2000.0, 3000.0],
            }
        )
        result = validator.validate(df)

        dup_issues = [i for i in result.issues if i.check == "duplicate_timestamps"]
        assert len(dup_issues) == 1
        assert dup_issues[0].severity == Severity.ERROR
        assert dup_issues[0].details["duplicate_count"] == 1

    def test_multiple_duplicate_groups_detected(self):
        """Multiple groups of duplicates counted correctly."""
        validator = OHLCVValidator()
        ts1 = datetime(2024, 1, 1, tzinfo=UTC)
        ts2 = datetime(2024, 1, 2, tzinfo=UTC)
        df = pl.DataFrame(
            {
                "timestamp": [ts1, ts1, ts1, ts2, ts2],  # 3 copies of ts1, 2 of ts2
                "open": [100.0] * 5,
                "high": [101.0] * 5,
                "low": [99.0] * 5,
                "close": [100.0] * 5,
                "volume": [1000.0] * 5,
            }
        )
        result = validator.validate(df)

        dup_issues = [i for i in result.issues if i.check == "duplicate_timestamps"]
        assert len(dup_issues) == 1
        # Total duplicates: (3-1) + (2-1) = 3
        assert dup_issues[0].details["duplicate_count"] == 3
        assert dup_issues[0].details["unique_duplicates"] == 2

    def test_duplicate_check_disabled(self):
        """Duplicate check can be disabled."""
        validator = OHLCVValidator(check_duplicate_timestamps=False)
        ts = datetime(2024, 1, 1, tzinfo=UTC)
        df = pl.DataFrame(
            {
                "timestamp": [ts, ts],
                "open": [100.0, 100.0],
                "high": [101.0, 101.0],
                "low": [99.0, 99.0],
                "close": [100.0, 100.0],
                "volume": [1000.0, 1000.0],
            }
        )
        result = validator.validate(df)

        dup_issues = [i for i in result.issues if i.check == "duplicate_timestamps"]
        assert len(dup_issues) == 0


class TestChronologicalOrderCheck:
    """Test timestamp ordering validation."""

    def test_sorted_timestamps_pass(self):
        """Sorted timestamps pass validation."""
        validator = OHLCVValidator()
        df = create_valid_ohlcv_df()
        result = validator.validate(df)

        order_issues = [i for i in result.issues if i.check == "chronological_order"]
        assert len(order_issues) == 0

    def test_unsorted_timestamps_detected(self):
        """Unsorted timestamps detected as error."""
        validator = OHLCVValidator()
        df = pl.DataFrame(
            {
                "timestamp": [
                    datetime(2024, 1, 3, tzinfo=UTC),
                    datetime(2024, 1, 1, tzinfo=UTC),  # Out of order
                    datetime(2024, 1, 2, tzinfo=UTC),
                ],
                "open": [100.0, 98.0, 99.0],
                "high": [101.0, 99.0, 100.0],
                "low": [99.0, 97.0, 98.0],
                "close": [100.0, 98.5, 99.5],
                "volume": [1000.0, 800.0, 900.0],
            }
        )
        result = validator.validate(df)

        order_issues = [i for i in result.issues if i.check == "chronological_order"]
        assert len(order_issues) == 1
        assert order_issues[0].severity == Severity.ERROR

    def test_single_row_passes(self):
        """Single row DataFrame passes ordering check."""
        validator = OHLCVValidator()
        df = pl.DataFrame(
            {
                "timestamp": [datetime(2024, 1, 1, tzinfo=UTC)],
                "open": [100.0],
                "high": [101.0],
                "low": [99.0],
                "close": [100.0],
                "volume": [1000.0],
            }
        )
        result = validator.validate(df)

        order_issues = [i for i in result.issues if i.check == "chronological_order"]
        assert len(order_issues) == 0

    def test_empty_dataframe_passes(self):
        """Empty DataFrame passes ordering check."""
        validator = OHLCVValidator()
        df = pl.DataFrame(
            {
                "timestamp": [],
                "open": [],
                "high": [],
                "low": [],
                "close": [],
                "volume": [],
            }
        ).cast(
            {
                "timestamp": pl.Datetime("us", "UTC"),
                "open": pl.Float64,
                "high": pl.Float64,
                "low": pl.Float64,
                "close": pl.Float64,
                "volume": pl.Float64,
            }
        )
        result = validator.validate(df)

        order_issues = [i for i in result.issues if i.check == "chronological_order"]
        assert len(order_issues) == 0

    def test_order_check_disabled(self):
        """Chronological order check can be disabled."""
        validator = OHLCVValidator(check_chronological_order=False)
        df = pl.DataFrame(
            {
                "timestamp": [
                    datetime(2024, 1, 2, tzinfo=UTC),
                    datetime(2024, 1, 1, tzinfo=UTC),  # Out of order
                ],
                "open": [100.0, 99.0],
                "high": [101.0, 100.0],
                "low": [99.0, 98.0],
                "close": [100.0, 99.0],
                "volume": [1000.0, 900.0],
            }
        )
        result = validator.validate(df)

        order_issues = [i for i in result.issues if i.check == "chronological_order"]
        assert len(order_issues) == 0


class TestPriceStalenessCheck:
    """Test stale price detection."""

    def test_varying_prices_pass(self):
        """Varying prices pass staleness check."""
        validator = OHLCVValidator(staleness_threshold=3)
        df = create_valid_ohlcv_df(n_rows=10)
        result = validator.validate(df)

        stale_issues = [i for i in result.issues if i.check == "price_staleness"]
        assert len(stale_issues) == 0

    def test_stale_prices_detected(self):
        """Stale prices detected as warning."""
        validator = OHLCVValidator(staleness_threshold=3)
        df = pl.DataFrame(
            {
                "timestamp": [datetime(2024, 1, i, tzinfo=UTC) for i in range(1, 8)],
                "open": [100.0] * 7,
                "high": [101.0] * 7,
                "low": [99.0] * 7,
                "close": [100.0] * 7,  # All identical
                "volume": [1000.0, 1100.0, 1200.0, 1300.0, 1400.0, 1500.0, 1600.0],
            }
        )
        result = validator.validate(df)

        stale_issues = [i for i in result.issues if i.check == "price_staleness"]
        assert len(stale_issues) == 1
        assert stale_issues[0].severity == Severity.WARNING
        assert stale_issues[0].details["threshold_days"] == 3

    def test_short_dataframe_skipped(self):
        """DataFrame shorter than threshold skipped."""
        validator = OHLCVValidator(staleness_threshold=5)
        df = pl.DataFrame(
            {
                "timestamp": [datetime(2024, 1, i, tzinfo=UTC) for i in range(1, 4)],
                "open": [100.0] * 3,
                "high": [101.0] * 3,
                "low": [99.0] * 3,
                "close": [100.0] * 3,  # All identical, but too short
                "volume": [1000.0] * 3,
            }
        )
        result = validator.validate(df)

        stale_issues = [i for i in result.issues if i.check == "price_staleness"]
        assert len(stale_issues) == 0

    def test_custom_staleness_threshold(self):
        """Custom staleness threshold respected."""
        validator = OHLCVValidator(staleness_threshold=10)
        # 9 identical prices should not trigger with threshold 10
        df = pl.DataFrame(
            {
                "timestamp": [datetime(2024, 1, i, tzinfo=UTC) for i in range(1, 10)],
                "open": [100.0] * 9,
                "high": [101.0] * 9,
                "low": [99.0] * 9,
                "close": [100.0] * 9,
                "volume": [1000.0] * 9,
            }
        )
        result = validator.validate(df)

        stale_issues = [i for i in result.issues if i.check == "price_staleness"]
        assert len(stale_issues) == 0

    def test_staleness_check_disabled(self):
        """Staleness check can be disabled."""
        validator = OHLCVValidator(check_price_staleness=False)
        df = pl.DataFrame(
            {
                "timestamp": [datetime(2024, 1, i, tzinfo=UTC) for i in range(1, 11)],
                "open": [100.0] * 10,
                "high": [101.0] * 10,
                "low": [99.0] * 10,
                "close": [100.0] * 10,
                "volume": [1000.0] * 10,
            }
        )
        result = validator.validate(df)

        stale_issues = [i for i in result.issues if i.check == "price_staleness"]
        assert len(stale_issues) == 0


class TestExtremeReturnsCheck:
    """Test extreme returns detection."""

    def test_normal_returns_pass(self):
        """Normal returns pass validation."""
        validator = OHLCVValidator(max_return_threshold=0.5)
        df = create_valid_ohlcv_df(n_rows=10)
        result = validator.validate(df)

        return_issues = [i for i in result.issues if i.check == "extreme_returns"]
        assert len(return_issues) == 0

    def test_extreme_positive_return_detected(self):
        """Extreme positive return detected as warning."""
        validator = OHLCVValidator(max_return_threshold=0.5)
        df = pl.DataFrame(
            {
                "timestamp": [
                    datetime(2024, 1, 1, tzinfo=UTC),
                    datetime(2024, 1, 2, tzinfo=UTC),
                ],
                "open": [100.0, 160.0],
                "high": [100.5, 165.0],
                "low": [99.0, 155.0],
                "close": [100.0, 160.0],  # 60% return
                "volume": [1000.0, 2000.0],
            }
        )
        result = validator.validate(df)

        return_issues = [i for i in result.issues if i.check == "extreme_returns"]
        assert len(return_issues) == 1
        assert return_issues[0].severity == Severity.WARNING

    def test_extreme_negative_return_detected(self):
        """Extreme negative return detected."""
        validator = OHLCVValidator(max_return_threshold=0.5)
        df = pl.DataFrame(
            {
                "timestamp": [
                    datetime(2024, 1, 1, tzinfo=UTC),
                    datetime(2024, 1, 2, tzinfo=UTC),
                ],
                "open": [100.0, 40.0],
                "high": [100.5, 45.0],
                "low": [99.0, 35.0],
                "close": [100.0, 40.0],  # -60% return
                "volume": [1000.0, 2000.0],
            }
        )
        result = validator.validate(df)

        return_issues = [i for i in result.issues if i.check == "extreme_returns"]
        assert len(return_issues) == 1

    def test_single_row_skipped(self):
        """Single row DataFrame skipped for returns."""
        validator = OHLCVValidator(max_return_threshold=0.5)
        df = pl.DataFrame(
            {
                "timestamp": [datetime(2024, 1, 1, tzinfo=UTC)],
                "open": [100.0],
                "high": [101.0],
                "low": [99.0],
                "close": [100.0],
                "volume": [1000.0],
            }
        )
        result = validator.validate(df)

        return_issues = [i for i in result.issues if i.check == "extreme_returns"]
        assert len(return_issues) == 0

    def test_custom_return_threshold(self):
        """Custom return threshold respected."""
        validator = OHLCVValidator(max_return_threshold=0.10)  # 10%
        df = pl.DataFrame(
            {
                "timestamp": [
                    datetime(2024, 1, 1, tzinfo=UTC),
                    datetime(2024, 1, 2, tzinfo=UTC),
                ],
                "open": [100.0, 112.0],
                "high": [100.5, 115.0],
                "low": [99.0, 110.0],
                "close": [100.0, 112.0],  # 12% return > 10% threshold
                "volume": [1000.0, 2000.0],
            }
        )
        result = validator.validate(df)

        return_issues = [i for i in result.issues if i.check == "extreme_returns"]
        assert len(return_issues) == 1

    def test_extreme_returns_check_disabled(self):
        """Extreme returns check can be disabled."""
        validator = OHLCVValidator(check_extreme_returns=False)
        df = pl.DataFrame(
            {
                "timestamp": [
                    datetime(2024, 1, 1, tzinfo=UTC),
                    datetime(2024, 1, 2, tzinfo=UTC),
                ],
                "open": [100.0, 200.0],
                "high": [100.5, 210.0],
                "low": [99.0, 190.0],
                "close": [100.0, 200.0],  # 100% return
                "volume": [1000.0, 2000.0],
            }
        )
        result = validator.validate(df)

        return_issues = [i for i in result.issues if i.check == "extreme_returns"]
        assert len(return_issues) == 0


class TestValidationResult:
    """Test ValidationResult metadata and behavior."""

    def test_result_includes_metadata(self):
        """Validation result includes row count and columns."""
        validator = OHLCVValidator()
        df = create_valid_ohlcv_df(n_rows=15)
        result = validator.validate(df)

        assert result.metadata["row_count"] == 15
        assert "timestamp" in result.metadata["columns"]
        assert "close" in result.metadata["columns"]

    def test_result_includes_duration(self):
        """Validation result includes duration."""
        validator = OHLCVValidator()
        df = create_valid_ohlcv_df()
        result = validator.validate(df)

        assert result.duration_seconds is not None
        assert result.duration_seconds >= 0

    def test_passed_false_on_critical_issue(self):
        """Result passed is False when CRITICAL issue found."""
        validator = OHLCVValidator()
        df = create_valid_ohlcv_df().drop("close")
        result = validator.validate(df)

        # Missing column is CRITICAL, should set passed=False
        assert result.passed is False

    def test_passed_true_for_clean_data(self):
        """Result passed is True for clean data."""
        validator = OHLCVValidator()
        df = create_valid_ohlcv_df()
        result = validator.validate(df)

        assert result.passed is True
        assert len(result.issues) == 0


class TestAllChecksDisabled:
    """Test behavior with all checks disabled."""

    def test_all_checks_disabled_passes_invalid_data(self):
        """With all checks disabled, invalid data passes."""
        validator = OHLCVValidator(
            check_nulls=False,
            check_price_consistency=False,
            check_negative_prices=False,
            check_negative_volume=False,
            check_duplicate_timestamps=False,
            check_chronological_order=False,
            check_price_staleness=False,
            check_extreme_returns=False,
        )

        # Extremely invalid data
        ts = datetime(2024, 1, 1, tzinfo=UTC)
        df = pl.DataFrame(
            {
                "timestamp": [ts, ts],  # Duplicates
                "open": [-100.0, None],  # Negative and null
                "high": [-50.0, 0.0],  # Negative
                "low": [-150.0, 10.0],  # Negative, and low > high
                "close": [-80.0, 0.0],  # Negative
                "volume": [-1000.0, -500.0],  # Negative
            }
        )

        result = validator.validate(df)

        # Only required columns check runs (and passes)
        assert result.passed is True
        assert len(result.issues) == 0


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_dataframe(self):
        """Empty DataFrame validated correctly."""
        validator = OHLCVValidator()
        df = pl.DataFrame(
            {
                "timestamp": [],
                "open": [],
                "high": [],
                "low": [],
                "close": [],
                "volume": [],
            }
        ).cast(
            {
                "timestamp": pl.Datetime("us", "UTC"),
                "open": pl.Float64,
                "high": pl.Float64,
                "low": pl.Float64,
                "close": pl.Float64,
                "volume": pl.Float64,
            }
        )

        result = validator.validate(df)

        assert result.passed is True
        assert result.metadata["row_count"] == 0

    def test_very_large_prices(self):
        """Very large prices handled correctly."""
        validator = OHLCVValidator()
        df = pl.DataFrame(
            {
                "timestamp": [datetime(2024, 1, 1, tzinfo=UTC)],
                "open": [1e12],
                "high": [1e12 + 1e10],
                "low": [1e12 - 1e10],
                "close": [1e12 + 5e9],
                "volume": [1e15],
            }
        )

        result = validator.validate(df)

        assert result.passed is True

    def test_very_small_prices(self):
        """Very small positive prices handled correctly."""
        validator = OHLCVValidator()
        df = pl.DataFrame(
            {
                "timestamp": [datetime(2024, 1, 1, tzinfo=UTC)],
                "open": [0.00001],
                "high": [0.00002],
                "low": [0.000005],
                "close": [0.000015],
                "volume": [1e9],
            }
        )

        result = validator.validate(df)

        # No issues for positive prices
        neg_issues = [i for i in result.issues if i.check == "negative_prices"]
        assert len(neg_issues) == 0
