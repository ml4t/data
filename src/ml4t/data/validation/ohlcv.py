"""OHLCV data validation."""

import time
from typing import Any, Literal
from warnings import warn

import polars as pl
import structlog

from ml4t.data.validation.base import Severity, ValidationIssue, ValidationResult, Validator

logger = structlog.get_logger()


def _numeric_scalar(value: object, metric: str) -> float:
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise TypeError(f"'{metric}' must contain non-null numeric values")
    return float(value)


NegativePricePolicy = Literal["forbid", "warn", "allow"]
_NUMERIC_COLUMNS = ("open", "high", "low", "close", "volume")
_PRICE_COLUMNS = ("open", "high", "low", "close")


class OHLCVValidator(Validator):
    """Validator for OHLCV (Open, High, Low, Close, Volume) data."""

    def __init__(
        self,
        check_nulls: bool = True,
        check_price_consistency: bool = True,
        negative_price_policy: NegativePricePolicy | bool = "forbid",
        check_negative_volume: bool = True,
        check_duplicate_timestamps: bool = True,
        check_chronological_order: bool = True,
        check_price_staleness: bool = True,
        check_extreme_returns: bool = True,
        max_return_threshold: float = 0.5,  # 50% return threshold
        staleness_threshold: int = 5,  # Days of identical prices
        *,
        check_negative_prices: bool | None = None,
    ) -> None:
        """
        Initialize OHLCV validator with configurable checks.

        Args:
            check_nulls: Check for null values
            check_price_consistency: Check high >= low, high >= close, etc.
            negative_price_policy: Whether negative prices fail, warn, or are accepted
            check_negative_volume: Check for negative volume
            check_duplicate_timestamps: Check for duplicate timestamps
            check_chronological_order: Check timestamps are in order
            check_price_staleness: Check for stale (unchanged) prices
            check_extreme_returns: Check for extreme price returns
            max_return_threshold: Threshold for extreme returns (as fraction)
            staleness_threshold: Days of identical prices to flag as stale
            check_negative_prices: Deprecated boolean alias for negative_price_policy

        Numeric-column and finite-value checks are structural and always run.
        """
        self.check_nulls = check_nulls
        self.check_price_consistency = check_price_consistency
        legacy_policy = negative_price_policy if isinstance(negative_price_policy, bool) else None
        if check_negative_prices is not None and legacy_policy is not None:
            raise ValueError("Specify check_negative_prices only once")
        if check_negative_prices is not None:
            legacy_policy = check_negative_prices
        if legacy_policy is not None:
            warn(
                "check_negative_prices is deprecated; use negative_price_policy",
                DeprecationWarning,
                stacklevel=2,
            )
            negative_price_policy = "forbid" if legacy_policy else "allow"
        if negative_price_policy not in {"forbid", "warn", "allow"}:
            raise ValueError("negative_price_policy must be 'forbid', 'warn', or 'allow'")
        self.negative_price_policy = negative_price_policy
        self.check_negative_volume = check_negative_volume
        self.check_duplicate_timestamps = check_duplicate_timestamps
        self.check_chronological_order = check_chronological_order
        self.check_price_staleness = check_price_staleness
        self.check_extreme_returns = check_extreme_returns
        self.max_return_threshold = max_return_threshold
        self.staleness_threshold = staleness_threshold

    def name(self) -> str:
        """Return validator name."""
        return "OHLCVValidator"

    def validate(self, df: pl.DataFrame, **kwargs: Any) -> ValidationResult:
        """
        Validate OHLCV DataFrame.

        Args:
            df: DataFrame with OHLCV columns
            **kwargs: Additional parameters

        Returns:
            ValidationResult with any issues found
        """
        start_time = time.time()
        result = ValidationResult(passed=True)

        # Store metadata
        result.metadata["row_count"] = len(df)
        result.metadata["columns"] = df.columns

        # Check for required columns
        required_columns = ["timestamp", "open", "high", "low", "close", "volume"]
        missing_columns = set(required_columns) - set(df.columns)
        if missing_columns:
            result.add_issue(
                ValidationIssue(
                    severity=Severity.CRITICAL,
                    check="required_columns",
                    message=f"Missing required columns: {missing_columns}",
                    details={"missing": list(missing_columns)},
                )
            )
            result.duration_seconds = time.time() - start_time
            return result

        # Perform validation checks
        if not self._check_numeric_columns(df, result):
            result.duration_seconds = time.time() - start_time
            return result

        self._check_finite_values(df, result)

        if self.check_nulls:
            self._check_nulls(df, result)

        if self.check_price_consistency:
            self._check_price_consistency(df, result)

        if self.negative_price_policy != "allow":
            self._check_negative_prices(df, result)

        if self.check_negative_volume:
            self._check_negative_volume(df, result)

        if self.check_duplicate_timestamps:
            self._check_duplicate_timestamps(df, result)

        if self.check_chronological_order:
            self._check_chronological_order(df, result)

        if self.check_price_staleness:
            self._check_price_staleness(df, result)

        if self.check_extreme_returns:
            self._check_extreme_returns(df, result)

        result.duration_seconds = time.time() - start_time
        return result

    def _check_numeric_columns(self, df: pl.DataFrame, result: ValidationResult) -> bool:
        """Require numeric OHLCV values before running numeric checks."""
        invalid_columns = [
            column
            for column in _NUMERIC_COLUMNS
            if df.schema[column] != pl.Null and not df.schema[column].is_numeric()
        ]
        if invalid_columns:
            result.add_issue(
                ValidationIssue(
                    severity=Severity.CRITICAL,
                    check="numeric_columns",
                    message=f"OHLCV columns must be numeric: {invalid_columns}",
                    details={"columns": invalid_columns},
                )
            )
            return False
        return True

    def _check_finite_values(self, df: pl.DataFrame, result: ValidationResult) -> None:
        """Reject NaN and positive or negative infinity in every numeric column."""
        counts = df.select(
            *[
                (pl.col(column).is_not_null() & ~pl.col(column).is_finite()).sum().alias(column)
                for column in _NUMERIC_COLUMNS
            ]
        ).row(0, named=True)
        invalid_columns = [column for column, count in counts.items() if count]
        if not invalid_columns:
            return

        row_column = "__ml4t_validation_row__"
        while row_column in df.columns:
            row_column = f"_{row_column}"
        indexed = df.with_row_index(row_column)
        for column in invalid_columns:
            invalid = indexed.filter(pl.col(column).is_not_null() & ~pl.col(column).is_finite())
            result.add_issue(
                ValidationIssue(
                    severity=Severity.ERROR,
                    check="non_finite_values",
                    message=f"Found {len(invalid)} non-finite values in '{column}' column",
                    details={"column": column, "non_finite_count": len(invalid)},
                    row_count=len(invalid),
                    sample_rows=invalid.get_column(row_column).to_list()[:10],
                )
            )

    def _check_nulls(self, df: pl.DataFrame, result: ValidationResult) -> None:
        """Check for null values in OHLCV columns."""
        for col in _NUMERIC_COLUMNS:
            null_count = df[col].null_count()
            if null_count > 0:
                null_rows = df.with_row_index().filter(pl.col(col).is_null())["index"].to_list()
                result.add_issue(
                    ValidationIssue(
                        severity=Severity.ERROR,
                        check="null_values",
                        message=f"Found {null_count} null values in '{col}' column",
                        details={"column": col, "null_count": null_count},
                        row_count=null_count,
                        sample_rows=null_rows[:10],  # First 10 rows with nulls
                    )
                )

    def _check_price_consistency(self, df: pl.DataFrame, result: ValidationResult) -> None:
        """Check OHLC price relationships."""
        # High should be >= Low
        invalid_high_low = df.filter(pl.col("high") < pl.col("low"))
        if len(invalid_high_low) > 0:
            result.add_issue(
                ValidationIssue(
                    severity=Severity.ERROR,
                    check="price_consistency",
                    message=f"Found {len(invalid_high_low)} rows where high < low",
                    details={"condition": "high < low"},
                    row_count=len(invalid_high_low),
                    sample_rows=invalid_high_low.with_row_index()["index"].to_list()[:10],
                )
            )

        # High should be >= Open and Close
        invalid_high = df.filter(
            (pl.col("high") < pl.col("open")) | (pl.col("high") < pl.col("close"))
        )
        if len(invalid_high) > 0:
            result.add_issue(
                ValidationIssue(
                    severity=Severity.ERROR,
                    check="price_consistency",
                    message=f"Found {len(invalid_high)} rows where high < open or high < close",
                    details={"condition": "high < open or high < close"},
                    row_count=len(invalid_high),
                    sample_rows=invalid_high.with_row_index()["index"].to_list()[:10],
                )
            )

        # Low should be <= Open and Close
        invalid_low = df.filter(
            (pl.col("low") > pl.col("open")) | (pl.col("low") > pl.col("close"))
        )
        if len(invalid_low) > 0:
            result.add_issue(
                ValidationIssue(
                    severity=Severity.ERROR,
                    check="price_consistency",
                    message=f"Found {len(invalid_low)} rows where low > open or low > close",
                    details={"condition": "low > open or low > close"},
                    row_count=len(invalid_low),
                    sample_rows=invalid_low.with_row_index()["index"].to_list()[:10],
                )
            )

    def _check_negative_prices(self, df: pl.DataFrame, result: ValidationResult) -> None:
        """Check for negative prices."""
        severity = Severity.ERROR if self.negative_price_policy == "forbid" else Severity.WARNING
        for col in _PRICE_COLUMNS:
            negative_prices = df.filter(pl.col(col) < 0)
            if len(negative_prices) > 0:
                result.add_issue(
                    ValidationIssue(
                        severity=severity,
                        check="negative_prices",
                        message=f"Found {len(negative_prices)} negative prices in '{col}'",
                        details={"column": col},
                        row_count=len(negative_prices),
                        sample_rows=negative_prices.with_row_index()["index"].to_list()[:10],
                    )
                )

    def _check_negative_volume(self, df: pl.DataFrame, result: ValidationResult) -> None:
        """Check for negative volume."""
        negative_volume = df.filter(pl.col("volume") < 0)
        if len(negative_volume) > 0:
            result.add_issue(
                ValidationIssue(
                    severity=Severity.ERROR,
                    check="negative_volume",
                    message=f"Found {len(negative_volume)} rows with negative volume",
                    row_count=len(negative_volume),
                    sample_rows=negative_volume.with_row_index()["index"].to_list()[:10],
                )
            )

    def _check_duplicate_timestamps(self, df: pl.DataFrame, result: ValidationResult) -> None:
        """Check for duplicate timestamps."""
        duplicate_count = len(df) - df["timestamp"].n_unique()
        if duplicate_count > 0:
            duplicates = (
                df.group_by("timestamp").agg(pl.len().alias("count")).filter(pl.col("count") > 1)
            )
            result.add_issue(
                ValidationIssue(
                    severity=Severity.ERROR,
                    check="duplicate_timestamps",
                    message=f"Found {duplicate_count} duplicate timestamps",
                    details={
                        "duplicate_count": duplicate_count,
                        "unique_duplicates": len(duplicates),
                    },
                    row_count=duplicate_count,
                )
            )

    def _check_chronological_order(self, df: pl.DataFrame, result: ValidationResult) -> None:
        """Check if timestamps are in chronological order."""
        if len(df) > 1:
            # Check if timestamps are sorted
            is_sorted = df["timestamp"].is_sorted()
            if not is_sorted:
                # Find out-of-order positions
                timestamps = df["timestamp"]
                out_of_order = []
                for i in range(1, len(timestamps)):
                    if timestamps[i] < timestamps[i - 1]:
                        out_of_order.append(i)

                result.add_issue(
                    ValidationIssue(
                        severity=Severity.ERROR,
                        check="chronological_order",
                        message=f"Timestamps are not in chronological order ({len(out_of_order)} violations)",
                        details={"out_of_order_count": len(out_of_order)},
                        row_count=len(out_of_order),
                        sample_rows=out_of_order[:10],
                    )
                )

    def _check_price_staleness(self, df: pl.DataFrame, result: ValidationResult) -> None:
        """Check for stale (unchanged) prices over multiple periods."""
        if len(df) < self.staleness_threshold:
            return

        # Check close prices for staleness
        close_prices = df["close"]
        stale_periods = []

        i = 0
        while i < len(close_prices) - self.staleness_threshold:
            # Check if next N prices are identical
            window = close_prices[i : i + self.staleness_threshold]
            if window.n_unique() == 1:
                stale_periods.append(i)
                # Skip ahead to avoid overlapping detections
                i += self.staleness_threshold
            else:
                i += 1

        if stale_periods:
            result.add_issue(
                ValidationIssue(
                    severity=Severity.WARNING,
                    check="price_staleness",
                    message=f"Found {len(stale_periods)} periods with {self.staleness_threshold}+ days of identical close prices",
                    details={
                        "threshold_days": self.staleness_threshold,
                        "period_count": len(stale_periods),
                    },
                    row_count=len(stale_periods) * self.staleness_threshold,
                    sample_rows=stale_periods[:10],
                )
            )

    def _check_extreme_returns(self, df: pl.DataFrame, result: ValidationResult) -> None:
        """Check for extreme price returns."""
        if len(df) < 2:
            return

        # Calculate returns
        df_with_returns = df.with_columns(
            (pl.col("close") / pl.col("close").shift(1) - 1).alias("return")
        )

        # Find extreme returns
        extreme_returns = df_with_returns.filter(pl.col("return").abs() > self.max_return_threshold)

        if len(extreme_returns) > 0:
            max_return = extreme_returns["return"].abs().max()
            result.add_issue(
                ValidationIssue(
                    severity=Severity.WARNING,
                    check="extreme_returns",
                    message=f"Found {len(extreme_returns)} extreme returns (>{self.max_return_threshold:.0%})",
                    details={
                        "threshold": self.max_return_threshold,
                        "max_return": _numeric_scalar(max_return, "return"),
                        "count": len(extreme_returns),
                    },
                    row_count=len(extreme_returns),
                    sample_rows=extreme_returns.with_row_index()["index"].to_list()[:10],
                )
            )
