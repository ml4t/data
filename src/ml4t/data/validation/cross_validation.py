"""Cross-validation checks for data consistency."""

from __future__ import annotations

import time
from typing import Any

import polars as pl
import structlog

from ml4t.data.validation.base import Severity, ValidationIssue, ValidationResult, Validator
from ml4t.specs import UNSET_CALENDAR, CalendarUnset, FeedSpec

logger = structlog.get_logger()


def _numeric_scalar(value: object, metric: str) -> float:
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise TypeError(f"'{metric}' must contain non-null numeric values")
    return float(value)


class CrossValidator(Validator):
    """Cross-validation checks across multiple data sources or timeframes."""

    def __init__(
        self,
        check_price_continuity: bool = True,
        check_volume_spikes: bool = True,
        check_weekend_trading: bool = True,
        check_market_hours: bool = True,
        volume_spike_threshold: float = 10.0,  # 10x average volume
        price_gap_threshold: float = 0.1,  # 10% price gap
        is_crypto: bool = False,  # Crypto trades 24/7
        *,
        calendar: str | None | CalendarUnset = UNSET_CALENDAR,
        feed_spec: Any | None = None,
    ) -> None:
        """
        Initialize cross-validator.

        Args:
            check_price_continuity: Check for price gaps between sessions
            check_volume_spikes: Check for abnormal volume spikes
            check_weekend_trading: Check for weekend trading (non-crypto)
            check_market_hours: Check for trades outside market hours
            volume_spike_threshold: Multiplier for volume spike detection
            price_gap_threshold: Threshold for price gap detection
            is_crypto: Whether this is cryptocurrency data (24/7 trading)
        """
        self.check_price_continuity = check_price_continuity
        self.check_volume_spikes = check_volume_spikes
        self.check_weekend_trading = check_weekend_trading
        self.check_market_hours = check_market_hours
        self.volume_spike_threshold = volume_spike_threshold
        self.price_gap_threshold = price_gap_threshold
        self.is_crypto = is_crypto
        self.calendar = calendar
        self.feed_spec = FeedSpec.from_any(feed_spec)

    def name(self) -> str:
        """Return validator name."""
        return "CrossValidator"

    def validate(
        self,
        df: pl.DataFrame,
        reference_df: pl.DataFrame | None = None,
        *,
        frequency: str | None = None,
        calendar: str | None | CalendarUnset = UNSET_CALENDAR,
        feed_spec: Any | None = None,
        **kwargs: Any,
    ) -> ValidationResult:
        """
        Validate DataFrame with cross-validation checks.

        Args:
            df: Primary DataFrame to validate
            reference_df: Optional reference DataFrame for comparison
            **kwargs: Additional parameters

        Returns:
            ValidationResult with any issues found
        """
        start_time = time.time()
        result = ValidationResult(passed=True)

        # Store metadata
        result.metadata["row_count"] = len(df)
        if reference_df is not None:
            result.metadata["reference_row_count"] = len(reference_df)

        # Perform validation checks
        if self.check_price_continuity:
            self._check_price_continuity(df, result)

        if self.check_volume_spikes:
            self._check_volume_spikes(df, result)

        feed = self.feed_spec if feed_spec is None else FeedSpec.from_any(feed_spec)
        if calendar is UNSET_CALENDAR:
            calendar = self.calendar
        if (
            calendar is UNSET_CALENDAR
            and self.is_crypto
            and (
                feed.calendar is UNSET_CALENDAR
                or (feed.time_contract_version == 1 and feed.calendar is None)
            )
        ):
            calendar = "24/7"
        if self.check_weekend_trading or self.check_market_hours:
            self._check_sessions(df, result, calendar, feed, frequency)

        # Cross-reference validation if reference data provided
        if reference_df is not None:
            self._cross_reference_validation(df, reference_df, result)

        result.duration_seconds = time.time() - start_time
        return result

    def _check_price_continuity(self, df: pl.DataFrame, result: ValidationResult) -> None:
        """Check for price gaps between consecutive periods."""
        if len(df) < 2:
            return

        # Calculate price gaps between close and next open
        df_with_gaps = df.select(
            [
                "timestamp",
                "close",
                pl.col("open").shift(-1).alias("next_open"),
            ]
        ).drop_nulls()

        # Calculate gap percentage
        df_with_gaps = df_with_gaps.with_columns(
            ((pl.col("next_open") - pl.col("close")).abs() / pl.col("close")).alias("gap_pct")
        )

        # Find significant gaps
        significant_gaps = df_with_gaps.filter(pl.col("gap_pct") > self.price_gap_threshold)

        if len(significant_gaps) > 0:
            max_gap = significant_gaps["gap_pct"].max()
            result.add_issue(
                ValidationIssue(
                    severity=Severity.WARNING,
                    check="price_continuity",
                    message=f"Found {len(significant_gaps)} price gaps >{self.price_gap_threshold:.0%}",
                    details={
                        "threshold": self.price_gap_threshold,
                        "max_gap": _numeric_scalar(max_gap, "gap_pct"),
                        "gap_count": len(significant_gaps),
                    },
                    row_count=len(significant_gaps),
                    sample_rows=significant_gaps.with_row_index()["index"].to_list()[:10],
                )
            )

    def _check_volume_spikes(self, df: pl.DataFrame, result: ValidationResult) -> None:
        """Check for abnormal volume spikes."""
        if len(df) < 20:  # Need enough data for meaningful average
            return

        # Calculate rolling average volume (20-period)
        df_with_avg = df.with_columns(
            pl.col("volume").rolling_mean(window_size=20, min_samples=10).alias("avg_volume")
        ).drop_nulls()

        # Find volume spikes
        df_with_spikes = df_with_avg.with_columns(
            (pl.col("volume") / pl.col("avg_volume")).alias("volume_ratio")
        )

        volume_spikes = df_with_spikes.filter(pl.col("volume_ratio") > self.volume_spike_threshold)

        if len(volume_spikes) > 0:
            max_spike = volume_spikes["volume_ratio"].max()
            result.add_issue(
                ValidationIssue(
                    severity=Severity.INFO,
                    check="volume_spikes",
                    message=f"Found {len(volume_spikes)} volume spikes >{self.volume_spike_threshold}x average",
                    details={
                        "threshold": self.volume_spike_threshold,
                        "max_spike": _numeric_scalar(max_spike, "volume_ratio"),
                        "spike_count": len(volume_spikes),
                    },
                    row_count=len(volume_spikes),
                    sample_rows=volume_spikes.with_row_index()["index"].to_list()[:10],
                )
            )

    def _check_sessions(
        self,
        df: pl.DataFrame,
        result: ValidationResult,
        calendar: str | None | CalendarUnset,
        feed: FeedSpec,
        frequency: str | None,
    ) -> None:
        from ml4t.data.sessions import SessionAssigner
        from ml4t.data.sessions.assigner import _uses_session_labels

        assigner = SessionAssigner(calendar=calendar, feed_spec=feed)
        if assigner.calendar is None or df.is_empty() or "timestamp" not in df.columns:
            return
        frequency = frequency or feed.data_frequency
        daily = frequency is not None and frequency.lower() in {"daily", "1day", "1d"}
        frame = df.select("timestamp").with_row_index("_validation_row")
        bar_frequency = "daily" if daily else "auto" if frequency is None else "intraday"
        assigned = assigner.assign_sessions(frame, bar_frequency=bar_frequency)
        outside = assigned.filter(pl.col("session_date").is_null())
        if outside.is_empty():
            return

        daily = _uses_session_labels(frame, feed, bar_frequency)
        clock = pl.col("timestamp")
        dtype = df.schema["timestamp"]
        if isinstance(dtype, pl.Datetime):
            if daily:
                if dtype.time_zone is not None and feed.timezone is not None:
                    clock = clock.dt.convert_time_zone(feed.timezone)
            else:
                if dtype.time_zone is None:
                    clock = clock.dt.replace_time_zone(feed.timezone or "UTC")
                clock = clock.dt.convert_time_zone(str(assigner.calendar.tz))
        weekend = outside.filter(clock.dt.weekday() >= 6)
        if self.check_weekend_trading and not weekend.is_empty():
            result.add_issue(
                ValidationIssue(
                    severity=Severity.WARNING,
                    check="weekend_trading",
                    message=f"Found {len(weekend)} weekend records outside the selected schedule",
                    details={
                        "weekend_count": len(weekend),
                        "total_count": len(df),
                        "percentage": len(weekend) / len(df) * 100,
                    },
                    row_count=len(weekend),
                    sample_rows=weekend["_validation_row"].head(10).to_list(),
                )
            )

        if self.check_market_hours:
            result.add_issue(
                ValidationIssue(
                    severity=Severity.INFO,
                    check="market_hours",
                    message=f"Found {len(outside)} records outside the selected market schedule",
                    details={
                        "outside_hours_count": len(outside),
                        "weekend_count": len(weekend),
                        "total_count": len(df),
                        "percentage": len(outside) / len(df) * 100,
                    },
                    row_count=len(outside),
                    sample_rows=outside["_validation_row"].head(10).to_list(),
                )
            )

    def _cross_reference_validation(
        self, df: pl.DataFrame, reference_df: pl.DataFrame, result: ValidationResult
    ) -> None:
        """Cross-reference validation between primary and reference data."""
        # Check overlapping timestamps
        df_timestamps = set(df["timestamp"].to_list())
        ref_timestamps = set(reference_df["timestamp"].to_list())

        common_timestamps = df_timestamps & ref_timestamps

        if not common_timestamps:
            result.add_issue(
                ValidationIssue(
                    severity=Severity.WARNING,
                    check="cross_reference",
                    message="No overlapping timestamps between primary and reference data",
                    details={
                        "primary_count": len(df_timestamps),
                        "reference_count": len(ref_timestamps),
                    },
                )
            )
            return

        # Compare prices at common timestamps
        df_common = df.filter(pl.col("timestamp").is_in(common_timestamps))
        ref_common = reference_df.filter(pl.col("timestamp").is_in(common_timestamps))

        # Join on timestamp and compare
        comparison = df_common.join(ref_common, on="timestamp", suffix="_ref")

        # Check close price differences
        comparison = comparison.with_columns(
            ((pl.col("close") - pl.col("close_ref")).abs() / pl.col("close")).alias(
                "price_diff_pct"
            )
        )

        # Flag significant differences (>1%)
        significant_diffs = comparison.filter(pl.col("price_diff_pct") > 0.01)

        if len(significant_diffs) > 0:
            max_diff = significant_diffs["price_diff_pct"].max()
            result.add_issue(
                ValidationIssue(
                    severity=Severity.WARNING,
                    check="cross_reference",
                    message=f"Found {len(significant_diffs)} timestamps with >1% price difference vs reference",
                    details={
                        "max_difference": _numeric_scalar(max_diff, "price_diff_pct"),
                        "diff_count": len(significant_diffs),
                        "common_timestamps": len(common_timestamps),
                    },
                    row_count=len(significant_diffs),
                    sample_rows=significant_diffs.with_row_index()["index"].to_list()[:10],
                )
            )
