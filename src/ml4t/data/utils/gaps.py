"""Gap detection and filling utilities for time series data."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from typing import Any, ClassVar

import polars as pl
import structlog

logger = structlog.get_logger()


@dataclass
class DataGap:
    """Represents a gap in time series data."""

    start: datetime
    end: datetime
    missing_periods: int
    duration: timedelta
    frequency: str

    @property
    def is_significant(self) -> bool:
        """Check if gap is significant (more than 1 period)."""
        return self.missing_periods > 1

    def __str__(self) -> str:
        """String representation of gap."""
        duration_str = str(self.duration).split(".")[0]  # Remove microseconds
        return (
            f"Gap: {self.start.isoformat()} to {self.end.isoformat()} "
            f"({self.missing_periods} periods, {duration_str})"
        )


class GapDetector:
    """Detect and analyze gaps in time series data."""

    # Expected periods per day for different frequencies
    PERIODS_PER_DAY: ClassVar[dict[str, float]] = {
        "minute": 390,  # US market hours (9:30-16:00)
        "5minute": 78,
        "15minute": 26,
        "30minute": 13,
        "hourly": 6.5,
        "daily": 1,
        "weekly": 0.2,  # ~1 per 5 days
        "monthly": 0.045,  # ~1 per 22 days
    }

    # Tolerance for gap detection (fraction of expected interval)
    DEFAULT_TOLERANCE = 0.1

    def __init__(self, tolerance: float = DEFAULT_TOLERANCE) -> None:
        """
        Initialize gap detector.

        Args:
            tolerance: Tolerance factor for gap detection (0.1 = 10%)
        """
        self.tolerance = tolerance

    def detect_gaps(
        self,
        df: pl.DataFrame,
        frequency: str = "daily",
        timestamp_col: str = "timestamp",
        is_crypto: bool = False,
    ) -> list[DataGap]:
        """
        Detect gaps in time series data.

        Args:
            df: DataFrame with time series data
            frequency: Data frequency (minute, hourly, daily, etc.)
            timestamp_col: Name of timestamp column
            is_crypto: If True, expect 24/7 data; if False, market hours only

        Returns:
            List of detected gaps
        """
        if df.is_empty() or len(df) < 2:
            return []

        # Ensure data is sorted by timestamp
        df = df.sort(timestamp_col)
        timestamps = df[timestamp_col]

        # Calculate expected interval
        expected_interval = self._get_expected_interval(frequency, is_crypto)
        tolerance_seconds = expected_interval.total_seconds() * self.tolerance

        gaps = []

        # Check each consecutive pair of timestamps
        for i in range(len(timestamps) - 1):
            current = timestamps[i]
            next_ts = timestamps[i + 1]

            # Skip if timestamps are not datetime
            if not isinstance(current, datetime) or not isinstance(next_ts, datetime):
                continue

            # Calculate actual interval
            actual_interval = next_ts - current
            actual_seconds = actual_interval.total_seconds()
            expected_seconds = expected_interval.total_seconds()

            # Check if this is a gap (considering market hours for non-crypto)
            if (
                not is_crypto
                and frequency
                in [
                    "minute",
                    "5minute",
                    "15minute",
                    "30minute",
                    "hourly",
                ]
                and self._spans_non_trading_hours(current, next_ts, frequency)
            ):
                continue

            # Detect gap if interval exceeds expected + tolerance
            if actual_seconds > expected_seconds + tolerance_seconds:
                missing_periods = int(actual_seconds / expected_seconds) - 1

                gap = DataGap(
                    start=current,
                    end=next_ts,
                    missing_periods=missing_periods,
                    duration=actual_interval,
                    frequency=frequency,
                )
                gaps.append(gap)

        if gaps:
            logger.info(
                f"Detected {len(gaps)} gaps in data",
                total_missing_periods=sum(g.missing_periods for g in gaps),
                largest_gap=max(g.duration for g in gaps),
                frequency=frequency,
            )

        return gaps

    def _get_expected_interval(self, frequency: str, is_crypto: bool = False) -> timedelta:
        """
        Get expected time interval between data points.

        Args:
            frequency: Data frequency
            is_crypto: If True, calculate for 24/7; if False, for market hours

        Returns:
            Expected timedelta between consecutive points
        """
        freq_lower = frequency.lower()

        # Handle minute-based frequencies
        if "minute" in freq_lower:
            if freq_lower == "minute" or freq_lower == "1minute":
                return timedelta(minutes=1)
            if freq_lower == "5minute":
                return timedelta(minutes=5)
            if freq_lower == "15minute":
                return timedelta(minutes=15)
            if freq_lower == "30minute":
                return timedelta(minutes=30)

        # Handle other frequencies
        elif freq_lower == "hourly" or freq_lower == "1hour":
            return timedelta(hours=1)
        elif freq_lower == "daily" or freq_lower == "1day":
            if is_crypto:
                return timedelta(days=1)
            # For stocks, next trading day could be 1-3 days away
            return timedelta(days=1)
        elif freq_lower == "weekly" or freq_lower == "1week":
            return timedelta(weeks=1)
        elif freq_lower == "monthly" or freq_lower == "1month":
            return timedelta(days=30)  # Approximate

        # Default to daily
        return timedelta(days=1)

    def _spans_non_trading_hours(
        self,
        start: datetime,
        end: datetime,
        frequency: str,  # noqa: ARG002
    ) -> bool:
        """
        Check if gap spans non-trading hours (nights/weekends).

        Args:
            start: Start of gap
            end: End of gap
            frequency: Data frequency

        Returns:
            True if gap is expected (non-trading hours)
        """
        # Convert to UTC if needed
        if start.tzinfo is None:
            start = start.replace(tzinfo=UTC)
        if end.tzinfo is None:
            end = end.replace(tzinfo=UTC)

        # Check for weekend
        if start.weekday() == 4 and end.weekday() == 0:  # Friday to Monday
            return True

        # Check for overnight gap (4pm to 9:30am ET)
        # Convert to ET (UTC-5 or UTC-4 for DST)
        market_close_hour = 21  # 4pm ET in UTC (during standard time)
        market_open_hour = 14  # 9:30am ET in UTC (during standard time)

        # Simplified check: if gap starts after market close and ends at/after market open
        return start.hour >= market_close_hour or end.hour < market_open_hour

    def summarize_gaps(self, gaps: list[DataGap]) -> dict[str, Any]:
        """
        Summarize detected gaps.

        Args:
            gaps: List of detected gaps

        Returns:
            Summary statistics
        """
        if not gaps:
            return {
                "count": 0,
                "total_missing_periods": 0,
                "average_gap_duration": timedelta(0),
                "largest_gap": None,
                "smallest_gap": None,
            }

        durations = [g.duration for g in gaps]

        return {
            "count": len(gaps),
            "total_missing_periods": sum(g.missing_periods for g in gaps),
            "average_gap_duration": sum(durations, timedelta(0)) / len(durations),
            "largest_gap": max(gaps, key=lambda g: g.duration),
            "smallest_gap": min(gaps, key=lambda g: g.duration),
            "significant_gaps": [g for g in gaps if g.is_significant],
        }

    def fill_gaps(
        self,
        df: pl.DataFrame,
        gaps: list[DataGap],
        method: str = "forward",
        timestamp_col: str = "timestamp",
    ) -> pl.DataFrame:
        """
        Fill detected gaps in data.

        Args:
            df: DataFrame with gaps
            gaps: List of detected gaps
            method: Fill method ('forward', 'backward', 'interpolate', 'zero')
            timestamp_col: Name of timestamp column

        Returns:
            DataFrame with gaps filled
        """
        if not gaps:
            return df

        logger.info(
            f"Filling {len(gaps)} gaps using {method} method",
            total_periods=sum(g.missing_periods for g in gaps),
        )

        # Create a complete time index
        min_ts = df[timestamp_col].min()
        max_ts = df[timestamp_col].max()
        if not isinstance(min_ts, date) or not isinstance(max_ts, date):
            raise TypeError(f"'{timestamp_col}' must contain non-null Date or Datetime values")

        if gaps:
            frequency = gaps[0].frequency
            interval = self._get_expected_interval(frequency)

            # Generate complete timestamp range
            complete_timestamps = []
            current = min_ts
            while current <= max_ts:
                complete_timestamps.append(current)
                current = current + interval

            # Create DataFrame with complete timestamps
            complete_df = pl.DataFrame({timestamp_col: complete_timestamps})

            # Cast to match original datetime precision to avoid join errors
            original_dtype = df[timestamp_col].dtype
            complete_df = complete_df.with_columns(pl.col(timestamp_col).cast(original_dtype))

            # Join with original data
            filled_df = complete_df.join(df, on=timestamp_col, how="left")

            # Fill missing values based on method
            if method == "forward":
                filled_df = filled_df.fill_null(strategy="forward")
            elif method == "backward":
                filled_df = filled_df.fill_null(strategy="backward")
            elif method == "interpolate":
                # Interpolate numeric columns
                numeric_cols = [
                    col
                    for col in filled_df.columns
                    if col != timestamp_col
                    and filled_df[col].dtype in [pl.Float32, pl.Float64, pl.Int32, pl.Int64]
                ]
                for col in numeric_cols:
                    filled_df = filled_df.with_columns(pl.col(col).interpolate())
            elif method == "zero":
                filled_df = filled_df.fill_null(0)

            return filled_df

        return df
