"""Gap detection and filling utilities for time series data."""

from __future__ import annotations

import re
from bisect import bisect_left, bisect_right
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, ClassVar

import polars as pl
import structlog

from ml4t.data.sessions.assigner import (
    SessionAssigner,
    _daily_session_labels,
    _schedule_for_instants,
    _session_intervals,
)
from ml4t.specs import (
    UNSET_CALENDAR,
    BarLabel,
    CalendarUnset,
    FeedSpec,
    PartialBarPolicy,
    TimestampSemantics,
)

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


class _IntradayGrid:
    def __init__(
        self,
        intervals: list[tuple[datetime, datetime, int, int]],
        interval: timedelta,
        right_labeled: bool,
        include_partial: bool,
        *,
        start: datetime,
        end: datetime,
    ) -> None:
        self.intervals = intervals
        self.interval = interval
        self.right_labeled = right_labeled
        self.include_partial = include_partial
        self.start = start
        self.end = end
        self.boundaries = [row[1] if right_labeled else row[0] for row in intervals]
        self.offsets = [row[3] for row in intervals]
        self.size = intervals[-1][3] + intervals[-1][2] if intervals else 0

    def position(self, timestamp: datetime) -> int:
        index = (
            bisect_left(self.boundaries, timestamp)
            if self.right_labeled
            else bisect_right(self.boundaries, timestamp) - 1
        )
        if index < 0 or index >= len(self.intervals):
            raise ValueError("intraday observations contain off-grid timestamps")
        opening, closing, count, offset = self.intervals[index]
        if self.right_labeled and timestamp == closing:
            if (closing - opening) % self.interval == timedelta(0) or self.include_partial:
                return offset + count - 1
            raise ValueError("intraday observations contain off-grid partial-close timestamps")
        position, remainder = divmod(timestamp - opening, self.interval)
        if self.right_labeled:
            position -= 1
        if remainder or timestamp < opening or timestamp >= closing or not 0 <= position < count:
            raise ValueError("intraday observations contain off-grid timestamps")
        return offset + position

    def timestamp(self, position: int) -> datetime:
        if not 0 <= position < self.size:
            raise IndexError("bar position outside the expected grid")
        opening, closing, _, offset = self.intervals[bisect_right(self.offsets, position) - 1]
        timestamp = opening + (position - offset + int(self.right_labeled)) * self.interval
        return min(timestamp, closing) if self.right_labeled else timestamp

    def bound(self, timestamp: datetime, *, inclusive: bool = False) -> int:
        left, right = 0, self.size
        while left < right:
            middle = (left + right) // 2
            candidate = self.timestamp(middle)
            if candidate < timestamp or (inclusive and candidate == timestamp):
                left = middle + 1
            else:
                right = middle
        return left


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
        frequency: str | None = None,
        timestamp_col: str = "timestamp",
        is_crypto: bool = False,
        *,
        calendar: str | None | CalendarUnset = UNSET_CALENDAR,
        feed_spec: Any | None = None,
    ) -> list[DataGap]:
        """
        Detect gaps in time series data.

        Args:
            df: DataFrame with time series data
            frequency: Data frequency; inherit the feed declaration, otherwise daily.
            timestamp_col: Name of timestamp column
            is_crypto: If True, expect 24/7 data; if False, market hours only
            calendar: Schedule override; None disables schedule-based detection.
            feed_spec: Shared feed declaration, resolved before the NYSE default.

        Returns:
            List of detected gaps
        """
        feed = FeedSpec.from_any(feed_spec)
        if frequency is None:
            frequency = str(feed.data_frequency) if feed.data_frequency is not None else "daily"
        # Preserve the old crypto hint only when the feed has no declaration.
        if (
            frequency.lower() in {"daily", "1day"}
            and calendar is UNSET_CALENDAR
            and is_crypto
            and (
                feed.calendar is UNSET_CALENDAR
                or (feed.time_contract_version == 1 and feed.calendar is None)
            )
        ):
            calendar = "24/7"
        selected_calendar = feed.resolve_calendar(calendar=calendar)
        if selected_calendar is None:
            return []
        df = self._observed_rows(df)
        declared_interval = (
            self._intraday_interval(str(feed.data_frequency))
            if feed.data_frequency is not None
            else None
        )
        if (
            declared_interval is not None
            and self._intraday_interval(frequency) != declared_interval
        ):
            raise ValueError("declared grid data_frequency must match the requested frequency")
        if frequency.lower() in {"daily", "1day"}:
            return self._detect_daily_gaps(df, timestamp_col, frequency, selected_calendar, feed)
        if self._intraday_interval(frequency) is not None:
            return self._detect_intraday_gaps(df, timestamp_col, frequency, selected_calendar, feed)
        if calendar is not UNSET_CALENDAR or feed_spec is not None:
            raise NotImplementedError(
                "calendar/feed declarations currently support daily gaps only"
            )
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

    @staticmethod
    def _observed_rows(df: pl.DataFrame) -> pl.DataFrame:
        if "is_imputed" not in df.columns:
            return df
        if df.schema["is_imputed"] != pl.Boolean:
            raise TypeError("is_imputed must be a Boolean placeholder marker")
        return df.filter(~pl.col("is_imputed").fill_null(False))

    @staticmethod
    def _intraday_interval(frequency: str) -> timedelta | None:
        value = frequency.lower().strip().replace("hourly", "hour")
        match = re.fullmatch(r"(\d+)?(minutes?|min|m|hours?|h)", value)
        if match is None:
            return None
        count = int(match[1] or 1)
        if count <= 0:
            raise ValueError("bar frequency must be positive")
        return timedelta(hours=count) if match[2].startswith("h") else timedelta(minutes=count)

    def _intraday_grid(
        self,
        df: pl.DataFrame,
        timestamp_col: str,
        frequency: str,
        calendar: str,
        feed: FeedSpec,
        *,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> tuple[_IntradayGrid | None, list[datetime]]:
        interval = self._intraday_interval(frequency)
        if interval is None:
            raise ValueError("unsupported intraday bar frequency")
        required = (
            "data_frequency",
            "timestamp_semantics",
            "bar_grid_anchor",
            "bar_label",
            "partial_bar_policy",
        )
        if feed.time_contract_version != 2 or any(getattr(feed, name) is None for name in required):
            raise ValueError("intraday detection requires a complete declared version-2 bar grid")
        if feed.timestamp_semantics == TimestampSemantics.SESSION_LABEL:
            raise ValueError("intraday detection requires a declared event/close bar grid")
        if feed.bar_type not in (None, "time"):
            raise ValueError("intraday grid detection requires fixed time bars")
        if interval != self._intraday_interval(str(feed.data_frequency)):
            raise ValueError("declared grid data_frequency must match the requested frequency")
        assigner = SessionAssigner(calendar, feed_spec=feed)
        if timestamp_col not in df.columns:
            raise ValueError(f"DataFrame must have '{timestamp_col}' column")
        dtype = df.schema[timestamp_col]
        if not isinstance(dtype, pl.Datetime):
            raise TypeError(f"'{timestamp_col}' must contain Datetime values for an intraday grid")
        if df[timestamp_col].null_count():
            raise ValueError(f"'{timestamp_col}' must not contain null timestamps")
        if dtype.time_zone is None and feed.timezone is None:
            raise ValueError("naive intraday timestamps require a declared timezone")
        expr = pl.col(timestamp_col)
        if dtype.time_zone is None:
            expr = expr.dt.replace_time_zone(feed.timezone)
        columns = [expr.dt.convert_time_zone("UTC").alias(timestamp_col)]
        if "is_imputed" in df.columns:
            columns.append(pl.col("is_imputed"))
        scoped = df.select(columns)
        if start is not None:
            scoped = scoped.filter(pl.col(timestamp_col) >= start)
        if end is not None:
            scoped = scoped.filter(pl.col(timestamp_col) <= end)
        timestamps = scoped[timestamp_col].unique().sort().to_list()
        if not timestamps and (start is None or end is None):
            return None, []
        start = start if start is not None else timestamps[0]
        end = end if end is not None else timestamps[-1]
        if start > end:
            raise ValueError("start_date must not be after end_date")
        actual = self._observed_rows(scoped)
        if (
            dtype.time_unit == "ns"
            and actual.select((pl.col(timestamp_col).dt.nanosecond() % 1000 != 0).any()).item()
        ):
            raise ValueError("intraday observations contain off-grid sub-microsecond timestamps")
        observed = actual[timestamp_col].unique().sort().to_list()
        schedule = _schedule_for_instants(assigner.calendar, start, end, padding_days=2)
        grid_intervals: list[tuple[datetime, datetime, int, int]] = []
        total_bars = 0
        for _, row in schedule.iterrows():
            for opening, closing in _session_intervals(row):
                opening, closing = opening.astimezone(UTC), closing.astimezone(UTC)
                count, remainder = divmod(closing - opening, interval)
                if remainder and feed.partial_bar_policy == PartialBarPolicy.INCLUDE:
                    count += 1
                if count <= 0:
                    continue
                if grid_intervals and opening < grid_intervals[-1][1]:
                    raise ValueError("calendar returned overlapping tradable intervals")
                grid_intervals.append((opening, closing, count, total_bars))
                total_bars += count
        return (
            _IntradayGrid(
                grid_intervals,
                interval,
                feed.bar_label == BarLabel.RIGHT,
                feed.partial_bar_policy == PartialBarPolicy.INCLUDE,
                start=start,
                end=end,
            ),
            observed,
        )

    def _detect_intraday_gaps(
        self,
        df: pl.DataFrame,
        timestamp_col: str,
        frequency: str,
        calendar: str,
        feed: FeedSpec,
    ) -> list[DataGap]:
        grid, observed = self._intraday_grid(df, timestamp_col, frequency, calendar, feed)
        if grid is None:
            return []
        positions = [grid.position(timestamp) for timestamp in observed]
        gaps = []
        for current, following, left_index, right_index in zip(
            observed, observed[1:], positions, positions[1:]
        ):
            missing = right_index - left_index - 1
            if missing > 0:
                gaps.append(DataGap(current, following, missing, following - current, frequency))
        return gaps

    def _detect_daily_gaps(
        self,
        df: pl.DataFrame,
        timestamp_col: str,
        frequency: str,
        calendar: str,
        feed: FeedSpec,
    ) -> list[DataGap]:
        assigner = SessionAssigner(calendar, feed_spec=feed)
        ordered = df.sort(timestamp_col) if timestamp_col in df.columns else df
        labels = _daily_session_labels(ordered, assigner, timestamp_col).to_list()
        if df.is_empty():
            return []
        timestamps = ordered[timestamp_col].to_list()
        sessions = [
            label.date()
            for label in assigner.calendar.schedule(start_date=labels[0], end_date=labels[-1]).index
        ]
        gaps = []
        for current, following, start, end in zip(timestamps, timestamps[1:], labels, labels[1:]):
            missing = bisect_left(sessions, end) - bisect_right(sessions, start)
            if missing > 0:
                current = (
                    current
                    if isinstance(current, datetime)
                    else datetime.combine(current, datetime.min.time())
                )
                following = (
                    following
                    if isinstance(following, datetime)
                    else datetime.combine(following, datetime.min.time())
                )
                gaps.append(DataGap(current, following, missing, following - current, frequency))
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
        interval = self._intraday_interval(frequency)
        if interval is not None:
            return interval

        if freq_lower == "daily" or freq_lower == "1day":
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
        method: str = "none",
        timestamp_col: str = "timestamp",
    ) -> pl.DataFrame:
        """Reject synthetic market values; an empty gap list remains a no-op."""
        if not gaps:
            return df
        raise ValueError(
            f"Synthetic gap imputation ({method!r}, {timestamp_col!r}) is unsupported; "
            "inspect missing observations "
            "and download available source data instead"
        )
