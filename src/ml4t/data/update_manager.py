"""Incremental update system with gap detection and resume capability."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from enum import Enum
from typing import Any

import polars as pl
import structlog

from ml4t.data.core.schemas import align_frames_for_concat
from ml4t.data.providers.base import BaseProvider
from ml4t.data.providers.protocols import TimestampRangeProvider
from ml4t.data.sessions.assigner import SessionAssigner, _daily_session_labels
from ml4t.data.storage.backend import StorageBackend, normalize_storage_metadata
from ml4t.data.storage.metadata_tracker import MetadataTracker, UpdateRecord
from ml4t.data.utils.gaps import GapDetector as ObservationGapDetector
from ml4t.specs import UNSET_CALENDAR, CalendarUnset, FeedSpec

logger = structlog.get_logger()


def _ensure_datetime(value: Any) -> datetime:
    """Return a UTC datetime scalar or reject an incompatible timestamp value."""
    if isinstance(value, datetime):
        return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)
    if isinstance(value, date):
        return datetime.combine(value, datetime.min.time(), UTC)
    raise TypeError(f"Expected a date or datetime, got {type(value).__name__}")


def _session_label(value: Any) -> datetime:
    """Encode a civil session date as UTC midnight without converting an instant."""
    if not isinstance(value, date):
        raise TypeError("'timestamp' must contain non-null Date or Datetime values")
    label = value.date() if isinstance(value, datetime) else value
    return datetime.combine(label, datetime.min.time(), UTC)


def _rewrite_preserving_metadata(
    storage: StorageBackend,
    key: str,
    data: pl.DataFrame,
) -> None:
    """Publish replacement data with retained identity and a recomputed date range."""
    if data.is_empty():
        raise ValueError("Data must not be empty")
    if "timestamp" not in data.columns:
        raise ValueError("Data must have a 'timestamp' column")
    min_ts = _ensure_datetime(data["timestamp"].min())
    max_ts = _ensure_datetime(data["timestamp"].max())
    updated_at = datetime.now(UTC)
    storage.write(
        data,
        key,
        {
            "start_date": min_ts,
            "end_date": max_ts,
            # Clear the inherited value so normalization uses the commit's fresh UTC stamp.
            "last_updated": None,
            "data_range": {"start": str(min_ts), "end": str(max_ts)},
            # BulkManager compares this legacy field with naive local cutoffs.
            "attributes": {"last_update": updated_at.astimezone().replace(tzinfo=None).isoformat()},
        },
        preserve_metadata=True,
    )


class UpdateStrategy(Enum):
    """Update strategy types."""

    INCREMENTAL = "incremental"  # Only fetch new data
    APPEND_ONLY = "append_only"  # Never update existing data
    FULL_REFRESH = "full_refresh"  # Replace all data
    BACKFILL = "backfill"  # Fill gaps in historical data


def _merge_backfill_rows(
    existing: pl.DataFrame, recovered: pl.DataFrame, feed: FeedSpec
) -> tuple[pl.DataFrame, int, int]:
    if recovered.is_empty():
        return existing, 0, 0
    identity = ["timestamp"]
    if feed.entity_col is not None or "symbol" in existing.columns:
        entity = feed.resolve(existing.columns, ("symbol",)).entity_col
        stored_ids = existing[entity].drop_nulls().unique()
        if len(stored_ids) and not recovered[entity].is_in(stored_ids.implode()).all():
            raise ValueError("Backfill cannot introduce a different instrument identity")
        if len(stored_ids) > 1:
            identity.append(entity)
    existing, recovered = align_frames_for_concat(
        existing, recovered, right_fill_values={"is_imputed": False}
    )
    recovered = recovered.unique(subset=identity, keep="first", maintain_order=True)
    observed = ObservationGapDetector._observed_rows(existing)
    recovered = recovered.join(observed.select(identity), on=identity, how="anti")
    if recovered.is_empty():
        return existing, 0, 0
    retained = existing.join(recovered.select(identity), on=identity, how="anti")
    replaced = len(existing) - len(retained)
    combined = pl.concat([retained, recovered]).sort(identity)
    return combined, len(recovered) - replaced, replaced


@dataclass
class UpdateResult:
    """Result of an update operation."""

    success: bool
    update_type: str
    rows_added: int
    rows_updated: int
    rows_before: int
    rows_after: int
    gaps_filled: int = 0
    duration_seconds: float = 0.0
    errors: list[str] | None = None
    remaining_gaps: list[dict[str, Any]] | None = None

    def __post_init__(self) -> None:
        """Initialize errors list if None."""
        if self.errors is None:
            self.errors = []


class GapDetector:
    """Detect gaps in time-series data."""

    def __init__(
        self,
        exclude_weekends: bool | None = None,
        *,
        calendar: str | None | CalendarUnset = UNSET_CALENDAR,
        feed_spec: Any | None = None,
    ) -> None:
        """
        Initialize gap detector.

        Args:
            exclude_weekends: Legacy hint; explicit False selects 24/7 absent a declaration.
            calendar: Explicit schedule selection; None disables schedule classification.
            feed_spec: Shared feed declaration, resolved before the NYSE default.
        """
        self.exclude_weekends = exclude_weekends
        self.calendar = calendar
        self.feed_spec = feed_spec

    def _feed_calendar(
        self, calendar: str | None | CalendarUnset, feed_spec: Any | None
    ) -> tuple[FeedSpec, str | None]:
        feed = FeedSpec.from_any(self.feed_spec if feed_spec is None else feed_spec)
        if calendar is UNSET_CALENDAR:
            calendar = self.calendar
        if (
            calendar is UNSET_CALENDAR
            and self.exclude_weekends is False
            and (
                feed.calendar is UNSET_CALENDAR
                or (feed.time_contract_version == 1 and feed.calendar is None)
            )
        ):
            calendar = "24/7"
        return feed, feed.resolve_calendar(calendar=calendar)

    @staticmethod
    def _label_frame(df: pl.DataFrame, feed: FeedSpec, calendar: str | None) -> pl.DataFrame:
        labels = _daily_session_labels(df, SessionAssigner(calendar, feed_spec=feed))
        return df.with_columns(labels.cast(pl.Datetime("us")).dt.replace_time_zone("UTC"))

    @staticmethod
    def _by_entity(
        df: pl.DataFrame,
        feed: FeedSpec,
        detect: Callable[[pl.DataFrame], list[dict[str, Any]]],
        *,
        storage: StorageBackend | None = None,
        key: str | None = None,
    ) -> list[dict[str, Any]]:
        if feed.entity_col is None and "symbol" not in df.columns:
            return detect(df)
        if storage is not None and key is not None and not storage.exists(key):
            return detect(df)
        entity = (
            feed.with_overrides(timestamp_col="timestamp")
            .resolve(df.columns, ("symbol",))
            .entity_col
        )
        if not isinstance(entity, str):
            raise ValueError("a single entity column is required")
        identity_columns = [entity] + (["is_imputed"] if "is_imputed" in df.columns else [])
        identity_frame = (
            storage.read(key).select(identity_columns).collect()
            if storage is not None and key is not None
            else df.select(identity_columns)
        )
        identifiers = ObservationGapDetector._observed_rows(identity_frame)[entity].unique()
        if identifiers.null_count():
            raise ValueError("entity identifiers must not be null")
        if len(identifiers) <= 1:
            return detect(df)
        return [
            {**gap, "entity": identifier}
            for identifier in identifiers.sort().to_list()
            for gap in detect(df.filter(pl.col(entity) == identifier))
        ]

    @staticmethod
    def _select_gap_rows(
        data: pl.DataFrame, gap: dict[str, Any], feed: FeedSpec, calendar: str
    ) -> pl.DataFrame:
        actual = ObservationGapDetector._observed_rows(data)
        if "entity" in gap:
            entity = feed.resolve(actual.columns, ("symbol",)).entity_col
            actual = actual.filter(pl.col(entity) == gap["entity"])
        frequency = feed.data_frequency or "daily"
        if frequency.lower() in {"daily", "1d", "1day"}:
            schedule = SessionAssigner(calendar).calendar.schedule(
                start_date=gap["start"].date(), end_date=gap["end"].date()
            )
            labels = GapDetector._label_frame(actual, feed, calendar)["timestamp"].dt.date()
            return actual.filter(labels.is_in([label.date() for label in schedule.index]))
        actual = actual.filter(
            (pl.col("timestamp") >= gap["start"]) & (pl.col("timestamp") <= gap["end"])
        )
        if not actual.is_empty():
            grid, timestamps = ObservationGapDetector()._intraday_grid(
                actual,
                "timestamp",
                frequency,
                calendar,
                feed,
                start=gap["start"],
                end=gap["end"],
            )
            if grid is not None:
                for timestamp in timestamps:
                    grid.position(timestamp)
        return actual

    @staticmethod
    def _daily_ranges(
        df: pl.DataFrame,
        start: datetime,
        end: datetime,
        calendar: str,
        tolerance_days: int = 0,
    ) -> list[dict[str, Any]]:
        if start.date() > end.date():
            raise ValueError("start_date must not be after end_date")
        assigner = SessionAssigner(calendar)
        observed = {
            value.date() if isinstance(value, datetime) else value
            for value in ObservationGapDetector._observed_rows(df)["timestamp"].to_list()
        }
        schedule = assigner.calendar.schedule(start_date=start.date(), end_date=end.date())
        gaps = []
        missing: list[date] = []

        def append_gap() -> None:
            if len(missing) > tolerance_days:
                gaps.append(
                    {
                        "start": datetime.combine(missing[0], datetime.min.time(), UTC),
                        "end": datetime.combine(missing[-1], datetime.min.time(), UTC),
                        "size_days": len(missing),
                    }
                )

        for label in schedule.index:
            session = label.date()
            if session not in observed:
                missing.append(session)
            elif missing:
                append_gap()
                missing = []
        if missing:
            append_gap()
        return gaps

    def detect_gaps(
        self,
        df: pl.DataFrame,
        frequency: str | None = None,
        tolerance_days: int = 0,
        *,
        calendar: str | None | CalendarUnset = UNSET_CALENDAR,
        feed_spec: Any | None = None,
        tolerance_bars: int = 0,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> list[dict[str, Any]]:
        """
        Detect gaps in a DataFrame's timestamp column.

        Args:
            df: DataFrame with timestamp column
            frequency: Inherit the feed frequency when omitted, otherwise daily.
            tolerance_days: Missing sessions tolerated for daily data; elapsed days otherwise.
            calendar: Per-call override of configured calendar and feed metadata.
            feed_spec: Per-call feed declaration.
            tolerance_bars: Number of consecutive missing intraday bars to tolerate.
            start_date: Inclusive requested coverage start; supply together with end_date.
            end_date: Inclusive coverage end. Daily bounds are labels; intraday bounds are instants.

        Returns:
            List of gap dictionaries with start, end, and size
        """
        if (start_date is None) != (end_date is None):
            raise ValueError("start_date and end_date must be supplied together")
        feed, selected = self._feed_calendar(calendar, feed_spec)
        frequency = (feed.data_frequency or "daily") if frequency is None else frequency
        if selected is None:
            return []
        if df.is_empty() and start_date is None:
            return []
        if feed.timestamp_col != "timestamp":
            if feed.timestamp_col not in df.columns:
                raise ValueError(f"declared timestamp column {feed.timestamp_col!r} is missing")
            df = df.select(
                pl.col(feed.timestamp_col).alias("timestamp"),
                *[name for name in df.columns if name not in {"timestamp", feed.timestamp_col}],
            )
            feed = feed.with_overrides(timestamp_col="timestamp")
        if "timestamp" not in df.columns:
            if not df.is_empty():
                raise ValueError("DataFrame must have 'timestamp' column")
            df = df.with_columns(pl.Series("timestamp", [], dtype=pl.Datetime("us", "UTC")))
        if frequency.lower() in {"daily", "1day", "1d"}:

            def daily_ranges(part: pl.DataFrame) -> list[dict[str, Any]]:
                labeled = self._label_frame(part, feed, selected)
                return self._daily_ranges(
                    labeled,
                    _session_label(
                        start_date if start_date is not None else labeled["timestamp"].min()
                    ),
                    _session_label(
                        end_date if end_date is not None else labeled["timestamp"].max()
                    ),
                    selected,
                    tolerance_days,
                )

            return self._by_entity(df, feed, daily_ranges)
        if ObservationGapDetector._intraday_interval(frequency) is not None:
            if tolerance_days:
                raise ValueError("use tolerance_bars, not tolerance_days, for intraday data")
            return self._by_entity(
                df,
                feed,
                lambda part: self._intraday_ranges(
                    part,
                    frequency,
                    selected,
                    feed,
                    tolerance_bars=tolerance_bars,
                    start=_ensure_datetime(start_date) if start_date is not None else None,
                    end=_ensure_datetime(end_date) if end_date is not None else None,
                ),
            )
        if (
            calendar is not UNSET_CALENDAR
            or self.calendar is not UNSET_CALENDAR
            or feed_spec is not None
            or self.feed_spec is not None
        ):
            raise NotImplementedError(
                "calendar/feed declarations currently support daily ranges only"
            )

        # Sort by timestamp
        df = df.sort("timestamp")
        # Ensure we have datetime objects, not date objects
        timestamps = [_ensure_datetime(ts) for ts in df["timestamp"].to_list()]

        gaps = []
        expected_delta = self._get_expected_delta(frequency)

        for i in range(len(timestamps) - 1):
            current = timestamps[i]
            next_ts = timestamps[i + 1]
            expected = current + expected_delta

            if next_ts > expected + timedelta(days=tolerance_days):
                gap_start = current + expected_delta
                gap_end = next_ts - expected_delta
                gaps.append(
                    {
                        "start": gap_start,
                        "end": gap_end,
                        "size_days": (gap_end - gap_start).days + 1,
                    }
                )

        return gaps

    @staticmethod
    def _intraday_ranges(
        df: pl.DataFrame,
        frequency: str,
        calendar: str,
        feed: FeedSpec,
        *,
        start: datetime | None = None,
        end: datetime | None = None,
        tolerance_bars: int = 0,
    ) -> list[dict[str, Any]]:
        if (
            isinstance(tolerance_bars, bool)
            or not isinstance(tolerance_bars, int)
            or tolerance_bars < 0
        ):
            raise ValueError("tolerance_bars must be a nonnegative integer")
        grid, observed = ObservationGapDetector()._intraday_grid(
            df, "timestamp", frequency, calendar, feed, start=start, end=end
        )
        if grid is None:
            return []
        positions = sorted({grid.position(timestamp) for timestamp in observed})
        first = grid.bound(grid.start)
        stop = grid.bound(grid.end, inclusive=True)
        gaps = []
        cursor = first
        for position in [p for p in positions if first <= p < stop] + [stop]:
            if position - cursor > tolerance_bars:
                gaps.append(
                    {
                        "start": grid.timestamp(cursor),
                        "end": grid.timestamp(position - 1),
                        "size_bars": position - cursor,
                    }
                )
            cursor = position + 1
        return gaps

    def detect_gaps_in_storage(
        self,
        storage: StorageBackend,
        key: str,
        start_date: datetime,
        end_date: datetime,
        frequency: str | None = None,
        *,
        calendar: str | None | CalendarUnset = UNSET_CALENDAR,
        feed_spec: Any | None = None,
    ) -> list[dict[str, Any]]:
        """
        Detect gaps through a storage backend.

        Args:
            storage: Storage backend
            key: Storage key for the dataset
            start_date: Start of range to check
            end_date: End of range to check
            frequency: Inherit the stored/feed frequency when omitted, otherwise daily.

        Returns:
            Read-only gap report with inclusive UTC-aware bounds. Daily size_days counts
            sessions; intraday size_bars counts expected bars, including boundary absences.
        """
        if feed_spec is None and self.feed_spec is None:
            try:
                feed_spec = normalize_storage_metadata(storage.get_metadata(key), key)
            except KeyError:
                pass
        feed, selected = self._feed_calendar(calendar, feed_spec)
        frequency = (feed.data_frequency or "daily") if frequency is None else frequency
        if selected is None:
            return []
        if frequency.lower() in {"daily", "1day", "1d"}:
            start_date = _session_label(start_date)
            end_date = _session_label(end_date)
            # Session labels retain their civil date even when stored with a timezone.
            start_bound = start_date - timedelta(days=1)
            end_bound = datetime.combine(end_date.date(), datetime.min.time(), UTC) + timedelta(
                days=2
            )
            try:
                df = storage.read(key, start_bound, end_bound).collect()
            except KeyError:
                df = pl.DataFrame(schema={"timestamp": pl.Datetime("us", "UTC")})
            return self._by_entity(
                df,
                feed,
                lambda part: self._daily_ranges(
                    self._label_frame(part, feed, selected), start_date, end_date, selected
                ),
                storage=storage,
                key=key,
            )
        start_date = _ensure_datetime(start_date)
        end_date = _ensure_datetime(end_date)
        if start_date > end_date:
            raise ValueError("start_date must not be after end_date")
        if ObservationGapDetector._intraday_interval(frequency) is not None:
            try:
                df = storage.read(key, start_date, end_date + timedelta(microseconds=1)).collect()
            except KeyError:
                df = pl.DataFrame(schema={"timestamp": pl.Datetime("us", "UTC")})
            return self._by_entity(
                df,
                feed,
                lambda part: self._intraday_ranges(
                    part, frequency, selected, feed, start=start_date, end=end_date
                ),
                storage=storage,
                key=key,
            )
        if (
            calendar is not UNSET_CALENDAR
            or self.calendar is not UNSET_CALENDAR
            or feed_spec is not None
            or self.feed_spec is not None
        ):
            raise NotImplementedError(
                "calendar/feed declarations currently support daily ranges only"
            )
        try:
            # Storage end bounds are exclusive. Open the inclusive end by one
            # microsecond so a bar timestamped on end_date is part of the range.
            df = storage.read(key, start_date, end_date + timedelta(microseconds=1)).collect()

            if df.is_empty():
                # No data in range means entire range is a gap
                return [
                    {
                        "start": start_date,
                        "end": end_date,
                        "size_days": (end_date - start_date).days + 1,
                    }
                ]

            # Detect gaps in the data
            gaps = self.detect_gaps(df, frequency)

            # Check for gap at the beginning
            first_timestamp = _ensure_datetime(df["timestamp"].min())
            if first_timestamp > start_date:
                gaps.insert(
                    0,
                    {
                        "start": start_date,
                        "end": first_timestamp - timedelta(days=1),
                        "size_days": (first_timestamp - start_date).days,
                    },
                )

            # Check for gap at the end
            last_timestamp = _ensure_datetime(df["timestamp"].max())
            if last_timestamp < end_date:
                gaps.append(
                    {
                        "start": last_timestamp + timedelta(days=1),
                        "end": end_date,
                        "size_days": (end_date - last_timestamp).days,
                    }
                )

            return gaps

        except KeyError:
            # No data exists for this key
            return [
                {
                    "start": start_date,
                    "end": end_date,
                    "size_days": (end_date - start_date).days + 1,
                }
            ]

    def _get_expected_delta(self, frequency: str) -> timedelta:
        """Get expected time delta between consecutive points."""
        if frequency == "daily":
            return timedelta(days=1)
        if frequency == "hourly":
            return timedelta(hours=1)
        if frequency == "weekly":
            return timedelta(weeks=1)
        raise ValueError(f"unsupported frequency: {frequency!r}")


class IncrementalUpdater:
    """Manages incremental updates with resume capability."""

    def __init__(
        self,
        strategy: UpdateStrategy = UpdateStrategy.INCREMENTAL,
    ) -> None:
        """
        Initialize incremental updater.

        Args:
            strategy: Default update strategy
        """
        self.strategy = strategy
        self.gap_detector = GapDetector(exclude_weekends=True)

    def determine_update_range(
        self,
        storage: StorageBackend,
        key: str,
        requested_start: datetime,
        requested_end: datetime,
        *,
        calendar: str | None | CalendarUnset = UNSET_CALENDAR,
        feed_spec: Any | None = None,
    ) -> tuple[datetime, datetime, str]:
        """
        Determine the actual range to update based on existing data.

        Args:
            storage: Storage backend
            key: Dataset key
            requested_start: Requested start date
            requested_end: Requested end date

        Returns:
            Tuple of UTC-aware (actual_start, actual_end, update_type)
        """
        requested_start = _ensure_datetime(requested_start)
        requested_end = _ensure_datetime(requested_end)
        if requested_start > requested_end:
            raise ValueError("requested_start must not be after requested_end")
        try:
            # Check if data exists
            if not storage.exists(key):
                return requested_start, requested_end, "full"

            # Get existing data range
            df = storage.read(key).collect()
            metadata = normalize_storage_metadata(storage.get_metadata(key), key)
        except Exception as e:
            logger.warning(f"Error checking existing data: {e}")
            return requested_start, requested_end, "full"

        observed = ObservationGapDetector._observed_rows(df)
        if observed.is_empty():
            return requested_start, requested_end, "full"
        feed, selected = self.gap_detector._feed_calendar(
            calendar, metadata if feed_spec is None else feed_spec
        )
        frequency = feed.data_frequency or "daily"
        if selected is not None and frequency.lower() in {"daily", "1d", "1day"}:
            observed = self.gap_detector._label_frame(observed, feed, selected)
        last_timestamp = _ensure_datetime(observed["timestamp"].max())
        if requested_end <= last_timestamp:
            return requested_end, requested_end, "none"
        lower = max(last_timestamp, requested_start)
        if selected is None:
            # Without a schedule, overlap the last real row rather than invent a next bar.
            return lower, requested_end, "incremental"
        gaps = self.gap_detector.detect_gaps_in_storage(
            storage, key, lower, requested_end, calendar=selected, feed_spec=feed
        )
        if not gaps:
            return requested_end, requested_end, "none"
        return min(gap["start"] for gap in gaps), requested_end, "incremental"

    def update_incremental(
        self,
        storage: StorageBackend,
        tracker: MetadataTracker,
        key: str,
        new_data: pl.DataFrame,
        provider: str,
        strategy: UpdateStrategy | None = None,
    ) -> UpdateResult:
        """
        Perform incremental update with atomic operations.

        Args:
            storage: Storage backend
            tracker: Metadata tracker
            key: Dataset key
            new_data: New data to add/update
            provider: Provider name
            strategy: Update strategy override

        Returns:
            UpdateResult with operation details
        """
        start_time = time.time()
        strategy = strategy or self.strategy
        errors = []
        rows_before = 0

        try:
            # Validate new data
            if "timestamp" not in new_data.columns:
                error_msg = "New data must have 'timestamp' column"
                errors.append(error_msg)
                return UpdateResult(
                    success=False,
                    update_type=strategy.value,
                    rows_added=0,
                    rows_updated=0,
                    rows_before=0,
                    rows_after=0,
                    errors=errors,
                )

            # Get existing data info
            existing_df = None

            if storage.exists(key):
                existing_df = storage.read(key).collect()
                rows_before = len(existing_df)

            # Apply strategy
            result = self.apply_strategy(storage, key, new_data, strategy)

            # Get final count
            final_df = storage.read(key).collect()
            rows_after = len(final_df)

            # Update metadata
            if tracker and result.success:
                new_start = _ensure_datetime(new_data["timestamp"].min())
                new_end = _ensure_datetime(new_data["timestamp"].max())
                final_start = _ensure_datetime(final_df["timestamp"].min())
                final_end = _ensure_datetime(final_df["timestamp"].max())
                update_record = UpdateRecord(
                    timestamp=datetime.now(),
                    update_type=strategy.value,
                    rows_before=rows_before,
                    rows_after=rows_after,
                    rows_added=result.rows_added,
                    rows_updated=result.rows_updated,
                    start_date=new_start,
                    end_date=new_end,
                    provider=provider,
                    duration_seconds=time.time() - start_time,
                    gaps_filled=result.gaps_filled,
                    errors=errors,
                )

                tracker.update_metadata(
                    key,
                    update_record,
                    rows_after,
                    final_start,
                    final_end,
                )

            result.duration_seconds = time.time() - start_time
            result.rows_before = rows_before
            result.rows_after = rows_after

            logger.info(
                f"Update completed for {key}",
                strategy=strategy.value,
                rows_added=result.rows_added,
                rows_updated=result.rows_updated,
                duration=result.duration_seconds,
            )

            return result

        except Exception as e:
            error_msg = f"Update failed: {e!s}"
            errors.append(error_msg)
            logger.error(error_msg, key=key, strategy=strategy.value)

            return UpdateResult(
                success=False,
                update_type=strategy.value,
                rows_added=0,
                rows_updated=0,
                rows_before=rows_before,
                rows_after=rows_before,
                duration_seconds=time.time() - start_time,
                errors=errors,
            )

    def apply_strategy(
        self,
        storage: StorageBackend,
        key: str,
        new_data: pl.DataFrame,
        strategy: UpdateStrategy,
    ) -> UpdateResult:
        """
        Apply specific update strategy.

        Args:
            storage: Storage backend
            key: Dataset key
            new_data: New data to process
            strategy: Update strategy to apply

        Returns:
            UpdateResult with operation details
        """
        if strategy == UpdateStrategy.FULL_REFRESH:
            # Replace all data
            _rewrite_preserving_metadata(storage, key, new_data)

            return UpdateResult(
                success=True,
                update_type=strategy.value,
                rows_added=len(new_data),
                rows_updated=0,
                rows_before=0,
                rows_after=len(new_data),
            )

        if strategy == UpdateStrategy.APPEND_ONLY:
            # Only add new data, never update existing
            if storage.exists(key):
                existing_df = storage.read(key).collect()
                existing_df, new_data = align_frames_for_concat(existing_df, new_data)
                max_existing = existing_df["timestamp"].max()

                # Filter new data to only include timestamps after existing
                new_rows = new_data.filter(pl.col("timestamp") > max_existing)

                if not new_rows.is_empty():
                    # Append new rows
                    combined = pl.concat([existing_df, new_rows])
                    _rewrite_preserving_metadata(storage, key, combined)

                    return UpdateResult(
                        success=True,
                        update_type=strategy.value,
                        rows_added=len(new_rows),
                        rows_updated=0,
                        rows_before=len(existing_df),
                        rows_after=len(combined),
                    )
                return UpdateResult(
                    success=True,
                    update_type=strategy.value,
                    rows_added=0,
                    rows_updated=0,
                    rows_before=len(existing_df),
                    rows_after=len(existing_df),
                )
            storage.write(new_data, key)
            return UpdateResult(
                success=True,
                update_type=strategy.value,
                rows_added=len(new_data),
                rows_updated=0,
                rows_before=0,
                rows_after=len(new_data),
            )

        if strategy == UpdateStrategy.BACKFILL:
            # Fill gaps in existing data
            if storage.exists(key):
                existing_df = storage.read(key).collect()
                existing_df, new_data = align_frames_for_concat(existing_df, new_data)

                # Detect gaps
                feed_metadata = normalize_storage_metadata(storage.get_metadata(key), key)
                gaps = self.gap_detector.detect_gaps(existing_df, feed_spec=feed_metadata)
                feed, selected_calendar = self.gap_detector._feed_calendar(
                    UNSET_CALENDAR, feed_metadata
                )
                parts = [
                    self.gap_detector._select_gap_rows(new_data, gap, feed, selected_calendar)
                    for gap in gaps
                    if selected_calendar is not None
                ]
                gap_data = pl.concat(parts) if parts else new_data.head(0)

                if not gap_data.is_empty():
                    combined, added, replaced = _merge_backfill_rows(existing_df, gap_data, feed)
                    if added or replaced:
                        _rewrite_preserving_metadata(storage, key, combined)
                    remaining = self.gap_detector.detect_gaps(combined, feed_spec=feed_metadata)

                    return UpdateResult(
                        success=True,
                        update_type=strategy.value,
                        rows_added=added,
                        rows_updated=replaced,
                        rows_before=len(existing_df),
                        rows_after=len(combined),
                        gaps_filled=sum(
                            not any(
                                missing.get("entity") == gap.get("entity")
                                and missing["start"] <= gap["end"]
                                and missing["end"] >= gap["start"]
                                for missing in remaining
                            )
                            for gap in gaps
                        ),
                        remaining_gaps=remaining,
                    )
                return UpdateResult(
                    success=True,
                    update_type=strategy.value,
                    rows_added=0,
                    rows_updated=0,
                    rows_before=len(existing_df),
                    rows_after=len(existing_df),
                    gaps_filled=0,
                    remaining_gaps=gaps,
                )
            storage.write(new_data, key)
            return UpdateResult(
                success=True,
                update_type=strategy.value,
                rows_added=len(new_data),
                rows_updated=0,
                rows_before=0,
                rows_after=len(new_data),
            )

        # INCREMENTAL - default
        # Update existing and add new
        if storage.exists(key):
            existing_df = storage.read(key).collect()
            existing_df, new_data = align_frames_for_concat(existing_df, new_data)

            # Separate overlapping and new data
            min_new = _ensure_datetime(new_data["timestamp"].min())
            max_existing = _ensure_datetime(existing_df["timestamp"].max())

            if min_new <= max_existing:
                # There's overlap
                overlap_data = new_data.filter(pl.col("timestamp") <= max_existing)
                new_only_data = new_data.filter(pl.col("timestamp") > max_existing)

                # Remove overlapping timestamps from existing
                overlap_timestamps = overlap_data["timestamp"].to_list()
                existing_filtered = existing_df.filter(
                    ~pl.col("timestamp").is_in(overlap_timestamps)
                )

                # Combine all data
                combined = pl.concat([existing_filtered, new_data]).sort("timestamp")

                _rewrite_preserving_metadata(storage, key, combined)

                return UpdateResult(
                    success=True,
                    update_type=strategy.value,
                    rows_added=len(new_only_data),
                    rows_updated=len(overlap_data),
                    rows_before=len(existing_df),
                    rows_after=len(combined),
                )
            # No overlap, just append
            combined = pl.concat([existing_df, new_data]).sort("timestamp")
            _rewrite_preserving_metadata(storage, key, combined)

            return UpdateResult(
                success=True,
                update_type=strategy.value,
                rows_added=len(new_data),
                rows_updated=0,
                rows_before=len(existing_df),
                rows_after=len(combined),
            )
        storage.write(new_data, key)
        return UpdateResult(
            success=True,
            update_type=strategy.value,
            rows_added=len(new_data),
            rows_updated=0,
            rows_before=0,
            rows_after=len(new_data),
        )


class BackfillManager:
    """Manages backfill operations for historical data gaps."""

    def __init__(
        self,
        storage: StorageBackend | None,
        tracker: MetadataTracker | None,
    ) -> None:
        """
        Initialize backfill manager.

        Args:
            storage: Storage backend
            tracker: Metadata tracker
        """
        self.storage = storage
        self.tracker = tracker
        self.gap_detector = GapDetector(exclude_weekends=True)

    def identify_candidates(
        self,
        min_gap_days: int = 5,
        max_age_days: int = 30,
        *,
        min_gap_bars: int = 1,
    ) -> list[dict[str, Any]]:
        """
        Identify datasets that need backfilling.

        Args:
            min_gap_days: Minimum gap size to consider for backfill
            max_age_days: Maximum age of data to consider
            min_gap_bars: Minimum expected-bar count for intraday datasets, not elapsed days.

        Returns:
            List of backfill candidates with gap information
        """
        if not self.storage:
            return []

        candidates = []

        for key in self.storage.list_keys():
            try:
                # Get data range
                df = self.storage.read(key).collect()
                if df.is_empty():
                    continue

                # Check data age
                last_update = _ensure_datetime(df["timestamp"].max())
                age_days = (datetime.now(UTC) - last_update).days

                if age_days > max_age_days:
                    continue

                # Detect gaps
                feed_metadata = normalize_storage_metadata(self.storage.get_metadata(key), key)
                gaps = self.gap_detector.detect_gaps(df, feed_spec=feed_metadata)

                # Filter significant gaps
                significant_gaps = [
                    gap
                    for gap in gaps
                    if gap.get("size_bars", gap.get("size_days", 0))
                    >= (min_gap_bars if "size_bars" in gap else min_gap_days)
                ]

                if significant_gaps:
                    intraday = "size_bars" in significant_gaps[0]
                    size_key = "size_bars" if intraday else "size_days"
                    total_key = "total_gap_bars" if intraday else "total_gap_days"

                    candidates.append(
                        {
                            "symbol": key,
                            "gaps": significant_gaps,
                            total_key: sum(g[size_key] for g in significant_gaps),
                            "last_update": last_update,
                            "importance": "normal",  # Could be enhanced with business logic
                        }
                    )

            except Exception as e:
                logger.warning(f"Error checking {key} for backfill: {e}")

        return candidates

    def prioritize_tasks(
        self,
        candidates: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """
        Prioritize backfill tasks based on importance and gap size.

        Args:
            candidates: List of backfill candidates

        Returns:
            Prioritized list of backfill tasks
        """
        # Score each candidate
        for candidate in candidates:
            score = 0

            # Factor 1: Gap size (larger gaps = higher priority)
            score += candidate.get("total_gap_days", candidate.get("total_gap_bars", 0)) * 2

            # Factor 2: Data staleness (older last update = higher priority)
            if "last_update" in candidate:
                last_update = _ensure_datetime(candidate["last_update"])
                days_old = (datetime.now(UTC) - last_update).days
                score += days_old

            # Factor 3: Importance
            importance_scores = {
                "critical": 100,
                "high": 50,
                "normal": 10,
                "low": 1,
            }
            score += importance_scores.get(candidate.get("importance", "normal"), 10)

            candidate["priority_score"] = score

        # Sort by priority score (highest first)
        return sorted(candidates, key=lambda x: x["priority_score"], reverse=True)

    def execute_backfill(
        self,
        symbol: str,
        gaps: list[dict[str, Any]],
        provider: BaseProvider,
        *,
        source_symbol: str | None = None,
        calendar: str | None | CalendarUnset = UNSET_CALENDAR,
        feed_spec: Any | None = None,
    ) -> UpdateResult:
        """Download real observations for selected missing ranges in one stored series.

        The symbol argument is the storage key. Supply source_symbol separately for a
        path-like key or a vendor alias. Intraday recovery requires the provider's
        optional timestamp-range capability and a matching left-labeled feed grid.
        Success describes processing, not completeness; inspect remaining_gaps.
        """
        started = time.time()
        errors: list[str] = []
        rows_before = rows_after = added = replaced = filled = 0
        remaining = None
        try:
            if self.storage is None:
                raise ValueError("Backfill requires a storage backend")
            existing = (
                self.storage.read(symbol).collect()
                if self.storage.exists(symbol)
                else pl.DataFrame(schema={"timestamp": pl.Datetime("us", "UTC")})
            )
            rows_before = rows_after = len(existing)
            if not gaps:
                return UpdateResult(
                    True, "backfill", 0, 0, rows_before, rows_after, remaining_gaps=[]
                )
            metadata = (
                normalize_storage_metadata(self.storage.get_metadata(symbol), symbol) or {}
                if self.storage.exists(symbol)
                else {}
            )
            feed, selected = self.gap_detector._feed_calendar(
                calendar, metadata if feed_spec is None else feed_spec
            )
            if selected is None:
                raise ValueError("calendar=None disables calendar-based recovery")
            frequency = feed.data_frequency or "daily"
            interval = ObservationGapDetector._intraday_interval(frequency)
            if interval is not None:
                if not isinstance(provider, TimestampRangeProvider):
                    raise NotImplementedError(
                        f"{provider.name} does not support exact timestamp-range downloads"
                    )
                if feed.bar_label != "left":
                    raise NotImplementedError("Targeted OHLCV recovery requires left-labeled bars")
            stored_provider = metadata.get("provider")
            if stored_provider and str(stored_provider).lower() not in {
                provider.name.lower(),
                provider.__class__.__name__.lower(),
            }:
                raise ValueError("Selected provider does not match the stored dataset provider")
            source_symbol = source_symbol or metadata.get("source_symbol")
            if source_symbol is None:
                if "/" in symbol:
                    raise ValueError("source_symbol is required separately from the storage key")
                source_symbol = symbol
            if not isinstance(source_symbol, str) or not source_symbol.strip():
                raise ValueError("source_symbol must identify one vendor instrument")
            entity = None
            entity_value = None
            if feed.entity_col is not None or "symbol" in existing.columns:
                entity = feed.resolve(existing.columns, ("symbol",)).entity_col
                if not isinstance(entity, str):
                    raise ValueError("Targeted recovery requires one entity column")
                source_rows = ObservationGapDetector._observed_rows(existing)
                identifiers = existing[entity].drop_nulls().unique()
                if source_rows[entity].null_count() or len(identifiers) > 1:
                    raise ValueError("Targeted recovery requires one instrument per storage key")
                entity_value = identifiers[0] if len(identifiers) else source_symbol.upper()

            def inspect(ranges: list[dict[str, Any]]) -> list[dict[str, Any]]:
                report: list[dict[str, Any]] = []
                for gap in ranges:
                    for missing in self.gap_detector.detect_gaps_in_storage(
                        self.storage,
                        symbol,
                        gap["start"],
                        gap["end"],
                        frequency,
                        calendar=selected,
                        feed_spec=feed,
                    ):
                        if missing not in report:
                            report.append(missing)
                return report

            fresh = inspect(gaps)
            remaining = fresh
            parts: list[pl.DataFrame] = []
            for gap in fresh:
                try:
                    if interval is not None and isinstance(provider, TimestampRangeProvider):
                        data = provider.fetch_ohlcv_range(
                            source_symbol, gap["start"], gap["end"] + interval, frequency
                        )
                    else:
                        data = provider.fetch_ohlcv(
                            source_symbol,
                            gap["start"].date().isoformat(),
                            gap["end"].date().isoformat(),
                            frequency,
                        )
                    if not isinstance(data, pl.DataFrame):
                        raise TypeError("Provider must return a DataFrame")
                    if data.is_empty():
                        continue
                    if "symbol" in data.columns:
                        identities = data["symbol"].unique().to_list()
                        if identities != [source_symbol.upper()]:
                            raise ValueError("Provider returned a different source instrument")
                        if "symbol" in existing.columns and not existing.is_empty():
                            symbols = existing["symbol"].drop_nulls().unique()
                            target_symbol = (
                                symbols[0] if len(symbols) == 1 else source_symbol.upper()
                            )
                            data = data.with_columns(pl.lit(target_symbol).alias("symbol"))
                    if entity is not None and not existing.is_empty():
                        data = data.with_columns(pl.lit(entity_value).alias(entity))
                    recovered = self.gap_detector._select_gap_rows(data, gap, feed, selected)
                    if not recovered.is_empty():
                        parts.append(recovered)
                except Exception as error:
                    errors.append(f"Failed to fetch gap {gap['start']} - {gap['end']}: {error}")

            if parts:
                recovered = pl.concat(parts, how="diagonal_relaxed")
                combined, proposed_added, proposed_replaced = _merge_backfill_rows(
                    existing, recovered, feed
                )
                if proposed_added or proposed_replaced:
                    remaining = None
                    _rewrite_preserving_metadata(self.storage, symbol, combined)
                    added, replaced = proposed_added, proposed_replaced
                    rows_after = len(combined)
                    if self.tracker is not None:
                        try:
                            self.tracker.update_metadata(
                                symbol,
                                UpdateRecord(
                                    timestamp=datetime.now(),
                                    update_type="backfill",
                                    rows_before=rows_before,
                                    rows_after=rows_after,
                                    rows_added=added,
                                    rows_updated=replaced,
                                    start_date=_ensure_datetime(recovered["timestamp"].min()),
                                    end_date=_ensure_datetime(recovered["timestamp"].max()),
                                    provider=provider.name,
                                    duration_seconds=time.time() - started,
                                    errors=errors,
                                ),
                                rows_after,
                                _ensure_datetime(combined["timestamp"].min()),
                                _ensure_datetime(combined["timestamp"].max()),
                            )
                        except Exception as error:
                            errors.append(f"Observations stored but update history failed: {error}")
            remaining = inspect(gaps)
            filled = sum(not inspect([gap]) for gap in fresh)
        except Exception as error:
            errors.append(f"Backfill failed for {symbol}: {error}")
        return UpdateResult(
            success=not errors,
            update_type="backfill",
            rows_added=added,
            rows_updated=replaced,
            rows_before=rows_before,
            rows_after=rows_after,
            gaps_filled=filled,
            duration_seconds=time.time() - started,
            errors=errors,
            remaining_gaps=remaining,
        )
