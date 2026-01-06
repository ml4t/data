"""Incremental update system with gap detection and resume capability."""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from enum import Enum
from typing import Any

import polars as pl
import structlog

from ml4t.data.providers.base import BaseProvider
from ml4t.data.storage.hive import HiveStorage
from ml4t.data.storage.metadata_tracker import MetadataTracker, UpdateRecord

logger = structlog.get_logger()


class UpdateStrategy(Enum):
    """Update strategy types."""

    INCREMENTAL = "incremental"  # Only fetch new data
    APPEND_ONLY = "append_only"  # Never update existing data
    FULL_REFRESH = "full_refresh"  # Replace all data
    BACKFILL = "backfill"  # Fill gaps in historical data


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

    def __post_init__(self) -> None:
        """Initialize errors list if None."""
        if self.errors is None:
            self.errors = []


class GapDetector:
    """Detect gaps in time-series data."""

    def __init__(self, exclude_weekends: bool = False) -> None:
        """
        Initialize gap detector.

        Args:
            exclude_weekends: Whether to exclude weekends from gap detection
        """
        self.exclude_weekends = exclude_weekends

    def _ensure_datetime(self, dt: date | datetime) -> datetime:
        """Ensure we have a datetime object."""
        if isinstance(dt, date) and not isinstance(dt, datetime):
            return datetime.combine(dt, datetime.min.time())
        return dt

    def detect_gaps(
        self,
        df: pl.DataFrame,
        frequency: str = "daily",
        tolerance_days: int = 0,
    ) -> list[dict[str, Any]]:
        """
        Detect gaps in a DataFrame's timestamp column.

        Args:
            df: DataFrame with timestamp column
            frequency: Data frequency (daily, hourly, etc.)
            tolerance_days: Number of days to tolerate as non-gap

        Returns:
            List of gap dictionaries with start, end, and size
        """
        if df.is_empty() or "timestamp" not in df.columns:
            return []

        # Sort by timestamp
        df = df.sort("timestamp")
        # Ensure we have datetime objects, not date objects
        timestamps = [self._ensure_datetime(ts) for ts in df["timestamp"].to_list()]

        gaps = []
        expected_delta = self._get_expected_delta(frequency)

        for i in range(len(timestamps) - 1):
            current = timestamps[i]
            next_ts = timestamps[i + 1]
            expected = current + expected_delta

            # Check if there's a gap
            if self.exclude_weekends and frequency == "daily":
                # Skip weekend days
                while expected.weekday() >= 5:  # Saturday = 5, Sunday = 6
                    expected += timedelta(days=1)

            if next_ts > expected + timedelta(days=tolerance_days):
                # Found a gap
                gap_start = current + expected_delta
                gap_end = next_ts - expected_delta

                # Adjust for weekends if needed
                if self.exclude_weekends and frequency == "daily":
                    # Skip weekend at start
                    while gap_start.weekday() >= 5:
                        gap_start += timedelta(days=1)
                    # Skip weekend at end
                    while gap_end.weekday() >= 5:
                        gap_end -= timedelta(days=1)

                    # Only add gap if it's still valid after weekend adjustment
                    if gap_start <= gap_end:
                        gaps.append(
                            {
                                "start": gap_start,
                                "end": gap_end,
                                "size_days": (gap_end - gap_start).days + 1,
                            }
                        )
                else:
                    gaps.append(
                        {
                            "start": gap_start,
                            "end": gap_end,
                            "size_days": (gap_end - gap_start).days + 1,
                        }
                    )

        return gaps

    def detect_gaps_in_storage(
        self,
        storage: HiveStorage,
        key: str,
        start_date: datetime,
        end_date: datetime,
        frequency: str = "daily",
    ) -> list[dict[str, Any]]:
        """
        Detect gaps directly in Hive storage structure.

        Args:
            storage: HiveStorage instance
            key: Storage key for the dataset
            start_date: Start of range to check
            end_date: End of range to check
            frequency: Data frequency

        Returns:
            List of gaps found in storage
        """
        try:
            # Read data from storage
            df = storage.read(key, start_date, end_date).collect()

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
            first_timestamp = self._ensure_datetime(df["timestamp"].min())
            start_date = self._ensure_datetime(start_date)
            end_date = self._ensure_datetime(end_date)

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
            last_timestamp = self._ensure_datetime(df["timestamp"].max())
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
        # Default to daily
        return timedelta(days=1)


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

    def _ensure_datetime(self, dt: date | datetime) -> datetime:
        """Ensure we have a datetime object."""
        if isinstance(dt, date) and not isinstance(dt, datetime):
            return datetime.combine(dt, datetime.min.time())
        return dt

    def determine_update_range(
        self,
        storage: HiveStorage,
        key: str,
        requested_start: datetime,
        requested_end: datetime,
    ) -> tuple[datetime, datetime, str]:
        """
        Determine the actual range to update based on existing data.

        Args:
            storage: Storage backend
            key: Dataset key
            requested_start: Requested start date
            requested_end: Requested end date

        Returns:
            Tuple of (actual_start, actual_end, update_type)
        """
        try:
            # Check if data exists
            if not storage.exists(key):
                return requested_start, requested_end, "full"

            # Get existing data range
            df = storage.read(key).select("timestamp").collect()

            if df.is_empty():
                return requested_start, requested_end, "full"

            last_timestamp = self._ensure_datetime(df["timestamp"].max())
            requested_end = self._ensure_datetime(requested_end)
            requested_start = self._ensure_datetime(requested_start)

            # If requested end is before last data, no update needed
            if requested_end <= last_timestamp:
                logger.info(
                    f"Data for {key} already up to date",
                    last_timestamp=last_timestamp,
                    requested_end=requested_end,
                )
                return requested_end, requested_end, "none"

            # Incremental update from day after last data
            actual_start = last_timestamp + timedelta(days=1)

            logger.info(
                f"Incremental update for {key}",
                last_data=last_timestamp,
                update_from=actual_start,
                update_to=requested_end,
            )

            return actual_start, requested_end, "incremental"

        except Exception as e:
            logger.warning(f"Error checking existing data: {e}")
            return requested_start, requested_end, "full"

    def update_incremental(
        self,
        storage: HiveStorage,
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
            rows_before = 0
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
                update_record = UpdateRecord(
                    timestamp=datetime.now(),
                    update_type=strategy.value,
                    rows_before=rows_before,
                    rows_after=rows_after,
                    rows_added=result.rows_added,
                    rows_updated=result.rows_updated,
                    start_date=new_data["timestamp"].min(),
                    end_date=new_data["timestamp"].max(),
                    provider=provider,
                    duration_seconds=time.time() - start_time,
                    gaps_filled=result.gaps_filled,
                    errors=errors,
                )

                tracker.update_metadata(
                    key,
                    update_record,
                    rows_after,
                    final_df["timestamp"].min(),
                    final_df["timestamp"].max(),
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
                rows_before=rows_before if "rows_before" in locals() else 0,
                rows_after=rows_before if "rows_before" in locals() else 0,
                duration_seconds=time.time() - start_time,
                errors=errors,
            )

    def apply_strategy(
        self,
        storage: HiveStorage,
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
            storage.delete(key)
            storage.write(new_data, key)

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
                max_existing = existing_df["timestamp"].max()

                # Filter new data to only include timestamps after existing
                new_rows = new_data.filter(pl.col("timestamp") > max_existing)

                if not new_rows.is_empty():
                    # Append new rows
                    combined = pl.concat([existing_df, new_rows])
                    storage.delete(key)
                    storage.write(combined, key)

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

                # Detect gaps
                gaps = self.gap_detector.detect_gaps(existing_df)
                gaps_filled = 0

                # Filter new data to only gap periods
                gap_data = pl.DataFrame()
                for gap in gaps:
                    gap_rows = new_data.filter(
                        (pl.col("timestamp") >= gap["start"]) & (pl.col("timestamp") <= gap["end"])
                    )
                    if not gap_rows.is_empty():
                        gap_data = (
                            pl.concat([gap_data, gap_rows]) if not gap_data.is_empty() else gap_rows
                        )
                        gaps_filled += 1

                if not gap_data.is_empty():
                    # Merge with existing data
                    combined = pl.concat([existing_df, gap_data]).sort("timestamp")
                    storage.delete(key)
                    storage.write(combined, key)

                    return UpdateResult(
                        success=True,
                        update_type=strategy.value,
                        rows_added=len(gap_data),
                        rows_updated=0,
                        rows_before=len(existing_df),
                        rows_after=len(combined),
                        gaps_filled=gaps_filled,
                    )
                return UpdateResult(
                    success=True,
                    update_type=strategy.value,
                    rows_added=0,
                    rows_updated=0,
                    rows_before=len(existing_df),
                    rows_after=len(existing_df),
                    gaps_filled=0,
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

            # Separate overlapping and new data
            min_new = new_data["timestamp"].min()
            max_existing = existing_df["timestamp"].max()

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

                storage.delete(key)
                storage.write(combined, key)

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
            storage.delete(key)
            storage.write(combined, key)

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
        storage: HiveStorage | None,
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

    def _ensure_datetime(self, dt: date | datetime) -> datetime:
        """Ensure we have a datetime object."""
        if isinstance(dt, date) and not isinstance(dt, datetime):
            return datetime.combine(dt, datetime.min.time())
        return dt

    def identify_candidates(
        self,
        min_gap_days: int = 5,
        max_age_days: int = 30,
    ) -> list[dict[str, Any]]:
        """
        Identify datasets that need backfilling.

        Args:
            min_gap_days: Minimum gap size to consider for backfill
            max_age_days: Maximum age of data to consider

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
                last_update = self._ensure_datetime(df["timestamp"].max())
                age_days = (datetime.now() - last_update).days

                if age_days > max_age_days:
                    continue

                # Detect gaps
                gaps = self.gap_detector.detect_gaps(df)

                # Filter significant gaps
                significant_gaps = [g for g in gaps if g["size_days"] >= min_gap_days]

                if significant_gaps:
                    total_gap_days = sum(g["size_days"] for g in significant_gaps)

                    candidates.append(
                        {
                            "symbol": key,
                            "gaps": significant_gaps,
                            "total_gap_days": total_gap_days,
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
            score += candidate["total_gap_days"] * 2

            # Factor 2: Data staleness (older last update = higher priority)
            if "last_update" in candidate:
                days_old = (datetime.now() - candidate["last_update"]).days
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
    ) -> UpdateResult:
        """
        Execute backfill operation for identified gaps.

        Args:
            symbol: Symbol to backfill
            gaps: List of gaps to fill
            provider: Data provider to fetch from

        Returns:
            UpdateResult with backfill details
        """
        total_rows_added = 0
        gaps_filled = 0
        errors = []
        start_time = time.time()

        try:
            # Fetch data for each gap
            all_gap_data = pl.DataFrame()

            for gap in gaps:
                try:
                    logger.info(
                        f"Backfilling gap for {symbol}",
                        start=gap["start"],
                        end=gap["end"],
                        size_days=gap["size_days"],
                    )

                    # Fetch data from provider
                    gap_data = provider.fetch_ohlcv(
                        symbol,
                        gap["start"].strftime("%Y-%m-%d"),
                        gap["end"].strftime("%Y-%m-%d"),
                        "daily",
                    )

                    if not gap_data.is_empty():
                        all_gap_data = (
                            pl.concat([all_gap_data, gap_data])
                            if not all_gap_data.is_empty()
                            else gap_data
                        )
                        gaps_filled += 1
                        total_rows_added += len(gap_data)

                except Exception as e:
                    error_msg = f"Failed to fetch gap {gap['start']} - {gap['end']}: {e}"
                    errors.append(error_msg)
                    logger.warning(error_msg)

            # Apply backfill update if we got any data
            if not all_gap_data.is_empty() and self.storage and self.tracker:
                updater = IncrementalUpdater(strategy=UpdateStrategy.BACKFILL)
                result = updater.update_incremental(
                    self.storage,
                    self.tracker,
                    symbol,
                    all_gap_data,
                    provider=provider.__class__.__name__,
                    strategy=UpdateStrategy.BACKFILL,
                )

                result.gaps_filled = gaps_filled
                return result

            return UpdateResult(
                success=gaps_filled > 0,
                update_type="backfill",
                rows_added=total_rows_added,
                rows_updated=0,
                rows_before=0,
                rows_after=0,
                gaps_filled=gaps_filled,
                duration_seconds=time.time() - start_time,
                errors=errors if errors else None,
            )

        except Exception as e:
            error_msg = f"Backfill failed for {symbol}: {e}"
            errors.append(error_msg)
            logger.error(error_msg)

            return UpdateResult(
                success=False,
                update_type="backfill",
                rows_added=0,
                rows_updated=0,
                rows_before=0,
                rows_after=0,
                duration_seconds=time.time() - start_time,
                errors=errors,
            )
