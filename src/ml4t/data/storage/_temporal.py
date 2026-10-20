"""Timestamp read semantics shared by the local storage layouts."""

from datetime import UTC, datetime
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import polars as pl

from ml4t.specs import FeedSpec

from .backend import normalize_storage_metadata


def storage_feed(metadata: dict[str, Any]) -> FeedSpec:
    """Validate the stored declaration without adding defaults to the source metadata."""
    feed = FeedSpec.from_any(normalize_storage_metadata(metadata))
    feed.resolve_calendar()
    if feed.timezone is not None:
        try:
            ZoneInfo(feed.timezone)
        except (TypeError, ValueError, ZoneInfoNotFoundError) as error:
            raise ValueError(f"Invalid declared timezone: {feed.timezone!r}") from error
    return feed


def read_bound(
    value: datetime | None,
    dtype: pl.DataType | None,
    metadata: dict[str, Any],
    *,
    partition: bool = False,
) -> datetime | None:
    """Interpret a bound in the observation domain, or its physical partition timezone."""
    if value is None:
        return None
    feed = storage_feed(metadata)
    if not isinstance(dtype, pl.Datetime):
        return value.replace(tzinfo=None)
    source_zone = ZoneInfo(dtype.time_zone or feed.timezone or "UTC")
    if feed.timestamp_semantics == "session_label":
        label_zone = ZoneInfo(feed.timezone) if feed.timezone else source_zone
        label = value.replace(tzinfo=label_zone) if value.tzinfo is None else value
        target_zone = source_zone if partition or feed.timezone is None else UTC
        return label.astimezone(target_zone)
    instant = value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)
    return instant.astimezone(source_zone) if partition else instant


def read_timestamps(
    frame: pl.LazyFrame,
    metadata: dict[str, Any],
    start: datetime | None,
    end: datetime | None,
) -> pl.LazyFrame:
    """Keep source precision and declared labels, retaining legacy UTC instant reads."""
    feed = storage_feed(metadata)
    timestamp_col = feed.timestamp_col
    dtype = frame.collect_schema().get(timestamp_col)
    if dtype is None:
        return frame
    if isinstance(dtype, pl.Datetime) and (
        feed.timestamp_semantics != "session_label"
        or feed.timezone is not None
        or dtype.time_zone is None
    ):
        timestamp = pl.col(timestamp_col)
        naive_zone = feed.timezone or "UTC"
        timestamp = (
            timestamp.dt.replace_time_zone(naive_zone).dt.convert_time_zone("UTC")
            if dtype.time_zone is None
            else timestamp.dt.convert_time_zone("UTC")
        )
        frame = frame.with_columns(timestamp)
    for bound, lower in ((start, True), (end, False)):
        normalized = read_bound(bound, dtype, metadata)
        if normalized is not None:
            value = normalized.date() if dtype == pl.Date else normalized
            predicate = pl.col(timestamp_col) >= value if lower else pl.col(timestamp_col) < value
            frame = frame.filter(predicate)
    return frame
