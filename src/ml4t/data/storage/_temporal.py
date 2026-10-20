"""Timestamp read semantics shared by the local storage layouts."""

from datetime import UTC, datetime
from typing import Any
from zoneinfo import ZoneInfo

import polars as pl

from ml4t.specs import FeedSpec

from .backend import normalize_storage_metadata


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
    feed = FeedSpec.from_any(normalize_storage_metadata(metadata))
    if not isinstance(dtype, pl.Datetime):
        return value.replace(tzinfo=None)
    source_zone = ZoneInfo(dtype.time_zone) if dtype.time_zone else UTC
    if feed.timestamp_semantics == "session_label":
        if dtype.time_zone is None and feed.timezone is not None:
            source_zone = ZoneInfo(feed.timezone)
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
    feed = FeedSpec.from_any(normalize_storage_metadata(metadata))
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
        naive_zone = (
            feed.timezone
            if feed.timestamp_semantics == "session_label" and feed.timezone is not None
            else "UTC"
        )
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
