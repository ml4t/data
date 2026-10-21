"""Availability-based revision selection for native input tables."""

from collections.abc import Sequence
from datetime import date, datetime

import pandas as pd
import polars as pl

from ml4t.data.utils.conversion import pandas_to_polars
from ml4t.specs import FeedSpec


def native_frame(frame: pl.DataFrame | pd.DataFrame) -> pl.DataFrame:
    return pandas_to_polars(frame) if isinstance(frame, pd.DataFrame) else frame.clone()


def utc_stamp(value: datetime | str) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    if pd.isna(stamp):
        raise ValueError("An as_of cutoff must be a known instant")
    return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")


def utc_cutoff(value: datetime | str) -> pl.Expr:
    stamp = utc_stamp(value)
    return pl.lit(stamp.value).cast(pl.Datetime("ns", "UTC"))


def session_labels_at(points: pl.Series, feed: FeedSpec) -> pl.Series:
    from ml4t.data.sessions import SessionAssigner

    selected = feed.resolve_calendar()
    if selected is None:
        return points.dt.convert_time_zone(feed.timezone or "UTC").dt.date()
    point_feed = FeedSpec(
        time_contract_version=2,
        calendar=selected,
        timestamp_semantics="event_time"
        if feed.timestamp_semantics == "session_label"
        else feed.timestamp_semantics,
        bar_label=None if feed.timestamp_semantics == "session_label" else feed.bar_label,
    )
    assigner = SessionAssigner(feed_spec=point_feed)
    assigned = assigner.assign_sessions(
        pl.DataFrame({"timestamp": points}), bar_frequency="intraday"
    )["session_date"]
    timezone = feed.timezone or str(getattr(assigner.calendar, "tz", "UTC"))
    return assigned.fill_null(points.dt.convert_time_zone(timezone).dt.date())


def session_label_at(value: datetime | str, feed: FeedSpec) -> date:
    stamp = utc_stamp(value)
    points = pl.Series([stamp.value], dtype=pl.Datetime("ns", "UTC"))
    return session_labels_at(
        points, feed.with_overrides(timestamp_semantics="event_time", bar_label=None)
    )[0]


def utc_column(frame: pl.DataFrame, name: str, timezone: str | None = None) -> pl.DataFrame:
    if name not in frame.columns or frame.schema[name] == pl.Null:
        return frame.with_columns(pl.lit(None, dtype=pl.Datetime("ns", "UTC")).alias(name))
    dtype = frame.schema[name]
    if not isinstance(dtype, pl.Datetime):
        raise ValueError(f"{name} must contain Datetime instants")
    expression = pl.col(name)
    if dtype.time_zone is None:
        expression = expression.dt.replace_time_zone(timezone or "UTC")
    return frame.with_columns(expression.dt.convert_time_zone("UTC"))


def select_revisions(
    frame: pl.DataFrame,
    keys: Sequence[str],
    as_of: datetime | str | None,
) -> pl.DataFrame:
    required = {*keys, "revision_id"}
    if missing := required.difference(frame.columns):
        raise ValueError(f"Missing revision identity columns: {sorted(missing)}")
    if any(frame[name].null_count() for name in required):
        raise ValueError("Revision identity keys cannot be null")
    if any(
        (frame.schema[name] == pl.String and frame[name].str.strip_chars().eq("").any())
        or (frame.schema[name].is_float() and not frame[name].is_finite().all())
        for name in required
    ):
        raise ValueError("Revision identity keys must be nonempty and finite")
    if frame.select(*keys, "revision_id").is_duplicated().any():
        raise ValueError("Duplicate source revision identities")
    frame = utc_column(frame, "available_at")
    repeated = frame.select(keys).is_duplicated()
    if frame.filter(
        pl.col("available_at").is_null() & (repeated | pl.lit(as_of is not None))
    ).height:
        raise ValueError("Unknown available_at cannot establish causal revision selection")
    if as_of is not None:
        frame = frame.filter(pl.col("available_at") <= utc_cutoff(as_of))
    if frame.select(*keys, "available_at").is_duplicated().any():
        raise ValueError("Conflicting revisions have indistinguishable availability")
    return frame.sort("available_at", nulls_last=False).unique(
        subset=keys, keep="last", maintain_order=True
    )
