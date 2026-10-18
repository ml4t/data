"""Expose absent minute positions without manufacturing market observations."""

from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Any

import polars as pl
import structlog

from ml4t.data.sessions.assigner import (
    SessionAssigner,
    _schedule_for_instants,
    _session_intervals,
)
from ml4t.specs import UNSET_CALENDAR, BarLabel, CalendarUnset, TimestampSemantics

logger = structlog.get_logger()


def _datetime_scalar(value: object, name: str) -> datetime:
    if not isinstance(value, datetime):
        raise TypeError(f"'{name}' must contain non-null Datetime values")
    return value


class SessionCompleter:
    """Left join observations to scheduled minutes, retaining absent values as nulls.

    The legacy is_imputed marker identifies placeholder rows, not fabricated prices.
    Use the disk gap report instead when expansion into a nullable grid is unnecessary.
    """

    def __init__(
        self,
        calendar: str | None | CalendarUnset = UNSET_CALENDAR,
        *,
        calendar_name: str | None | CalendarUnset = UNSET_CALENDAR,
        exchange: str | None = None,
        feed_spec: Any | None = None,
    ) -> None:
        """Initialize with exchange calendar.

        Args:
            calendar: Explicit schedule; None disables completion.
            calendar_name: Compatibility alias.
            exchange: Compatibility selector.
            feed_spec: Shared temporal declaration.
        """
        assigner = SessionAssigner(
            calendar, calendar_name=calendar_name, exchange=exchange, feed_spec=feed_spec
        )
        self.calendar_name = assigner.calendar_name
        self.calendar = assigner.calendar
        self.feed_spec = assigner.feed_spec

    def complete_sessions(
        self,
        df: pl.DataFrame,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        fill_method: str = "none",
        zero_volume: bool = False,
    ) -> pl.DataFrame:
        """Represent missing scheduled minutes with nulls, preserving observed values.

        Args:
            df: Input DataFrame with timestamp, open, high, low, close, volume
            start_date: Optional start date (auto-detected if not provided)
            end_date: Optional exclusive end date (auto-detected if not provided)
            fill_method: Only "none" is supported; synthetic prices are rejected.
            zero_volume: Must be False; an absent observation has no known volume.

        Returns:
            DataFrame with complete sessions, an is_imputed column, and timestamp order

        Raises:
            ValueError: If required columns are missing, timestamps are duplicated or
                not minute-aligned, multiple symbols share a timestamp, or an observation
                falls outside a scheduled session or the requested half-open range
        """
        if self.calendar is None:
            return df
        if (
            self.feed_spec.bar_grid_anchor is not None
            or self.feed_spec.partial_bar_policy is not None
        ):
            raise NotImplementedError(
                "native minute completion does not support explicit general bar-grid policies"
            )
        if self.feed_spec.bar_type not in (None, "time"):
            raise NotImplementedError("minute completion requires fixed time bars")
        if self.feed_spec.data_frequency is not None and str(
            self.feed_spec.data_frequency
        ).lower() not in {"minute", "1minute", "1min", "1m"}:
            raise ValueError("minute completion requires a declared one-minute frequency")
        if self.feed_spec.timestamp_semantics not in (
            None,
            TimestampSemantics.EVENT_TIME,
        ) or self.feed_spec.bar_label not in (None, BarLabel.LEFT):
            raise NotImplementedError(
                "minute completion currently requires open-labeled event_time bars"
            )
        if "timestamp" not in df.columns:
            raise ValueError("DataFrame must have 'timestamp' column")
        if fill_method != "none" or zero_volume:
            raise ValueError(
                "Synthetic price/volume imputation is unsupported; "
                "use fill_method='none' and zero_volume=False"
            )

        if df.is_empty():
            logger.warning("DataFrame is empty, cannot complete sessions")
            return df

        timestamp_dtype = df.schema["timestamp"]
        if not isinstance(timestamp_dtype, pl.Datetime):
            raise TypeError("'timestamp' must contain Datetime values")

        # Auto-detect date range
        if start_date is None:
            start_date = _datetime_scalar(df["timestamp"].min(), "timestamp")
        if end_date is None:
            end_date = _datetime_scalar(df["timestamp"].max(), "timestamp") + timedelta(minutes=1)

        logger.info(
            f"Completing sessions for {len(df)} rows",
            start_date=str(start_date),
            end_date=str(end_date),
            fill_method=fill_method,
        )

        try:
            import pandas as pd

            request_start = pd.Timestamp(start_date)
            request_end = pd.Timestamp(end_date)
            if not isinstance(request_start, pd.Timestamp) or not isinstance(
                request_end, pd.Timestamp
            ):
                raise ValueError("start_date and end_date must not be NaT")
            if request_start.tzinfo is None:
                request_start = request_start.tz_localize(
                    self.feed_spec.timezone or "UTC"
                ).tz_convert("UTC")
            else:
                request_start = request_start.tz_convert("UTC")
            if request_end.tzinfo is None:
                request_end = request_end.tz_localize(self.feed_spec.timezone or "UTC").tz_convert(
                    "UTC"
                )
            else:
                request_end = request_end.tz_convert("UTC")
            if request_start >= request_end:
                raise ValueError("start_date must be earlier than the exclusive end_date")
            if any(
                value.second or value.microsecond or value.nanosecond
                for value in (request_start, request_end)
            ):
                raise ValueError("start_date and end_date must be minute-aligned")

            schedule = _schedule_for_instants(self.calendar, request_start, request_end)

            if len(schedule) == 0:
                logger.warning("No trading sessions found in date range")
                return df

            logger.debug(f"Got {len(schedule)} sessions from calendar")

            # Generate complete minute timestamps for all sessions
            all_minutes: list[pd.Timestamp] = []
            session_dates: list[date] = []

            for session_date, row in schedule.iterrows():
                if not isinstance(session_date, pd.Timestamp):
                    raise TypeError("calendar returned a non-datetime session label")
                for interval_start, interval_end in _session_intervals(row):
                    clipped_start = max(interval_start, request_start)
                    clipped_end = min(interval_end, request_end)
                    if clipped_start >= clipped_end:
                        continue
                    minutes = pd.date_range(
                        start=clipped_start,
                        end=clipped_end,
                        freq="1min",
                        inclusive="left",
                    )
                    all_minutes.extend(minutes)
                    session_dates.extend([session_date.date()] * len(minutes))

            # Create complete minute template
            minute_template = pl.DataFrame(
                {
                    "timestamp": pl.Series(
                        "timestamp",
                        [minute.to_pydatetime() for minute in all_minutes],
                        dtype=pl.Datetime("ns", "UTC"),
                    ),
                    "session_date": pl.Series("session_date", session_dates, dtype=pl.Date),
                }
            )

            # Ensure input data has matching timestamp type
            timestamp_expr = pl.col("timestamp")
            if timestamp_dtype.time_zone is None:
                timestamp_expr = timestamp_expr.dt.replace_time_zone(
                    self.feed_spec.timezone or "UTC"
                )
            timestamp_expr = timestamp_expr.dt.convert_time_zone("UTC").cast(
                pl.Datetime("ns", "UTC")
            )
            df_with_tz = df.with_columns(
                timestamp_expr.alias("timestamp"), pl.lit(True).alias("_is_observed")
            )
            if df_with_tz.get_column("timestamp").n_unique() != df_with_tz.height:
                duplicates = (
                    df_with_tz.group_by("timestamp")
                    .len()
                    .filter(pl.col("len") > 1)
                    .get_column("timestamp")
                    .head(5)
                    .to_list()
                )
                raise ValueError(
                    "Session completion requires one symbol and unique input timestamps; "
                    f"duplicates: {', '.join(str(value) for value in duplicates)}"
                )

            unmatched = df_with_tz.join(
                minute_template.select("timestamp"), on="timestamp", how="anti"
            )
            if not unmatched.is_empty():
                timestamps = ", ".join(
                    str(value) for value in unmatched.get_column("timestamp").head(5).to_list()
                )
                raise ValueError(
                    f"Input observations fall outside the completion template: {timestamps}"
                )

            # Left join: keep all minutes from template
            complete_df = minute_template.join(df_with_tz, on="timestamp", how="left")

            # Drop duplicate session_date column if exists
            if "session_date_right" in complete_df.columns:
                complete_df = complete_df.drop("session_date_right")
            complete_df = complete_df.with_columns(
                pl.col("_is_observed").is_null().alias("is_imputed")
            )

            complete_df = complete_df.drop("_is_observed")
            rows_added = complete_df.get_column("is_imputed").sum()
            logger.info(f"Completed sessions: added {rows_added} rows ({len(complete_df)} total)")

            return complete_df.sort("timestamp")

        except Exception as e:
            logger.error(f"Failed to complete sessions: {e}", exc_info=True)
            raise

    def get_session_info(
        self, date_or_datetime: datetime | date
    ) -> dict[str, datetime | date | None]:
        """Get session start/end times for a given date.

        Args:
            date_or_datetime: Date to get session info for

        Returns:
            Dict with 'session_date', 'market_open', 'market_close' keys
        """
        import pandas as pd

        pd_date = pd.Timestamp(date_or_datetime)
        schedule = self.calendar.schedule(start_date=pd_date, end_date=pd_date)

        if len(schedule) == 0:
            return {
                "session_date": None,
                "market_open": None,
                "market_close": None,
            }

        row = schedule.iloc[0]
        return {
            "session_date": schedule.index[0].to_pydatetime().date(),
            "market_open": row["market_open"].to_pydatetime(),
            "market_close": row["market_close"].to_pydatetime(),
        }
