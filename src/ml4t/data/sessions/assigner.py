"""Session date assignment using exchange calendars."""

from __future__ import annotations

from datetime import date, datetime
from typing import Any, Literal

import polars as pl
import structlog

from ml4t.specs import UNSET_CALENDAR, BarLabel, CalendarUnset, FeedSpec, TimestampSemantics

logger = structlog.get_logger()


def _datetime_scalar(value: object, name: str) -> datetime:
    if not isinstance(value, datetime):
        raise TypeError(f"'{name}' must contain non-null Datetime values")
    return value


def _session_intervals(row: Any) -> list[tuple[datetime, datetime]]:
    import pandas as pd

    intervals = [(row["market_open"].to_pydatetime(), row["market_close"].to_pydatetime())]
    columns = [("break_start", "break_end")] if "break_start" in row.index else []
    columns.extend(
        (name, name.replace("interruption_start_", "interruption_end_"))
        for name in row.index
        if name.startswith("interruption_start_")
    )
    closures = []
    for start_column, end_column in columns:
        start, end = row[start_column], row[end_column]
        if pd.isna(start) and pd.isna(end):
            continue
        if pd.isna(start) or pd.isna(end) or start > end:
            raise ValueError("calendar returned invalid closure boundaries")
        if start != end:
            closures.append((start.to_pydatetime(), end.to_pydatetime()))
    for start, end in sorted(closures):
        remaining = []
        for opening, closing in intervals:
            if end <= opening or start >= closing:
                remaining.append((opening, closing))
            else:
                if opening < start:
                    remaining.append((opening, start))
                if end < closing:
                    remaining.append((end, closing))
        intervals = remaining
    return intervals


def _schedule_for_instants(calendar: Any, start: Any, end: Any, *, padding_days: int = 1) -> Any:
    import pandas as pd

    from ml4t.data.sessions._cme_crypto import CMEGlobexCryptoSchedule

    if isinstance(calendar, CMEGlobexCryptoSchedule):
        return calendar.schedule_for_instants(start, end)
    return calendar.schedule(
        start_date=pd.Timestamp(start).normalize() - pd.Timedelta(days=padding_days),
        end_date=pd.Timestamp(end).normalize() + pd.Timedelta(days=padding_days),
        interruptions=True,
    )


# Exchange to calendar mapping
EXCHANGE_CALENDARS = {
    "CME": "CME Globex Crypto",
    "NYSE": "NYSE",
    "NASDAQ": "NASDAQ",
    "LSE": "LSE",
    "TSE": "TSE",
    "HKEX": "HKEX",
    "ASX": "ASX",
    "SSE": "SSE",
    "TSX": "TSX",
}


def _uses_session_labels(df: pl.DataFrame, feed: FeedSpec, bar_frequency: str) -> bool:
    if feed.timestamp_semantics is not None:
        return feed.timestamp_semantics == TimestampSemantics.SESSION_LABEL
    if feed.bar_label is not None:
        return False
    if df.schema["timestamp"] == pl.Date:
        return True
    timestamps = df["timestamp"].drop_nulls().to_list()
    inferred_daily = bool(timestamps) and all(
        value.hour == value.minute == value.second == value.microsecond == 0
        for value in timestamps
        if isinstance(value, datetime)
    )
    return bar_frequency == "daily" or (bar_frequency == "auto" and inferred_daily)


def _daily_session_labels(
    df: pl.DataFrame, assigner: SessionAssigner, timestamp_col: str = "timestamp"
) -> pl.Series:
    if timestamp_col not in df.columns:
        raise ValueError(f"DataFrame must have '{timestamp_col}' column")
    feed = assigner.feed_spec
    if feed.data_frequency is not None and str(feed.data_frequency).lower() not in {
        "daily",
        "1day",
        "1d",
    }:
        raise ValueError("daily session projection requires a declared daily frequency")
    if df.is_empty():
        return pl.Series(timestamp_col, [], dtype=pl.Date)
    dtype = df.schema[timestamp_col]
    if dtype != pl.Date and not isinstance(dtype, pl.Datetime):
        raise TypeError(f"'{timestamp_col}' must contain Date or Datetime values")
    if df[timestamp_col].null_count():
        raise ValueError(f"'{timestamp_col}' must not contain null timestamps")
    if feed.timestamp_semantics in (
        TimestampSemantics.EVENT_TIME,
        TimestampSemantics.BAR_CLOSE,
    ) or (
        feed.timestamp_semantics != TimestampSemantics.SESSION_LABEL and feed.bar_label is not None
    ):
        assigned = assigner.assign_sessions(
            df.select(pl.col(timestamp_col).alias("timestamp")),
            outside_session="raise",
            bar_frequency="daily",
        )
        return assigned["session_date"].rename(timestamp_col)
    labels = df[timestamp_col]
    if isinstance(dtype, pl.Datetime):
        if dtype.time_zone is not None and feed.timezone is not None:
            labels = labels.dt.convert_time_zone(feed.timezone)
        labels = labels.dt.date()
    return labels


class SessionAssigner:
    """Assigns session dates to trading data based on exchange calendars.

    Uses pandas_market_calendars to determine trading sessions and assigns
    a session_date column to each timestamp.

    CME crypto uses business trade dates across overnight and weekend trading.
    """

    def __init__(
        self,
        calendar: str | None | CalendarUnset = UNSET_CALENDAR,
        *,
        calendar_name: str | None | CalendarUnset = UNSET_CALENDAR,
        exchange: str | None = None,
        feed_spec: Any | None = None,
    ) -> None:
        """Initialize session assigner.

        Args:
            calendar: Explicit name; None disables session handling.
            calendar_name: Compatibility alias for calendar.
            exchange: Compatibility selector, not inferred from descriptive metadata.
            feed_spec: Shared calendar, timezone and timestamp declarations.
        """
        selectors = [calendar_name]
        if exchange is not None:
            if exchange.upper() not in EXCHANGE_CALENDARS:
                raise ValueError(f"Unknown exchange '{exchange}'")
            selectors.append(EXCHANGE_CALENDARS[exchange.upper()])
        for alias in selectors:
            if alias is not UNSET_CALENDAR:
                if calendar is not UNSET_CALENDAR and calendar != alias:
                    raise ValueError("conflicting calendar selectors")
                calendar = alias
        self.feed_spec = FeedSpec.from_any(feed_spec)
        self.calendar_name = self.feed_spec.resolve_calendar(calendar=calendar)
        self.calendar = None
        if self.calendar_name is None:
            return
        if self.feed_spec.session_start_time is not None:
            raise NotImplementedError(
                "session_start_time overrides require a registered calendar with that schedule"
            )
        timezone = self.feed_spec.timezone
        if timezone is not None:
            if not isinstance(timezone, str) or not timezone.strip():
                raise ValueError("timezone must be a non-empty name or None")
            try:
                pl.Series([], dtype=pl.Datetime("us", "UTC")).dt.convert_time_zone(timezone)
            except (pl.exceptions.PolarsError, TypeError) as error:
                raise ValueError(f"Unknown timezone '{timezone}'") from error
        try:
            import pandas_market_calendars as mcal
        except ImportError:
            raise ImportError(
                "pandas_market_calendars is required for session assignment. "
                "Install with: pip install pandas-market-calendars"
            )

        try:
            self.calendar = mcal.get_calendar(self.calendar_name)
            from pandas_market_calendars.calendars.cme_globex_crypto import (
                CMEGlobexCryptoExchangeCalendar,
            )

            if type(self.calendar) is CMEGlobexCryptoExchangeCalendar:
                from ml4t.data.sessions._cme_crypto import CMEGlobexCryptoSchedule

                self.calendar = CMEGlobexCryptoSchedule(
                    self.calendar, mcal.get_calendar("CME_TradeDate")
                )
        except Exception as e:
            raise ValueError(f"Unknown calendar '{self.calendar_name}': {e}")

        logger.info(f"Initialized SessionAssigner with calendar: {self.calendar_name}")

    @classmethod
    def from_exchange(cls, exchange: str) -> SessionAssigner:
        """Create SessionAssigner from exchange code.

        Args:
            exchange: Exchange code (e.g., "CME", "NYSE", "NASDAQ")

        Returns:
            SessionAssigner instance

        Raises:
            ValueError: If exchange not recognized
        """
        calendar_name = EXCHANGE_CALENDARS.get(exchange.upper())
        if not calendar_name:
            raise ValueError(
                f"Unknown exchange '{exchange}'. "
                f"Known exchanges: {', '.join(EXCHANGE_CALENDARS.keys())}"
            )
        return cls(calendar_name)

    def assign_sessions(
        self,
        df: pl.DataFrame,
        start_date: datetime | date | str | None = None,
        end_date: datetime | date | str | None = None,
        outside_session: Literal["null", "raise", "drop"] = "null",
        bar_frequency: Literal["auto", "daily", "intraday"] = "auto",
    ) -> pl.DataFrame:
        """Assign session_date column to DataFrame.

        Args:
            df: DataFrame with timestamp column
            start_date: Optional start date (auto-detected from data if not provided)
            end_date: Optional end date (auto-detected from data if not provided)
            outside_session: Treatment for observations outside a trading interval
            bar_frequency: Treat timestamps as daily period labels, intraday instants,
                or infer daily labels when every non-null timestamp is midnight

        Returns:
            DataFrame with session_date column added

        Raises:
            ValueError: If timestamp column missing
        """
        if self.calendar is None:
            return df
        if "timestamp" not in df.columns:
            raise ValueError("DataFrame must have 'timestamp' column")
        if outside_session not in {"null", "raise", "drop"}:
            raise ValueError("outside_session must be 'null', 'raise', or 'drop'")
        if bar_frequency not in {"auto", "daily", "intraday"}:
            raise ValueError("bar_frequency must be 'auto', 'daily', or 'intraday'")

        if df.is_empty():
            logger.warning("DataFrame is empty, cannot assign sessions")
            return df.with_columns(pl.lit(None).cast(pl.Date).alias("session_date"))

        timestamp_dtype = df.schema["timestamp"]
        if timestamp_dtype != pl.Date and not isinstance(timestamp_dtype, pl.Datetime):
            raise TypeError("'timestamp' must contain Date or Datetime values")
        semantics = self.feed_spec.timestamp_semantics
        if timestamp_dtype == pl.Date and (
            semantics in (TimestampSemantics.EVENT_TIME, TimestampSemantics.BAR_CLOSE)
            or self.feed_spec.bar_label is not None
        ):
            raise ValueError("event_time/bar_close timestamps must contain Datetime values")

        # Auto-detect date range from data
        if start_date is None:
            start_date = df["timestamp"].min()
        if end_date is None:
            end_date = df["timestamp"].max()
        if start_date is None or end_date is None:
            raise TypeError("'timestamp' must contain non-null Datetime or Date values")

        logger.info(
            f"Assigning sessions for {len(df)} rows",
            start_date=str(start_date),
            end_date=str(end_date),
        )

        # Get trading schedule from calendar
        try:
            import pandas as pd

            start_timestamp = pd.Timestamp(start_date)
            end_timestamp = pd.Timestamp(end_date)
            if not isinstance(start_timestamp, pd.Timestamp) or not isinstance(
                end_timestamp, pd.Timestamp
            ):
                raise ValueError("start_date and end_date must not be NaT")
            schedule = _schedule_for_instants(self.calendar, start_timestamp, end_timestamp)
            logger.debug(f"Got {len(schedule)} sessions from calendar")

            session_map = []
            for session_date, row in schedule.iterrows():
                if not isinstance(session_date, pd.Timestamp):
                    raise TypeError("calendar returned a non-datetime session label")
                for opening, closing in _session_intervals(row):
                    session_map.append(
                        {
                            "_calendar_session_start": opening,
                            "_calendar_session_end": closing,
                            "_assigned_session_date": session_date.date(),
                        }
                    )

            if not session_map:
                logger.warning("No trading sessions found in date range")
                result = df.with_columns(pl.lit(None).cast(pl.Date).alias("session_date"))
                if outside_session == "raise":
                    raise ValueError("Observations outside a trading session: no sessions found")
                if outside_session == "drop":
                    return result.head(0)
                return result

            timestamp_expr = pl.col("timestamp")
            if timestamp_dtype == pl.Date:
                timestamp_expr = timestamp_expr.cast(pl.Datetime("us"))
            if timestamp_dtype == pl.Date or timestamp_dtype.time_zone is None:
                timestamp_expr = timestamp_expr.dt.replace_time_zone(
                    self.feed_spec.timezone or "UTC"
                )
            comparison_unit = (
                "ns"
                if isinstance(timestamp_dtype, pl.Datetime) and timestamp_dtype.time_unit == "ns"
                else "us"
            )
            comparison_dtype = pl.Datetime(comparison_unit, "UTC")
            timestamp_expr = timestamp_expr.dt.convert_time_zone("UTC").cast(comparison_dtype)

            session_df = (
                pl.DataFrame(session_map)
                .with_columns(
                    pl.col("_calendar_session_start", "_calendar_session_end").cast(
                        comparison_dtype
                    )
                )
                .sort("_calendar_session_end")
            )
            daily_labels = _uses_session_labels(df, self.feed_spec, bar_frequency)
            if daily_labels:
                label_expr = pl.col("timestamp")
                if (
                    isinstance(timestamp_dtype, pl.Datetime)
                    and timestamp_dtype.time_zone is not None
                    and self.feed_spec.timezone is not None
                ):
                    label_expr = label_expr.dt.convert_time_zone(self.feed_spec.timezone)
                if isinstance(timestamp_dtype, pl.Datetime):
                    label_expr = label_expr.dt.date()
                result = (
                    df.with_row_index("_session_row_index")
                    .with_columns(label_expr.alias("_session_date_label"))
                    .join(
                        session_df.select(
                            pl.col("_assigned_session_date").alias("_calendar_session_date_label"),
                            "_assigned_session_date",
                        ).unique(),
                        left_on="_session_date_label",
                        right_on="_calendar_session_date_label",
                        how="left",
                    )
                    .with_columns(pl.col("_assigned_session_date").alias("session_date"))
                )
                outside = result.filter(pl.col("session_date").is_null())
                if outside_session == "raise" and not outside.is_empty():
                    timestamps = ", ".join(
                        str(value) for value in outside.get_column("timestamp").head(5).to_list()
                    )
                    raise ValueError(f"Observations outside a trading session: {timestamps}")
                if outside_session == "drop":
                    result = result.filter(pl.col("session_date").is_not_null())
                return result.sort("_session_row_index").drop(
                    "_session_row_index", "_session_date_label", "_assigned_session_date"
                )

            after_last_session = _datetime_scalar(
                session_df.get_column("_calendar_session_end").max(),
                "_calendar_session_end",
            ) + pd.Timedelta(minutes=1)
            observations = (
                df.with_row_index("_session_row_index")
                .with_columns(timestamp_expr.alias("_session_timestamp"))
                .with_columns(
                    pl.col("_session_timestamp")
                    .fill_null(pl.lit(after_last_session, dtype=comparison_dtype))
                    .alias("_session_join_timestamp")
                )
                .sort("_session_join_timestamp")
            )
            close_labeled = (
                semantics == TimestampSemantics.BAR_CLOSE
                or self.feed_spec.bar_label == BarLabel.RIGHT
            )
            join_boundary = "_calendar_session_end" if close_labeled else "_calendar_session_start"
            joined = observations.join_asof(
                session_df.sort(join_boundary),
                left_on="_session_join_timestamp",
                right_on=join_boundary,
                strategy="forward" if close_labeled else "backward",
            )
            if close_labeled:
                in_session = (pl.col("_session_timestamp") > pl.col("_calendar_session_start")) & (
                    pl.col("_session_timestamp") <= pl.col("_calendar_session_end")
                )
            else:
                in_session = (pl.col("_session_timestamp") >= pl.col("_calendar_session_start")) & (
                    pl.col("_session_timestamp") < pl.col("_calendar_session_end")
                )
            result = joined.with_columns(
                pl.when(in_session)
                .then(pl.col("_assigned_session_date"))
                .otherwise(None)
                .alias("session_date")
            )
            outside = result.filter(pl.col("session_date").is_null())
            if outside_session == "raise" and not outside.is_empty():
                timestamps = ", ".join(
                    str(value) for value in outside.get_column("timestamp").head(5).to_list()
                )
                raise ValueError(f"Observations outside a trading session: {timestamps}")
            if outside_session == "drop":
                result = result.filter(pl.col("session_date").is_not_null())

            helper_columns = [
                "_session_row_index",
                "_session_timestamp",
                "_session_join_timestamp",
                "_calendar_session_start",
                "_calendar_session_end",
                "_assigned_session_date",
            ]
            df_with_sessions = result.sort("_session_row_index").drop(
                *[column for column in helper_columns if column in result.columns]
            )

            logger.info(f"Assigned sessions to {len(df_with_sessions)} rows")
            return df_with_sessions

        except Exception as e:
            logger.error(f"Failed to assign sessions: {e}", exc_info=True)
            raise
