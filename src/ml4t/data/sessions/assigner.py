"""Session date assignment using exchange calendars."""

from __future__ import annotations

from datetime import date, datetime
from typing import Literal

import polars as pl
import structlog

logger = structlog.get_logger()


def _datetime_scalar(value: object, name: str) -> datetime:
    if not isinstance(value, datetime):
        raise TypeError(f"'{name}' must contain non-null Datetime values")
    return value


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


class SessionAssigner:
    """Assigns session dates to trading data based on exchange calendars.

    Uses pandas_market_calendars to determine trading sessions and assigns
    a session_date column to each timestamp.

    For CME futures, sessions start at 5pm CT Sunday and end at 4pm CT Friday.
    The session_date is the date when the session ENDS (4pm date).
    """

    def __init__(self, calendar_name: str):
        """Initialize session assigner.

        Args:
            calendar_name: Name of calendar from pandas_market_calendars
                          (e.g., "CME_Globex_Crypto", "NYSE", "NASDAQ")
        """
        try:
            import pandas_market_calendars as mcal
        except ImportError:
            raise ImportError(
                "pandas_market_calendars is required for session assignment. "
                "Install with: pip install pandas-market-calendars"
            )

        self.calendar_name = calendar_name
        try:
            self.calendar = mcal.get_calendar(calendar_name)
        except Exception as e:
            raise ValueError(f"Unknown calendar '{calendar_name}': {e}")

        logger.info(f"Initialized SessionAssigner with calendar: {calendar_name}")

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
        if not isinstance(timestamp_dtype, pl.Datetime):
            raise TypeError("'timestamp' must contain Datetime values")

        # Auto-detect date range from data
        if start_date is None:
            start_date = _datetime_scalar(df["timestamp"].min(), "timestamp")
        if end_date is None:
            end_date = _datetime_scalar(df["timestamp"].max(), "timestamp")

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
            start_pd = start_timestamp.normalize() - pd.Timedelta(days=1)
            end_pd = end_timestamp.normalize() + pd.Timedelta(days=1)
            schedule = self.calendar.schedule(start_date=start_pd, end_date=end_pd)
            logger.debug(f"Got {len(schedule)} sessions from calendar")

            session_map = []
            for session_date, row in schedule.iterrows():
                if not isinstance(session_date, pd.Timestamp):
                    raise TypeError("calendar returned a non-datetime session label")
                entry = {
                    "_calendar_session_start": row["market_open"].to_pydatetime(),
                    "_calendar_session_end": row["market_close"].to_pydatetime(),
                    "_assigned_session_date": session_date.date(),
                }
                if "break_start" in schedule.columns:
                    entry["_calendar_break_start"] = (
                        None if pd.isna(row["break_start"]) else row["break_start"].to_pydatetime()
                    )
                    entry["_calendar_break_end"] = (
                        None if pd.isna(row["break_end"]) else row["break_end"].to_pydatetime()
                    )
                session_map.append(entry)

            if not session_map:
                logger.warning("No trading sessions found in date range")
                result = df.with_columns(pl.lit(None).cast(pl.Date).alias("session_date"))
                if outside_session == "raise":
                    raise ValueError("Observations outside a trading session: no sessions found")
                if outside_session == "drop":
                    return result.head(0)
                return result

            timestamp_expr = pl.col("timestamp")
            if timestamp_dtype.time_zone is None:
                timestamp_expr = timestamp_expr.dt.replace_time_zone("UTC")
            timestamp_expr = timestamp_expr.dt.convert_time_zone("UTC").cast(
                pl.Datetime("us", "UTC")
            )

            session_df = pl.DataFrame(session_map).sort("_calendar_session_end")
            non_null_timestamps = df.get_column("timestamp").drop_nulls().to_list()
            inferred_daily = bool(non_null_timestamps) and all(
                value.hour == value.minute == value.second == value.microsecond == 0
                for value in non_null_timestamps
                if isinstance(value, datetime)
            )
            daily_labels = bar_frequency == "daily" or (bar_frequency == "auto" and inferred_daily)
            if daily_labels:
                result = (
                    df.with_row_index("_session_row_index")
                    .with_columns(timestamp_expr.dt.date().alias("_session_date_label"))
                    .join(
                        session_df.select(
                            pl.col("_assigned_session_date").alias("_calendar_session_date_label"),
                            "_assigned_session_date",
                        ),
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
                    .fill_null(pl.lit(after_last_session))
                    .alias("_session_join_timestamp")
                )
                .sort("_session_join_timestamp")
            )
            joined = observations.join_asof(
                session_df,
                left_on="_session_join_timestamp",
                right_on="_calendar_session_end",
                strategy="forward",
            )
            in_session = (pl.col("_session_timestamp") >= pl.col("_calendar_session_start")) & (
                pl.col("_session_timestamp") < pl.col("_calendar_session_end")
            )
            if "_calendar_break_start" in joined.columns:
                in_break = (
                    pl.col("_calendar_break_start").is_not_null()
                    & (pl.col("_session_timestamp") >= pl.col("_calendar_break_start"))
                    & (pl.col("_session_timestamp") < pl.col("_calendar_break_end"))
                )
                in_session &= ~in_break
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
                "_calendar_break_start",
                "_calendar_break_end",
            ]
            df_with_sessions = result.sort("_session_row_index").drop(
                *[column for column in helper_columns if column in result.columns]
            )

            logger.info(f"Assigned sessions to {len(df_with_sessions)} rows")
            return df_with_sessions

        except Exception as e:
            logger.error(f"Failed to assign sessions: {e}", exc_info=True)
            raise
