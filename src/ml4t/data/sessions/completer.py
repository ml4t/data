"""Session completion with gap filling for trading data.

Ported from crypto-data-pipeline with enhancements for ml4t.data.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta

import polars as pl
import structlog

logger = structlog.get_logger()


def _datetime_scalar(value: object, name: str) -> datetime:
    if not isinstance(value, datetime):
        raise TypeError(f"'{name}' must contain non-null Datetime values")
    return value


class SessionCompleter:
    """Fill gaps in trading data to create complete sessions.

    For each trading session:
    1. Generate all minute timestamps in session (e.g., 1380 for 23-hour CME sessions)
    2. Left join with actual data
    3. Forward fill OHLC prices from last close
    4. Set volume=0 for filled rows
    5. Add session_date column

    Example:
        ```python
        completer = SessionCompleter("CME_Globex_Crypto")
        df_complete = completer.complete_sessions(df)
        # Now has continuous timestamps with no gaps
        ```
    """

    def __init__(self, calendar_name: str):
        """Initialize with exchange calendar.

        Args:
            calendar_name: Name of pandas_market_calendars calendar
                          (e.g., "CME_Globex_Crypto", "NYSE", "NASDAQ")
        """
        try:
            import pandas_market_calendars as mcal
        except ImportError:
            raise ImportError(
                "pandas_market_calendars is required for session completion. "
                "Install with: pip install pandas-market-calendars"
            )

        self.calendar_name = calendar_name
        try:
            self.calendar = mcal.get_calendar(calendar_name)
        except Exception as e:
            raise ValueError(f"Unknown calendar '{calendar_name}': {e}")

        logger.info(f"Initialized SessionCompleter with calendar: {calendar_name}")

    def complete_sessions(
        self,
        df: pl.DataFrame,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        fill_method: str = "forward",
        zero_volume: bool = True,
    ) -> pl.DataFrame:
        """Fill gaps in data to create complete trading sessions.

        Args:
            df: Input DataFrame with timestamp, open, high, low, close, volume
            start_date: Optional start date (auto-detected if not provided)
            end_date: Optional end date (auto-detected if not provided)
            fill_method: Method for filling prices ("forward", "backward", "none")
            zero_volume: If True, set volume=0 for filled rows; if False, use NaN

        Returns:
            DataFrame with complete sessions (no gaps), sorted by timestamp

        Raises:
            ValueError: If required columns missing or data is invalid
        """
        if "timestamp" not in df.columns:
            raise ValueError("DataFrame must have 'timestamp' column")
        if fill_method not in {"forward", "backward", "none"}:
            raise ValueError("fill_method must be 'forward', 'backward', or 'none'")

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
                request_start = request_start.tz_localize("UTC")
            else:
                request_start = request_start.tz_convert("UTC")
            if request_end.tzinfo is None:
                request_end = request_end.tz_localize("UTC")
            else:
                request_end = request_end.tz_convert("UTC")
            if request_start >= request_end:
                raise ValueError("start_date must be earlier than the exclusive end_date")

            start_pd = request_start.normalize() - pd.Timedelta(days=1)
            end_pd = request_end.normalize() + pd.Timedelta(days=1)

            # Get trading schedule
            schedule = self.calendar.schedule(start_date=start_pd, end_date=end_pd)

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
                market_open = row["market_open"]
                market_close = row["market_close"]
                intervals = [(market_open, market_close)]
                if "break_start" in schedule.columns and not pd.isna(row["break_start"]):
                    intervals = [
                        (market_open, row["break_start"]),
                        (row["break_end"], market_close),
                    ]
                for interval_start, interval_end in intervals:
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
                timestamp_expr = timestamp_expr.dt.replace_time_zone("UTC")
            timestamp_expr = timestamp_expr.dt.convert_time_zone("UTC").cast(
                pl.Datetime("ns", "UTC")
            )
            df_with_tz = df.with_columns(
                timestamp_expr.alias("timestamp"), pl.lit(True).alias("_is_observed")
            )
            if df_with_tz.get_column("timestamp").n_unique() != df_with_tz.height:
                raise ValueError("Session completion requires unique input timestamps")

            # Left join: keep all minutes from template
            complete_df = minute_template.join(df_with_tz, on="timestamp", how="left")

            # Drop duplicate session_date column if exists
            if "session_date_right" in complete_df.columns:
                complete_df = complete_df.drop("session_date_right")
            complete_df = complete_df.with_columns(
                pl.col("_is_observed").is_null().alias("is_imputed")
            )

            # Fill missing data based on method
            if fill_method != "none":
                complete_df = self._fill_missing_data(
                    complete_df,
                    method=fill_method,
                    zero_volume=zero_volume,
                )

            complete_df = complete_df.drop("_is_observed")
            rows_added = complete_df.get_column("is_imputed").sum()
            logger.info(f"Completed sessions: added {rows_added} rows ({len(complete_df)} total)")

            return complete_df.sort("timestamp")

        except Exception as e:
            logger.error(f"Failed to complete sessions: {e}", exc_info=True)
            raise

    def _fill_missing_data(
        self,
        df: pl.DataFrame,
        method: str = "forward",
        zero_volume: bool = True,
    ) -> pl.DataFrame:
        """Fill missing data with forward-filled prices and zero/NaN volume.

        Strategy:
        1. Forward/backward fill OHLC from last/next available close
        2. Set volume=0 (or NaN) for filled rows
        3. Preserve all other columns

        Args:
            df: DataFrame with nulls in OHLCV columns
            method: Fill method ("forward" or "backward")
            zero_volume: If True, use 0 for missing volume; if False, keep NaN

        Returns:
            DataFrame with filled data
        """
        price_columns = ["open", "high", "low", "close"]

        reference_price: pl.Expr | None = None
        if method == "forward":
            if "close" in df.columns:
                reference_price = pl.col("close").forward_fill().over("session_date")
        elif method == "backward":
            if "close" in df.columns:
                reference_price = pl.col("close").backward_fill().over("session_date")

        filled_df = df
        if reference_price is not None:
            filled_df = filled_df.with_columns(
                *[
                    pl.when(pl.col("is_imputed"))
                    .then(reference_price)
                    .otherwise(pl.col(column))
                    .alias(column)
                    for column in price_columns
                    if column in df.columns
                ]
            )

        # Handle volume
        if "volume" in df.columns:
            if zero_volume:
                filled_df = filled_df.with_columns(
                    pl.when(pl.col("is_imputed"))
                    .then(pl.lit(0.0))
                    .otherwise(pl.col("volume"))
                    .alias("volume")
                )
            # else: keep NaN for missing volume

        # Forward fill common metadata columns if present
        metadata_columns = [
            "instrument_id",
            "symbol",
            "raw_symbol",
            "base_symbol",
            "rtype",
            "publisher_id",
        ]

        for col in metadata_columns:
            if col in filled_df.columns:
                filled_df = filled_df.with_columns(pl.col(col).forward_fill().over("session_date"))

        return filled_df

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
