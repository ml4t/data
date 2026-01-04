"""Session completion with gap filling for trading data.

Ported from crypto-data-pipeline with enhancements for ml4t.data.
"""

from __future__ import annotations

from datetime import date, datetime

import polars as pl
import structlog

logger = structlog.get_logger()


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

        if df.is_empty():
            logger.warning("DataFrame is empty, cannot complete sessions")
            return df

        # Auto-detect date range
        if start_date is None:
            start_date = df["timestamp"].min()
        if end_date is None:
            end_date = df["timestamp"].max()

        logger.info(
            f"Completing sessions for {len(df)} rows",
            start_date=str(start_date),
            end_date=str(end_date),
            fill_method=fill_method,
        )

        try:
            import pandas as pd

            # Convert to pandas Timestamps
            start_pd = pd.Timestamp(start_date.date())
            # Add 1 day to capture sessions that contain end_date
            end_pd = pd.Timestamp(end_date.date()) + pd.Timedelta(days=1)

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
                market_open = row["market_open"]
                market_close = row["market_close"]

                # Generate minute range for session
                # Use inclusive="left" to exclude market_close (avoid overlap)
                minutes = pd.date_range(
                    start=market_open,
                    end=market_close,
                    freq="1min",
                    inclusive="left",
                )

                all_minutes.extend(minutes)
                # Session date is the END date (market close date)
                session_dates.extend([session_date.date()] * len(minutes))

            # Create complete minute template
            minute_template = pl.DataFrame(
                {
                    "timestamp": [m.to_pydatetime() for m in all_minutes],
                    "session_date": session_dates,
                }
            )

            # Ensure proper timezone and types
            minute_template = minute_template.with_columns(
                [
                    pl.col("timestamp").dt.replace_time_zone("UTC").cast(pl.Datetime("ns", "UTC")),
                    pl.col("session_date").cast(pl.Date),
                ]
            )

            # Ensure input data has matching timestamp type
            df_with_tz = df.with_columns(pl.col("timestamp").cast(pl.Datetime("ns", "UTC")))

            # Left join: keep all minutes from template
            complete_df = minute_template.join(df_with_tz, on="timestamp", how="left")

            # Drop duplicate session_date column if exists
            if "session_date_right" in complete_df.columns:
                complete_df = complete_df.drop("session_date_right")

            # Fill missing data based on method
            if fill_method != "none":
                complete_df = self._fill_missing_data(
                    complete_df,
                    method=fill_method,
                    zero_volume=zero_volume,
                )

            rows_added = len(complete_df) - len(df)
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

        # Fill price columns
        if method == "forward":
            filled_df = df.with_columns(
                [pl.col(col).forward_fill() for col in price_columns if col in df.columns]
            )
        elif method == "backward":
            filled_df = df.with_columns(
                [pl.col(col).backward_fill() for col in price_columns if col in df.columns]
            )
        else:
            filled_df = df

        # Handle volume
        if "volume" in df.columns:
            if zero_volume:
                filled_df = filled_df.with_columns(
                    pl.when(pl.col("volume").is_null())
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
                filled_df = filled_df.with_columns(pl.col(col).forward_fill())

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
