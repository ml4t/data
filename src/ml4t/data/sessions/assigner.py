"""Session date assignment using exchange calendars."""

from __future__ import annotations

from datetime import date, datetime

import polars as pl
import structlog

logger = structlog.get_logger()

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
    ) -> pl.DataFrame:
        """Assign session_date column to DataFrame.

        Args:
            df: DataFrame with timestamp column
            start_date: Optional start date (auto-detected from data if not provided)
            end_date: Optional end date (auto-detected from data if not provided)

        Returns:
            DataFrame with session_date column added

        Raises:
            ValueError: If timestamp column missing
        """
        if "timestamp" not in df.columns:
            raise ValueError("DataFrame must have 'timestamp' column")

        if df.is_empty():
            logger.warning("DataFrame is empty, cannot assign sessions")
            return df.with_columns(pl.lit(None).cast(pl.Date).alias("session_date"))

        # Auto-detect date range from data
        if start_date is None:
            start_date = df["timestamp"].min()
        if end_date is None:
            end_date = df["timestamp"].max()

        logger.info(
            f"Assigning sessions for {len(df)} rows",
            start_date=str(start_date),
            end_date=str(end_date),
        )

        # Get trading schedule from calendar
        try:
            import pandas as pd

            # Convert to pandas Timestamp
            if (
                isinstance(start_date, str)
                or isinstance(start_date, date)
                and not isinstance(start_date, datetime)
            ):
                start_pd = pd.Timestamp(start_date)
            else:
                start_pd = pd.Timestamp(start_date)

            if (
                isinstance(end_date, str)
                or isinstance(end_date, date)
                and not isinstance(end_date, datetime)
            ):
                end_pd = pd.Timestamp(end_date)
            else:
                end_pd = pd.Timestamp(end_date)

            schedule = self.calendar.schedule(start_date=start_pd, end_date=end_pd)
            logger.debug(f"Got {len(schedule)} sessions from calendar")

            # Build session mapping: timestamp → session_date
            session_map = []
            for session_date, row in schedule.iterrows():
                market_open = row["market_open"]
                market_close = row["market_close"]

                # Session date is the END date (when market closes)
                session_map.append(
                    {
                        "session_start": market_open.to_pydatetime(),
                        "session_end": market_close.to_pydatetime(),
                        "session_date": session_date.date(),
                    }
                )

            if not session_map:
                logger.warning("No trading sessions found in date range")
                return df.with_columns(pl.lit(None).cast(pl.Date).alias("session_date"))

            # Create session mapping DataFrame
            session_df = pl.DataFrame(session_map)

            # Join with data using inequality join
            # For each timestamp, find the session where:
            # session_start <= timestamp < session_end
            df_with_sessions = df.join_asof(
                session_df.select(["session_end", "session_date"]),
                left_on="timestamp",
                right_on="session_end",
                strategy="forward",
            )

            logger.info(f"Assigned sessions to {len(df_with_sessions)} rows")
            return df_with_sessions

        except Exception as e:
            logger.error(f"Failed to assign sessions: {e}", exc_info=True)
            raise
