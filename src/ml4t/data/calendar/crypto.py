"""24/7 market calendar for cryptocurrency trading."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import polars as pl
import structlog

logger = structlog.get_logger()


class CryptoCalendar:
    """
    24/7 market calendar for cryptocurrency markets.

    Cryptocurrencies trade continuously except during:
    - Exchange maintenance windows
    - Major outages
    - Regulatory halts (rare)
    """

    def __init__(self, exchange: str | None = None) -> None:
        """
        Initialize crypto calendar.

        Args:
            exchange: Optional exchange name for specific maintenance windows
        """
        self.exchange = exchange
        self.maintenance_windows = self._load_maintenance_windows()

    def is_open(self, timestamp: datetime) -> bool:
        """
        Check if crypto market is open at given time.

        Args:
            timestamp: Time to check

        Returns:
            True if market is open (almost always for crypto)
        """
        # Ensure timezone-aware
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=UTC)

        # Check maintenance windows
        return not self.is_maintenance(timestamp)

    def is_maintenance(self, timestamp: datetime) -> bool:
        """
        Check if exchange is in maintenance at given time.

        Args:
            timestamp: Time to check

        Returns:
            True if in maintenance window
        """
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=UTC)

        for window_start, window_end in self.maintenance_windows:
            if window_start <= timestamp < window_end:
                return True
        return False

    def get_sessions(self, start: datetime, end: datetime) -> list[tuple[datetime, datetime]]:
        """
        Get trading sessions between dates.

        For crypto, sessions are 24-hour periods (UTC days).

        Args:
            start: Start date
            end: End date

        Returns:
            List of (session_start, session_end) tuples
        """
        if start.tzinfo is None:
            start = start.replace(tzinfo=UTC)
        if end.tzinfo is None:
            end = end.replace(tzinfo=UTC)

        sessions = []
        current = start.replace(hour=0, minute=0, second=0, microsecond=0)

        while current < end:
            session_start = current
            session_end = current + timedelta(days=1)

            # Check if entire session is in maintenance
            if not self._is_full_maintenance(session_start, session_end):
                sessions.append((session_start, min(session_end, end)))

            current = session_end

        return sessions

    def get_expected_sessions_count(
        self, start: datetime, end: datetime, frequency: str = "daily"
    ) -> int:
        """
        Get expected number of data points for period.

        Args:
            start: Start date
            end: End date
            frequency: Data frequency

        Returns:
            Expected number of data points
        """
        if start.tzinfo is None:
            start = start.replace(tzinfo=UTC)
        if end.tzinfo is None:
            end = end.replace(tzinfo=UTC)

        duration = end - start

        # Map frequency to expected points
        if frequency == "daily" or frequency == "1d":
            # One point per day
            return duration.days
        if frequency == "hourly" or frequency == "1h":
            # 24 points per day
            return int(duration.total_seconds() / 3600)
        if frequency == "minute" or frequency == "1m":
            # 1440 points per day
            return int(duration.total_seconds() / 60)
        if frequency == "5minute" or frequency == "5m":
            # 288 points per day
            return int(duration.total_seconds() / 300)
        if frequency == "15minute" or frequency == "15m":
            # 96 points per day
            return int(duration.total_seconds() / 900)
        if frequency == "30minute" or frequency == "30m":
            # 48 points per day
            return int(duration.total_seconds() / 1800)
        if frequency == "4hour" or frequency == "4h":
            # 6 points per day
            return int(duration.total_seconds() / 14400)
        # Default to daily
        return duration.days

    def find_gaps(
        self,
        df: pl.DataFrame,
        frequency: str = "daily",
        threshold_multiplier: float = 1.5,
    ) -> list[tuple[datetime, datetime]]:
        """
        Find gaps in crypto data.

        Args:
            df: DataFrame with timestamp column
            frequency: Expected data frequency
            threshold_multiplier: Multiplier for gap threshold

        Returns:
            List of (gap_start, gap_end) tuples
        """
        if df.is_empty() or "timestamp" not in df.columns:
            return []

        # Sort by timestamp
        df = df.sort("timestamp")
        timestamps = df["timestamp"].to_list()

        # Get expected interval in seconds
        expected_interval = self._get_interval_seconds(frequency)
        threshold = expected_interval * threshold_multiplier

        gaps = []
        for i in range(1, len(timestamps)):
            time_diff = (timestamps[i] - timestamps[i - 1]).total_seconds()

            # Check if gap is larger than threshold and not maintenance
            if time_diff > threshold:
                gap_start = timestamps[i - 1]
                gap_end = timestamps[i]

                # Only report if not during maintenance
                if not self._is_full_maintenance(gap_start, gap_end):
                    gaps.append((gap_start, gap_end))

        return gaps

    def _load_maintenance_windows(self) -> list[tuple[datetime, datetime]]:
        """
        Load known maintenance windows for exchange.

        Returns:
            List of maintenance windows
        """
        windows: list[tuple[datetime, datetime]] = []

        # Common maintenance windows (example)
        if self.exchange == "binance":
            # Binance typically does maintenance on Wednesdays
            # This is simplified - real implementation would load from config
            pass
        elif self.exchange == "coinbase":
            # Coinbase maintenance windows
            pass

        return windows

    def _is_full_maintenance(self, start: datetime, end: datetime) -> bool:
        """
        Check if entire period is in maintenance.

        Args:
            start: Period start
            end: Period end

        Returns:
            True if entire period is maintenance
        """
        # For crypto, maintenance is rare and brief
        # This is a simplified check
        for window_start, window_end in self.maintenance_windows:
            if start >= window_start and end <= window_end:
                return True
        return False

    def _get_interval_seconds(self, frequency: str) -> int:
        """
        Get expected interval in seconds for frequency.

        Args:
            frequency: Data frequency

        Returns:
            Interval in seconds
        """
        intervals = {
            "1m": 60,
            "minute": 60,
            "3m": 180,
            "3minute": 180,
            "5m": 300,
            "5minute": 300,
            "15m": 900,
            "15minute": 900,
            "30m": 1800,
            "30minute": 1800,
            "1h": 3600,
            "hourly": 3600,
            "2h": 7200,
            "4h": 14400,
            "4hour": 14400,
            "6h": 21600,
            "8h": 28800,
            "12h": 43200,
            "1d": 86400,
            "daily": 86400,
            "3d": 259200,
            "1w": 604800,
            "weekly": 604800,
        }
        return intervals.get(frequency, 86400)  # Default to daily


class CryptoSessionValidator:
    """Validator for crypto trading sessions."""

    def __init__(self, calendar: CryptoCalendar | None = None) -> None:
        """
        Initialize validator.

        Args:
            calendar: Crypto calendar to use
        """
        self.calendar = calendar or CryptoCalendar()

    def validate_continuity(
        self,
        df: pl.DataFrame,
        frequency: str = "daily",
    ) -> list[str]:
        """
        Validate data continuity for 24/7 market.

        Args:
            df: DataFrame to validate
            frequency: Expected data frequency

        Returns:
            List of validation issues
        """
        issues = []

        if df.is_empty():
            return ["No data to validate"]

        if "timestamp" not in df.columns:
            return ["Missing timestamp column"]

        # Find gaps
        gaps = self.calendar.find_gaps(df, frequency)

        for gap_start, gap_end in gaps:
            duration = gap_end - gap_start
            issues.append(
                f"Gap detected: {gap_start.isoformat()} to {gap_end.isoformat()} "
                f"({duration.total_seconds() / 3600:.1f} hours)"
            )

        # Check for weekend data (crypto should have it)
        timestamps = df["timestamp"].to_list()
        weekend_dates = [
            ts
            for ts in timestamps
            if ts.weekday() in [5, 6]  # Saturday, Sunday
        ]

        if not weekend_dates and len(timestamps) >= 7:
            issues.append("Missing weekend data (crypto trades 24/7)")

        return issues

    def validate_volume_profile(
        self,
        df: pl.DataFrame,
    ) -> list[str]:
        """
        Validate volume profile for crypto data.

        Args:
            df: DataFrame to validate

        Returns:
            List of validation issues
        """
        issues = []

        if "volume" not in df.columns:
            return ["Missing volume column"]

        # Check for zero volume periods (unusual for major cryptos)
        zero_volume = df.filter(pl.col("volume") == 0)
        if len(zero_volume) > 0:
            issues.append(f"Found {len(zero_volume)} periods with zero volume")

        # Check for extreme volume spikes
        if len(df) > 10:
            df["volume"].to_numpy()
            median_volume = df["volume"].median()

            if median_volume and median_volume > 0:
                extreme_spikes = df.filter(pl.col("volume") > median_volume * 100)
                if len(extreme_spikes) > 0:
                    issues.append(
                        f"Found {len(extreme_spikes)} extreme volume spikes (>100x median)"
                    )

        return issues
