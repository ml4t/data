"""Mock provider for testing."""

from __future__ import annotations

import random
from datetime import UTC, datetime, timedelta
from enum import Enum
from typing import Any

import polars as pl

from ml4t.data.providers.base import BaseProvider


class DataPattern(str, Enum):
    """Data generation patterns for mock provider."""

    RANDOM = "random"  # Random walk
    TREND = "trend"  # Trending data
    VOLATILE = "volatile"  # High volatility
    FLAT = "flat"  # Range-bound/flat
    GAPS = "gaps"  # Data with price gaps


class MockProvider(BaseProvider):
    """Mock data provider for testing purposes."""

    def __init__(
        self,
        seed: int = 42,
        pattern: DataPattern = DataPattern.RANDOM,
        base_price: float = 100.0,
        volatility: float = 0.01,
        trend_rate: float = 0.0,
        gap_probability: float = 0.0,
        range_bound: float = 0.05,
    ) -> None:
        """
        Initialize mock provider with configurable parameters.

        Args:
            seed: Random seed for reproducibility
            pattern: Data generation pattern
            base_price: Starting price level
            volatility: Price volatility (as fraction)
            trend_rate: Daily trend rate (for TREND pattern)
            gap_probability: Probability of price gaps (for GAPS pattern)
            range_bound: Range bound for FLAT pattern
        """
        # Initialize base provider with no rate limiting (mock doesn't need it)
        super().__init__(rate_limit=None)

        self.seed = seed
        self.pattern = pattern
        self.base_price = base_price
        self.volatility = volatility
        self.trend_rate = trend_rate
        self.gap_probability = gap_probability
        self.range_bound = range_bound

    @property
    def name(self) -> str:
        """Return the provider name."""
        return "mock"

    def _fetch_raw_data(self, symbol: str, start: str, end: str, frequency: str) -> Any:  # noqa: ARG002
        """
        Fetch raw data from provider API (provider-specific).

        For MockProvider, this generates synthetic data.

        Args:
            symbol: The symbol to fetch data for
            start: Start date in YYYY-MM-DD format
            end: End date in YYYY-MM-DD format
            frequency: Data frequency (daily, minute, etc.)

        Returns:
            Dictionary containing raw mock data
        """
        # Parse dates
        start_date = datetime.strptime(start, "%Y-%m-%d").replace(tzinfo=UTC)
        end_date = datetime.strptime(end, "%Y-%m-%d").replace(tzinfo=UTC)

        # Generate timestamps based on frequency
        timestamps = self._generate_timestamps(start_date, end_date, frequency)

        if not timestamps:
            # Return empty raw data
            return {
                "timestamps": [],
                "data": [],
            }

        # Generate price data based on pattern
        ohlcv_data = self._generate_price_data(timestamps)

        # Return raw data format
        return {
            "timestamps": ohlcv_data["timestamp"],
            "opens": ohlcv_data["open"],
            "highs": ohlcv_data["high"],
            "lows": ohlcv_data["low"],
            "closes": ohlcv_data["close"],
            "volumes": ohlcv_data["volume"],
        }

    def _transform_data(self, raw_data: Any, symbol: str) -> pl.DataFrame:
        """
        Transform raw data to standardized schema (provider-specific).

        Args:
            raw_data: Raw data from _fetch_raw_data
            symbol: Symbol for the data

        Returns:
            Polars DataFrame with standardized schema
        """
        # Handle empty data
        if not raw_data["timestamps"]:
            return pl.DataFrame(
                {
                    "timestamp": [],
                    "symbol": [],
                    "open": [],
                    "high": [],
                    "low": [],
                    "close": [],
                    "volume": [],
                },
                schema={
                    "timestamp": pl.Datetime("ns", "UTC"),
                    "symbol": pl.Utf8,
                    "open": pl.Float64,
                    "high": pl.Float64,
                    "low": pl.Float64,
                    "close": pl.Float64,
                    "volume": pl.Float64,
                },
            )

        # Create DataFrame from raw data (symbol is always uppercase for consistency)
        df = pl.DataFrame(
            {
                "timestamp": raw_data["timestamps"],
                "symbol": [symbol.upper()] * len(raw_data["timestamps"]),
                "open": raw_data["opens"],
                "high": raw_data["highs"],
                "low": raw_data["lows"],
                "close": raw_data["closes"],
                "volume": raw_data["volumes"],
            }
        )

        # Ensure proper data types
        df = df.with_columns(
            [
                pl.col("timestamp").cast(pl.Datetime("ns", "UTC")),
                pl.col("symbol").cast(pl.Utf8),
                pl.col("open").cast(pl.Float64),
                pl.col("high").cast(pl.Float64),
                pl.col("low").cast(pl.Float64),
                pl.col("close").cast(pl.Float64),
                pl.col("volume").cast(pl.Float64),
            ]
        )

        return df

    def _generate_timestamps(
        self, start_date: datetime, end_date: datetime, frequency: str
    ) -> list[datetime]:
        """Generate timestamps for the given date range and frequency."""
        timestamps = []

        if frequency == "daily":
            delta = timedelta(days=1)
            # Set time to market open (9:30 AM ET = 14:30 UTC)
            current = start_date.replace(hour=14, minute=30)
            end_date = end_date.replace(hour=14, minute=30)

            while current <= end_date:
                # Skip weekends
                if current.weekday() < 5:
                    timestamps.append(current)
                current += delta

        elif frequency == "minute":
            # Generate minute bars for market hours (9:30 AM - 4:00 PM ET)
            current_day = start_date.replace(hour=14, minute=30)  # 9:30 AM ET

            while current_day.date() <= end_date.date():
                # Skip weekends
                if current_day.weekday() < 5:
                    # Market hours: 9:30 AM - 4:00 PM ET (390 minutes)
                    market_open = current_day.replace(hour=14, minute=30)

                    current_minute = market_open
                    # Generate exactly 390 minutes
                    for _ in range(390):
                        if current_minute.date() > end_date.date():
                            break
                        timestamps.append(current_minute)
                        current_minute += timedelta(minutes=1)

                current_day += timedelta(days=1)

        else:
            # Default to hourly
            delta = timedelta(hours=1)
            current = start_date
            while current <= end_date:
                timestamps.append(current)
                current += delta

        return timestamps

    def _generate_price_data(self, timestamps: list[datetime]) -> dict[str, Any]:
        """Generate OHLCV data based on the configured pattern."""
        random.seed(self.seed)
        n_points = len(timestamps)

        opens = []
        highs = []
        lows = []
        closes = []
        volumes = []

        current_price = self.base_price
        range_center = self.base_price

        for i in range(n_points):
            # For gaps pattern, add gap at the open
            if (
                self.pattern == DataPattern.GAPS
                and i > 0
                and random.random() < self.gap_probability
            ):
                # Create a gap at market open
                gap_size = random.gauss(0, 0.02) * current_price
                open_price = current_price + gap_size
            else:
                open_price = current_price

            # Apply pattern-specific logic for the close
            if self.pattern == DataPattern.TREND:
                # Add trend component
                trend = self.trend_rate * current_price
                drift = trend + random.gauss(0, self.volatility * current_price)

            elif self.pattern == DataPattern.VOLATILE:
                # High volatility
                drift = random.gauss(0, self.volatility * current_price * 3)

            elif self.pattern == DataPattern.FLAT:
                # Keep price within range
                if abs(current_price - range_center) > range_center * self.range_bound:
                    # Pull back toward center
                    drift = (range_center - current_price) * 0.5
                else:
                    drift = random.gauss(0, self.volatility * current_price * 0.5)

            elif self.pattern == DataPattern.GAPS:
                # Normal drift for intraday movement
                drift = random.gauss(0, self.volatility * current_price)

            else:  # RANDOM pattern (default)
                drift = random.gauss(0, self.volatility * current_price)

            # Generate close price
            close_price = max(0.01, open_price + drift)  # Ensure positive

            # Add intraday volatility
            intraday_vol = abs(random.gauss(0, self.volatility * current_price * 0.5))
            high_price = max(open_price, close_price) + intraday_vol
            low_price = max(0.01, min(open_price, close_price) - intraday_vol)

            # Generate volume (higher volume on larger price moves)
            base_volume = 1000000
            volume_multiplier = 1 + abs(drift / current_price) * 10
            volume = base_volume * volume_multiplier + random.randint(-100000, 500000)

            opens.append(open_price)
            highs.append(high_price)
            lows.append(low_price)
            closes.append(close_price)
            volumes.append(max(0, volume))

            # Update current price for next iteration
            current_price = close_price

        return {
            "timestamp": timestamps,
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": volumes,
        }

    def validate_data(self, df: pl.DataFrame) -> dict[str, Any]:
        """Validate mock data quality.

        Args:
            df: DataFrame to validate

        Returns:
            Validation results dictionary
        """
        if df.is_empty():
            return {"valid": False, "reason": "Empty DataFrame"}

        validation_results = {
            "valid": True,
            "total_records": len(df),
            "date_range": (df["timestamp"].min(), df["timestamp"].max()),
            "issues": [],
        }

        # Check for missing values
        null_counts = df.null_count()
        for col in df.columns:
            if null_counts[col][0] > 0:
                validation_results["issues"].append(
                    f"Column {col} has {null_counts[col][0]} null values"
                )

        # Check OHLC invariants
        invalid_ohlc = (
            (df["high"] < df["low"])
            | (df["high"] < df["open"])
            | (df["high"] < df["close"])
            | (df["low"] > df["open"])
            | (df["low"] > df["close"])
        )

        if invalid_ohlc.any():
            n_invalid = invalid_ohlc.sum()
            validation_results["issues"].append(
                f"Found {n_invalid} rows with invalid OHLC relationships"
            )
            validation_results["valid"] = False

        # Check for negative prices
        for col in ["open", "high", "low", "close"]:
            if (df[col] <= 0).any():
                validation_results["issues"].append(f"Column {col} has negative or zero values")
                validation_results["valid"] = False

        # Check for negative volumes
        if (df["volume"] < 0).any():
            validation_results["issues"].append("Volume has negative values")
            validation_results["valid"] = False

        return validation_results
