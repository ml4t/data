"""
Price adjustment methods for futures continuous contracts.

When rolling from one contract month to the next, there's usually a price gap.
Adjustment methods handle this gap to create a smooth continuous series.
"""

from abc import ABC, abstractmethod
from datetime import date

import polars as pl


class AdjustmentMethod(ABC):
    """Abstract base class for price adjustment methods."""

    @abstractmethod
    def adjust(self, data: pl.DataFrame, roll_dates: list[date]) -> pl.DataFrame:
        """
        Adjust prices to create continuous series across rolls.

        Args:
            data: Clean continuous data (single row per date) with unadjusted prices
                  Must have columns: date, open, high, low, close
            roll_dates: Dates when rolls occur

        Returns:
            DataFrame with additional adjusted price columns
        """


class BackAdjustment(AdjustmentMethod):
    """
    Back-adjustment (additive/shift adjustment).

    Shifts historical prices by the roll gap to create a continuous series.
    Preserves price differences (useful for spread trading).
    Can produce negative prices.

    Formula:
        gap = new_front_close - old_front_close (on roll date)
        adjusted_price[t < roll] = price[t] + gap

    Cumulative for multiple rolls (work backwards from most recent).
    """

    def adjust(self, data: pl.DataFrame, roll_dates: list[date]) -> pl.DataFrame:
        """
        Apply back-adjustment to create continuous series.

        Args:
            data: Unadjusted continuous data
            roll_dates: Dates when rolls occur

        Returns:
            DataFrame with adjusted_* columns (open, high, low, close)
        """
        if len(roll_dates) == 0:
            # No rolls = no adjustment needed
            return data.with_columns(
                [
                    pl.col("open").alias("adjusted_open"),
                    pl.col("high").alias("adjusted_high"),
                    pl.col("low").alias("adjusted_low"),
                    pl.col("close").alias("adjusted_close"),
                ]
            )

        # Sort roll dates chronologically (work backwards from most recent)
        sorted_rolls = sorted(roll_dates, reverse=True)

        # Start with unadjusted prices
        result = data.with_columns(
            [
                pl.col("open").alias("adjusted_open"),
                pl.col("high").alias("adjusted_high"),
                pl.col("low").alias("adjusted_low"),
                pl.col("close").alias("adjusted_close"),
            ]
        )

        # Apply adjustments working backwards from most recent roll
        cumulative_adjustment = 0.0

        for roll_date in sorted_rolls:
            # Get the close price on the day before the roll
            prev_data = result.filter(pl.col("date") < roll_date).tail(1)
            # Get the close price on the roll date
            roll_data = result.filter(pl.col("date") == roll_date)

            if len(prev_data) == 0 or len(roll_data) == 0:
                continue  # Can't calculate gap, skip

            # Calculate gap: price jump from old front to new front at roll
            # Gap = close[roll_date] - close[day_before_roll]
            close_before = prev_data["close"].item()  # Use original close, not adjusted
            close_after = roll_data["close"].item()

            gap = close_after - close_before

            # Subtract gap from all prices BEFORE the roll (back-adjustment)
            # This aligns historical prices with current contract level
            cumulative_adjustment -= gap

            # Apply adjustment to all dates before this roll
            result = result.with_columns(
                [
                    pl.when(pl.col("date") < roll_date)
                    .then(pl.col("adjusted_open") + cumulative_adjustment)
                    .otherwise(pl.col("adjusted_open"))
                    .alias("adjusted_open"),
                    pl.when(pl.col("date") < roll_date)
                    .then(pl.col("adjusted_high") + cumulative_adjustment)
                    .otherwise(pl.col("adjusted_high"))
                    .alias("adjusted_high"),
                    pl.when(pl.col("date") < roll_date)
                    .then(pl.col("adjusted_low") + cumulative_adjustment)
                    .otherwise(pl.col("adjusted_low"))
                    .alias("adjusted_low"),
                    pl.when(pl.col("date") < roll_date)
                    .then(pl.col("adjusted_close") + cumulative_adjustment)
                    .otherwise(pl.col("adjusted_close"))
                    .alias("adjusted_close"),
                ]
            )

        return result


class RatioAdjustment(AdjustmentMethod):
    """
    Ratio-adjustment (multiplicative/proportional adjustment).

    Scales historical prices by the roll ratio to create a continuous series.
    Preserves percentage returns (useful for return analysis).
    Always maintains positive prices.

    Formula:
        ratio = new_front_close / old_front_close (on roll date)
        adjusted_price[t < roll] = price[t] * ratio

    Cumulative for multiple rolls (work backwards from most recent).
    """

    def adjust(self, data: pl.DataFrame, roll_dates: list[date]) -> pl.DataFrame:
        """
        Apply ratio-adjustment to create continuous series.

        Args:
            data: Unadjusted continuous data
            roll_dates: Dates when rolls occur

        Returns:
            DataFrame with adjusted_* columns (open, high, low, close)
        """
        if len(roll_dates) == 0:
            # No rolls = no adjustment needed
            return data.with_columns(
                [
                    pl.col("open").alias("adjusted_open"),
                    pl.col("high").alias("adjusted_high"),
                    pl.col("low").alias("adjusted_low"),
                    pl.col("close").alias("adjusted_close"),
                ]
            )

        # Sort roll dates chronologically (work backwards)
        sorted_rolls = sorted(roll_dates, reverse=True)

        # Start with unadjusted prices
        result = data.with_columns(
            [
                pl.col("open").alias("adjusted_open"),
                pl.col("high").alias("adjusted_high"),
                pl.col("low").alias("adjusted_low"),
                pl.col("close").alias("adjusted_close"),
            ]
        )

        # Apply adjustments working backwards
        cumulative_ratio = 1.0

        for roll_date in sorted_rolls:
            # Get the close price on the day before the roll
            prev_data = result.filter(pl.col("date") < roll_date).tail(1)
            # Get the close price on the roll date
            roll_data = result.filter(pl.col("date") == roll_date)

            if len(prev_data) == 0 or len(roll_data) == 0:
                continue

            # Calculate ratio: close after roll / close before roll
            close_before = prev_data["close"].item()  # Use original close
            close_after = roll_data["close"].item()

            if close_before == 0:
                continue  # Avoid division by zero

            ratio = close_after / close_before

            # Divide by ratio to align historical prices (inverse of ratio)
            # If new contract is 10% higher, divide historical by 1.1
            cumulative_ratio /= ratio

            # Apply ratio to all dates before this roll
            result = result.with_columns(
                [
                    pl.when(pl.col("date") < roll_date)
                    .then(pl.col("adjusted_open") * cumulative_ratio)
                    .otherwise(pl.col("adjusted_open"))
                    .alias("adjusted_open"),
                    pl.when(pl.col("date") < roll_date)
                    .then(pl.col("adjusted_high") * cumulative_ratio)
                    .otherwise(pl.col("adjusted_high"))
                    .alias("adjusted_high"),
                    pl.when(pl.col("date") < roll_date)
                    .then(pl.col("adjusted_low") * cumulative_ratio)
                    .otherwise(pl.col("adjusted_low"))
                    .alias("adjusted_low"),
                    pl.when(pl.col("date") < roll_date)
                    .then(pl.col("adjusted_close") * cumulative_ratio)
                    .otherwise(pl.col("adjusted_close"))
                    .alias("adjusted_close"),
                ]
            )

        return result


class NoAdjustment(AdjustmentMethod):
    """
    No adjustment (Panama method).

    Returns prices as-is with no adjustment for roll gaps.
    Shows actual market prices but creates discontinuous series.
    Useful for understanding roll impact and actual trading prices.
    """

    def adjust(self, data: pl.DataFrame, _roll_dates: list[date]) -> pl.DataFrame:
        """
        No adjustment - pass through original prices.

        Args:
            data: Unadjusted continuous data
            _roll_dates: Dates when rolls occur (ignored)

        Returns:
            DataFrame with adjusted_* columns equal to original prices
        """
        return data.with_columns(
            [
                pl.col("open").alias("adjusted_open"),
                pl.col("high").alias("adjusted_high"),
                pl.col("low").alias("adjusted_low"),
                pl.col("close").alias("adjusted_close"),
            ]
        )
