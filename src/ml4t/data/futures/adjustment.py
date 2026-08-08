"""Price adjustments based on paired futures roll observations."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Sequence
from functools import reduce
from operator import mul

import polars as pl

from ml4t.data.futures.roll import RollEvent

PRICE_COLUMNS = ("open", "high", "low", "close")


class AdjustmentMethod(ABC):
    """Transform a selected futures series using explicit roll events."""

    @abstractmethod
    def adjust(self, data: pl.DataFrame, roll_events: Sequence[RollEvent]) -> pl.DataFrame:
        """Return the series with adjusted OHLC columns."""


class BackAdjustment(AdjustmentMethod):
    """Add each same-date contract gap once to the preceding history."""

    def adjust(self, data: pl.DataFrame, roll_events: Sequence[RollEvent]) -> pl.DataFrame:
        _validate_inputs(data, roll_events)
        adjustments = [
            pl.when(pl.col("date") < event.date)
            .then(pl.lit(event.new_close - event.old_close))
            .otherwise(pl.lit(0.0))
            for event in roll_events
        ]
        offset = pl.sum_horizontal(adjustments) if adjustments else pl.lit(0.0)
        return data.with_columns(
            (pl.col(column) + offset).alias(f"adjusted_{column}") for column in PRICE_COLUMNS
        )


class RatioAdjustment(AdjustmentMethod):
    """Multiply preceding history by each valid same-date contract ratio once."""

    def adjust(self, data: pl.DataFrame, roll_events: Sequence[RollEvent]) -> pl.DataFrame:
        _validate_inputs(data, roll_events)
        ratios = []
        for event in roll_events:
            if event.old_close == 0 or event.new_close == 0:
                raise ValueError(f"Ratio adjustment is undefined for a zero close on {event.date}")
            ratio = event.new_close / event.old_close
            if ratio <= 0:
                raise ValueError(
                    f"Ratio adjustment requires old and new closes with the same sign on {event.date}"
                )
            ratios.append(
                pl.when(pl.col("date") < event.date).then(pl.lit(ratio)).otherwise(pl.lit(1.0))
            )
        multiplier = reduce(mul, ratios, pl.lit(1.0))
        return data.with_columns(
            (pl.col(column) * multiplier).alias(f"adjusted_{column}") for column in PRICE_COLUMNS
        )


class NoAdjustment(AdjustmentMethod):
    """Expose unchanged prices through the common adjusted-column contract."""

    def adjust(
        self,
        data: pl.DataFrame,
        roll_events: Sequence[RollEvent],  # noqa: ARG002
    ) -> pl.DataFrame:
        _validate_inputs(data, roll_events)
        return data.with_columns(
            pl.col(column).alias(f"adjusted_{column}") for column in PRICE_COLUMNS
        )


def _validate_inputs(data: pl.DataFrame, roll_events: Sequence[RollEvent]) -> None:
    _validate_price_columns(data)
    if any(not isinstance(event, RollEvent) for event in roll_events):
        raise TypeError("roll_events must contain RollEvent instances with paired contract prices")
    event_dates = [event.date for event in roll_events]
    if len(set(event_dates)) != len(event_dates):
        raise ValueError("Only one roll event is allowed per date")
    data_dates = set(data["date"].to_list())
    missing = sorted(set(event_dates) - data_dates)
    if missing:
        raise ValueError(f"Roll event dates are absent from the selected series: {missing}")
    ordered = sorted(roll_events, key=lambda event: event.date)
    for event in ordered:
        if event.old_symbol == event.new_symbol:
            raise ValueError(f"Roll event on {event.date} does not change contract")
        if not math.isfinite(event.old_close) or not math.isfinite(event.new_close):
            raise ValueError(f"Roll event on {event.date} contains a non-finite close")
        if "symbol" in data.columns:
            selected = data.filter(pl.col("date") == event.date)["symbol"].to_list()
            if selected != [event.new_symbol]:
                raise ValueError(
                    f"Selected contract on {event.date} must be '{event.new_symbol}', got {selected}"
                )
    for previous, current in zip(ordered, ordered[1:], strict=False):
        if previous.new_symbol != current.old_symbol:
            raise ValueError(
                f"Roll event chain changes from '{previous.new_symbol}' to "
                f"unrelated '{current.old_symbol}'"
            )


def _validate_price_columns(data: pl.DataFrame) -> None:
    required = {"date", *PRICE_COLUMNS}
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"Adjustment data is missing columns: {missing}")
