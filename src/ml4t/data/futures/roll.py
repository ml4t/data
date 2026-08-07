"""Point-in-time futures contract selection and roll events."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import date, timedelta

import polars as pl

from ml4t.data.futures.schema import ContractSpec


@dataclass(frozen=True)
class RollEvent:
    """One contract switch with paired closes observed on the switch date."""

    date: date
    old_symbol: str
    new_symbol: str
    old_close: float
    new_close: float


class RollStrategy(ABC):
    """Select one identified contract for each observation date."""

    @abstractmethod
    def select_contracts(
        self, data: pl.DataFrame, contract_spec: ContractSpec | None = None
    ) -> pl.DataFrame:
        """Return unique ``date`` and ``symbol`` selections in chronological order."""

    def identify_rolls(
        self, data: pl.DataFrame, contract_spec: ContractSpec | None = None
    ) -> list[date]:
        """Return switch dates as a compatibility view of the symbol selections."""
        return [
            event_date
            for event_date, _, _ in _selection_changes(self.select_contracts(data, contract_spec))
        ]

    def identify_roll_events(
        self, data: pl.DataFrame, contract_spec: ContractSpec | None = None
    ) -> list[RollEvent]:
        """Return switches with old and new contract closes from the same date."""
        selections = self.select_contracts(data, contract_spec)
        return _build_roll_events(data, selections)


class VolumeBasedRoll(RollStrategy):
    """Select by confirmed previous-observation volume rank."""

    def __init__(self, lookback_days: int = 1, min_days_between_rolls: int = 20):
        if lookback_days < 1:
            raise ValueError("lookback_days must be at least 1")
        if min_days_between_rolls < 0:
            raise ValueError("min_days_between_rolls cannot be negative")
        self.lookback_days = lookback_days
        self.min_days_between_rolls = min_days_between_rolls

    def select_contracts(
        self,
        data: pl.DataFrame,
        contract_spec: ContractSpec | None = None,  # noqa: ARG002
    ) -> pl.DataFrame:
        return _lagged_rank_selections(
            data,
            metric="volume",
            rank=0,
            minimum=0,
            confirmation=self.lookback_days,
            min_days_between_rolls=self.min_days_between_rolls,
        )


class OpenInterestBasedRoll(RollStrategy):
    """Select by confirmed previous-observation closing open-interest rank."""

    def __init__(self, lookback_days: int = 1):
        if lookback_days < 1:
            raise ValueError("lookback_days must be at least 1")
        self.lookback_days = lookback_days

    def select_contracts(
        self,
        data: pl.DataFrame,
        contract_spec: ContractSpec | None = None,  # noqa: ARG002
    ) -> pl.DataFrame:
        return _lagged_rank_selections(
            data,
            metric="open_interest",
            rank=0,
            minimum=0,
            confirmation=self.lookback_days,
            min_days_between_rolls=0,
        )


class TimeBasedRoll(RollStrategy):
    """Select the nearest contract until its configured pre-expiry roll date."""

    def __init__(
        self,
        days_before_expiration: int = 5,
        use_business_days: bool = True,
    ):
        if days_before_expiration < 0:
            raise ValueError("days_before_expiration cannot be negative")
        self.days_before_expiration = days_before_expiration
        self.use_business_days = use_business_days

    def select_contracts(
        self,
        data: pl.DataFrame,
        contract_spec: ContractSpec | None = None,  # noqa: ARG002
    ) -> pl.DataFrame:
        _require_columns(data, "date", "symbol", "expiration")
        dates = _dates(data)
        if not dates:
            return _empty_selections()
        contracts = _contract_expirations(data)
        records = []
        for observation_date in dates:
            eligible = [
                (symbol, expiration)
                for symbol, expiration in contracts
                if observation_date < self._calculate_roll_date(expiration, dates)
            ]
            if eligible:
                records.append(
                    {"date": observation_date, "symbol": min(eligible, key=lambda item: item[1])[0]}
                )
        return _selection_frame(records)

    def _calculate_roll_date(self, expiration: date, trading_dates: list[date]) -> date:
        """Calculate the last configured roll date on the observed calendar."""
        target = expiration
        if self.use_business_days:
            days_counted = 0
            while days_counted < self.days_before_expiration:
                target -= timedelta(days=1)
                if target.weekday() < 5:
                    days_counted += 1
        else:
            target -= timedelta(days=self.days_before_expiration)
        observed = [trading_date for trading_date in trading_dates if trading_date <= target]
        return max(observed) if observed else target


class FirstNoticeDateRoll(RollStrategy):
    """Select physical contracts before the configured first-notice interval."""

    def __init__(self, days_before_first_notice: int = 1):
        if days_before_first_notice < 0:
            raise ValueError("days_before_first_notice cannot be negative")
        self.days_before_first_notice = days_before_first_notice

    def select_contracts(
        self, data: pl.DataFrame, contract_spec: ContractSpec | None = None
    ) -> pl.DataFrame:
        if contract_spec is None or contract_spec.is_cash_settled:
            return TimeBasedRoll(days_before_expiration=5).select_contracts(data, contract_spec)
        days = (contract_spec.first_notice_days or 25) + self.days_before_first_notice
        return TimeBasedRoll(days_before_expiration=days).select_contracts(data, contract_spec)


class CalendarRoll(RollStrategy):
    """Select the requested unexpired contract rank by expiration."""

    def __init__(self, rank: int = 0):
        if rank < 0:
            raise ValueError("rank cannot be negative")
        self.rank = rank

    def select_contracts(
        self,
        data: pl.DataFrame,
        contract_spec: ContractSpec | None = None,  # noqa: ARG002
    ) -> pl.DataFrame:
        _require_columns(data, "date", "symbol", "expiration")
        records = []
        for observation_date in _dates(data):
            ranked = (
                data.filter(
                    (pl.col("date") == observation_date)
                    & pl.col("expiration").is_not_null()
                    & (pl.col("expiration") > observation_date)
                )
                .sort(["expiration", "symbol"])
                .select("symbol")
            )
            if ranked.height > self.rank:
                records.append({"date": observation_date, "symbol": ranked["symbol"][self.rank]})
        return _selection_frame(records)


class HighestVolumeRoll(RollStrategy):
    """Select by previous-observation volume, matching Databento's volume rule."""

    def __init__(self, rank: int = 0, min_volume: float = 0):
        if rank < 0:
            raise ValueError("rank cannot be negative")
        if min_volume < 0:
            raise ValueError("min_volume cannot be negative")
        self.rank = rank
        self.min_volume = min_volume

    def select_contracts(
        self,
        data: pl.DataFrame,
        contract_spec: ContractSpec | None = None,  # noqa: ARG002
    ) -> pl.DataFrame:
        return _lagged_rank_selections(
            data,
            metric="volume",
            rank=self.rank,
            minimum=self.min_volume,
            confirmation=1,
            min_days_between_rolls=0,
        )


class HighestOpenInterestRoll(RollStrategy):
    """Select by previous-observation closing open interest."""

    def __init__(self, rank: int = 0, min_oi: float = 0):
        if rank < 0:
            raise ValueError("rank cannot be negative")
        if min_oi < 0:
            raise ValueError("min_oi cannot be negative")
        self.rank = rank
        self.min_oi = min_oi

    def select_contracts(
        self,
        data: pl.DataFrame,
        contract_spec: ContractSpec | None = None,  # noqa: ARG002
    ) -> pl.DataFrame:
        return _lagged_rank_selections(
            data,
            metric="open_interest",
            rank=self.rank,
            minimum=self.min_oi,
            confirmation=1,
            min_days_between_rolls=0,
        )


def _lagged_rank_selections(
    data: pl.DataFrame,
    *,
    metric: str,
    rank: int,
    minimum: float,
    confirmation: int,
    min_days_between_rolls: int,
) -> pl.DataFrame:
    _require_columns(data, "date", "symbol", metric)
    non_null = data.filter(pl.col(metric).is_not_null())
    if non_null.is_empty() and metric == "open_interest":
        raise ValueError("No open interest data available")
    valid = non_null.filter(pl.col(metric) >= minimum)
    if valid.is_empty():
        return _empty_selections()

    observation_dates = _dates(data)
    leaders: list[str | None] = []
    for ranking_date in observation_dates:
        ranked = (
            valid.filter(pl.col("date") == ranking_date)
            .sort([metric, "symbol"], descending=[True, False])
            .select("symbol")
        )
        leaders.append(ranked["symbol"][rank] if ranked.height > rank else None)

    records = []
    selected: str | None = None
    last_roll: date | None = None
    for index in range(confirmation, len(observation_dates)):
        history = leaders[index - confirmation : index]
        candidate = history[0] if history[0] is not None and len(set(history)) == 1 else None
        if candidate is not None:
            effective_date = observation_dates[index]
            if selected is None:
                selected = candidate
            elif candidate != selected and (
                last_roll is None or (effective_date - last_roll).days >= min_days_between_rolls
            ):
                selected = candidate
                last_roll = effective_date
        if selected is not None:
            records.append({"date": observation_dates[index], "symbol": selected})
    return _selection_frame(records)


def _selection_changes(selections: pl.DataFrame) -> list[tuple[date, str, str]]:
    if selections.is_empty():
        return []
    ordered = selections.sort("date")
    changes = []
    previous_symbol = ordered["symbol"][0]
    for row in ordered.iter_rows(named=True):
        symbol = row["symbol"]
        if symbol != previous_symbol:
            changes.append((row["date"], previous_symbol, symbol))
        previous_symbol = symbol
    return changes


def _build_roll_events(data: pl.DataFrame, selections: pl.DataFrame) -> list[RollEvent]:
    _require_columns(data, "date", "symbol", "close")
    events = []
    for roll_date, old_symbol, new_symbol in _selection_changes(selections):
        old_close = _paired_close(data, roll_date, old_symbol)
        new_close = _paired_close(data, roll_date, new_symbol)
        events.append(
            RollEvent(
                date=roll_date,
                old_symbol=old_symbol,
                new_symbol=new_symbol,
                old_close=old_close,
                new_close=new_close,
            )
        )
    return events


def _paired_close(data: pl.DataFrame, roll_date: date, symbol: str) -> float:
    rows = data.filter((pl.col("date") == roll_date) & (pl.col("symbol") == symbol))
    if rows.height != 1:
        raise ValueError(
            f"Roll on {roll_date} requires exactly one close for contract '{symbol}', "
            f"found {rows.height}"
        )
    value = rows["close"].item()
    if value is None or not math.isfinite(float(value)):
        raise ValueError(f"Roll on {roll_date} has an invalid close for contract '{symbol}'")
    return float(value)


def _contract_expirations(data: pl.DataFrame) -> list[tuple[str, date]]:
    contracts = (
        data.select("symbol", "expiration")
        .filter(pl.col("expiration").is_not_null())
        .group_by("symbol")
        .agg(
            pl.col("expiration").n_unique().alias("expiration_count"),
            pl.col("expiration").first().alias("expiration"),
        )
        .sort(["expiration", "symbol"])
    )
    conflicts = contracts.filter(pl.col("expiration_count") != 1)["symbol"].to_list()
    if conflicts:
        raise ValueError(f"Contracts have conflicting expiration dates: {conflicts}")
    return [(row["symbol"], row["expiration"]) for row in contracts.iter_rows(named=True)]


def _dates(data: pl.DataFrame) -> list[date]:
    if "date" not in data.columns:
        raise ValueError("Data must have 'date' column")
    return sorted(data.select("date").unique().to_series().to_list())


def _require_columns(data: pl.DataFrame, *columns: str) -> None:
    missing = [column for column in columns if column not in data.columns]
    if missing:
        raise ValueError(f"Data must have columns: {', '.join(missing)}")


def _selection_frame(records: list[dict[str, object]]) -> pl.DataFrame:
    if not records:
        return _empty_selections()
    frame = pl.DataFrame(records, schema={"date": pl.Date, "symbol": pl.String})
    if frame.select("date").n_unique() != frame.height:
        raise ValueError("Roll strategy selected more than one contract for a date")
    return frame.sort("date")


def _empty_selections() -> pl.DataFrame:
    return pl.DataFrame(schema={"date": pl.Date, "symbol": pl.String})
