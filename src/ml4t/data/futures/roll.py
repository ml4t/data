"""Point-in-time futures contract selection and roll events."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any

import polars as pl

from ml4t.data.futures.schema import ContractSpec
from ml4t.data.sessions.assigner import SessionAssigner
from ml4t.specs import UNSET_CALENDAR, CalendarUnset, FeedSpec


@dataclass(frozen=True)
class RollEvent:
    """One contract switch with paired closes observed on the switch date."""

    date: date
    old_symbol: str
    new_symbol: str
    old_close: float
    new_close: float


class RollStrategy(ABC):
    """Select one identified contract after the configured point-in-time warm-up."""

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
        return build_roll_events(data, selections)


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
        *,
        tie_breaker_col: str = "symbol",
    ) -> pl.DataFrame:
        return _lagged_rank_selections(
            data,
            metric="volume",
            rank=0,
            minimum=0,
            confirmation=self.lookback_days,
            min_days_between_rolls=self.min_days_between_rolls,
            tie_breaker_col=tie_breaker_col,
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
        *,
        tie_breaker_col: str = "symbol",
    ) -> pl.DataFrame:
        return _lagged_rank_selections(
            data,
            metric="open_interest",
            rank=0,
            minimum=0,
            confirmation=self.lookback_days,
            min_days_between_rolls=0,
            tie_breaker_col=tie_breaker_col,
        )


class TimeBasedRoll(RollStrategy):
    """Select the nearest contract until its configured pre-expiry roll date.

    ``use_business_days=True`` counts authoritative sessions strictly before expiration.
    Explicit ``calendar`` overrides ``feed_spec``, then NYSE; an exchange identity does not
    select a calendar. A missing target session never moves the trigger to an earlier observation.
    ``use_business_days=False`` selects elapsed civil days, not an observation count.
    Calendar-free row-based expiry offsets are unsupported in session-count mode.
    Equal expirations use the declared tie_breaker_col,then contract identity;the default is symbol.
    """

    def __init__(
        self,
        days_before_expiration: int = 5,
        use_business_days: bool = True,
        *,
        calendar: str | None | CalendarUnset = UNSET_CALENDAR,
        feed_spec: Any | None = None,
    ) -> None:
        if days_before_expiration < 0:
            raise ValueError("days_before_expiration cannot be negative")
        self.days_before_expiration = days_before_expiration
        self.use_business_days = use_business_days
        self.feed_spec = FeedSpec.from_any(feed_spec)
        self.calendar = self.feed_spec.resolve_calendar(calendar=calendar)
        if self.calendar is None and use_business_days:
            raise ValueError("calendar-free row-based expiry rolls are not supported")
        self._schedule_calendar = (
            SessionAssigner(self.calendar).calendar if self.calendar is not None else None
        )

    def select_contracts(
        self,
        data: pl.DataFrame,
        contract_spec: ContractSpec | None = None,  # noqa: ARG002
        *,
        tie_breaker_col: str = "symbol",
    ) -> pl.DataFrame:
        _require_columns(data, "date", "symbol", "expiration")
        ties = _tie_values(data, tie_breaker_col)
        dates = _dates(data)
        if not dates:
            return _empty_selections()
        contracts = dict(_contract_expirations(data))
        roll_dates = {
            symbol: self._calculate_roll_date(expiration)
            for symbol, expiration in contracts.items()
        }
        available_by_date = _available_symbols_by_date(data)
        records: list[dict[str, object]] = []
        for observation_date in dates:
            eligible = [
                (symbol, contracts[symbol])
                for symbol in available_by_date[observation_date]
                if symbol in contracts and observation_date < roll_dates[symbol]
            ]
            if eligible:
                records.append(
                    {
                        "date": observation_date,
                        "symbol": min(
                            eligible,
                            key=lambda item: (item[1], ties[(observation_date, item[0])], item[0]),
                        )[0],
                    }
                )
        return _selection_frame(records)

    def _calculate_roll_date(self, expiration: date) -> date:
        """Calculate a schedule or explicit elapsed-day trigger independently of observations."""
        if not self.use_business_days:
            return expiration - timedelta(days=self.days_before_expiration)
        if self.days_before_expiration == 0:
            return expiration
        span = max(14, self.days_before_expiration * 2)
        while True:
            try:
                start = expiration - timedelta(days=span)
            except OverflowError as error:
                raise ValueError(
                    "not enough calendar history for the requested expiry offset"
                ) from error
            schedule = self._schedule_calendar.schedule(
                start_date=start, end_date=expiration - timedelta(days=1)
            )
            if len(schedule) >= self.days_before_expiration:
                return schedule.index[-self.days_before_expiration].date()
            span *= 2


class FirstNoticeDateRoll(RollStrategy):
    """Select contracts before supplied dated notice facts,or a legacy root-offset estimate.

    A first_notice_date column takes precedence over root estimates. Without actual dates,the
    existing ContractSpec fallback remains;zero notice days is distinct from an unknown offset.
    """

    def __init__(
        self,
        days_before_first_notice: int = 1,
        *,
        calendar: str | None | CalendarUnset = UNSET_CALENDAR,
        feed_spec: Any | None = None,
    ) -> None:
        if days_before_first_notice < 0:
            raise ValueError("days_before_first_notice cannot be negative")
        self.days_before_first_notice = days_before_first_notice
        self._time_roll = TimeBasedRoll(calendar=calendar, feed_spec=feed_spec)

    def select_contracts(
        self,
        data: pl.DataFrame,
        contract_spec: ContractSpec | None = None,
        *,
        tie_breaker_col: str = "symbol",
    ) -> pl.DataFrame:
        if "first_notice_date" in data.columns and data["first_notice_date"].is_not_null().any():
            _require_columns(data, "date", "symbol", "expiration")
            if data.schema["first_notice_date"] != pl.Date:
                raise ValueError("first_notice_date must contain explicit civil dates")
            expirations = dict(_contract_expirations(data))
            ties = _tie_values(data, tie_breaker_col)
            notice_offset = TimeBasedRoll(
                self.days_before_first_notice,
                calendar=self._time_roll.calendar,
                feed_spec=self._time_roll.feed_spec,
            )
            triggers = {}
            for symbol, expiration in expirations.items():
                notices = data.filter(pl.col("symbol") == symbol)["first_notice_date"].unique()
                if len(notices) != 1:
                    raise ValueError(f"Contract {symbol} has conflicting first-notice dates")
                notice = notices[0]
                triggers[symbol] = (
                    self._time_roll._calculate_roll_date(expiration)
                    if notice is None
                    else notice_offset._calculate_roll_date(notice)
                )
            available = _available_symbols_by_date(data)
            records = []
            for observation_date in _dates(data):
                eligible = [
                    symbol
                    for symbol in available[observation_date]
                    if symbol in triggers and observation_date < triggers[symbol]
                ]
                if eligible:
                    selected = min(
                        eligible,
                        key=lambda symbol: (
                            expirations[symbol],
                            ties[(observation_date, symbol)],
                            symbol,
                        ),
                    )
                    records.append({"date": observation_date, "symbol": selected})
            return _selection_frame(records)
        if contract_spec is None or contract_spec.is_cash_settled:
            return self._time_roll.select_contracts(
                data, contract_spec, tie_breaker_col=tie_breaker_col
            )
        notice_days = (
            25 if contract_spec.first_notice_days is None else contract_spec.first_notice_days
        )
        days = notice_days + self.days_before_first_notice
        return TimeBasedRoll(
            days_before_expiration=days,
            calendar=self._time_roll.calendar,
            feed_spec=self._time_roll.feed_spec,
        ).select_contracts(data, contract_spec, tie_breaker_col=tie_breaker_col)


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
        *,
        tie_breaker_col: str = "symbol",
    ) -> pl.DataFrame:
        _require_columns(data, "date", "symbol", "expiration")
        tie_columns = _tie_columns(data, tie_breaker_col)
        records = []
        for observation_date in _dates(data):
            ranked = (
                data.filter(
                    (pl.col("date") == observation_date)
                    & pl.col("expiration").is_not_null()
                    & (pl.col("expiration") > observation_date)
                )
                .sort(["expiration", *tie_columns])
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
        *,
        tie_breaker_col: str = "symbol",
    ) -> pl.DataFrame:
        return _lagged_rank_selections(
            data,
            metric="volume",
            rank=self.rank,
            minimum=self.min_volume,
            confirmation=1,
            min_days_between_rolls=0,
            tie_breaker_col=tie_breaker_col,
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
        *,
        tie_breaker_col: str = "symbol",
    ) -> pl.DataFrame:
        return _lagged_rank_selections(
            data,
            metric="open_interest",
            rank=self.rank,
            minimum=self.min_oi,
            confirmation=1,
            min_days_between_rolls=0,
            tie_breaker_col=tie_breaker_col,
        )


def _tie_columns(data: pl.DataFrame, column: str) -> list[str]:
    _require_columns(data, column)
    if data[column].null_count():
        raise ValueError("Contract tie-breaker values must be known")
    return list(dict.fromkeys([column, "symbol"]))


def _tie_values(data: pl.DataFrame, column: str) -> dict[tuple[date, str], Any]:
    _require_columns(data, "date", "symbol")
    _tie_columns(data, column)
    facts = data.select("date", "symbol", pl.col(column).alias("__tie_breaker")).unique()
    if facts.select("date", "symbol").is_duplicated().any():
        raise ValueError("Conflicting tie-breaker facts for one contract observation")
    return {
        (observation_date, symbol): value for observation_date, symbol, value in facts.iter_rows()
    }


def _lagged_rank_selections(
    data: pl.DataFrame,
    *,
    metric: str,
    rank: int,
    minimum: float,
    confirmation: int,
    min_days_between_rolls: int,
    tie_breaker_col: str,
) -> pl.DataFrame:
    _require_columns(data, "date", "symbol", metric)
    tie_columns = _tie_columns(data, tie_breaker_col)
    non_null = data.filter(pl.col(metric).is_not_null())
    if non_null.is_empty() and metric == "open_interest":
        raise ValueError("No open interest data available")
    valid = non_null.filter(pl.col(metric) >= minimum)
    if valid.is_empty():
        return _empty_selections()

    observation_dates = _dates(data)
    ranked = (
        valid.sort(
            ["date", metric, *tie_columns], descending=[False, True, *[False] * len(tie_columns)]
        )
        .group_by("date", maintain_order=True)
        .agg(pl.col("symbol"))
    )
    rankings = {row["date"]: row["symbol"] for row in ranked.iter_rows(named=True)}
    leaders = [
        rankings[ranking_date][rank] if len(rankings.get(ranking_date, [])) > rank else None
        for ranking_date in observation_dates
    ]
    available_by_date = _available_symbols_by_date(data)

    records = []
    selected: str | None = None
    last_roll: date | None = None
    for index in range(confirmation, len(observation_dates)):
        history = leaders[index - confirmation : index]
        candidate = history[0] if history[0] is not None and len(set(history)) == 1 else None
        if candidate is not None:
            effective_date = observation_dates[index]
            available = available_by_date[effective_date]
            if selected is None:
                prior_ranking = rankings.get(observation_dates[index - 1], [])
                selected = next(
                    (symbol for symbol in prior_ranking[rank:] if symbol in available),
                    None,
                )
            elif (
                candidate != selected
                and (
                    last_roll is None or (effective_date - last_roll).days >= min_days_between_rolls
                )
                and candidate in available
            ):
                selected = candidate
                last_roll = effective_date
        if selected is not None:
            effective_date = observation_dates[index]
            if selected not in available_by_date[effective_date]:
                raise ValueError(
                    f"Selection on {effective_date} cannot use contract '{selected}': "
                    "it was not observed on that date"
                )
            records.append({"date": effective_date, "symbol": selected})
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


def build_roll_events(data: pl.DataFrame, selections: pl.DataFrame) -> list[RollEvent]:
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


def _available_symbols_by_date(data: pl.DataFrame) -> dict[date, set[str]]:
    _require_columns(data, "date", "symbol")
    available = {observation_date: set() for observation_date in _dates(data)}
    for observation_date, symbol in data.select("date", "symbol").unique().iter_rows():
        available[observation_date].add(symbol)
    return available


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
