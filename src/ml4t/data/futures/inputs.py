"""Versioned individual-contract facts and source-known selection inputs,not execution."""

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal

import pandas as pd
import polars as pl

from ml4t.data.assets import InstrumentRegistry
from ml4t.data.core.feeds import resolve_feed_spec
from ml4t.data.core.revisions import (
    native_frame,
    select_revisions,
    session_label_at,
    utc_column,
    utc_cutoff,
    utc_stamp,
)
from ml4t.data.futures.roll import (
    FirstNoticeDateRoll,
    HighestOpenInterestRoll,
    HighestVolumeRoll,
    OpenInterestBasedRoll,
    RollStrategy,
    TimeBasedRoll,
    VolumeBasedRoll,
    build_roll_events,
)
from ml4t.data.futures.schema import price_conversion_factor
from ml4t.specs import UNSET_CALENDAR, CalendarUnset, FeedSpec

_DATE = "__ml4t_futures_date"
_ID = "__ml4t_futures_id"
_DEFINITION_FIELDS = (
    "root",
    "exchange",
    "currency",
    "multiplier",
    "tick_size",
    "price_quote_unit",
    "settlement_type",
    "expiration",
    "first_notice_date",
    "last_trade_date",
    "auto_close_date",
    "volume_unit",
    "open_interest_unit",
    "source",
    "revision_id",
    "available_at",
)


@dataclass(frozen=True)
class FuturesInputs:
    """Source bars,dated definitions,eligible metric facts and one current decision per chain.

    Bars retain their raw source fields and field-specific availability declarations. A later OI
    publication in a captured bar is not eligible merely because its price was already available.
    Only ``metrics`` and the derived selection use that field's known-at cutoff. Roll proposals
    contain actual raw paired observations,not continuous prices,execution fills or account policy.
    """

    bars: pl.DataFrame
    definitions: pl.DataFrame
    chain: pl.DataFrame
    metrics: pl.DataFrame
    selections: pl.DataFrame
    rolls: pl.DataFrame
    coverage: pl.DataFrame
    metadata: dict[str, Any]


def _dated_bars(frame: pl.DataFrame, feed: FeedSpec) -> pl.DataFrame:
    if any(name.startswith("__ml4t_futures_") for name in frame.columns):
        raise ValueError("Source fields collide with internal futures keys")
    column = feed.timestamp_col
    if column not in frame.columns or frame[column].null_count():
        raise ValueError("Futures bars require known source timestamps")
    if frame.schema[column] == pl.Date:
        if feed.timestamp_semantics != "session_label":
            raise ValueError("Declare session_label semantics for Date contract observations")
        return frame.with_columns(pl.col(column).alias(_DATE))
    frame = utc_column(frame, column, feed.timezone)
    point = pl.col(column)
    if feed.timezone is not None:
        point = point.dt.convert_time_zone(feed.timezone)
    return frame.with_columns(point.dt.date().alias(_DATE))


def _contract_definitions(frame: pl.DataFrame) -> pl.DataFrame:
    required = {
        "root",
        "exchange",
        "instrument_type",
        "currency",
        "multiplier",
        "tick_size",
        "price_quote_unit",
        "settlement_type",
        "expiration",
    }
    if missing := required.difference(frame.columns):
        raise ValueError(f"Missing required contract facts: {sorted(missing)}")
    for name in required:
        if frame[name].null_count() or (
            frame.schema[name] == pl.String and frame[name].str.strip_chars().eq("").any()
        ):
            raise ValueError(f"Required contract fact {name} must be known")
    if not frame["instrument_type"].eq("future").all():
        raise ValueError("Declare future instrument_type;do not inherit equity defaults")
    for name in ("multiplier", "tick_size"):
        if (
            not frame.schema[name].is_numeric()
            or frame.filter(~pl.col(name).is_finite() | (pl.col(name) <= 0)).height
        ):
            raise ValueError(f"Contract {name} must be positive and finite")
    if frame.schema["expiration"] != pl.Date:
        raise ValueError("Contract expiration must be an explicit civil Date")
    for name in ("first_notice_date", "last_trade_date", "auto_close_date"):
        if name not in frame.columns or frame.schema[name] == pl.Null:
            frame = frame.with_columns(pl.lit(None, dtype=pl.Date).alias(name))
        elif frame.schema[name] != pl.Date:
            raise ValueError(f"{name} must be an explicit civil Date")
    for name in ("volume_unit", "open_interest_unit"):
        if name not in frame.columns:
            frame = frame.with_columns(pl.lit(None, dtype=pl.String).alias(name))
    if not frame["settlement_type"].is_in(["cash", "physical"]).all():
        raise ValueError("Declare cash or physical settlement_type")
    return frame


def _normalized_prices(frame: pl.DataFrame, feed: FeedSpec, target: str | None) -> pl.DataFrame:
    for name in ("price_quote_unit", "currency", "mark_kind"):
        if name not in frame.columns or frame[name].null_count():
            raise ValueError(f"Raw bars require declared {name};do not infer source units")
    if frame.filter(pl.col("currency") != pl.col("instrument_currency")).height:
        raise ValueError("Known bar and contract currency units conflict")
    outputs = {
        "price_conversion_factor",
        "normalized_multiplier",
        "normalized_tick_size",
        "normalized_price_quote_unit",
    } | {
        f"normalized_{getattr(feed, name)}"
        for name in ("open_col", "high_col", "low_col", "close_col", "price_col")
    }
    if outputs.intersection(frame.columns):
        raise ValueError("Normalized outputs collide with raw source fields")
    price_scale, contract_scale, units = [], [], []
    for source, native in frame.select(
        "price_quote_unit", "instrument_price_quote_unit"
    ).iter_rows():
        output = source if target is None else target
        price_scale.append(price_conversion_factor(source, output))
        contract_scale.append(price_conversion_factor(native, output))
        units.append(output)
    frame = (
        frame.with_columns(
            pl.Series("price_conversion_factor", price_scale, dtype=pl.Float64),
            pl.Series("__ml4t_futures_contract_scale", contract_scale, dtype=pl.Float64),
            pl.Series("normalized_price_quote_unit", units, dtype=pl.String),
        )
        .with_columns(
            (pl.col("instrument_multiplier") / pl.col("__ml4t_futures_contract_scale")).alias(
                "normalized_multiplier"
            ),
            (pl.col("instrument_tick_size") * pl.col("__ml4t_futures_contract_scale")).alias(
                "normalized_tick_size"
            ),
        )
        .drop("__ml4t_futures_contract_scale")
    )
    prices = list(
        dict.fromkeys(
            name
            for name in (feed.open_col, feed.high_col, feed.low_col, feed.close_col, feed.price_col)
            if name in frame.columns
        )
    )
    if feed.price_col not in prices:
        raise ValueError("The declared reference price is missing")
    for name in prices:
        if (
            not frame.schema[name].is_numeric()
            or frame.filter(pl.col(name).is_not_null() & ~pl.col(name).is_finite()).height
        ):
            raise ValueError("Known raw contract prices must be finite numeric observations")
    frame = frame.with_columns(
        (pl.col(name) * pl.col("price_conversion_factor")).alias(f"normalized_{name}")
        for name in prices
    )
    if frame.filter(
        pl.any_horizontal(
            ~pl.col(name).is_finite() & pl.col(name).is_not_null()
            for name in [
                "normalized_multiplier",
                "normalized_tick_size",
                *(f"normalized_{n}" for n in prices),
            ]
        )
    ).height:
        raise ValueError("Unit normalization produced nonfinite derived values")
    return frame


def _metric_facts(
    frame: pl.DataFrame,
    feed: FeedSpec,
    as_of: datetime | str | None,
    requested: str | None,
) -> pl.DataFrame:
    asset = FeedSpec._coerce_entity_col(feed.entity_col) or "asset"
    parts = []
    for metric in ("volume", "open_interest"):
        physical = feed.volume_col if metric == "volume" else metric
        if physical not in frame.columns:
            if requested == metric:
                raise ValueError(f"Missing required {metric} observations")
            continue
        available = f"{metric}_available_at"
        observed = frame.filter(pl.col(physical).is_not_null())
        if (
            not observed.schema[physical].is_numeric()
            or observed.filter(~pl.col(physical).is_finite() | (pl.col(physical) < 0)).height
        ):
            raise ValueError(f"Known {metric} must be nonnegative and finite")
        if observed.filter(pl.col(f"instrument_{metric}_unit") != "contracts").height or (
            observed[f"instrument_{metric}_unit"].null_count()
        ):
            if requested == metric:
                raise ValueError(f"Declare {metric} in contracts,not inferred units")
            continue
        if metric == "volume" and available not in observed.columns:
            observed = observed.with_columns(pl.col("available_at").alias(available))
        observed = utc_column(observed, available, feed.timezone)
        if as_of is not None:
            if observed[available].null_count() and requested == metric:
                raise ValueError(f"Unknown {available} cannot qualify a causal selection")
            observed = observed.filter(pl.col(available) <= utc_cutoff(as_of))
        observed = observed.with_columns(
            pl.col("available_at").alias("capture_available_at"),
            pl.col(available).alias("available_at"),
        ).select(
            feed.timestamp_col,
            asset,
            _DATE,
            _ID,
            "source",
            "revision_id",
            "available_at",
            "capture_available_at",
            pl.lit(metric).alias("metric"),
            pl.col(physical).alias(metric),
        )
        observed = select_revisions(observed, [asset, feed.timestamp_col, "source"], as_of)
        if observed.select(asset, feed.timestamp_col).is_duplicated().any():
            raise ValueError("Choose one source metric stream per contract observation")
        parts.append(observed)
    if not parts:
        return (
            frame.select(
                feed.timestamp_col, asset, _DATE, _ID, "source", "revision_id", "available_at"
            )
            .head(0)
            .with_columns(
                pl.lit(None, dtype=pl.String).alias("metric"),
                pl.lit(None, dtype=pl.Float64).alias("volume"),
                pl.lit(None, dtype=pl.Float64).alias("open_interest"),
            )
        )
    return pl.concat(parts, how="diagonal")


def prepare_futures_inputs(
    bars: pl.DataFrame | pd.DataFrame,
    definitions: pl.DataFrame | pd.DataFrame,
    *,
    as_of: datetime | str | None,
    roll_strategy: RollStrategy | None = None,
    feed_spec: Any | None = None,
    calendar: str | None | CalendarUnset = UNSET_CALENDAR,
    price_quote_unit: str | None = None,
    definitions_feed_spec: Any | None = None,
    identity_mode: Literal["stable_id", "symbol"] = "stable_id",
) -> FuturesInputs:
    """Prepare individual source bars and one current research selection per root/venue.

    A causal call uses only source revisions and ranking metrics published by as_of. Call again
    at each actual decision cutoff for historical decisions;this does not backdate a current
    source-vintage selection into previously executed account history. The existing strategy's
    observed-session lag and confirmation defaults remain unchanged. Bars are unadjusted captures;
    metrics carries separately eligible OI/volume facts,not a guarantee that every raw bar field
    was published together. Missing OI availability is never copied from a price timestamp.
    """
    feed = resolve_feed_spec(
        FeedSpec(entity_col="asset", time_contract_version=2) if feed_spec is None else feed_spec
    )
    if calendar is not UNSET_CALENDAR:
        feed = feed.with_overrides(calendar=calendar, time_contract_version=2)
    selected_calendar = feed.resolve_calendar()
    registry_feed = (
        feed if definitions_feed_spec is None else resolve_feed_spec(definitions_feed_spec)
    )
    facts = _contract_definitions(native_frame(definitions))
    registry = InstrumentRegistry.from_frame(facts, feed_spec=registry_feed)
    asset = FeedSpec._coerce_entity_col(feed.entity_col) or "asset"
    source = _dated_bars(native_frame(bars), feed)
    source = utc_column(source, "available_at", feed.timezone)
    if as_of is not None:
        source = source.filter(pl.col(_DATE) <= pl.lit(session_label_at(as_of, feed)))
    source = select_revisions(source, [asset, feed.timestamp_col, "source"], as_of)
    source, used, asset = registry._normalize(
        source,
        feed,
        as_of,
        identity_mode,
        include_fields=_DEFINITION_FIELDS,
    )
    feed = feed.with_overrides(entity_col=asset)
    source = source.with_columns(pl.col(asset).cast(pl.String).alias(_ID))
    if source.select(asset, _DATE).is_duplicated().any():
        raise ValueError("Roll inputs require one declared source observation per contract/session")
    normalized = _normalized_prices(source, feed, price_quote_unit)
    strategy = roll_strategy or VolumeBasedRoll()
    if type(strategy) is TimeBasedRoll:
        strategy = TimeBasedRoll(
            strategy.days_before_expiration,
            strategy.use_business_days,
            calendar=selected_calendar,
            feed_spec=feed,
        )
    elif type(strategy) is FirstNoticeDateRoll:
        strategy = FirstNoticeDateRoll(
            strategy.days_before_first_notice,
            calendar=selected_calendar,
            feed_spec=feed,
        )
        if normalized.filter(
            (pl.col("instrument_settlement_type") == "physical")
            & pl.col("instrument_first_notice_date").is_null()
        ).height:
            raise ValueError(
                "Dated physical notice selection requires actual first_notice_date facts"
            )
    requested = (
        "open_interest"
        if isinstance(strategy, HighestOpenInterestRoll | OpenInterestBasedRoll)
        else "volume"
        if isinstance(strategy, HighestVolumeRoll | VolumeBasedRoll)
        else None
    )
    metrics = _metric_facts(normalized, feed, as_of, requested)
    decisions, proposals = [], []
    for group in normalized.partition_by(
        ["instrument_root", "instrument_exchange"], maintain_order=True
    ):
        projection = group.select(
            pl.col(_DATE).alias("date"),
            pl.col(_ID).alias("symbol"),
            pl.col("instrument_expiration").alias("expiration"),
            pl.col("instrument_first_notice_date").alias("first_notice_date"),
            pl.col(feed.price_col).alias("close"),
        )
        for metric in ("volume", "open_interest"):
            values = metrics.filter(pl.col("metric") == metric)
            if metric in values.columns:
                projection = projection.join(
                    values.select(pl.col(_DATE).alias("date"), pl.col(_ID).alias("symbol"), metric),
                    on=["date", "symbol"],
                    how="left",
                    validate="1:1",
                )
            else:
                projection = projection.with_columns(pl.lit(None, dtype=pl.Float64).alias(metric))
        selections = strategy.select_contracts(projection)
        if selections.height == 0:
            continue
        current = selections.tail(1)
        chosen = group.filter(
            (pl.col(_ID) == current["symbol"][0]) & (pl.col(_DATE) == current["date"][0])
        )
        if chosen.height != 1:
            raise ValueError("Current selection must have exactly one raw source observation")
        decision = chosen.select(
            pl.col("instrument_root").alias("root"),
            pl.col("instrument_exchange").alias("exchange"),
            pl.col(asset).alias("selected_instrument_id"),
            pl.col(_DATE).alias("observation_label"),
            pl.col("instrument_source").alias("definition_source"),
            pl.col("instrument_revision_id").alias("definition_revision_id"),
        )
        if requested is not None:
            prior_dates = projection.filter(pl.col("date") < current["date"][0])["date"]
            prior = None if not len(prior_dates) else prior_dates.max()
            ranked = metrics.filter(
                (pl.col("metric") == requested)
                & (pl.col(_DATE) == prior)
                & pl.col(_ID).is_in(group[_ID].unique().implode())
            )
            clock = ranked.select(
                pl.col(feed.timestamp_col).max().alias("metric_observed_at"),
                pl.col("available_at").max().alias("metric_available_at"),
            )
            decision = decision.join(clock, how="cross")
        else:
            decision = decision.with_columns(
                pl.lit(None, dtype=pl.Date).alias("metric_observed_at"),
                pl.lit(None, dtype=pl.Datetime("ns", "UTC")).alias("metric_available_at"),
            )
        point = pl.lit(None, dtype=pl.Datetime("ns", "UTC")) if as_of is None else utc_cutoff(as_of)
        decisions.append(
            decision.with_columns(
                point.alias("decision_at"),
                point.alias("effective_at"),
                pl.lit(type(strategy).__name__).alias("selection_method"),
            )
        )
        for event in build_roll_events(projection, selections.tail(2)):
            old = group.filter((pl.col(_ID) == event.old_symbol) & (pl.col(_DATE) == event.date))
            new = group.filter((pl.col(_ID) == event.new_symbol) & (pl.col(_DATE) == event.date))
            if old["mark_kind"][0] != new["mark_kind"][0]:
                raise ValueError("Paired roll marks have conflicting source mark kinds")
            proposals.append(
                old.select(
                    pl.col("instrument_root").alias("root"),
                    pl.col(asset).alias("old_instrument_id"),
                    pl.col(feed.price_col).alias("old_mark"),
                    pl.col(feed.timestamp_col).alias("old_mark_observed_at"),
                    pl.col("available_at").alias("old_mark_available_at"),
                    pl.col("price_quote_unit").alias("old_price_quote_unit"),
                    "mark_kind",
                )
                .join(
                    new.select(
                        pl.col(asset).alias("new_instrument_id"),
                        pl.col(feed.price_col).alias("new_mark"),
                        pl.col(feed.timestamp_col).alias("new_mark_observed_at"),
                        pl.col("available_at").alias("new_mark_available_at"),
                        pl.col("price_quote_unit").alias("new_price_quote_unit"),
                    ),
                    how="cross",
                )
                .with_columns(point.alias("decision_at"), point.alias("effective_at"))
            )
    selections_frame = (
        pl.concat(decisions)
        if decisions
        else pl.DataFrame(
            schema={
                "root": pl.String,
                "exchange": pl.String,
                "selected_instrument_id": source.schema[asset],
                "decision_at": pl.Datetime("ns", "UTC"),
                "effective_at": pl.Datetime("ns", "UTC"),
            }
        )
    )
    rolls_frame = (
        pl.concat(proposals)
        if proposals
        else pl.DataFrame(
            schema={
                "old_instrument_id": source.schema[asset],
                "new_instrument_id": source.schema[asset],
                "old_mark": source.schema[feed.price_col],
                "new_mark": source.schema[feed.price_col],
            }
        )
    )
    chain = used.select(
        asset,
        "symbol",
        "root",
        "exchange",
        "valid_from",
        "valid_to",
        "source",
        "revision_id",
        "available_at",
    )
    coverage = normalized.group_by(asset, maintain_order=True).agg(pl.len().alias("observed_rows"))
    return FuturesInputs(
        bars=normalized.drop(_DATE, _ID),
        definitions=used,
        chain=chain,
        metrics=metrics.drop(_DATE, _ID),
        selections=selections_frame,
        rolls=rolls_frame,
        coverage=coverage,
        metadata={
            "feed_spec": feed.to_dict(),
            "calendar": selected_calendar,
            "price_basis": "raw",
            "perspective": "retrospective" if as_of is None else "causal_current_decision",
            "as_of": None if as_of is None else utc_stamp(as_of).isoformat(),
            "selection_scope": "one current source-vintage decision per root/venue,not backdated account history",
            "selection_policy": {
                "kind": f"{type(strategy).__module__}.{type(strategy).__name__}",
                "parameters": {
                    name: getattr(strategy, name)
                    for name in (
                        "rank",
                        "min_volume",
                        "min_oi",
                        "lookback_days",
                        "min_days_between_rolls",
                        "days_before_expiration",
                        "use_business_days",
                        "days_before_first_notice",
                    )
                    if hasattr(strategy, name)
                },
                "calendar": selected_calendar,
            },
            "raw_metric_availability": "raw bar fields retain source values;only metrics qualifies independent OI/volume publication",
            "roll_scope": "paired raw source marks and selection switches,not fills or continuous-price execution",
            "notice_method": "source_dates"
            if isinstance(strategy, FirstNoticeDateRoll)
            and normalized["instrument_first_notice_date"].is_not_null().any()
            else "expiration_only"
            if isinstance(strategy, FirstNoticeDateRoll)
            else "not_requested",
        },
    )
