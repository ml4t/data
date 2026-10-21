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
    session_labels_at,
    utc_column,
    utc_cutoff,
    utc_stamp,
)
from ml4t.data.futures.roll import (
    CalendarRoll,
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
from ml4t.data.sessions import SessionAssigner
from ml4t.data.sessions.assigner import _daily_session_labels
from ml4t.specs import UNSET_CALENDAR, CalendarUnset, FeedSpec

_DATE = "__ml4t_futures_date"
_ID = "__ml4t_futures_id"
_SOURCE_SYMBOL_TIES = (
    VolumeBasedRoll,
    OpenInterestBasedRoll,
    HighestVolumeRoll,
    HighestOpenInterestRoll,
    TimeBasedRoll,
    FirstNoticeDateRoll,
    CalendarRoll,
)
_DEFINITION_FIELDS = (
    "symbol",
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
    if feed.timestamp_semantics != "session_label":
        frame = utc_column(frame, column, feed.timezone)
    labels = _daily_session_labels(frame, SessionAssigner(feed_spec=feed), column)
    return frame.with_columns(labels.alias(_DATE))


def _contract_definitions(frame: pl.DataFrame, *, validate_values: bool = True) -> pl.DataFrame:
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
    if validate_values:
        for name in required:
            if frame[name].null_count() or (
                frame.schema[name] == pl.String and frame[name].str.strip_chars().eq("").any()
            ):
                raise ValueError(f"Required contract fact {name} must be known")
    if validate_values and not frame["instrument_type"].eq("future").all():
        raise ValueError("Declare future instrument_type;do not inherit equity defaults")
    for name in ("multiplier", "tick_size"):
        if (
            not frame.schema[name].is_numeric()
            or validate_values
            and frame.filter(~pl.col(name).is_finite() | (pl.col(name) <= 0)).height
        ):
            raise ValueError(f"Contract {name} must be positive and finite")
    if frame.schema["expiration"] != pl.Date and not isinstance(
        frame.schema["expiration"], pl.Datetime
    ):
        raise ValueError("Contract expiration must contain Date labels or Datetime instants")
    for name in ("first_notice_date", "last_trade_date", "auto_close_date"):
        if name not in frame.columns or frame.schema[name] == pl.Null:
            frame = frame.with_columns(pl.lit(None, dtype=pl.Date).alias(name))
        elif frame.schema[name] != pl.Date and not isinstance(frame.schema[name], pl.Datetime):
            raise ValueError(f"{name} must contain Date labels or Datetime instants")
    for name in ("volume_unit", "open_interest_unit"):
        if name not in frame.columns:
            frame = frame.with_columns(pl.lit(None, dtype=pl.String).alias(name))
    if validate_values and not frame["settlement_type"].is_in(["cash", "physical"]).all():
        raise ValueError("Declare cash or physical settlement_type")
    return frame


def _lifecycle_labels(frame: pl.DataFrame, name: str, feed: FeedSpec) -> pl.Series:
    if frame.schema[name] == pl.Date:
        return frame[name]
    points = utc_column(frame, name, feed.timezone)[name]
    if points.drop_nulls().is_empty():
        return pl.Series(name, [None] * frame.height, dtype=pl.Date)
    labels = iter(
        session_labels_at(
            points.drop_nulls(),
            FeedSpec(
                time_contract_version=2,
                calendar=feed.resolve_calendar(),
                timezone=feed.timezone,
                timestamp_semantics="event_time",
            ),
        )
    )
    return pl.Series(
        name, [None if missing else next(labels) for missing in points.is_null()], dtype=pl.Date
    )


def _normalized_prices(frame: pl.DataFrame, feed: FeedSpec, target: str | None) -> pl.DataFrame:
    if target is not None:
        price_conversion_factor(target, target)
    for name in ("price_quote_unit", "currency", "mark_kind"):
        if (
            name not in frame.columns
            or frame.schema[name] != pl.String
            or frame[name].null_count()
            or frame[name].str.strip_chars().eq("").any()
        ):
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
        (
            pl.col(name) if target is None else pl.col(name) * pl.col("price_conversion_factor")
        ).alias(f"normalized_{name}")
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
    if frame.filter(
        (pl.col("normalized_multiplier") <= 0) | (pl.col("normalized_tick_size") <= 0)
    ).height:
        raise ValueError("Normalized multiplier and tick size must remain positive;unit underflow")
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
        if observed.height and (
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
            pl.col("revision_id").alias("capture_revision_id"),
            pl.col(available).cast(pl.Datetime("ns", "UTC")).alias("available_at"),
        )
        revision = f"{metric}_revision_id"
        if revision in observed.columns:
            observed = observed.with_columns(pl.col(revision).alias("revision_id"))
        observed = observed.select(
            feed.timestamp_col,
            asset,
            _DATE,
            _ID,
            "source",
            "revision_id",
            "available_at",
            "capture_available_at",
            "capture_revision_id",
            pl.lit(metric).alias("metric"),
            pl.col(physical).alias(metric),
        )
        observed = observed.sort("capture_available_at").unique(
            subset=[asset, feed.timestamp_col, "source", "available_at", metric]
            + (["revision_id"] if revision in frame.columns else []),
            keep="first",
            maintain_order=True,
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


def _leg_marks(frame: pl.DataFrame, feed: FeedSpec, side: str) -> pl.DataFrame:
    asset = FeedSpec._coerce_entity_col(feed.entity_col) or "asset"
    fields = {
        "instrument_id": asset,
        "mark": feed.price_col,
        "mark_observed_at": feed.timestamp_col,
        "mark_available_at": "available_at",
        "mark_source": "source",
        "mark_revision_id": "revision_id",
        "price_quote_unit": "price_quote_unit",
        "currency": "currency",
        "normalized_mark": f"normalized_{feed.price_col}",
        "normalized_multiplier": "normalized_multiplier",
        "normalized_price_quote_unit": "normalized_price_quote_unit",
        "definition_source": "instrument_source",
        "definition_revision_id": "instrument_revision_id",
    }
    return frame.select(pl.col(column).alias(f"{side}_{name}") for name, column in fields.items())


def _roll_proposal(
    old: pl.DataFrame, new: pl.DataFrame, feed: FeedSpec, point: pl.Expr
) -> pl.DataFrame:
    return (
        old.select(
            pl.col("instrument_root").alias("root"),
            pl.col("instrument_exchange").alias("exchange"),
            "mark_kind",
        )
        .hstack(_leg_marks(old, feed, "old"))
        .join(_leg_marks(new, feed, "new"), how="cross")
        .with_columns(point.alias("decision_at"), point.alias("effective_at"))
    )


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
    registry_feed = registry_feed.with_overrides(
        calendar=selected_calendar, time_contract_version=2
    )
    facts = utc_column(native_frame(definitions), "available_at", registry_feed.timezone)
    if as_of is not None:
        facts = facts.filter(
            pl.col("available_at").is_null() | (pl.col("available_at") <= utc_cutoff(as_of))
        )
    facts = _contract_definitions(facts, validate_values=False)
    registry = InstrumentRegistry.from_frame(facts, feed_spec=registry_feed)
    asset = FeedSpec._coerce_entity_col(feed.entity_col) or "asset"
    source = _dated_bars(native_frame(bars), feed)
    source = utc_column(source, "available_at", feed.timezone)
    if as_of is not None:
        if (
            source.schema[feed.timestamp_col] != pl.Date
            and feed.timestamp_semantics != "session_label"
        ):
            source = source.filter(pl.col(feed.timestamp_col) <= utc_cutoff(as_of))
        source = source.filter(pl.col(_DATE) <= pl.lit(session_label_at(as_of, feed)))
        source = source.filter(
            pl.col("available_at").is_null() | (pl.col("available_at") <= utc_cutoff(as_of))
        )
    source, used, asset = registry._normalize(
        source,
        feed,
        as_of,
        identity_mode,
        include_fields=_DEFINITION_FIELDS,
    )
    used = _contract_definitions(used)
    feed = feed.with_overrides(entity_col=asset)
    source = source.with_columns(pl.col(asset).cast(pl.String).alias(_ID))
    selected_prices = select_revisions(source, [asset, feed.timestamp_col, "source"], as_of)
    if selected_prices.select(asset, _DATE).is_duplicated().any():
        raise ValueError("Roll inputs require one declared source observation per contract/session")
    normalized = _normalized_prices(selected_prices, feed, price_quote_unit)
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
    metrics = _metric_facts(source, feed, as_of, requested)
    point = pl.lit(None, dtype=pl.Datetime("ns", "UTC")) if as_of is None else utc_cutoff(as_of)
    decisions, proposals = [], []
    for group in normalized.partition_by(
        ["instrument_root", "instrument_exchange"], maintain_order=True
    ):
        projection = group.select(
            pl.col(_DATE).alias("date"),
            pl.col(_ID).alias("symbol"),
            pl.col("instrument_symbol").alias("source_symbol"),
            _lifecycle_labels(group, "instrument_expiration", registry_feed).alias("expiration"),
            _lifecycle_labels(group, "instrument_first_notice_date", registry_feed).alias(
                "first_notice_date"
            ),
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
        if isinstance(strategy, _SOURCE_SYMBOL_TIES) and type(strategy) in _SOURCE_SYMBOL_TIES:
            selections = strategy.select_contracts(projection, tie_breaker_col="source_symbol")
        else:
            selections = strategy.select_contracts(projection)
        if selections.height == 0:
            continue
        current = selections.tail(1)
        chosen = group.filter(
            (pl.col(_ID) == current["symbol"][0]) & (pl.col(_DATE) == current["date"][0])
        )
        if chosen.height != 1:
            raise ValueError("Current selection must have exactly one raw source observation")
        decision_label = current["date"][0] if as_of is None else session_label_at(as_of, feed)
        identity_point = (
            decision_label
            if registry.identities.schema["valid_from"] == pl.Date
            else utc_stamp(as_of)
            if as_of is not None
            else chosen[feed.timestamp_col][0]
        )
        current_facts = InstrumentRegistry(
            registry.identities.filter(pl.col(asset) == chosen[asset][0]), None, registry_feed
        ).lifetimes(on=identity_point, as_of=as_of)
        if current_facts.height != 1:
            raise ValueError("The current selection lacks dated identity coverage")
        current_facts = _contract_definitions(current_facts)
        for name in ("expiration", "last_trade_date"):
            if isinstance(current_facts.schema[name], pl.Datetime) and as_of is not None:
                past = (
                    utc_column(current_facts, name, registry_feed.timezone)
                    .filter(pl.col(name) <= utc_cutoff(as_of))
                    .height
                )
            else:
                past = (
                    _lifecycle_labels(current_facts, name, registry_feed).lt(decision_label).any()
                )
            if past:
                raise ValueError("The current selection is past its declared lifecycle date")
        if isinstance(strategy, TimeBasedRoll | FirstNoticeDateRoll):
            current_policy = current_facts.select(
                pl.lit(decision_label).alias("date"),
                pl.lit(current["symbol"][0]).alias("symbol"),
                _lifecycle_labels(current_facts, "expiration", registry_feed),
                _lifecycle_labels(current_facts, "first_notice_date", registry_feed),
            )
            if strategy.select_contracts(current_policy).is_empty():
                raise ValueError("The current selection is past its dated policy trigger")
        used = pl.concat([used, current_facts]).unique(maintain_order=True)
        decision = chosen.select(
            pl.col("instrument_root").alias("root"),
            pl.col("instrument_exchange").alias("exchange"),
            pl.col(asset).alias("selected_instrument_id"),
            pl.col(_DATE).alias("observation_label"),
        ).join(
            current_facts.select(
                pl.col(name).alias("definition_" + name) for name in _DEFINITION_FIELDS
            ),
            how="cross",
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
                pl.lit(None, dtype=source.schema[feed.timestamp_col]).alias("metric_observed_at"),
                pl.lit(None, dtype=pl.Datetime("ns", "UTC")).alias("metric_available_at"),
            )
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
            if old["currency"][0] != new["currency"][0]:
                raise ValueError(
                    "Paired roll marks have conflicting currency units;no FX is inferred"
                )
            proposals.append(_roll_proposal(old, new, feed, point))
    selections_frame = (
        pl.concat(decisions)
        if decisions
        else pl.DataFrame(
            schema={
                "root": used.schema["root"],
                "exchange": used.schema["exchange"],
                "selected_instrument_id": source.schema[asset],
                "observation_label": pl.Date,
                **{"definition_" + name: used.schema[name] for name in _DEFINITION_FIELDS},
                "metric_observed_at": source.schema[feed.timestamp_col],
                "metric_available_at": pl.Datetime("ns", "UTC"),
                "decision_at": pl.Datetime("ns", "UTC"),
                "effective_at": pl.Datetime("ns", "UTC"),
                "selection_method": pl.String,
            }
        )
    )
    rolls_frame = (
        pl.concat(proposals)
        if proposals
        else _roll_proposal(normalized.head(0), normalized.head(0), feed, point)
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
            "normalized_feed_spec": feed.with_overrides(
                **{
                    name: f"normalized_{getattr(feed, name)}"
                    for name in ("open_col", "high_col", "low_col", "close_col", "price_col")
                }
            ).to_dict(),
            "calendar": selected_calendar,
            "price_basis": "raw",
            "perspective": "retrospective" if as_of is None else "causal_current_decision",
            "as_of": None if as_of is None else utc_stamp(as_of).isoformat(),
            "selection_scope": "one current source-vintage decision per root/venue,not backdated account history",
            "selection_policy": {
                "kind": f"{type(strategy).__module__}.{type(strategy).__name__}",
                "tie_breaker": "observation_definition_symbol_then_stable_id_string"
                if type(strategy) in _SOURCE_SYMBOL_TIES
                else "strategy_defined",
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
