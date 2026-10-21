"""Known-at source history preparation without a runtime history or broker engine."""

from dataclasses import dataclass
from datetime import date, datetime
from math import isfinite
from typing import Any, Literal

import pandas as pd
import polars as pl

from ml4t.data.assets.registry import InstrumentRegistry
from ml4t.data.core.feeds import resolve_feed_spec
from ml4t.data.core.revisions import (
    native_frame,
    select_revisions,
    session_label_at,
    utc_column,
    utc_cutoff,
    utc_stamp,
)
from ml4t.data.utils.gaps import GapDetector
from ml4t.specs import UNSET_CALENDAR, CalendarUnset, FeedSpec

from ._history_factors import prepare_supplied_factors
from .core import _event_adjustment_factors
from .inputs import apply_adjustment_factors, prepare_action_inputs

_CLOCK = "__ml4t_history_clock"
_CASH = {"cash_dividend", "cash_distribution"}


@dataclass(frozen=True)
class AdjustedHistory:
    """Prepared native prices,facts,coverage and explicit source/basis declarations."""

    prices: pl.DataFrame
    actions: pl.DataFrame
    coverage: pl.DataFrame
    metadata: dict[str, Any]
    identities: pl.DataFrame | None = None


def _through(
    frame: pl.DataFrame, column: str, cutoff: datetime | str, feed: FeedSpec
) -> pl.DataFrame:
    bound = (
        pl.lit(session_label_at(cutoff, feed))
        if frame.schema[column] == pl.Date
        else utc_cutoff(cutoff)
    )
    return frame.filter(pl.col(column) <= bound)


def _price_clock(frame: pl.DataFrame, feed: FeedSpec) -> pl.DataFrame:
    if _CLOCK in frame.columns:
        raise ValueError("Source columns collide with an internal history key")
    timestamp = feed.timestamp_col
    if timestamp not in frame.columns or frame[timestamp].null_count():
        raise ValueError("History requires known observation timestamps")
    if frame.schema[timestamp] == pl.Date:
        if feed.timestamp_semantics != "session_label":
            raise ValueError("Declare session_label semantics for Date observations")
        clock = pl.col(timestamp).cast(pl.Datetime("ns", "UTC"))
    else:
        dtype = frame.schema[timestamp]
        preserve_label_zone = (
            feed.timestamp_semantics == "session_label"
            and isinstance(dtype, pl.Datetime)
            and dtype.time_zone is not None
            and feed.timezone is None
        )
        if not preserve_label_zone:
            frame = utc_column(frame, timestamp, feed.timezone)
        clock = pl.col(timestamp)
        if feed.timestamp_semantics == "session_label":
            if feed.timezone is not None:
                clock = clock.dt.convert_time_zone(feed.timezone)
            clock = clock.dt.date()
        clock = clock.cast(pl.Datetime("ns", "UTC"))
    return frame.with_columns(clock.alias(_CLOCK))


def _actions_at(
    actions: pl.DataFrame | pd.DataFrame | None,
    prices: pl.DataFrame,
    feed: FeedSpec,
    as_of: datetime | str | None,
    basis: str,
    imported: bool = False,
    action_feed_spec: Any | None = None,
) -> pl.DataFrame:
    asset = FeedSpec._coerce_entity_col(feed.entity_col) or "asset"
    if basis == "raw":
        actions = None
    if actions is None:
        frame = pl.DataFrame(
            schema={
                asset: prices.schema[asset],
                "source": pl.String,
                "event_id": pl.String,
                "revision_id": pl.String,
                "action_kind": pl.String,
                "effective_at": pl.Datetime("ns", "UTC"),
            }
        )
    else:
        frame = native_frame(actions)
        if action_feed_spec is not None:
            declaration = resolve_feed_spec(action_feed_spec)
            aliases = {
                FeedSpec._coerce_entity_col(declaration.entity_col) or "asset": asset,
                declaration.timestamp_col: "effective_at",
            }
            for source, target in aliases.items():
                if source not in frame.columns:
                    raise ValueError(f"Missing declared action field {source}")
                if source != target:
                    if target in frame.columns:
                        raise ValueError("Declared action mappings collide with source fields")
                    frame = frame.with_columns(pl.col(source).alias(target))
            frame = utc_column(frame, "available_at", declaration.timezone)
            if frame.schema["effective_at"] != pl.Date:
                frame = utc_column(frame, "effective_at", declaration.timezone)
    if _CLOCK in frame.columns or "__ml4t_history_limit" in frame.columns:
        raise ValueError("Source action columns collide with an internal history key")
    if asset not in frame.columns:
        raise ValueError("Declare the stable action asset column through the feed")
    frame = frame.filter(pl.col(asset).is_in(prices[asset].unique().implode()))
    frame = utc_column(frame, "available_at", feed.timezone)
    if "effective_at" not in frame.columns or frame["effective_at"].null_count():
        raise ValueError("Unknown effective_at cannot establish history adjustment coverage")
    if frame.schema["effective_at"] != pl.Date:
        frame = utc_column(frame, "effective_at", feed.timezone)
    keys = [asset, "source", "event_id"]
    if as_of is not None:
        relevant = _through(frame, "effective_at", as_of, feed).select(keys).unique()
        frame = frame.join(relevant, on=keys, how="semi")
        if basis == "raw":
            frame = frame.filter(pl.col("available_at").is_not_null())
    frame = select_revisions(frame, keys, as_of)
    if as_of is not None:
        frame = _through(frame, "effective_at", as_of, feed)
    elif not imported:
        limits = prices.group_by(asset).agg(pl.col(_CLOCK).max().alias("__ml4t_history_limit"))
        frame = (
            _action_clock(frame, feed)
            .join(limits, on=asset, how="inner")
            .filter(pl.col(_CLOCK) <= pl.col("__ml4t_history_limit"))
            .drop(_CLOCK, "__ml4t_history_limit")
        )
    required = ["split_ratio"] if basis == "split_adjusted" and not imported else []
    if basis == "total_return" and not imported:
        required += ["split_ratio", "cash_per_share", "currency", "share_basis"]
    action_feed = feed.with_overrides(timestamp_col="effective_at")
    if frame.schema["effective_at"] == pl.Date:
        action_feed = action_feed.with_overrides(timestamp_semantics="session_label")
    facts = prepare_action_inputs(frame, feed_spec=action_feed, required_terms=required)
    if (
        basis != "raw"
        and not imported
        and facts.filter(~pl.col("action_kind").is_in(["split", *_CASH])).height
    ):
        raise ValueError("Merger/stock distribution price history requires supplied source factors")
    return _action_clock(facts, feed)


def _action_clock(frame: pl.DataFrame, feed: FeedSpec) -> pl.DataFrame:
    effective = pl.col("effective_at")
    if feed.timestamp_semantics == "session_label":
        if frame.schema["effective_at"] != pl.Date:
            effective = effective.dt.convert_time_zone(feed.timezone or "UTC").dt.date()
        if "ex_date" in frame.columns:
            effective = pl.coalesce("ex_date", effective)
    return frame.with_columns(effective.cast(pl.Datetime("ns", "UTC")).alias(_CLOCK))


def _native_factors(
    group: pl.DataFrame,
    facts: pl.DataFrame,
    feed: FeedSpec,
    basis: str,
    closes: pl.DataFrame | None,
    close_times: dict[date, pd.Timestamp] | None,
) -> pl.DataFrame:
    asset = FeedSpec._coerce_entity_col(feed.entity_col) or "asset"
    close_col = feed.close_col if feed.close_col in group.columns else feed.price_col
    event_rows = []
    for event in facts.partition_by(_CLOCK, maintain_order=True):
        split_rows = event.filter(pl.col("action_kind") == "split")
        ratio = float(split_rows["split_ratio"].product()) if split_rows.height else 1.0
        cash_rows = event.filter(pl.col("action_kind").is_in(_CASH))
        cash = 0.0
        quoted_close = None
        if basis == "total_return" and cash_rows.height:
            if cash_rows["ex_date"].null_count() or cash_rows["ex_date"].n_unique() != 1:
                raise ValueError("Cash factors require one declared ex_date per effective event")
            ex_date = cash_rows["ex_date"][0]
            currencies = cash_rows["currency"].unique().to_list()
            if len(currencies) != 1 or "currency" not in group.columns:
                raise ValueError("Declare compatible price/cash currency units for native factors")
            if group.filter(
                pl.col("currency").is_null() | (pl.col("currency") != currencies[0])
            ).height:
                raise ValueError("Native cash adjustments cannot infer FX or currency units")
            cash = sum(
                amount / ratio if share_basis == "pre_event" else amount
                for amount, share_basis in cash_rows.select(
                    "cash_per_share", "share_basis"
                ).iter_rows()
            )
            if closes is None:
                if feed.timestamp_semantics != "session_label" and feed.data_frequency not in {
                    "daily",
                    "1d",
                    "1day",
                }:
                    raise ValueError(
                        "Intraday cash factors require captured available ex-date closes"
                    )
                dates = (
                    pl.col(_CLOCK).dt.date()
                    if feed.timestamp_semantics == "session_label"
                    else pl.col(feed.timestamp_col)
                    .dt.convert_time_zone(feed.timezone or "UTC")
                    .dt.date()
                )
                quote = group.filter(dates == ex_date)
            else:
                quote = closes.filter(
                    (pl.col(asset) == group[asset][0]) & (pl.col("ex_date") == ex_date)
                )
            if quote.height != 1 or quote[close_col].null_count():
                raise ValueError(
                    "An actually available ex-date close is required for a native cash factor"
                )
            if closes is not None and (
                "currency" not in quote.columns
                or quote["currency"].null_count()
                or not quote["currency"].eq(currencies[0]).all()
            ):
                raise ValueError("Captured ex-date close currency must match price/cash currency")
            if close_times is not None:
                actual_close = close_times.get(ex_date)
                if (
                    actual_close is None
                    or quote.filter(
                        pl.col("available_at").is_null()
                        | (pl.col("available_at") < utc_cutoff(actual_close))
                    ).height
                ):
                    raise ValueError(
                        "The selected quote is not an actually available ex-date close"
                    )
            quoted_close = quote[close_col][0]
            if not isfinite(quoted_close) or not quoted_close > 0:
                raise ValueError("Native cash factors require a positive finite ex-date close")
        if not isfinite(ratio) or ratio <= 0 or not isfinite(cash):
            raise ValueError("Combined native event factors require finite positive share ratios")
        local = (
            event.select(asset, _CLOCK)
            .head(1)
            .with_columns(
                pl.lit(ratio).alias("split_ratio"),
                pl.lit(cash).alias("ex-dividend"),
                pl.lit(quoted_close, dtype=pl.Float64).alias("close"),
            )
        )
        local = _event_adjustment_factors(
            local,
            split_col="split_ratio",
            dividend_col="ex-dividend" if basis == "total_return" and cash_rows.height else None,
        )
        event_rows.append(local.select(_CLOCK, "_price_event_factor", "_volume_event_factor"))
    if not event_rows:
        return group.select(feed.timestamp_col, asset).with_columns(
            pl.lit(1.0).alias("price_adjustment_factor"),
            pl.lit(1.0).alias("volume_adjustment_factor"),
        )
    curve = (
        pl.concat(event_rows)
        .sort(_CLOCK)
        .with_columns(
            pl.col("_price_event_factor")
            .reverse()
            .cum_prod()
            .reverse()
            .alias("price_adjustment_factor"),
            pl.col("_volume_event_factor")
            .reverse()
            .cum_prod()
            .reverse()
            .alias("volume_adjustment_factor"),
        )
        .select(_CLOCK, "price_adjustment_factor", "volume_adjustment_factor")
    )
    return (
        group.select(feed.timestamp_col, asset, _CLOCK)
        .sort(_CLOCK)
        .join_asof(
            curve,
            on=_CLOCK,
            strategy="forward",
            allow_exact_matches=False,
        )
        .drop(_CLOCK)
        .with_columns(pl.col("price_adjustment_factor", "volume_adjustment_factor").fill_null(1.0))
    )


def prepare_adjusted_history(
    prices: pl.DataFrame | pd.DataFrame,
    actions: pl.DataFrame | pd.DataFrame | None,
    *,
    as_of: datetime | str | None,
    feed_spec: Any | None = None,
    calendar: str | None | CalendarUnset = UNSET_CALENDAR,
    basis: Literal["raw", "split_adjusted", "total_return"] | None = None,
    action_coverage: Literal["complete", "unknown"] = "unknown",
    ex_date_closes: pl.DataFrame | pd.DataFrame | None = None,
    factors: pl.DataFrame | pd.DataFrame | None = None,
    factor_method: str | None = None,
    instruments: InstrumentRegistry | None = None,
    identity_mode: Literal["stable_id", "symbol"] = "stable_id",
    action_feed_spec: Any | None = None,
) -> AdjustedHistory:
    """Prepare source revisions available at a cutoff,or explicitly retrospective latest history.

    Price revisions require source,revision_id,available_at and unique asset/timestamp keys per
    revision. New handoffs use timestamp/asset; FeedSpec maps existing physical price columns.
    Raw values and missing observations are retained. Adjustment requires an explicit declaration
    of complete source action coverage for the supplied range,not inference from missing events.
    Native total-return factors use the actual available ex-date close. Intraday callers supply
    a separately captured ex_date_closes table with the same units and revision/availability keys.
    No action cash,positions,synthetic prices,FX or historical availability is inferred.

    Args:
        prices: Raw observations with source,revision_id,available_at and declared entity/clock.
        actions: Captured action revisions,or None. Absence does not declare complete coverage.
        as_of: Availability cutoff instant;None explicitly selects retrospective latest revisions.
        feed_spec: Shared observation clock,entity,price,calendar and bar-grid declaration.
        calendar: Explicit schedule override;None disables schedules,omission preserves the feed.
        basis: raw,split_adjusted or total_return. Native default is total_return. Supplied
            factors require an explicit basis and method;native defaults are not substituted.
        action_coverage: Caller declaration of complete or unknown source event coverage for
            the requested price range. Native adjusted history requires complete coverage.
        ex_date_closes: Captured native cash-factor close facts keyed by asset/ex_date with
            source,revision_id,available_at,reference price and currency. Required intraday.
        factors: Exact per-observation supplied factors. Causal inputs require version identity,
            actual availability and one known basis_at instant per asset,not a future/stale basis.
        factor_method: Name of the supplied financial definition,retained in output provenance.
        instruments: Optional captured dated registry for ID validation or symbol normalization.
        identity_mode: stable_id validates declared IDs;symbol resolves the feed entity column
            by date and optional exchange. Source symbols remain and stable IDs are added.
        action_feed_spec: Separate physical action clock/entity mappings when source names differ.

    Returns:
        Prepared raw/derived prices,selected actions,used identity provenance and between-endpoint
        missing/null coverage. metadata.feed_spec points consumers to the requested basis and
        stable IDs;input_feed_spec and source_feed_spec describe input/raw physical fields.
    """
    if basis is None:
        if factors is not None:
            raise ValueError("Declare the supplied factor price/share basis explicitly")
        basis = "total_return"
    if factors is not None and (not isinstance(factor_method, str) or not factor_method.strip()):
        raise ValueError("Declare the supplied factor method explicitly")
    if factors is not None and basis == "raw":
        raise ValueError("Raw history cannot simultaneously apply supplied adjustment factors")
    if basis not in {"raw", "split_adjusted", "total_return"}:
        raise ValueError("Declare raw,split_adjusted or total_return history basis")
    if action_coverage not in {"complete", "unknown"}:
        raise ValueError("Declare complete or unknown action coverage")
    if (
        basis != "raw"
        and action_coverage != "complete"
        and not (factors is not None and as_of is None)
    ):
        raise ValueError("Declare complete source action coverage before adjusting history")
    feed = resolve_feed_spec(
        FeedSpec(entity_col="asset", time_contract_version=2) if feed_spec is None else feed_spec
    )
    if calendar is not UNSET_CALENDAR:
        feed = feed.with_overrides(calendar=calendar, time_contract_version=2)
    input_feed = feed
    selected_calendar = feed.resolve_calendar()
    asset = FeedSpec._coerce_entity_col(feed.entity_col) or "asset"
    frame = _price_clock(native_frame(prices), feed)
    if feed.price_col not in frame.columns:
        raise ValueError("History requires the declared reference price column")
    if as_of is not None:
        if feed.timestamp_semantics == "session_label":
            limit = pl.lit(session_label_at(as_of, feed)).cast(pl.Datetime("ns", "UTC"))
            frame = frame.filter(pl.col(_CLOCK) <= limit)
        else:
            frame = _through(frame, feed.timestamp_col, as_of, feed)
    frame = utc_column(frame, "available_at", feed.timezone)
    identities = None
    if as_of is not None:
        frame = frame.filter(
            pl.col("available_at").is_null() | (pl.col("available_at") <= utc_cutoff(as_of))
        )
    if instruments is not None:
        frame, identities, asset = instruments._normalize(frame, feed, as_of, identity_mode)
        feed = feed.with_overrides(entity_col=asset)
    elif identity_mode != "stable_id":
        raise ValueError("Symbol history requires captured dated instruments")
    frame = select_revisions(frame, [asset, _CLOCK, "source"], as_of)
    if frame.select(asset, _CLOCK).is_duplicated().any():
        raise ValueError("Choose one source revision stream per observation")
    frame = frame.sort([asset, feed.timestamp_col])
    facts = _actions_at(
        actions,
        frame,
        feed,
        as_of,
        basis,
        imported=factors is not None,
        action_feed_spec=action_feed_spec,
    )
    supplied = None
    factor_metadata = {}
    if factors is not None:
        supplied, factor_metadata = prepare_supplied_factors(
            _price_clock(native_frame(factors), feed),
            frame,
            facts,
            feed,
            as_of,
            _CLOCK,
        )
        if "factor_adjustment_method" in supplied.columns and not (
            supplied["factor_adjustment_method"].drop_nulls().eq(factor_method).all()
        ):
            raise ValueError("Requested factor method conflicts with the source declaration")
    closes = None
    if ex_date_closes is not None and basis == "total_return" and supplied is None:
        closes = utc_column(native_frame(ex_date_closes), "available_at", feed.timezone)
        if as_of is not None:
            closes = _through(closes, "ex_date", as_of, feed)
        closes = select_revisions(closes, [asset, "ex_date", "source"], as_of)
    close_times = None
    cash_dates = facts.filter(pl.col("action_kind").is_in(_CASH))["ex_date"].drop_nulls()
    if as_of is not None and basis == "total_return" and len(cash_dates) and supplied is None:
        if selected_calendar is None:
            if closes is None:
                raise ValueError("Row mode requires explicit available ex-date close facts")
        else:
            from ml4t.data.sessions import SessionAssigner

            schedule = SessionAssigner(feed_spec=feed).calendar.schedule(
                start_date=cash_dates.min(), end_date=cash_dates.max()
            )
            close_times = {label.date(): row["market_close"] for label, row in schedule.iterrows()}
    adjusted = []
    coverage = []
    for group in frame.partition_by(asset, maintain_order=True):
        events = facts.filter(pl.col(asset) == group[asset][0])
        if basis == "raw":
            events = events.head(0)
        group_factors = (
            _native_factors(group, events, feed, basis, closes, close_times)
            if supplied is None
            else supplied.filter(pl.col(asset) == group[asset][0])
        )
        adjusted.append(
            apply_adjustment_factors(
                group.drop(_CLOCK),
                group_factors,
                feed_spec=feed,
                method=factor_method
                if supplied is not None
                else "raw"
                if basis == "raw"
                else "native_ex_date_close"
                if basis == "total_return"
                else "native_split",
            )
        )
        close = feed.close_col if feed.close_col in group.columns else feed.price_col
        missing = None
        if feed.data_frequency is not None:
            missing = sum(
                gap.missing_periods
                for gap in GapDetector().detect_gaps(
                    group,
                    timestamp_col=feed.timestamp_col,
                    feed_spec=feed,
                )
            )
        coverage.append(
            {
                asset: group[asset][0],
                "observed_rows": group.height,
                "null_prices": group[close].null_count(),
                "missing_periods": missing,
            }
        )
    if adjusted:
        result = pl.concat(adjusted)
    else:
        empty_factors = (
            supplied
            if supplied is not None
            else frame.select(feed.timestamp_col, asset).with_columns(
                pl.lit(1.0).alias("price_adjustment_factor"),
                pl.lit(1.0).alias("volume_adjustment_factor"),
            )
        )
        result = apply_adjustment_factors(
            frame.drop(_CLOCK),
            empty_factors,
            feed_spec=feed,
            method=factor_method if supplied is not None else "supplied",
        )
    coverage_frame = pl.DataFrame(
        coverage,
        schema={
            asset: frame.schema[asset],
            "observed_rows": pl.UInt64,
            "null_prices": pl.UInt64,
            "missing_periods": pl.UInt64,
        },
    )
    consumer = feed
    if basis != "raw":
        columns = {
            name: f"adj_{getattr(feed, name)}"
            for name in ("price_col", "open_col", "high_col", "low_col", "close_col", "volume_col")
            if f"adj_{getattr(feed, name)}" in result.columns
        }
        consumer = feed.with_overrides(**columns)
        if consumer.close_col not in result.columns:
            consumer = consumer.with_overrides(close_col=consumer.price_col)
    return AdjustedHistory(
        prices=result,
        actions=facts.drop(_CLOCK),
        coverage=coverage_frame,
        metadata={
            "feed_spec": consumer.to_dict(),
            "source_feed_spec": feed.to_dict(),
            "input_feed_spec": input_feed.to_dict(),
            "calendar": selected_calendar,
            "perspective": "retrospective" if as_of is None else "causal",
            "as_of": None if as_of is None else utc_stamp(as_of).isoformat(),
            "price_basis": basis,
            "input_price_basis": "raw",
            "action_coverage": action_coverage,
            "actions_used": basis != "raw",
            "factor_source": "native" if supplied is None else "supplied",
            **factor_metadata,
            "coverage_scope": "between supplied observed endpoints;not a completeness claim outside that range",
        },
        identities=identities,
    )
