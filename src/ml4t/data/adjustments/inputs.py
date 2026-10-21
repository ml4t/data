"""Native event facts and supplied-factor preparation, without account effects."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal

import pandas as pd
import polars as pl

from ml4t.data.core.feeds import resolve_feed_spec
from ml4t.data.utils.conversion import pandas_to_polars
from ml4t.specs import UNSET_CALENDAR, CalendarUnset, FeedSpec

_CASH = {"cash_dividend", "cash_distribution"}
_STOCK = {"stock_dividend", "stock_distribution"}
_KINDS = {"split", "merger", *_CASH, *_STOCK}
_DATES = ("ex_date", "entitlement_date", "payment_date")
_NUMERIC = ("split_ratio", "cash_per_share", "quantity_ratio", "cash_component")


def _frame(data: pl.DataFrame | pd.DataFrame) -> pl.DataFrame:
    return pandas_to_polars(data) if isinstance(data, pd.DataFrame) else data.clone()


def _keys(frame: pl.DataFrame, names: list[str]) -> None:
    missing = set(names).difference(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if any(frame[name].null_count() for name in names):
        raise ValueError(f"Observation/event identity keys cannot be null: {names}")
    if any(frame.schema[name].is_float() and not frame[name].is_finite().all() for name in names):
        raise ValueError("Observation/event identity keys must be finite")
    if any(
        frame.schema[name] == pl.String and frame[name].str.strip_chars().eq("").any()
        for name in names
    ):
        raise ValueError("Observation/event identity keys cannot be empty")
    if frame.select(names).is_duplicated().any():
        raise ValueError(f"Duplicate observation/event revision keys: {names}")


def _event_times(frame: pl.DataFrame, feed: FeedSpec) -> pl.DataFrame:
    for name in (feed.timestamp_col, "available_at", "announced_at"):
        if name not in frame.columns or frame.schema[name] == pl.Null:
            frame = frame.with_columns(pl.lit(None, dtype=pl.Datetime("ns", "UTC")).alias(name))
            continue
        dtype = frame.schema[name]
        if (
            dtype == pl.Date
            and name == feed.timestamp_col
            and feed.timestamp_semantics == "session_label"
        ):
            continue
        if not isinstance(dtype, pl.Datetime):
            raise ValueError(
                f"{name} must be a Datetime instant or an explicit effective session label"
            )
        expression = pl.col(name)
        if dtype.time_zone is None:
            expression = expression.dt.replace_time_zone(feed.timezone or "UTC")
        frame = frame.with_columns(expression.dt.convert_time_zone("UTC"))
    for name in _DATES:
        if name not in frame.columns or frame.schema[name] == pl.Null:
            frame = frame.with_columns(pl.lit(None, dtype=pl.Date).alias(name))
        elif frame.schema[name] != pl.Date:
            raise ValueError(f"{name} must contain civil Date values, not inferred instants")
    return frame


def prepare_action_inputs(
    actions: pl.DataFrame | pd.DataFrame,
    instruments: pl.DataFrame | pd.DataFrame | None = None,
    *,
    feed_spec: Any | None = None,
    calendar: str | None | CalendarUnset = UNSET_CALENDAR,
    split_convention: Literal["new_per_old", "price_factor"] | None = None,
    required_terms: Sequence[str] = (),
) -> pl.DataFrame:
    """Validate versioned action facts and retain unavailable terms as explicit coverage.

    Defaults use ``asset`` and ``effective_at``; map existing names with ``feed_spec``. Stable
    IDs, event/revision IDs, action kind and source are required. Optional ``instruments`` is a
    captured unique-ID table, not a dated registry. All dates, source columns and revisions remain
    in the result. Unknown availability is never copied from an ex-date or a download timestamp.

    Split ratios are new shares per old share. Explicit ``price_factor`` reciprocates imported
    legacy ratios, retaining ``raw_split_ratio`` and ``source_split_convention``. Cash, stock
    distributions and mergers remain facts, not positions or cash flows. ``missing_terms`` records
    unavailable applicable terms; requesting any of them in ``required_terms`` fails atomically.
    A declared calendar is retained in ``preparation_calendar``; omission is not replaced by NYSE.
    """
    feed = resolve_feed_spec(
        FeedSpec(timestamp_col="effective_at", entity_col="asset", time_contract_version=2)
        if feed_spec is None
        else feed_spec
    )
    if calendar is not UNSET_CALENDAR:
        feed = feed.with_overrides(calendar=calendar, time_contract_version=2)
    feed.resolve_calendar()
    asset = FeedSpec._coerce_entity_col(feed.entity_col) or "asset"
    frame = _frame(actions)
    _keys(frame, [asset, "source", "event_id", "revision_id"])
    if "action_kind" not in frame.columns or frame["action_kind"].null_count():
        raise ValueError("action_kind is required")
    if set(frame["action_kind"].unique().to_list()).difference(_KINDS):
        raise ValueError(
            "Unknown action_kind; declare split, cash/stock distribution or merger facts"
        )
    if instruments is not None:
        identities = _frame(instruments)
        _keys(identities, [asset])
        if not frame[asset].is_in(identities[asset].implode()).all():
            raise ValueError("Action identity is not in the declared instruments")
        if "destination_asset" in frame.columns and not (
            frame["destination_asset"].drop_nulls().is_in(identities[asset].implode()).all()
        ):
            raise ValueError("Destination identity is not in the declared instruments")
    frame = _event_times(frame, feed)
    for name in _NUMERIC:
        if name not in frame.columns or frame.schema[name] == pl.Null:
            frame = frame.with_columns(pl.lit(None, dtype=pl.Float64).alias(name))
        elif not frame.schema[name].is_numeric():
            raise ValueError(f"{name} must be numeric")
        if frame.filter(pl.col(name).is_not_null() & ~pl.col(name).is_finite()).height:
            raise ValueError(f"{name} must be finite when known")
    for name in ("currency", "share_basis", "source_record_id"):
        if name not in frame.columns:
            frame = frame.with_columns(pl.lit(None, dtype=pl.String).alias(name))
    if "destination_asset" not in frame.columns:
        frame = frame.with_columns(
            pl.lit(None, dtype=frame.schema[asset]).alias("destination_asset")
        )
    previous = (
        set(frame["split_ratio_convention"].drop_nulls())
        if "split_ratio_convention" in frame.columns
        else set()
    )
    if split_convention is not None and previous and previous != {split_convention}:
        raise ValueError("split convention conflicts with the input declaration")
    convention = split_convention or (next(iter(previous)) if len(previous) == 1 else "new_per_old")
    if len(previous) > 1 or convention not in {"new_per_old", "price_factor"}:
        raise ValueError("Declare one supported split convention")
    if "raw_split_ratio" not in frame.columns:
        frame = frame.with_columns(pl.col("split_ratio").alias("raw_split_ratio"))
    if "source_split_convention" not in frame.columns:
        frame = frame.with_columns(pl.lit(convention).alias("source_split_convention"))
    if frame.filter((pl.col("action_kind") == "split") & (pl.col("split_ratio") <= 0)).height:
        raise ValueError("Known split ratio must be positive")
    ratio = 1.0 / pl.col("split_ratio") if convention == "price_factor" else pl.col("split_ratio")
    frame = frame.with_columns(
        ratio.alias("split_ratio"), pl.lit("new_per_old").alias("split_ratio_convention")
    )
    fractions = ("split_numerator", "split_denominator")
    if any(name in frame.columns for name in fractions):
        for name in fractions:
            if name not in frame.columns:
                frame = frame.with_columns(pl.lit(None, dtype=pl.Float64).alias(name))
            if (
                not frame.schema[name].is_numeric()
                or frame.filter(
                    pl.col(name).is_not_null() & (~pl.col(name).is_finite() | (pl.col(name) <= 0))
                ).height
            ):
                raise ValueError("Known split ratio fraction terms must be positive and finite")
        rational = pl.col("split_numerator") / pl.col("split_denominator")
        if frame.filter(
            pl.col("split_ratio").is_not_null()
            & rational.is_not_null()
            & ((pl.col("split_ratio") - rational).abs() > rational.abs() * 1e-12)
        ).height:
            raise ValueError("Split ratio conflicts with supplied rational terms")
        frame = frame.with_columns(pl.coalesce("split_ratio", rational).alias("split_ratio"))
    if frame.filter(
        (pl.col("action_kind") == "split")
        & pl.col("split_ratio").is_not_null()
        & (~pl.col("split_ratio").is_finite() | (pl.col("split_ratio") <= 0))
    ).height:
        raise ValueError("Normalized split ratio must be positive and finite")
    if frame.filter(pl.col("action_kind").is_in(_CASH) & (pl.col("cash_per_share") < 0)).height:
        raise ValueError("Cash distribution amounts cannot be negative")
    if frame.filter(pl.col("action_kind").is_in(_STOCK) & (pl.col("quantity_ratio") <= 0)).height:
        raise ValueError("Stock distribution quantity ratio must be positive")
    if frame.filter((pl.col("action_kind") == "merger") & (pl.col("quantity_ratio") < 0)).height:
        raise ValueError("Merger quantity ratio cannot be negative")
    if frame.filter(
        pl.col("share_basis").is_not_null()
        & ~pl.col("share_basis").is_in(["pre_event", "post_event"])
    ).height:
        raise ValueError("Cash share_basis must be pre_event or post_event")
    requested = set(required_terms)
    allowed = set(frame.columns) | {"destination_asset", "quantity_ratio", "cash_component"}
    if requested.difference(allowed):
        raise ValueError(f"Unknown required terms: {sorted(requested.difference(allowed))}")
    coverage = []
    kind_fields = {
        "split_ratio",
        "cash_per_share",
        "currency",
        "share_basis",
        "payment_date",
        "destination_asset",
        "quantity_ratio",
        "cash_component",
    }
    for row in frame.iter_rows(named=True):
        relevant = {feed.timestamp_col, "available_at", "source_record_id"} | (
            requested - kind_fields
        )
        kind = row["action_kind"]
        if kind == "split":
            relevant.add("split_ratio")
        elif kind in _CASH:
            relevant.update({"cash_per_share", "currency", "share_basis", "payment_date"})
        elif kind in _STOCK:
            relevant.update({"destination_asset", "quantity_ratio"})
        elif kind == "merger":
            relevant.update({"quantity_ratio", "cash_component", "currency"})
            if row["quantity_ratio"] != 0:
                relevant.add("destination_asset")
        missing = sorted(
            name
            for name in relevant
            if row.get(name) is None or (isinstance(row[name], str) and not row[name].strip())
        )
        if requested.intersection(missing):
            raise ValueError(f"Unavailable required action terms: {missing}")
        coverage.append(missing)
    frame = frame.with_columns(pl.Series("missing_terms", coverage, dtype=pl.List(pl.String)))
    if "calendar" in feed.to_dict():
        frame = frame.with_columns(
            pl.lit(feed.to_dict()["calendar"], dtype=pl.String).alias("preparation_calendar")
        )
    return frame


def apply_adjustment_factors(
    prices: pl.DataFrame | pd.DataFrame,
    factors: pl.DataFrame | pd.DataFrame,
    *,
    feed_spec: Any | None = None,
    price_cols: Sequence[str] | None = None,
    method: str = "supplied",
) -> pl.DataFrame:
    """Multiply raw prices by supplied per-observation factors without changing their definition.

    Price factors must cover exactly the unique timestamp/entity keys. Preserve raw columns,
    add ``adj_<price>`` and explicit factors, and retain the supplied method. Volume is adjusted
    only when a volume factor is supplied. No dividend factor, currency conversion, missing row,
    source availability or unknown volume factor is inferred.
    """
    feed = resolve_feed_spec(
        FeedSpec(entity_col="asset", time_contract_version=2) if feed_spec is None else feed_spec
    )
    asset = FeedSpec._coerce_entity_col(feed.entity_col) or "asset"
    keys = [feed.timestamp_col, asset]
    frame, supplied = _frame(prices), _frame(factors)
    _keys(frame, keys)
    _keys(supplied, keys)
    if "price_adjustment_factor" not in supplied.columns:
        raise ValueError("price_adjustment_factor is required")
    factor_columns = [
        name
        for name in ("price_adjustment_factor", "volume_adjustment_factor")
        if name in supplied.columns
    ]
    for name in factor_columns:
        if (
            not supplied.schema[name].is_numeric()
            or supplied.filter(
                pl.col(name).is_null() | ~pl.col(name).is_finite() | (pl.col(name) <= 0)
            ).height
        ):
            raise ValueError(f"{name} must contain positive finite factors")
    if (
        frame.join(supplied.select(keys), on=keys, how="anti").height
        or supplied.join(frame.select(keys), on=keys, how="anti").height
    ):
        raise ValueError("Factor key coverage must exactly match the supplied prices")
    if not isinstance(method, str) or not method.strip():
        raise ValueError("An explicit factor method label is required")
    if "currency" in frame.columns and "currency" in supplied.columns:
        units = frame.select(*keys, pl.col("currency").alias("price_currency")).join(
            supplied.select(*keys, pl.col("currency").alias("factor_currency")), on=keys
        )
        if units.filter(
            pl.col("price_currency").is_not_null()
            & pl.col("factor_currency").is_not_null()
            & (pl.col("price_currency") != pl.col("factor_currency"))
        ).height:
            raise ValueError("Price and factor currency units are incompatible")
        supplied = supplied.rename({"currency": "factor_currency"})
    extra = [name for name in supplied.columns if name not in keys]
    if set(extra).intersection(frame.columns):
        raise ValueError(
            "Factor provenance collides with raw price columns; declare distinct fields"
        )
    selected = (
        list(price_cols)
        if price_cols is not None
        else list(
            dict.fromkeys(
                name
                for name in (
                    feed.open_col,
                    feed.high_col,
                    feed.low_col,
                    feed.close_col,
                    feed.price_col,
                )
                if name in frame.columns
            )
        )
    )
    if not selected or any(
        name not in frame.columns or not frame.schema[name].is_numeric() for name in selected
    ):
        raise ValueError("Declare existing numeric price columns")
    outputs = {f"adj_{name}" for name in selected} | {"adjustment_method"}
    if "volume_adjustment_factor" in supplied.columns and feed.volume_col in frame.columns:
        outputs.add(f"adj_{feed.volume_col}")
    if outputs.intersection(frame.columns):
        raise ValueError("Supply raw prices, not an already-adjusted output")
    if outputs.intersection(extra):
        raise ValueError("Factor provenance uses reserved adjustment output fields")
    joined = frame.join(supplied, on=keys, how="left", maintain_order="left", validate="1:1")
    expressions = [
        (pl.col(name) * pl.col("price_adjustment_factor")).alias(f"adj_{name}") for name in selected
    ]
    if "volume_adjustment_factor" in supplied.columns and feed.volume_col in frame.columns:
        expressions.append(
            (pl.col(feed.volume_col) * pl.col("volume_adjustment_factor")).alias(
                f"adj_{feed.volume_col}"
            )
        )
    return joined.with_columns(*expressions, pl.lit(method).alias("adjustment_method"))
