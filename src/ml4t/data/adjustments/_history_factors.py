"""Select supplied factor curves without recomputing their financial definition."""

from datetime import datetime
from typing import Any

import pandas as pd
import polars as pl

from ml4t.data.core.revisions import select_revisions, session_label_at, utc_column, utc_cutoff
from ml4t.specs import FeedSpec


def prepare_supplied_factors(
    factors: pl.DataFrame,
    prices: pl.DataFrame,
    actions: pl.DataFrame,
    feed: FeedSpec,
    as_of: datetime | str | None,
    clock_col: str,
) -> tuple[pl.DataFrame, dict[str, Any]]:
    asset = FeedSpec._coerce_entity_col(feed.entity_col) or "asset"
    keys = [asset, clock_col]
    if "price_adjustment_factor" not in factors.columns:
        raise ValueError("A supplied history requires price_adjustment_factor")
    candidates = factors.join(prices.select(keys).unique(), on=keys, how="semi")
    candidates = utc_column(candidates, "available_at", feed.timezone)
    versioned = {"source", "revision_id"}.issubset(candidates.columns) and all(
        not candidates[name].null_count() for name in ("source", "revision_id")
    )
    selected = (
        select_revisions(candidates, [*keys, "source"], as_of)
        if as_of is not None or versioned
        else candidates
    )
    if selected.select(keys).is_duplicated().any():
        raise ValueError("Choose one supplied factor revision per observation")
    if prices.select(keys).join(selected.select(keys), on=keys, how="anti").height:
        raise ValueError("Supplied factor coverage is incomplete at the requested cutoff")
    if "basis_at" not in selected.columns:
        selected = selected.with_columns(
            pl.lit(None, dtype=pl.Datetime("ns", "UTC")).alias("basis_at")
        )
    if selected.schema["basis_at"] != pl.Date:
        selected = utc_column(selected, "basis_at", feed.timezone)
    if as_of is not None:
        if selected.schema["basis_at"] == pl.Date or selected["basis_at"].null_count():
            raise ValueError("Known basis_at instants are required for causal supplied factors")
        if selected.filter(pl.col("basis_at") > utc_cutoff(as_of)).height:
            raise ValueError("Supplied basis_at cannot be later than the cutoff")
        counts = selected.group_by(asset).agg(pl.col("basis_at").n_unique())
        if counts.filter(pl.col("basis_at") != 1).height:
            raise ValueError("Declare one supplied share basis per asset")
        bases = selected.select(asset, "basis_at").unique()
        if actions.height:
            latest = actions.group_by(asset).agg(
                pl.col("effective_at").max().alias("last_effective_at")
            )
            if actions.schema["effective_at"] == pl.Date:
                stamps = bases["basis_at"].cast(pl.Datetime("ns", "UTC")).cast(pl.Int64)
                labels = [
                    session_label_at(pd.Timestamp(value, unit="ns", tz="UTC"), feed)
                    for value in stamps
                ]
                bases = bases.with_columns(pl.Series("basis_session", labels, dtype=pl.Date))
                checked = bases.join(latest, on=asset, how="left")
                stale = checked.filter(pl.col("last_effective_at") > pl.col("basis_session"))
            else:
                checked = bases.join(latest, on=asset, how="left")
                stale = checked.filter(pl.col("last_effective_at") > pl.col("basis_at"))
            if stale.height:
                raise ValueError("Supplied share basis predates an applicable known source action")
    if "currency" in selected.columns and "currency" in prices.columns:
        units = prices.select(*keys, pl.col("currency").alias("price_currency")).join(
            selected.select(*keys, pl.col("currency").alias("factor_currency")),
            on=keys,
        )
        if units.filter(
            pl.col("price_currency").is_not_null()
            & pl.col("factor_currency").is_not_null()
            & (pl.col("price_currency") != pl.col("factor_currency"))
        ).height:
            raise ValueError("Known price and supplied factor currency units conflict")
    numeric = [
        name
        for name in ("price_adjustment_factor", "volume_adjustment_factor")
        if name in selected.columns
    ]
    provenance = [name for name in selected.columns if name not in {*keys, *numeric}]
    projected = selected.select(
        *keys,
        *numeric,
        *(pl.col(name).alias(f"factor_{name}") for name in provenance),
    )
    output = (
        prices.select(asset, feed.timestamp_col, clock_col)
        .join(
            projected,
            on=keys,
            how="left",
            validate="1:1",
        )
        .drop(clock_col)
    )
    return output, {
        "factor_revision_identity": "declared" if versioned else "unknown unversioned capture",
        "factor_basis_known": not selected["basis_at"].null_count(),
        "factor_availability_known": not selected["available_at"].null_count(),
        "factor_rows_outside_window": factors.height - candidates.height,
        "volume_factor_supplied": "volume_adjustment_factor" in numeric,
    }
