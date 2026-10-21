"""Offline event-table migration with native and imported adjustment definitions."""

from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory

import polars as pl
from polars.testing import assert_frame_equal

from ml4t.data.adjustments import (
    apply_adjustment_factors,
    apply_corporate_actions,
    prepare_action_inputs,
)
from ml4t.data.storage import FlatStorage, StorageConfig
from ml4t.specs import FeedSpec


def migrate_actions(directory: Path) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    source = pl.read_csv(
        StringIO(
            "asset,event_id,revision_id,action_kind,source,source_record_id,effective_at,available_at,ex_date,payment_date,split_ratio,cash_per_share,currency,share_basis\n"
            "17,s1,r1,split,captured,101,2025-07-03T00:00:00Z,2025-07-01T12:00:00Z,2025-07-03,,0.5,,,\n"
            "17,c1,r1,cash_dividend,captured,102,2025-07-03T00:00:00Z,2025-07-01T12:00:00Z,2025-07-03,2025-07-15,,3,USD,post_event\n"
        ),
        try_parse_dates=True,
        schema_overrides={"split_ratio": pl.Float64, "cash_per_share": pl.Float64},
    )
    feed = FeedSpec(
        time_contract_version=2,
        timestamp_col="effective_at",
        entity_col="asset",
        calendar="NYSE",
    )
    facts = prepare_action_inputs(
        source,
        pl.DataFrame({"asset": [17]}),
        feed_spec=feed,
        split_convention="price_factor",
        required_terms=[
            "available_at",
            "source_record_id",
            "split_ratio",
            "cash_per_share",
            "currency",
            "share_basis",
        ],
    )
    storage = FlatStorage(StorageConfig(base_path=directory))
    storage.write(facts, "captured/actions", feed.to_dict())
    restored = storage.read("captured/actions").collect()
    assert_frame_equal(restored, facts)
    assert restored["raw_split_ratio"].to_list() == [0.5, None]
    assert restored["split_ratio"].to_list() == [2.0, None]

    prices = pl.read_csv(
        StringIO(
            "timestamp,asset,close,volume,action_state\n"
            "2025-07-02,17,200,1000,no_action\n2025-07-03,17,98,7,actions\n"
        ),
        try_parse_dates=True,
    )
    events = restored.group_by("ex_date").agg(
        pl.col("split_ratio").fill_null(1).product(),
        pl.col("cash_per_share").fill_null(0).sum().alias("ex-dividend"),
    )
    native_input = (
        prices.rename({"timestamp": "date"})
        .join(events, left_on="date", right_on="ex_date", how="left")
        .with_columns(
            pl.when(pl.col("action_state") == "no_action")
            .then(1)
            .otherwise(pl.col("split_ratio"))
            .alias("split_ratio"),
            pl.when(pl.col("action_state") == "no_action")
            .then(0)
            .otherwise(pl.col("ex-dividend"))
            .alias("ex-dividend"),
        )
    )
    native = apply_corporate_actions(native_input)
    assert abs(native["price_adjustment_factor"][0] - 98 / 202) < 1e-14
    assert native["adj_volume"].to_list() == [2000, 7]
    assert abs(native["adj_close"][1] / native["adj_close"][0] - 1.01) < 1e-14

    factors = prices.select("timestamp", "asset").with_columns(
        pl.Series("price_adjustment_factor", [0.485, 1.0]),
        pl.Series("volume_adjustment_factor", [2.0, 1.0]),
    )
    supplied = apply_adjustment_factors(prices, factors, method="vendor_prior_close")
    assert supplied["adj_close"].to_list() == [97, 98]
    assert supplied["close"].to_list() == [200, 98]
    assert supplied["volume"].to_list() == [1000, 7]
    return restored, native, supplied


if __name__ == "__main__":
    with TemporaryDirectory(prefix="ml4t-action-inputs-") as temporary:
        facts, native, supplied = migrate_actions(Path(temporary))
        print(facts.select("asset", "event_id", "split_ratio", "missing_terms"))
        print(native.select("date", "close", "adj_close", "adj_volume"))
        print(supplied.select("timestamp", "close", "adj_close", "adjustment_method"))
