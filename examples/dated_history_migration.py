"""Offline dated identity and known-at history inputs,including pre-entry warmup."""

from datetime import UTC, date, datetime
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory

import polars as pl
from polars.testing import assert_frame_equal

from ml4t.data.adjustments import AdjustedHistory, prepare_adjusted_history
from ml4t.data.assets import InstrumentRegistry
from ml4t.data.storage import FlatStorage, StorageConfig
from ml4t.specs import FeedSpec


def migrate_history(directory: Path) -> tuple[AdjustedHistory, AdjustedHistory, AdjustedHistory]:
    feed = FeedSpec(
        time_contract_version=2,
        entity_col="ticker",
        calendar="NYSE",
        timestamp_semantics="session_label",
        timezone="America/New_York",
        data_frequency="daily",
    )
    prices = pl.read_csv(
        StringIO(
            "timestamp,ticker,exchange,close,volume,currency,available_at,source,revision_id\n"
            "2025-07-01,OLD,NYSE,198,1000,USD,2025-07-01T20:00:00Z,capture,r1\n"
            "2025-07-02,OLD,NYSE,200,1000,USD,2025-07-02T20:00:00Z,capture,r1\n"
            "2025-07-03,OLD,NYSE,98,7,USD,2025-07-03T17:00:00Z,capture,r1\n"
            "2025-07-07,NEW,NYSE,101,9,USD,2025-07-07T20:00:00Z,capture,r1\n"
        ),
        try_parse_dates=True,
    )
    identities = pl.read_csv(
        StringIO(
            "asset,symbol,exchange,instrument_type,valid_from,valid_to,available_at,source,revision_id,listing_date\n"
            "17,OLD,NYSE,equity,2025-07-01,2025-07-07,2025-06-01T12:00:00Z,capture,r1,2000-01-01\n"
            "17,NEW,NYSE,equity,2025-07-07,,2025-06-01T12:00:00Z,capture,r1,2000-01-01\n"
        ),
        try_parse_dates=True,
    )
    members = pl.read_csv(
        StringIO(
            "asset,universe,valid_from,available_at,source,revision_id\n"
            "17,captured_index,2025-07-01,2025-06-01T12:00:00Z,capture,r1\n"
        ),
        try_parse_dates=True,
    )
    actions = pl.read_csv(
        StringIO(
            "asset,event_id,revision_id,action_kind,source,effective_at,available_at,ex_date,split_ratio,cash_per_share,currency,share_basis\n"
            "17,s1,r1,split,capture,2025-07-03T00:00:00Z,2025-07-01T12:00:00Z,2025-07-03,2,,,\n"
            "17,c1,r1,cash_dividend,capture,2025-07-03T00:00:00Z,2025-07-01T12:00:00Z,2025-07-03,,3,USD,post_event\n"
        ),
        try_parse_dates=True,
        schema_overrides={"split_ratio": pl.Float64, "cash_per_share": pl.Float64},
    )
    storage = FlatStorage(StorageConfig(base_path=directory))
    storage.write(prices, "capture/prices", feed.to_dict())
    identity_feed = feed.with_overrides(entity_col="asset", timestamp_col="valid_from")
    storage.write(identities, "capture/identities", identity_feed.to_dict())
    storage.write(members, "capture/membership", identity_feed.to_dict())
    action_feed = feed.with_overrides(entity_col="asset", timestamp_col="effective_at")
    storage.write(actions, "capture/actions", action_feed.to_dict())
    restored = storage.read("capture/prices").collect()
    events = storage.read("capture/actions").collect()
    registry = InstrumentRegistry.from_frame(
        storage.read("capture/identities").collect(),
        storage.read("capture/membership").collect(),
        feed_spec=identity_feed,
    )
    cutoff = datetime(2025, 7, 2, 21, tzinfo=UTC)
    assert registry.lifetimes(date(2025, 7, 2), as_of=cutoff, universe="captured_index")[
        "asset"
    ].to_list() == [17]
    past = prepare_adjusted_history(
        restored,
        events,
        as_of=cutoff,
        feed_spec=feed,
        instruments=registry,
        identity_mode="symbol",
        action_coverage="complete",
    )
    assert past.prices["asset"].to_list() == [17, 17]
    assert past.prices["adj_close"].to_list() == [198, 200]
    assert past.actions.height == 0

    late_revision = restored.filter(pl.col("close") == 200).with_columns(
        pl.lit(999, dtype=restored.schema["close"]).alias("close"),
        pl.lit("r2").alias("revision_id"),
        pl.lit(datetime(2025, 7, 20, tzinfo=UTC)).alias("available_at"),
    )
    perturbed = prepare_adjusted_history(
        pl.concat([restored, late_revision]),
        events,
        as_of=cutoff,
        feed_spec=feed,
        instruments=registry,
        identity_mode="symbol",
        action_coverage="complete",
    )
    assert_frame_equal(past.prices, perturbed.prices)
    later = prepare_adjusted_history(
        restored,
        events,
        as_of=datetime(2025, 7, 7, 21, tzinfo=UTC),
        feed_spec=feed,
        instruments=registry,
        identity_mode="symbol",
        action_coverage="complete",
    )
    assert later.prices["ticker"].to_list() == ["OLD", "OLD", "OLD", "NEW"]
    assert later.prices["asset"].to_list() == [17] * 4
    assert later.prices["close"].to_list() == [198, 200, 98, 101]
    assert later.prices["adj_volume"].to_list() == [2000, 2000, 7, 9]
    assert abs(later.prices["adj_close"][2] / later.prices["adj_close"][1] - 1.01) < 1e-14
    assert later.coverage["missing_periods"].to_list() == [0]
    return past, perturbed, later


if __name__ == "__main__":
    with TemporaryDirectory(prefix="ml4t-dated-history-") as temporary:
        past, _, later = migrate_history(Path(temporary))
        print(past.prices.select("timestamp", "ticker", "asset", "adj_close"))
        print(later.prices.select("timestamp", "ticker", "asset", "adj_close", "adj_volume"))
