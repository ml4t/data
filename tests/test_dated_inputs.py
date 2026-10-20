"""Dated source intervals must not merge tickers or leak future revisions."""

from datetime import UTC, date, datetime
from io import StringIO

import pandas as pd
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from ml4t.data.assets import InstrumentRegistry
from ml4t.data.storage import FlatStorage, StorageConfig
from ml4t.specs import FeedSpec


def captured_identities():
    return pl.read_csv(
        StringIO(
            "asset,symbol,exchange,instrument_type,valid_from,valid_to,available_at,source,revision_id,listing_date\n"
            "17,OLD,NYSE,equity,2025-07-01,2025-07-07,2025-06-01T12:00:00Z,captured,r1,2000-01-01\n"
            "17,NEW,NYSE,equity,2025-07-07,,2025-06-01T12:00:00Z,captured,r1,2000-01-01\n"
            "23,OLD,NYSE,equity,2025-07-07,,2025-06-01T12:00:00Z,captured,r1,2025-07-07\n"
            "37,NEW,NASDAQ,equity,2025-07-01,,2025-06-01T12:00:00Z,captured,r1,2010-01-01\n"
        ),
        try_parse_dates=True,
    )


def test_csv_storage_registry_retains_rename_and_ticker_reuse_without_merging_ids(tmp_path):
    source = captured_identities()
    feed = FeedSpec(
        time_contract_version=2,
        timestamp_col="valid_from",
        entity_col="asset",
        timestamp_semantics="session_label",
        timezone="America/New_York",
    )
    storage = FlatStorage(StorageConfig(base_path=tmp_path))
    storage.write(source, "captured/identities", feed.to_dict())
    restored = storage.read("captured/identities").collect()
    registry = InstrumentRegistry.from_frame(restored, feed_spec=feed)
    assert registry.resolve("OLD", date(2025, 7, 6), "NYSE") == 17
    assert registry.resolve("NEW", date(2025, 7, 7), "NYSE") == 17
    assert registry.resolve("OLD", date(2025, 7, 7), "NYSE") == 23
    assert registry.resolve("NEW", date(2025, 7, 7), "NASDAQ") == 37
    with pytest.raises(ValueError, match="ambiguous"):
        registry.resolve("NEW", date(2025, 7, 7))
    assert_frame_equal(registry.identities, source)
    assert registry.lifetimes(date(2025, 7, 8))["asset"].to_list() == [17, 23, 37]


def test_future_symbol_revisions_and_unknown_future_availability_cannot_change_past_identity():
    source = captured_identities()
    cutoff = datetime(2025, 7, 8, tzinfo=UTC)
    old = InstrumentRegistry.from_frame(source).lifetimes(as_of=cutoff)
    revised = source.filter((pl.col("asset") == 17) & (pl.col("symbol") == "NEW")).with_columns(
        pl.lit(date(2025, 8, 1)).alias("valid_to"),
        pl.lit(datetime(2025, 7, 20, tzinfo=UTC)).alias("available_at"),
        pl.lit("r2").alias("revision_id"),
    )
    future = source.head(1).with_columns(
        pl.lit("AFTER").alias("symbol"),
        pl.lit(date(2025, 8, 1)).alias("valid_from"),
        pl.lit(None, dtype=pl.Date).alias("valid_to"),
        pl.lit(None, dtype=source.schema["available_at"]).alias("available_at"),
    )
    changed = InstrumentRegistry.from_frame(pl.concat([source, revised, future]))
    assert_frame_equal(changed.lifetimes(as_of=cutoff), old)
    assert changed.resolve("NEW", date(2025, 7, 8), "NYSE", as_of=cutoff) == 17


def test_historical_membership_uses_captured_intervals_not_current_constituents():
    memberships = pl.read_csv(
        StringIO(
            "asset,universe,valid_from,valid_to,available_at,source,revision_id\n"
            "17,captured_index,2025-07-01,2025-07-07,2025-06-01T12:00:00Z,captured,r1\n"
            "23,captured_index,2025-07-07,,2025-06-01T12:00:00Z,captured,r1\n"
        ),
        try_parse_dates=True,
    )
    registry = InstrumentRegistry.from_frame(captured_identities(), memberships)
    assert_frame_equal(registry.memberships, memberships)
    cutoff = datetime(2025, 7, 8, tzinfo=UTC)
    assert registry.lifetimes(date(2025, 7, 6), as_of=cutoff, universe="captured_index")[
        "asset"
    ].to_list() == [17]
    assert registry.lifetimes(date(2025, 7, 7), as_of=cutoff, universe="captured_index")[
        "asset"
    ].to_list() == [23]
    with pytest.raises(ValueError, match="membership intervals"):
        InstrumentRegistry.from_frame(captured_identities()).lifetimes(
            date(2025, 7, 7), universe="current_list"
        )
    with pytest.raises(ValueError, match="source coverage"):
        registry.lifetimes(date(2025, 7, 7), universe="unknown_index")


def test_a_late_membership_revision_cannot_change_the_universe_known_at_an_earlier_cutoff():
    members = pl.read_csv(
        StringIO(
            "asset,universe,valid_from,valid_to,available_at,source,revision_id\n"
            "17,captured_index,2025-07-01,,2025-06-01T12:00:00Z,capture,r1\n"
            "17,captured_index,2025-07-01,2025-07-02,2025-07-20T12:00:00Z,capture,r2\n"
        ),
        try_parse_dates=True,
    )
    registry = InstrumentRegistry.from_frame(captured_identities(), members)
    old = registry.lifetimes(
        date(2025, 7, 2), as_of=datetime(2025, 7, 2, 21, tzinfo=UTC), universe="captured_index"
    )
    assert old["asset"].to_list() == [17]
    assert registry.lifetimes(date(2025, 7, 2), universe="captured_index").height == 0
    assert registry.memberships["revision_id"].to_list() == ["r1", "r2"]


def test_unknown_availability_blocks_causal_resolution_but_not_retrospective_inspection():
    source = captured_identities().with_columns(pl.lit(None).alias("available_at"))
    registry = InstrumentRegistry.from_frame(source)
    assert registry.resolve("OLD", date(2025, 7, 6), "NYSE") == 17
    with pytest.raises(ValueError, match="available_at"):
        registry.resolve("OLD", date(2025, 7, 6), "NYSE", as_of=datetime(2025, 7, 8, tzinfo=UTC))
    assert_frame_equal(
        source, captured_identities().with_columns(pl.lit(None).alias("available_at"))
    )


def test_overlapping_or_duplicate_source_intervals_fail_before_publishing_a_registry():
    source = captured_identities()
    conflicting = source.with_columns(
        pl.when(pl.col("symbol") == "OLD")
        .then(date(2025, 7, 9))
        .otherwise(pl.col("valid_to"))
        .alias("valid_to")
    )
    before = conflicting.clone()
    with pytest.raises(ValueError, match="Overlapping"):
        InstrumentRegistry.from_frame(conflicting)
    assert_frame_equal(conflicting, before)
    with pytest.raises(ValueError, match="Duplicate"):
        InstrumentRegistry.from_frame(pl.concat([source, source.head(1)]))


def test_declared_label_timezone_governs_the_date_of_an_aware_identity_lookup():
    registry = InstrumentRegistry.from_frame(
        captured_identities(), feed_spec=FeedSpec(entity_col="asset", timezone="America/New_York")
    )
    assert registry.resolve("OLD", datetime(2025, 7, 7, 1, tzinfo=UTC), "NYSE") == 17


def test_nanosecond_identity_boundaries_are_not_rounded_by_overlap_checks_or_lookup():
    source = pl.read_csv(
        StringIO(
            "asset,symbol,exchange,instrument_type,valid_from,valid_to,available_at,source,revision_id\n"
            "17,A,NYSE,equity,2025-07-07T00:00:00.000000000Z,2025-07-07T00:00:00.000000001Z,2025-07-01T00:00:00Z,captured,r1\n"
            "17,B,NYSE,equity,2025-07-07T00:00:00.000000001Z,,2025-07-01T00:00:00Z,captured,r1\n"
        )
    ).with_columns(
        pl.col("valid_from", "valid_to", "available_at").str.to_datetime(
            time_unit="ns", time_zone="UTC"
        )
    )
    registry = InstrumentRegistry.from_frame(source)
    assert registry.resolve("A", pd.Timestamp("2025-07-07T00:00:00.000000000Z"), "NYSE") == 17
    assert registry.resolve("B", pd.Timestamp("2025-07-07T00:00:00.000000001Z"), "NYSE") == 17
    with pytest.raises(ValueError, match="Missing"):
        registry.resolve("A", pd.Timestamp("2025-07-07T00:00:00.000000001Z"), "NYSE")


def test_revision_availability_cutoff_retains_nanoseconds_and_rejects_indistinguishable_revisions():
    source = pl.read_csv(
        StringIO(
            "asset,symbol,exchange,instrument_type,valid_from,available_at,source,revision_id\n"
            "17,A,NYSE,equity,2025-07-01,2025-07-07T00:00:00.000000001Z,captured,r1\n"
            "17,B,NYSE,equity,2025-07-01,2025-07-07T00:00:00.000000002Z,captured,r2\n"
        )
    ).with_columns(
        pl.col("valid_from").str.to_date(),
        pl.col("available_at").str.to_datetime(time_unit="ns", time_zone="UTC"),
    )
    registry = InstrumentRegistry.from_frame(source)
    before_revision = pd.Timestamp("2025-07-07T00:00:00.000000001Z")
    after_revision = pd.Timestamp("2025-07-07T00:00:00.000000002Z")
    assert registry.resolve("A", date(2025, 7, 7), "NYSE", as_of=before_revision) == 17
    assert registry.resolve("B", date(2025, 7, 7), "NYSE", as_of=after_revision) == 17
    with pytest.raises(ValueError, match="indistinguishable"):
        InstrumentRegistry.from_frame(source.with_columns(pl.col("available_at").min()))


def test_cme_dated_identity_uses_the_same_current_overnight_session_as_history():
    source = pl.read_csv(
        StringIO(
            "asset,symbol,exchange,instrument_type,valid_from,available_at,source,revision_id\n"
            "17,FUT,CME,future,2025-07-07,2025-07-06T23:30:00Z,captured,r1\n"
        ),
        try_parse_dates=True,
    )
    feed = FeedSpec(
        time_contract_version=2,
        entity_col="asset",
        calendar="CME_Equity",
        timestamp_semantics="session_label",
        timezone="America/New_York",
    )
    registry = InstrumentRegistry.from_frame(source, feed_spec=feed)
    instant = datetime(2025, 7, 6, 23, 45, tzinfo=UTC)
    assert registry.resolve("FUT", instant, "CME", as_of=instant) == 17
