"""History normalization must resolve dated source identities before joining action facts."""

from datetime import UTC, date, datetime
from io import StringIO
from pathlib import Path
from runpy import run_path

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from ml4t.data.adjustments import prepare_adjusted_history
from ml4t.data.assets import InstrumentRegistry
from ml4t.data.storage import FlatStorage, StorageConfig
from ml4t.specs import FeedSpec


def identity_registry():
    return InstrumentRegistry.from_frame(
        pl.read_csv(
            StringIO(
                "asset,symbol,exchange,instrument_type,valid_from,valid_to,available_at,source,revision_id\n"
                "17,OLD,NYSE,equity,2025-07-01,2025-07-07,2025-06-01T12:00:00Z,capture,r1\n"
                "17,NEW,NYSE,equity,2025-07-07,,2025-06-01T12:00:00Z,capture,r1\n"
                "23,OLD,NYSE,equity,2025-07-07,,2025-06-01T12:00:00Z,capture,r1\n"
                "37,NEW,NASDAQ,equity,2025-07-01,,2025-06-01T12:00:00Z,capture,r1\n"
            ),
            try_parse_dates=True,
        )
    )


def source_prices():
    return pl.read_csv(
        StringIO(
            "date,ticker,exchange,close,volume,available_at,source,revision_id\n"
            "2025-07-03,OLD,NYSE,100,7,2025-07-03T17:00:00Z,capture,r1\n"
            "2025-07-07,NEW,NYSE,101,8,2025-07-07T20:00:00Z,capture,r1\n"
            "2025-07-07,OLD,NYSE,50,9,2025-07-07T20:00:00Z,capture,r1\n"
            "2025-07-07,NEW,NASDAQ,60,10,2025-07-07T20:00:00Z,capture,r1\n"
        ),
        try_parse_dates=True,
    )


def source_feed():
    return FeedSpec(
        time_contract_version=2,
        timestamp_col="date",
        entity_col="ticker",
        timestamp_semantics="session_label",
        timezone="America/New_York",
        calendar="NYSE",
        data_frequency="daily",
    )


def test_disk_symbol_history_preserves_rename_reuse_and_venue_with_stable_consumer_ids(tmp_path):
    source = source_prices()
    storage = FlatStorage(StorageConfig(base_path=tmp_path))
    storage.write(source, "capture/symbols", source_feed().to_dict())
    restored = storage.read("capture/symbols").collect()
    result = prepare_adjusted_history(
        restored,
        None,
        instruments=identity_registry(),
        identity_mode="symbol",
        as_of=datetime(2025, 7, 8, tzinfo=UTC),
        feed_spec=source_feed(),
        basis="raw",
    )
    assert result.prices["asset"].to_list() == [17, 17, 23, 37]
    assert result.prices["ticker"].to_list() == ["OLD", "NEW", "OLD", "NEW"]
    assert result.prices["close"].to_list() == [100, 101, 50, 60]
    assert result.identities is not None and result.identities.height == 4
    assert result.identities["revision_id"].to_list() == ["r1"] * 4
    assert FeedSpec.from_any(result.metadata["feed_spec"]).entity_col == "asset"
    assert FeedSpec.from_any(result.metadata["input_feed_spec"]).entity_col == "ticker"
    assert result.coverage["observed_rows"].to_list() == [2, 1, 1]
    assert_frame_equal(source, source_prices())


def test_ambiguous_symbols_invalid_lifetimes_and_stable_id_collisions_fail_atomically():
    source = source_prices()
    with pytest.raises(ValueError, match="Missing or ambiguous"):
        prepare_adjusted_history(
            source.drop("exchange"),
            None,
            instruments=identity_registry(),
            identity_mode="symbol",
            as_of=None,
            feed_spec=source_feed(),
            basis="raw",
        )
    with pytest.raises(ValueError, match="collid"):
        prepare_adjusted_history(
            source.with_columns(pl.lit(99).alias("asset")),
            None,
            instruments=identity_registry(),
            identity_mode="symbol",
            as_of=None,
            feed_spec=source_feed(),
            basis="raw",
        )
    invalid = source.head(1).rename({"ticker": "sid"}).with_columns(pl.lit(23).alias("sid"))
    with pytest.raises(ValueError, match="Missing or ambiguous"):
        prepare_adjusted_history(
            invalid,
            None,
            instruments=identity_registry(),
            as_of=None,
            feed_spec=source_feed().with_overrides(entity_col="sid"),
            basis="raw",
        )
    assert_frame_equal(source, source_prices())


def test_future_price_symbols_and_identity_revisions_do_not_change_known_at_history():
    prices = source_prices()
    cutoff = datetime(2025, 7, 8, tzinfo=UTC)
    future_price = prices.head(1).with_columns(
        pl.lit("UNPUBLISHED").alias("ticker"),
        pl.lit(datetime(2025, 8, 1, tzinfo=UTC)).alias("available_at"),
        pl.lit("r2").alias("revision_id"),
    )
    registry = identity_registry()
    revision = registry.identities.filter(pl.col("asset") == 23).with_columns(
        pl.lit("CHANGED").alias("symbol"),
        pl.lit(datetime(2025, 8, 1, tzinfo=UTC)).alias("available_at"),
        pl.lit("r2").alias("revision_id"),
    )
    before = prepare_adjusted_history(
        prices,
        None,
        instruments=registry,
        identity_mode="symbol",
        as_of=cutoff,
        feed_spec=source_feed(),
        basis="raw",
    )
    after = prepare_adjusted_history(
        pl.concat([prices, future_price]),
        None,
        instruments=InstrumentRegistry.from_frame(pl.concat([registry.identities, revision])),
        identity_mode="symbol",
        as_of=cutoff,
        feed_spec=source_feed(),
        basis="raw",
    )
    assert_frame_equal(before.prices, after.prices)
    assert_frame_equal(before.identities, after.identities)


def test_custom_action_feed_maps_source_keys_without_overwriting_captured_columns():
    prices = source_prices().filter(pl.col("exchange") == "NYSE").head(2)
    actions = pl.read_csv(
        StringIO(
            "effective,security,source,event_id,revision_id,action_kind,split_ratio,available_at\n"
            "2025-07-07,17,capture,s1,r1,split,2,2025-07-01T12:00:00Z\n"
        ),
        try_parse_dates=True,
    )
    result = prepare_adjusted_history(
        prices,
        actions,
        instruments=identity_registry(),
        identity_mode="symbol",
        as_of=datetime(2025, 7, 8, tzinfo=UTC),
        feed_spec=source_feed(),
        action_feed_spec=FeedSpec(
            timestamp_col="effective",
            entity_col="security",
            timestamp_semantics="session_label",
            time_contract_version=2,
        ),
        basis="split_adjusted",
        action_coverage="complete",
    )
    assert result.prices["asset"].to_list() == [17, 17]
    assert result.prices["adj_close"].to_list() == [50, 101]
    assert result.actions["security"].to_list() == [17]
    assert result.actions["effective"].to_list() == [date(2025, 7, 7)]
    assert result.actions["asset"].to_list() == [17]
    assert result.actions["effective_at"].to_list() == [date(2025, 7, 7)]
    assert_frame_equal(actions, actions.clone())


def test_overnight_instant_prices_use_declared_session_identity_intervals():
    feed = FeedSpec(
        time_contract_version=2,
        entity_col="asset",
        calendar="CME_Equity",
        timezone="America/New_York",
        timestamp_semantics="session_label",
    )
    identities = pl.read_csv(
        StringIO(
            "asset,symbol,exchange,instrument_type,valid_from,available_at,source,revision_id\n"
            "17,FUT,CME,future,2025-07-07,2025-07-06T23:00:00Z,capture,r1\n"
        ),
        try_parse_dates=True,
    )
    prices = pl.read_csv(
        StringIO(
            "timestamp,asset,close,available_at,source,revision_id\n"
            "2025-07-06T23:15:00Z,17,100,2025-07-06T23:20:00Z,capture,r1\n"
        ),
        try_parse_dates=True,
    )
    result = prepare_adjusted_history(
        prices,
        None,
        instruments=InstrumentRegistry.from_frame(identities, feed_spec=feed),
        feed_spec=feed.with_overrides(timestamp_semantics="event_time"),
        basis="raw",
        as_of=datetime(2025, 7, 6, 23, 45, tzinfo=UTC),
    )
    assert result.prices["close"].to_list() == [100]
    assert result.identities["valid_from"].to_list() == [date(2025, 7, 7)]


def test_empty_symbol_history_retains_identity_and_source_schemas():
    result = prepare_adjusted_history(
        source_prices().head(0),
        None,
        instruments=identity_registry(),
        identity_mode="symbol",
        as_of=datetime(2025, 7, 8, tzinfo=UTC),
        feed_spec=source_feed(),
        basis="raw",
    )
    assert result.prices.height == 0
    assert result.prices.schema["asset"] == pl.Int64
    assert result.prices.schema["ticker"] == pl.String
    assert result.identities.height == 0
    assert result.identities.schema["valid_from"] == pl.Date


def test_offline_migration_retains_pre_entry_history_and_future_revision_invariance(tmp_path):
    recipe = run_path(
        str(Path(__file__).resolve().parents[1] / "examples/dated_history_migration.py")
    )
    past, changed, later = recipe["migrate_history"](tmp_path)
    assert_frame_equal(past.prices, changed.prices)
    assert past.prices["adj_close"].to_list() == [198, 200]
    assert past.prices["asset"].to_list() == [17, 17]
    assert past.prices["adj_close"][-1] > 199
    assert later.actions["event_id"].to_list() == ["s1", "c1"]
    assert later.coverage["missing_periods"].to_list() == [0]
