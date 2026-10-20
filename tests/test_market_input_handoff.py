"""Imported source observations must survive disk and analytical handoffs unchanged."""

import runpy
from datetime import UTC, date, datetime
from io import StringIO
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from ml4t.data import DataManager
from ml4t.data.sessions import SessionAssigner
from ml4t.data.storage.config import StorageConfig
from ml4t.data.storage.flat import FlatStorage
from ml4t.data.storage.hive import HiveStorage
from ml4t.data.utils.conversion import from_price_panel, polars_to_pandas, to_price_panel
from ml4t.specs import FeedSpec


def source_observations(*, labels=False):
    text = (
        "timestamp,asset,open,high,low,close,volume,available_at\n"
        "2025-07-07T00:15:00.000000001,17,100,101,99,100,0,2025-07-08\n"
        "2025-07-07T00:15:00.000000002,17,100,101,99,100,,2025-07-08\n"
    )
    return pl.read_csv(StringIO(text)).with_columns(
        pl.col("timestamp")
        .str.to_datetime(time_unit="ns")
        .dt.replace_time_zone("Pacific/Kiritimati" if labels else "UTC")
    )


@pytest.fixture(params=[FlatStorage, HiveStorage])
def storage(request, tmp_path):
    return request.param(StorageConfig(base_path=tmp_path, partition_granularity="day"))


@pytest.mark.parametrize("labels", [False, True])
def test_imported_adjacent_nanoseconds_and_session_labels_survive_disk(storage, labels):
    source = source_observations(labels=labels)
    declaration = FeedSpec(
        time_contract_version=2,
        calendar=None,
        timestamp_semantics="session_label" if labels else "event_time",
        timezone="Pacific/Kiritimati" if labels else "UTC",
    )
    with DataManager(storage=storage, enable_validation=False) as manager:
        key = manager.import_data(
            source, "captured", "file", frequency="tick", feed_spec=declaration
        )
        restored = storage.read(key).collect().select(source.columns)
        expected = source.with_columns(pl.col("timestamp").dt.convert_time_zone("UTC"))
        assert_frame_equal(restored, expected)
        assert restored["timestamp"].dt.convert_time_zone(
            declaration.timezone
        ).dt.date().to_list() == (source["timestamp"].dt.date().to_list())
        metadata = manager.get_metadata_for_key(key)
        assert metadata["calendar"] is None
        assert metadata["timestamp_semantics"] == declaration.timestamp_semantics


def test_utc_bounds_do_not_prune_source_timezone_partition(storage):
    source = source_observations(labels=True)
    declaration = FeedSpec(time_contract_version=2, calendar=None, timestamp_semantics="event_time")
    with DataManager(storage=storage, enable_validation=False) as manager:
        key = manager.import_data(
            source, "captured", "file", frequency="tick", feed_spec=declaration
        )
        restored = storage.read(
            key,
            datetime(2025, 7, 6, 10, tzinfo=UTC),
            datetime(2025, 7, 6, 11, tzinfo=UTC),
            columns=["asset", "close", "volume"],
        ).collect()
        assert_frame_equal(restored, source.select("asset", "close", "volume"))


def test_naive_bounds_on_declared_labels_use_label_dates_not_utc_instants(storage):
    source = source_observations(labels=True)
    declaration = FeedSpec(
        time_contract_version=2,
        calendar="24/7",
        timezone="Pacific/Kiritimati",
        timestamp_semantics="session_label",
    )
    with DataManager(storage=storage, enable_validation=False) as manager:
        key = manager.import_data(
            source, "captured", "file", frequency="tick", feed_spec=declaration
        )
        restored = storage.read(key, datetime(2025, 7, 7), datetime(2025, 7, 8)).collect()
        assert_frame_equal(
            restored.select(source.columns),
            source.with_columns(pl.col("timestamp").dt.convert_time_zone("UTC")),
        )
        assert (
            storage.read(
                key,
                datetime(2025, 7, 8, tzinfo=ZoneInfo("Pacific/Kiritimati")),
            )
            .collect()
            .is_empty()
        )


def captured_prices():
    return pl.read_csv(
        StringIO(
            "timestamp,asset,close\n"
            "2025-07-02,17,100\n2025-07-02,23,50\n"
            "2025-07-03,17,101\n2025-07-03,23,52\n"
            "2025-07-07,17,98\n2025-07-07,23,\n"
            "2025-07-08,17,103\n2025-07-08,23,53\n"
        ),
        try_parse_dates=True,
        schema_overrides={"close": pl.Float64},
    )


@pytest.mark.parametrize("calendar", [None, "NYSE", "24/7"])
def test_panel_retains_complete_clock_typed_ids_and_observed_null_without_filling(calendar):
    prices = captured_prices()
    feed = FeedSpec(time_contract_version=2, entity_col="asset", calendar=calendar)
    provenance = {"price_basis": "raw", "currency": "USD", "source_revision": "captured-1"}
    panel = to_price_panel(prices, feed_spec=feed, metadata=provenance)
    assert panel.columns.tolist() == [17, 23]
    assert panel[17].tolist() == [100, 101, 98, 103]
    assert panel[23].iloc[:2].tolist() == [50, 52]
    assert pd.isna(panel[23].iloc[2])
    assert panel[23].iloc[3] == 53
    assert len(panel.index) == 4
    assert panel.attrs["ml4t_price_panel"]["metadata"] == provenance
    assert panel.attrs["ml4t_price_panel"]["feed_spec"]["calendar"] == calendar
    assert_frame_equal(from_price_panel(panel), prices)
    assert_frame_equal(prices, captured_prices())


def test_ragged_panel_distinguishes_absent_observation_from_observed_null():
    source = captured_prices().filter(
        ~((pl.col("asset") == 23) & (pl.col("timestamp").dt.day() == 8))
    )
    panel = to_price_panel(source)
    assert pd.isna(panel[23].iloc[2]) and pd.isna(panel[23].iloc[3])
    observed = panel.attrs["ml4t_price_panel"]["observed"]
    assert [row[1] for row in observed] == [True, True, True, False]
    assert_frame_equal(from_price_panel(panel), source)
    panel.iloc[3, 1] = 42
    with pytest.raises(ValueError, match="absent observation"):
        from_price_panel(panel)


def test_custom_panel_mappings_and_nanosecond_timezone_roundtrip():
    source = (
        source_observations(labels=True)
        .select("timestamp", "asset", "close")
        .rename({"timestamp": "stamp", "asset": "instrument", "close": "raw_price"})
    )
    feed = FeedSpec(
        timestamp_col="stamp",
        entity_col="instrument",
        price_col="raw_price",
        calendar=None,
        time_contract_version=2,
        timestamp_semantics="session_label",
        timezone="Pacific/Kiritimati",
    )
    panel = to_price_panel(source, feed_spec=feed)
    assert panel.index[1].value - panel.index[0].value == 1
    assert_frame_equal(from_price_panel(panel), source)


@pytest.mark.parametrize(
    "bad_feed",
    [
        {"entity_col": "asset", "symbol_col": "symbol"},
        {"timestamp_col": "stamp", "time_col": "timestamp"},
    ],
)
def test_panel_rejects_conflicting_aliases_without_mutating_source(bad_feed):
    source = captured_prices()
    with pytest.raises(ValueError, match="Conflicting"):
        to_price_panel(source, feed_spec=bad_feed)
    assert_frame_equal(source, captured_prices())


def test_duplicate_keys_ambiguous_entities_and_stale_masks_are_not_silently_repaired():
    source = captured_prices()
    with pytest.raises(ValueError, match="Duplicate"):
        to_price_panel(pl.concat([source, source.head(1)]))
    with pytest.raises(ValueError, match="ambiguous"):
        to_price_panel(source.with_columns(pl.col("asset").alias("symbol")))
    panel = to_price_panel(source)
    panel.index = panel.index[::-1]
    with pytest.raises(ValueError, match="mask"):
        from_price_panel(panel)


def test_empty_storage_retains_schema_and_calendar_declaration(storage):
    source = source_observations().head(0)
    declaration = FeedSpec(time_contract_version=2, calendar=None).to_dict()
    storage.write(source, "empty", declaration)
    restored = storage.read("empty").collect()
    assert_frame_equal(restored.select(source.columns), source)
    assert storage.get_metadata("empty")["custom"]["calendar"] is None


def test_omitted_calendar_is_not_serialized_as_null_by_panel():
    panel = to_price_panel(captured_prices(), feed_spec=FeedSpec(time_contract_version=2))
    assert "calendar" not in panel.attrs["ml4t_price_panel"]["feed_spec"]


def test_panel_keeps_a_whole_missing_price_session_instead_of_compressing_the_clock():
    source = captured_prices().with_columns(
        pl.when(pl.col("timestamp") == date(2025, 7, 7))
        .then(None)
        .otherwise(pl.col("close"))
        .alias("close")
    )
    panel = to_price_panel(source)
    assert panel.index.day.tolist() == [2, 3, 7, 8]
    assert panel.iloc[2].isna().all()
    assert_frame_equal(from_price_panel(panel), source)


@pytest.mark.parametrize("calendar", ["omitted", None, "NYSE", "24/7"])
def test_storage_keeps_source_facts_and_calendar_presence(storage, calendar):
    metadata = {
        "time_contract_version": 2,
        "timestamp_semantics": "event_time",
        "source_revision": "file-v2",
        "price_basis": "raw",
        "volume_basis": "shares",
        "availability": "available_at",
        "numerical_scale": "USD",
    }
    if calendar != "omitted":
        metadata["calendar"] = calendar
    source = source_observations()
    storage.write(source, "captured", metadata)
    assert storage.get_metadata("captured")["custom"] == metadata
    assert_frame_equal(storage.read("captured").collect().select(source.columns), source)


def test_label_timezone_is_not_erased_when_the_declaration_has_no_timezone(storage):
    source = source_observations(labels=True)
    storage.write(
        source,
        "labels",
        {
            "time_contract_version": 2,
            "calendar": None,
            "timestamp_semantics": "session_label",
        },
    )
    assert_frame_equal(storage.read("labels").collect().select(source.columns), source)


def test_naive_late_session_labels_are_not_pruned_as_utc_instants(storage):
    source = source_observations().with_columns(
        pl.col("timestamp").dt.replace_time_zone(None).dt.offset_by("23h")
    )
    feed = FeedSpec(
        time_contract_version=2,
        calendar="24/7",
        timezone="Asia/Tokyo",
        timestamp_semantics="session_label",
    )
    storage.write(source, "labels", feed.to_dict())
    restored = storage.read("labels", datetime(2025, 7, 7), datetime(2025, 7, 8)).collect()
    assert restored.height == 2
    assert restored["timestamp"].dt.convert_time_zone("Asia/Tokyo").dt.date().to_list() == [
        date(2025, 7, 7),
        date(2025, 7, 7),
    ]


@pytest.mark.parametrize("semantics", ["invalid", "event_time"])
def test_invalid_declaration_is_not_silently_used_by_storage(storage, semantics):
    source = source_observations()
    metadata = {"time_contract_version": 2, "timestamp_semantics": semantics}
    if semantics == "event_time":
        metadata["bar_grid_anchor"] = "not-a-grid"
    storage.write(source, "valid")
    previous = storage.get_metadata("valid")
    with pytest.raises(ValueError):
        storage.write(source.head(1), "valid", metadata)
    assert storage.get_metadata("valid") == previous
    assert_frame_equal(storage.read("valid").collect().select(source.columns), source)


def test_empty_price_panel_and_plain_pandas_panel_do_not_drop_null_cells():
    empty = captured_prices().head(0)
    assert_frame_equal(from_price_panel(to_price_panel(empty)), empty)
    plain = pd.DataFrame(
        {17: [100.0, float("nan")]}, index=pd.to_datetime(["2025-07-02", "2025-07-03"])
    )
    restored = from_price_panel(plain)
    assert restored.height == 2
    assert restored["close"].to_list() == [100, None]


@pytest.mark.parametrize("dtype,base", [(pl.Int64, 2**53), (pl.UInt64, 2**63)])
def test_price_panel_preserves_integer_price_scale_larger_than_float_exactness(dtype, base):
    source = captured_prices().with_columns(
        pl.Series("close", [base + 1, None, base + 3, 4, 5, 6, 7, 8], dtype=dtype)
    )
    panel = to_price_panel(source)
    assert_frame_equal(from_price_panel(panel), source)


@pytest.mark.parametrize(
    "calendar,instants,expected",
    [
        ("CME_Equity", ["2025-07-06T23:30:00Z", "2025-07-07T00:30:00Z"], ["2025-07-07"] * 2),
        ("NYSE", ["2025-07-03T16:30:00Z", "2025-07-03T17:00:00Z"], ["2025-07-03"] * 2),
        ("NYSE", ["2025-03-07T15:00:00Z", "2025-03-10T14:00:00Z"], ["2025-03-07", "2025-03-10"]),
        ("24/7", ["2025-07-05T12:00:00Z", "2025-07-06T12:00:00Z"], ["2025-07-05", "2025-07-06"]),
    ],
)
def test_calendar_session_identity_survives_actual_disk_and_panel_handoff(
    storage, calendar, instants, expected
):
    source = pl.read_csv(
        StringIO("timestamp,asset,close\n" + "\n".join(f"{instant},17,100" for instant in instants))
    ).with_columns(pl.col("timestamp").str.to_datetime(time_unit="ns", time_zone="UTC"))
    feed = FeedSpec(
        time_contract_version=2,
        calendar=calendar,
        entity_col="asset",
        timestamp_semantics="event_time",
        data_frequency="30min",
        bar_grid_anchor="session_open",
        bar_label="right",
        partial_bar_policy="exclude",
    )
    storage.write(source, "calendar", feed.to_dict())
    feed = FeedSpec.from_any(storage.get_metadata("calendar")["custom"])
    frame = from_price_panel(to_price_panel(storage.read("calendar").collect(), feed_spec=feed))
    assert_frame_equal(frame, source)
    sessions = SessionAssigner(feed_spec=feed).assign_sessions(
        frame, bar_frequency="intraday", outside_session="raise"
    )
    assert sessions["session_date"].to_list() == [date.fromisoformat(value) for value in expected]


def test_offline_migration_recipe_executes_the_public_storage_and_panel_workflow(tmp_path):
    recipe = runpy.run_path(str(Path(__file__).parents[1] / "examples/price_panel_migration.py"))
    native, panel = recipe["migrate_prices"](tmp_path)
    assert native.height == 8
    assert panel.columns.to_list() == ["A", "B"]


def test_panel_metadata_does_not_break_ordinary_pandas_concatenation():
    first = to_price_panel(captured_prices())
    second = to_price_panel(captured_prices())
    combined = pd.concat([first, second])
    assert combined.shape == (8, 2)


def test_declared_physical_columns_survive_bounded_storage_to_panel_without_renaming(storage):
    source = (
        source_observations()
        .select("timestamp", "asset", "close")
        .rename({"timestamp": "stamp", "asset": "instrument", "close": "raw_price"})
    )
    feed = FeedSpec(
        time_contract_version=2,
        calendar=None,
        timestamp_col="stamp",
        entity_col="instrument",
        price_col="raw_price",
        data_frequency="1ns",
        timestamp_semantics="event_time",
    )
    storage.write(source, "custom", feed.to_dict())
    native = storage.read(
        "custom", datetime(2025, 7, 7, tzinfo=UTC), datetime(2025, 7, 8, tzinfo=UTC)
    ).collect()
    panel = to_price_panel(native, feed_spec=feed)
    assert_frame_equal(from_price_panel(panel), source)
    assert panel.attrs["ml4t_price_panel"]["feed_spec"]["data_frequency"] == "1ns"
    assert storage.read("custom", datetime(2025, 7, 8, tzinfo=UTC)).collect().is_empty()


@pytest.mark.parametrize("local_hour,utc_day,utc_hour", [(9, 7, 13), (23, 8, 3)])
def test_imported_naive_instants_honor_the_declared_source_timezone(
    storage, local_hour, utc_day, utc_hour
):
    source = source_observations().with_columns(
        pl.col("timestamp").dt.replace_time_zone(None).dt.offset_by(f"{local_hour}h15m")
    )
    feed = FeedSpec(
        time_contract_version=2,
        calendar="NYSE",
        timezone="America/New_York",
        timestamp_semantics="event_time",
    )
    with DataManager(storage=storage, enable_validation=False) as manager:
        key = manager.import_data(source, "captured", "file", frequency="tick", feed_spec=feed)
        restored = storage.read(key).collect()
        assert restored["timestamp"].dt.hour().to_list() == [utc_hour, utc_hour]
        assert restored["timestamp"].dt.day().to_list() == [utc_day, utc_day]
        bounded = storage.read(
            key,
            datetime(2025, 7, utc_day, utc_hour, tzinfo=UTC),
            datetime(2025, 7, utc_day, utc_hour + 1, tzinfo=UTC),
        ).collect()
        assert_frame_equal(bounded.select(source.columns), restored.select(source.columns))


@pytest.mark.parametrize("timezone", ["", "Not/A_Zone", 7])
def test_bad_timezone_metadata_cannot_replace_a_valid_source_generation(storage, timezone):
    source = source_observations()
    storage.write(source, "valid")
    before = storage.get_metadata("valid")
    with pytest.raises(ValueError, match="timezone"):
        storage.write(source.head(1), "valid", {"timezone": timezone})
    assert storage.get_metadata("valid") == before
    assert_frame_equal(storage.read("valid").collect().select(source.columns), source)


def test_price_panel_keeps_native_nan_distinct_from_an_explicit_null_source_cell():
    source = pl.read_csv(
        StringIO("timestamp,asset,close\n2025-07-07,A,NaN\n2025-07-07,B,\n"),
        try_parse_dates=True,
        schema_overrides={"close": pl.Float64},
    )
    restored = from_price_panel(to_price_panel(source))
    assert_frame_equal(restored, source)
    assert restored["close"].null_count() == 1
    assert restored["close"].is_nan().sum() == 1


def test_pandas_input_retains_a_late_local_session_label_without_a_timezone_declaration():
    source = (
        source_observations(labels=True)
        .select("timestamp", "asset", "close")
        .with_columns(pl.col("timestamp").dt.offset_by("23h"))
    )
    feed = FeedSpec(time_contract_version=2, calendar=None, timestamp_semantics="session_label")
    restored = from_price_panel(to_price_panel(polars_to_pandas(source), feed_spec=feed))
    assert_frame_equal(restored, source)


def test_price_panel_attrs_do_not_break_pandas_parquet_and_sparse_inverse(tmp_path):
    pytest.importorskip("pyarrow")
    source = captured_prices().filter(
        ~((pl.col("asset") == 23) & (pl.col("timestamp").dt.day() == 8))
    )
    panel = to_price_panel(source)
    path = tmp_path / "prices.parquet"
    panel.to_parquet(path)
    restored = from_price_panel(pd.read_parquet(path))
    assert_frame_equal(restored, source)
