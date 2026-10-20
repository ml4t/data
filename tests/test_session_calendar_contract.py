"""Real provider/session workflows preserve declared calendar and timestamp meaning."""

from datetime import date, time
from zoneinfo import ZoneInfo

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from ml4t.data import DataManager
from ml4t.data.core.models import Metadata
from ml4t.data.providers.synthetic import SyntheticProvider
from ml4t.data.sessions import SessionAssigner, SessionCompleter
from ml4t.data.storage.config import StorageConfig
from ml4t.data.storage.hive import HiveStorage
from ml4t.specs import FeedSpec


def provider_frame(frequency="daily", *, continuous=False):
    with SyntheticProvider(calendar_mode="continuous" if continuous else "equity") as provider:
        return provider.fetch_ohlcv("SPY", "2025-07-07", "2025-07-07", frequency)


@pytest.fixture
def interrupted_calendar():
    from pandas_market_calendars.market_calendar import MarketCalendar

    class DataSessionHaltCalendar(MarketCalendar):
        name = "DataSessionHalt"
        aliases = ["DataSessionHalt"]
        tz = ZoneInfo("UTC")
        regular_market_times = {
            "market_open": ((None, time(9)),),
            "market_close": ((None, time(14)),),
        }

        @property
        def interruptions(self):
            return [("2025-07-07", time(10), time(10, 30), time(12), time(12, 15))]

    return "DataSessionHalt"


@pytest.mark.parametrize(
    "semantics,expected",
    [
        ("event_time", [True, False, False, True, True, False, False, True]),
        ("bar_close", [True, True, False, False, True, True, False, False]),
    ],
)
def test_scheduled_interruptions_keep_declared_interval_sides(
    interrupted_calendar, semantics, expected
):
    minutes = [
        time(9, 59),
        time(10),
        time(10, 15),
        time(10, 30),
        time(11, 59),
        time(12),
        time(12, 10),
        time(12, 15),
    ]
    frame = provider_frame("minute", continuous=True).filter(
        pl.col("timestamp").dt.time().is_in(minutes)
    )
    feed = FeedSpec(calendar=interrupted_calendar, timestamp_semantics=semantics)
    assigned = SessionAssigner(feed_spec=feed).assign_sessions(frame)
    assert assigned["session_date"].is_not_null().to_list() == expected
    assert_frame_equal(assigned.select(frame.columns), frame)


def test_minute_completion_cannot_impute_a_scheduled_interruption(manager, interrupted_calendar):
    frame = provider_frame("minute", continuous=True).filter(
        pl.col("timestamp").dt.time().is_in([time(9, 59), time(10, 30)])
    )
    completed = manager.complete_sessions(frame, calendar=interrupted_calendar)
    assert completed.height == 2
    assert completed["is_imputed"].to_list() == [False, False]
    assert_frame_equal(completed.select(frame.columns).cast(frame.schema), frame)


def test_minute_completion_rejects_observed_rows_during_a_scheduled_halt(
    manager, interrupted_calendar
):
    frame = provider_frame("minute", continuous=True).filter(
        pl.col("timestamp").dt.time().is_in([time(9, 59), time(10, 15), time(10, 30)])
    )
    original = frame.clone()
    with pytest.raises(ValueError, match="outside"):
        manager.complete_sessions(frame, calendar=interrupted_calendar)
    assert_frame_equal(frame, original)


def test_interrupted_session_has_one_daily_label_not_one_per_tradable_interval(
    interrupted_calendar,
):
    frame = provider_frame()
    feed = FeedSpec(calendar=interrupted_calendar, timestamp_semantics="session_label")
    assigned = SessionAssigner(feed_spec=feed).assign_sessions(frame)
    assert assigned.height == 1
    assert assigned["session_date"].to_list() == [date(2025, 7, 7)]


@pytest.fixture(params=[False, True])
def manager(request, tmp_path):
    storage = HiveStorage(StorageConfig(base_path=tmp_path)) if request.param else None
    with DataManager(storage=storage) as instance:
        yield instance


def test_facade_default_is_nyse_not_a_missing_argument_error(manager):
    frame = provider_frame()
    assigned = manager.assign_sessions(frame, bar_frequency="daily")
    assert assigned["session_date"].to_list() == [date(2025, 7, 7)]


@pytest.mark.parametrize("method", ["assign_sessions", "complete_sessions"])
def test_explicit_none_preserves_provider_rows_without_session_or_imputation(manager, method):
    frame = provider_frame()
    assert_frame_equal(getattr(manager, method)(frame, calendar=None), frame)


def test_continuous_midnight_event_belongs_to_the_new_session():
    frame = provider_frame("hourly", continuous=True).head(2)
    assigned = SessionAssigner("24/7").assign_sessions(frame, bar_frequency="intraday")
    assert assigned["session_date"].to_list() == [date(2025, 7, 7), date(2025, 7, 7)]


def test_no_fill_flag_does_not_expand_a_provider_gap_without_storage(manager):
    frame = provider_frame("minute", continuous=True).filter(
        (pl.col("timestamp").dt.hour() == 13) & pl.col("timestamp").dt.minute().is_in([30, 32])
    )
    assigned = manager.complete_sessions(frame, calendar="NYSE", fill_gaps=False)
    assert assigned.height == 2
    assert "is_imputed" not in assigned.columns
    assert_frame_equal(assigned.select(frame.columns), frame)


def test_contradictory_explicit_exchange_alias_fails_without_input_mutation(manager):
    frame = provider_frame()
    original = frame.clone()
    with pytest.raises(ValueError, match="conflict"):
        manager.assign_sessions(frame, exchange="NYSE", calendar="LSE")
    assert_frame_equal(frame, original)


@pytest.mark.parametrize(
    "semantics, expected", [("event_time", date(2025, 7, 7)), ("bar_close", date(2025, 7, 6))]
)
def test_explicit_timestamp_meaning_overrides_daily_and_midnight_heuristics(semantics, expected):
    frame = provider_frame("hourly", continuous=True).head(1)
    feed = FeedSpec(calendar="24/7", timestamp_semantics=semantics)
    assigned = SessionAssigner(feed_spec=feed).assign_sessions(frame, bar_frequency="daily")
    assert assigned["session_date"].to_list() == [expected]


@pytest.mark.parametrize("calendar, hour", [("NYSE", 20), ("HKEX", 4)])
def test_close_labeled_provider_bar_includes_close_or_break_start(calendar, hour):
    frame = provider_frame("hourly", continuous=True).filter(
        pl.col("timestamp").dt.hour() == hour - 1
    )
    frame = frame.with_columns(pl.col("timestamp") + pl.duration(hours=1))
    feed = FeedSpec(calendar=calendar, timestamp_semantics="bar_close")
    assigned = SessionAssigner(feed_spec=feed).assign_sessions(frame)
    assert assigned["session_date"].to_list() == [date(2025, 7, 7)]
    event = SessionAssigner(
        feed_spec=FeedSpec(calendar=calendar, timestamp_semantics="event_time")
    ).assign_sessions(frame)
    assert event["session_date"].to_list() == [None]


@pytest.mark.parametrize(
    "declaration, expected",
    [
        ({}, "NYSE"),
        ({"calendar": None}, "NYSE"),
        ({"time_contract_version": 2}, "NYSE"),
        ({"time_contract_version": 2, "calendar": None}, None),
        ({"time_contract_version": 2, "calendar": "24/7"}, "24/7"),
    ],
)
def test_data_metadata_json_preserves_calendar_declaration_presence(declaration, expected):
    metadata = Metadata(provider="synthetic", symbol="SPY", asset_class="equities", **declaration)
    restored = Metadata.model_validate_json(metadata.model_dump_json())
    assert FeedSpec.from_any(restored).resolve_calendar() == expected
    assert FeedSpec.from_mapping(metadata.model_dump()).resolve_calendar() == expected
    assert FeedSpec.from_any(restored.model_copy()).resolve_calendar() == expected
    if declaration == {"time_contract_version": 2}:
        assert "calendar" not in restored.model_dump()


def test_feed_disable_survives_metadata_and_public_facade(manager):
    metadata = Metadata(
        provider="synthetic",
        symbol="SPY",
        asset_class="equities",
        time_contract_version=2,
        calendar=None,
    )
    restored = Metadata.model_validate_json(metadata.model_dump_json())
    frame = provider_frame()
    assert_frame_equal(manager.assign_sessions(frame, feed_spec=restored), frame)
    assigned = manager.assign_sessions(
        frame, feed_spec=restored, calendar="NYSE", bar_frequency="daily"
    )
    assert assigned["session_date"].to_list() == [date(2025, 7, 7)]


def test_calendar_name_alias_is_compatible_but_conflicts_are_not_silently_ignored():
    assert SessionAssigner(calendar_name="NYSE").calendar_name == "NYSE"
    assert SessionCompleter(calendar_name="NYSE").calendar_name == "NYSE"
    with pytest.raises(ValueError, match="conflict"):
        SessionAssigner(calendar="NYSE", calendar_name="LSE")


def test_calendar_free_low_level_completion_is_not_a_continuous_grid():
    frame = provider_frame()
    assert_frame_equal(SessionCompleter(calendar=None).complete_sessions(frame), frame)


def test_declared_bar_side_is_not_inferred_from_event_time():
    frame = provider_frame("hourly", continuous=True).head(1)
    feed = FeedSpec(
        calendar="24/7",
        time_contract_version=2,
        timestamp_semantics="event_time",
        bar_label="right",
    )
    assert SessionAssigner(feed_spec=feed).assign_sessions(frame)["session_date"].to_list() == [
        date(2025, 7, 6)
    ]


def test_persisted_data_metadata_retains_calendar_disable_and_bar_grid(tmp_path):
    import yaml

    metadata = Metadata(
        provider="synthetic",
        symbol="SPY",
        asset_class="equities",
        calendar=None,
        time_contract_version=2,
        timezone="UTC",
        timestamp_semantics="event_time",
        bar_grid_anchor="session_open",
        bar_label="right",
        partial_bar_policy="include",
    )
    restored = Metadata.model_validate(
        yaml.safe_load(yaml.safe_dump(metadata.model_dump(mode="json")))
    )
    feed = FeedSpec.from_any(restored)
    assert feed.resolve_calendar() is None
    assert feed.bar_label == "right"
    assert feed.bar_grid_anchor == "session_open"
    assert feed.partial_bar_policy == "include"
    frame = provider_frame("hourly", continuous=True).head(1)
    storage = HiveStorage(StorageConfig(base_path=tmp_path))
    storage.write(frame, "equities/hourly/SPY", restored.model_dump())
    with DataManager(storage=storage) as manager:
        declared = manager.get_metadata_for_key("equities/hourly/SPY")
        assert_frame_equal(manager.assign_sessions(frame, feed_spec=declared), frame)


def test_explicit_session_labels_keep_civil_dates_in_non_utc_timezone():
    frame = provider_frame().with_columns(
        pl.col("timestamp")
        .dt.replace_time_zone(None)
        .dt.truncate("1d")
        .dt.replace_time_zone("Asia/Tokyo")
    )
    feed = FeedSpec(calendar="NYSE", timestamp_semantics="session_label", timezone="Asia/Tokyo")
    result = SessionAssigner(feed_spec=feed).assign_sessions(frame, bar_frequency="intraday")
    assert result["session_date"].to_list() == [date(2025, 7, 7)]


def test_naive_event_time_uses_declared_timezone_and_rejects_ambiguous_clock():
    frame = provider_frame("hourly", continuous=True).filter(pl.col("timestamp").dt.hour() == 14)
    local = frame.with_columns(
        pl.col("timestamp").dt.convert_time_zone("America/New_York").dt.replace_time_zone(None)
    )
    feed = FeedSpec(calendar="NYSE", timestamp_semantics="event_time", timezone="America/New_York")
    assert SessionAssigner(feed_spec=feed).assign_sessions(local)["session_date"].to_list() == [
        date(2025, 7, 7)
    ]
    with SyntheticProvider(calendar_mode="continuous") as provider:
        ambiguous = (
            provider.fetch_ohlcv("SPY", "2025-11-02", "2025-11-02", "hourly")
            .filter(pl.col("timestamp").dt.hour() == 1)
            .with_columns(pl.col("timestamp").dt.replace_time_zone(None))
        )
    with pytest.raises(pl.exceptions.ComputeError, match="ambiguous"):
        SessionAssigner(feed_spec=feed).assign_sessions(ambiguous)


def test_real_provider_update_preserves_a_stored_explicit_calendar_disable(tmp_path):
    frame = provider_frame()
    storage = HiveStorage(StorageConfig(base_path=tmp_path))
    storage.write(
        frame,
        "equities/daily/SPY",
        {"provider": "synthetic", "calendar": None, "time_contract_version": 2},
    )
    with DataManager(storage=storage) as manager:
        assert (
            FeedSpec.from_mapping(
                manager.get_metadata_for_key("equities/daily/SPY")
            ).resolve_calendar()
            is None
        )
        manager.update("SPY", fill_gaps=False, provider="synthetic")
        assert (
            FeedSpec.from_mapping(
                manager.get_metadata_for_key("equities/daily/SPY")
            ).resolve_calendar()
            is None
        )


@pytest.mark.parametrize("invalid", [{"calendar": {"bad": True}}, {"time_contract_version": 3}])
def test_real_provider_data_cannot_replace_a_dataset_with_invalid_temporal_metadata(
    tmp_path, invalid
):
    frame = provider_frame()
    storage = HiveStorage(StorageConfig(base_path=tmp_path))
    storage.write(frame, "equities/daily/SPY", {"provider": "synthetic"})
    before = storage.get_metadata("equities/daily/SPY")
    with pytest.raises(ValueError, match="calendar|time_contract_version"):
        storage.write(frame.head(0), "equities/daily/SPY", {"provider": "synthetic", **invalid})
    assert storage.get_metadata("equities/daily/SPY") == before
    assert_frame_equal(storage.read("equities/daily/SPY").collect(), frame)


@pytest.mark.parametrize("timezone", ["", "Not/A_Zone"])
@pytest.mark.parametrize("adapter", [SessionAssigner, SessionCompleter])
def test_invalid_declared_timezone_does_not_silently_become_utc(timezone, adapter):
    frame = provider_frame().with_columns(pl.col("timestamp").dt.replace_time_zone(None))
    feed = FeedSpec.from_mapping({"timezone": timezone, "timestamp_semantics": "event_time"})
    original = frame.clone()
    with pytest.raises((ValueError, pl.exceptions.ComputeError), match="time.?zone"):
        instance = adapter(feed_spec=feed)
        if adapter is SessionAssigner:
            instance.assign_sessions(frame)
        else:
            instance.complete_sessions(frame)
    assert_frame_equal(frame, original)


@pytest.mark.parametrize(
    "declaration, message",
    [({"bar_type": "volume"}, "time bars"), ({"data_frequency": "hourly"}, "minute")],
)
def test_minute_completion_rejects_a_declared_non_minute_or_event_count_feed(declaration, message):
    frame = provider_frame("hourly", continuous=True).filter(pl.col("timestamp").dt.hour() == 14)
    with pytest.raises((ValueError, NotImplementedError), match=message):
        SessionCompleter(feed_spec=FeedSpec(**declaration)).complete_sessions(frame)


def test_session_start_override_is_preserved_but_not_silently_applied_as_native_schedule():
    metadata = Metadata(
        provider="synthetic",
        symbol="SPY",
        asset_class="equities",
        time_contract_version=2,
        session_start_time="12:00",
    )
    restored = Metadata.model_validate_json(metadata.model_dump_json())
    assert FeedSpec.from_any(restored).session_start_time == "12:00"
    with pytest.raises(NotImplementedError, match="session_start_time"):
        SessionAssigner(feed_spec=restored)


def test_native_minute_completion_does_not_silently_ignore_general_grid_policies():
    frame = provider_frame("hourly", continuous=True).filter(pl.col("timestamp").dt.hour() == 14)
    feed = FeedSpec(
        time_contract_version=2, bar_grid_anchor="session_open", partial_bar_policy="exclude"
    )
    with pytest.raises(NotImplementedError, match="bar-grid"):
        SessionCompleter(feed_spec=feed).complete_sessions(frame)
