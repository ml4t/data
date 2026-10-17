"""Real provider/session workflows preserve declared calendar and timestamp meaning."""

from datetime import date

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
