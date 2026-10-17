"""Declared calendar choices must survive real ingestion and update workflows."""

from datetime import UTC, date, datetime, timedelta

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from ml4t.data import DataManager
from ml4t.data.core.models import Metadata
from ml4t.data.providers.synthetic import SyntheticProvider
from ml4t.data.storage.config import StorageConfig
from ml4t.data.storage.flat import FlatStorage
from ml4t.data.storage.hive import HiveStorage
from ml4t.data.utils.gaps import GapDetector
from ml4t.specs import FeedSpec


def provider_frame(start="2025-07-15", end="2025-07-17"):
    with SyntheticProvider() as provider:
        return provider.fetch_ohlcv("SPY", start, end, "daily")


@pytest.fixture(params=[FlatStorage, HiveStorage])
def manager(request, tmp_path):
    storage = request.param(StorageConfig(base_path=tmp_path))
    with DataManager(storage=storage) as instance:
        yield instance, storage


def test_explicit_disabled_import_does_not_become_nyse_downstream(manager):
    instance, storage = manager
    frame = provider_frame().filter(pl.col("timestamp").dt.date() != date(2025, 7, 16))
    key = instance.import_data(frame, "SPY", "synthetic", calendar=None)
    declared = instance.get_metadata_for_key(key)
    assert GapDetector().detect_gaps(storage.read(key).collect(), feed_spec=declared) == []
    assert_frame_equal(instance.assign_sessions(frame, feed_spec=declared), frame)


def test_omitted_import_inherits_nyse_but_preserves_omission(manager):
    instance, storage = manager
    frame = provider_frame().filter(pl.col("timestamp").dt.date() != date(2025, 7, 16))
    key = instance.import_data(frame, "SPY", "synthetic", exchange="LSE")
    declared = instance.get_metadata_for_key(key)
    assert declared["time_contract_version"] == 2
    assert "calendar" not in declared
    gaps = GapDetector().detect_gaps(storage.read(key).collect(), feed_spec=declared)
    assert sum(gap.missing_periods for gap in gaps) == 1


def test_load_explicit_none_survives_public_provider_storage_and_session_mapping(manager):
    instance, storage = manager
    key = instance.load("SPY", "2025-07-15", "2025-07-17", provider="synthetic", calendar=None)
    frame = storage.read(key).collect()
    assert_frame_equal(
        instance.assign_sessions(frame, feed_spec=instance.get_metadata_for_key(key)), frame
    )


def test_feed_calendar_is_stored_and_explicit_argument_wins(manager):
    instance, storage = manager
    frame = provider_frame("2025-07-11", "2025-07-14")
    feed = FeedSpec(calendar="24/7", time_contract_version=2, timestamp_semantics="session_label")
    key = instance.import_data(frame, "SPY", "synthetic", feed_spec=feed)
    gaps = GapDetector().detect_gaps(
        storage.read(key).collect(), feed_spec=instance.get_metadata_for_key(key)
    )
    assert sum(gap.missing_periods for gap in gaps) == 2
    key = instance.import_data(frame, "SPY", "synthetic", feed_spec=feed, calendar=None)
    assert (
        GapDetector().detect_gaps(
            storage.read(key).collect(), feed_spec=instance.get_metadata_for_key(key)
        )
        == []
    )


def test_invalid_calendar_does_not_replace_existing_generation(manager):
    instance, storage = manager
    frame = provider_frame()
    key = instance.import_data(frame, "SPY", "synthetic")
    before = storage.get_metadata(key)
    with pytest.raises(ValueError, match="Unknown calendar"):
        instance.import_data(frame, "SPY", "synthetic", calendar="not-a-calendar")
    assert storage.get_metadata(key) == before
    assert_frame_equal(storage.read(key).collect(), frame)


def test_disabled_update_does_not_impute_an_old_provider_outage(manager):
    instance, storage = manager
    historic = provider_frame().filter(pl.col("timestamp").dt.date() != date(2025, 7, 16))
    recent_day = (datetime.now(UTC) - timedelta(days=3)).date()
    recent = provider_frame(str(recent_day - timedelta(days=3)), str(recent_day))
    frame = pl.concat([historic, recent]).unique("timestamp").sort("timestamp")
    key = instance.import_data(frame, "SPY", "synthetic", calendar=None)
    instance.update("SPY", provider="synthetic", check_gaps=True)
    stored = storage.read(key).collect()
    assert stored.filter(pl.col("timestamp").dt.date() == date(2025, 7, 16)).is_empty()
    assert FeedSpec.from_mapping(instance.get_metadata_for_key(key)).resolve_calendar() is None


def test_conflicting_feed_frequency_is_rejected_before_storage_publication(manager):
    instance, storage = manager
    frame = provider_frame()
    with pytest.raises(ValueError, match="frequency"):
        instance.import_data(
            frame,
            "SPY",
            "synthetic",
            frequency="daily",
            feed_spec=FeedSpec(data_frequency="hourly"),
        )
    assert not storage.exists("equities/daily/SPY")


def test_initial_update_passes_explicit_disable_through_real_acquisition(manager):
    instance, storage = manager
    key = instance.update(
        "SPY",
        provider="synthetic",
        calendar=None,
        initial_start="2025-07-15",
        initial_end="2025-07-17",
    )
    assert FeedSpec.from_mapping(instance.get_metadata_for_key(key)).resolve_calendar() is None
    assert storage.read(key).collect().height == 3


def test_declared_grid_and_timezone_survive_import_and_model_projection(manager):
    instance, _ = manager
    feed = FeedSpec(
        time_contract_version=2,
        calendar="24/7",
        data_frequency="hourly",
        bar_type="time",
        timezone="Asia/Tokyo",
        timestamp_semantics="event_time",
        bar_grid_anchor="session_open",
        bar_label="left",
        partial_bar_policy="exclude",
    )
    with SyntheticProvider(calendar_mode="continuous") as provider:
        frame = provider.fetch_ohlcv("SPY", "2025-07-15", "2025-07-15", "hourly")
    key = instance.import_data(frame, "SPY", "synthetic", frequency="hourly", feed_spec=feed)
    model = Metadata.model_validate(instance.get_metadata_for_key(key))
    restored = FeedSpec.from_any(model)
    for name in (
        "calendar",
        "data_frequency",
        "bar_type",
        "timezone",
        "timestamp_semantics",
        "bar_grid_anchor",
        "bar_label",
        "partial_bar_policy",
        "time_contract_version",
    ):
        assert getattr(restored, name) == getattr(feed, name)


def test_explicit_update_disable_overrides_stored_schedule_without_filling(manager):
    instance, storage = manager
    historic = provider_frame().filter(pl.col("timestamp").dt.date() != date(2025, 7, 16))
    recent_day = (datetime.now(UTC) - timedelta(days=3)).date()
    recent = provider_frame(str(recent_day - timedelta(days=3)), str(recent_day))
    frame = pl.concat([historic, recent]).unique("timestamp").sort("timestamp")
    key = instance.import_data(frame, "SPY", "synthetic", calendar="24/7")
    instance.update("SPY", provider="synthetic", calendar=None, check_gaps=True)
    assert (
        storage.read(key)
        .collect()
        .filter(pl.col("timestamp").dt.date() == date(2025, 7, 16))
        .is_empty()
    )
    assert FeedSpec.from_mapping(instance.get_metadata_for_key(key)).resolve_calendar() is None


def test_legacy_unspecified_metadata_is_not_promoted_by_an_ordinary_update(manager):
    instance, storage = manager
    frame = provider_frame()
    key = instance.import_data(frame, "SPY", "synthetic", feed_spec=FeedSpec(calendar=None))
    instance.update("SPY", provider="synthetic", fill_gaps=False)
    declared = FeedSpec.from_mapping(instance.get_metadata_for_key(key))
    assert declared.time_contract_version == 1
    assert declared.resolve_calendar() == "NYSE"
    assert storage.read(key).collect().height >= frame.height


def test_invalid_update_selection_cannot_replace_an_existing_generation(manager):
    instance, storage = manager
    frame = provider_frame()
    key = instance.import_data(frame, "SPY", "synthetic", calendar="24/7")
    before = storage.get_metadata(key)
    with pytest.raises(ValueError, match="Unknown calendar"):
        instance.update("SPY", provider="synthetic", calendar="not-a-calendar", fill_gaps=False)
    assert storage.get_metadata(key) == before
    assert_frame_equal(storage.read(key).collect(), frame)
