"""Daily consumers must preserve declared label dates and map explicit instants to sessions."""

from datetime import date

import polars as pl
import pytest

from ml4t.data import DataManager
from ml4t.data.core.models import Metadata
from ml4t.data.providers.synthetic import SyntheticProvider
from ml4t.data.storage.config import StorageConfig
from ml4t.data.storage.flat import FlatStorage
from ml4t.data.storage.hive import HiveStorage
from ml4t.data.update_manager import GapDetector as RangeDetector
from ml4t.data.utils.gaps import GapDetector
from ml4t.specs import FeedSpec


def provider_frame(start="2025-07-15", end="2025-07-17"):
    with SyntheticProvider() as provider:
        return provider.fetch_ohlcv("SPY", start, end, "daily")


@pytest.fixture(params=[FlatStorage, HiveStorage])
def stored_labels(request, tmp_path):
    storage = request.param(StorageConfig(base_path=tmp_path))
    frame = provider_frame("2025-07-11", "2025-07-14").with_columns(
        pl.col("timestamp").dt.date().cast(pl.Datetime("us")).dt.replace_time_zone("Asia/Tokyo")
    )
    metadata = Metadata(
        provider="synthetic",
        symbol="SPY",
        asset_class="equities",
        bar_params={"frequency": "daily"},
        calendar="JPX",
        time_contract_version=2,
        timezone="Asia/Tokyo",
        timestamp_semantics="session_label",
    )
    key = "equities/daily/SPY"
    storage.write(frame, key, metadata.model_dump())
    with DataManager(storage=storage) as manager:
        yield storage.read(key).collect(), manager.get_metadata_for_key(key)


@pytest.mark.parametrize("consumer", [GapDetector, RangeDetector])
def test_real_utc_storage_roundtrip_does_not_turn_tokyo_labels_into_a_false_friday_outage(
    stored_labels, consumer
):
    frame, metadata = stored_labels
    assert frame["timestamp"].dt.date().to_list() == [date(2025, 7, 10), date(2025, 7, 13)]
    assert consumer().detect_gaps(frame, feed_spec=metadata) == []


@pytest.mark.parametrize("consumer", [GapDetector, RangeDetector])
@pytest.mark.parametrize("semantics", ["event_time", "bar_close"])
def test_explicit_daily_instants_use_their_session_and_keep_a_real_missing_session(
    consumer, semantics
):
    frame = provider_frame().filter(pl.col("timestamp").dt.date() != date(2025, 7, 16))
    feed = FeedSpec(time_contract_version=2, calendar="NYSE", timestamp_semantics=semantics)
    found = consumer().detect_gaps(frame, feed_spec=feed)
    count = sum(g.missing_periods if consumer is GapDetector else g["size_days"] for g in found)
    assert count == 1


@pytest.mark.parametrize("semantics,expected", [("event_time", [12, 13]), ("bar_close", [11, 12])])
def test_continuous_midnight_right_labels_change_the_missing_session_dates(semantics, expected):
    frame = provider_frame("2025-07-11", "2025-07-14").with_columns(
        pl.col("timestamp").dt.date().cast(pl.Datetime("us")).dt.replace_time_zone("UTC")
    )
    found = RangeDetector().detect_gaps(
        frame,
        feed_spec=FeedSpec(
            time_contract_version=2,
            calendar="24/7",
            timestamp_semantics=semantics,
        ),
    )
    assert len(found) == 1
    assert [found[0]["start"].day, found[0]["end"].day] == expected


@pytest.mark.parametrize("consumer", [GapDetector, RangeDetector])
def test_disabled_daily_event_declaration_does_not_restore_a_schedule(consumer):
    frame = provider_frame().gather_every(2)
    feed = FeedSpec(time_contract_version=2, calendar=None, timestamp_semantics="event_time")
    assert consumer().detect_gaps(frame, feed_spec=feed) == []


def test_declared_outside_session_instant_is_not_reinterpreted_as_a_daily_label():
    with SyntheticProvider(calendar_mode="continuous") as provider:
        frame = provider.fetch_ohlcv("SPY", "2025-07-07", "2025-07-07", "hourly").head(2)
    feed = FeedSpec(time_contract_version=2, calendar="NYSE", timestamp_semantics="event_time")
    with pytest.raises(ValueError, match="outside a trading session"):
        GapDetector().detect_gaps(frame, feed_spec=feed)


def test_declared_instants_support_the_public_custom_timestamp_column():
    frame = (
        provider_frame()
        .filter(pl.col("timestamp").dt.date() != date(2025, 7, 16))
        .rename({"timestamp": "stamp"})
    )
    feed = FeedSpec(time_contract_version=2, calendar="NYSE", timestamp_semantics="event_time")
    found = GapDetector().detect_gaps(frame, timestamp_col="stamp", feed_spec=feed)
    assert sum(g.missing_periods for g in found) == 1
