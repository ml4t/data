"""Backfill planning must request missing sessions, not exchange closures."""

from datetime import UTC, date, datetime

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from ml4t.data.providers.synthetic import SyntheticProvider
from ml4t.data.storage.config import StorageConfig
from ml4t.data.storage.flat import FlatStorage
from ml4t.data.storage.hive import HiveStorage
from ml4t.data.update_manager import GapDetector, IncrementalUpdater, UpdateStrategy
from ml4t.specs import FeedSpec


def provider_frame(start: str, end: str) -> pl.DataFrame:
    with SyntheticProvider() as provider:
        return provider.fetch_ohlcv("SPY", start, end, "daily")


@pytest.fixture(params=[FlatStorage, HiveStorage])
def storage(request, tmp_path):
    return request.param(StorageConfig(base_path=tmp_path))


def test_range_planner_does_not_request_a_provider_holiday() -> None:
    frame = provider_frame("2025-07-03", "2025-07-07")
    frame = frame.filter(pl.col("timestamp").dt.date() != date(2025, 7, 4))
    assert GapDetector(exclude_weekends=True).detect_gaps(frame) == []


def test_range_planner_default_excludes_exchange_weekends() -> None:
    frame = provider_frame("2025-07-11", "2025-07-14")
    assert GapDetector().detect_gaps(frame) == []


def test_storage_closed_boundary_is_not_a_gap(storage) -> None:
    storage.write(provider_frame("2025-07-03", "2025-07-03"), "equities/daily/SPY")
    gaps = GapDetector().detect_gaps_in_storage(
        storage, "equities/daily/SPY", datetime(2025, 7, 3), datetime(2025, 7, 6)
    )
    assert gaps == []


@pytest.mark.parametrize("calendar, count", [("NYSE", 0), ("24/7", 2), (None, 0)])
def test_entirely_absent_dataset_uses_the_selected_schedule(storage, calendar, count) -> None:
    gaps = GapDetector(calendar=calendar).detect_gaps_in_storage(
        storage, "equities/daily/SPY", datetime(2025, 7, 12), datetime(2025, 7, 13)
    )
    assert sum(gap["size_days"] for gap in gaps) == count


def test_all_assets_missing_a_stored_session_stays_missing(storage) -> None:
    frame = provider_frame("2025-07-14", "2025-07-18")
    frame = frame.filter(pl.col("timestamp").dt.date() != date(2025, 7, 16))
    for symbol in ("SPY", "QQQ"):
        key = f"equities/daily/{symbol}"
        storage.write(frame.with_columns(pl.lit(symbol).alias("symbol")), key)
        gaps = GapDetector().detect_gaps_in_storage(
            storage, key, datetime(2025, 7, 14), datetime(2025, 7, 18)
        )
        assert gaps == [
            {
                "start": datetime(2025, 7, 16, tzinfo=UTC),
                "end": datetime(2025, 7, 16, tzinfo=UTC),
                "size_days": 1,
            }
        ]


def test_missing_early_close_and_boundary_sessions_are_not_hidden(storage) -> None:
    frame = provider_frame("2025-07-03", "2025-07-03")
    storage.write(frame, "equities/daily/SPY")
    gaps = GapDetector().detect_gaps_in_storage(
        storage, "equities/daily/SPY", datetime(2025, 7, 2), datetime(2025, 7, 7)
    )
    assert [gap["start"].date() for gap in gaps] == [date(2025, 7, 2), date(2025, 7, 7)]
    assert [gap["size_days"] for gap in gaps] == [1, 1]


@pytest.mark.parametrize("calendar, count", [("NYSE", 0), ("24/7", 2), (None, 0)])
def test_persisted_feed_selection_is_preserved(storage, calendar, count) -> None:
    frame = provider_frame("2025-07-11", "2025-07-14")
    feed = FeedSpec(calendar=calendar, time_contract_version=2)
    storage.write(frame, "equities/daily/SPY", feed.to_dict())
    gaps = GapDetector().detect_gaps_in_storage(
        storage, "equities/daily/SPY", datetime(2025, 7, 11), datetime(2025, 7, 14)
    )
    assert sum(gap["size_days"] for gap in gaps) == count
    assert (
        GapDetector(calendar=None).detect_gaps_in_storage(
            storage, "equities/daily/SPY", datetime(2025, 7, 11), datetime(2025, 7, 14)
        )
        == []
    )


def test_explicit_calendar_overrides_stored_crypto_schedule(storage) -> None:
    frame = provider_frame("2025-07-11", "2025-07-14")
    storage.write(frame, "equities/daily/SPY", FeedSpec(calendar="24/7").to_dict())
    assert (
        GapDetector(calendar="NYSE").detect_gaps_in_storage(
            storage, "equities/daily/SPY", datetime(2025, 7, 11), datetime(2025, 7, 14)
        )
        == []
    )


def test_date_label_range_includes_non_midnight_provider_bar_during_backfill(storage) -> None:
    complete = provider_frame("2025-07-14", "2025-07-18")
    existing = complete.filter(pl.col("timestamp").dt.date() != date(2025, 7, 16))
    new = complete.filter(pl.col("timestamp").dt.date() == date(2025, 7, 16))
    storage.write(existing, "equities/daily/SPY", {"calendar": "NYSE"})
    result = IncrementalUpdater().apply_strategy(
        storage, "equities/daily/SPY", new, UpdateStrategy.BACKFILL
    )
    assert result.rows_added == 1
    assert_frame_equal(storage.read("equities/daily/SPY").collect(), complete)


def test_invalid_calendar_leaves_stored_generation_unchanged(storage) -> None:
    frame = provider_frame("2025-07-11", "2025-07-14")
    storage.write(frame, "equities/daily/SPY")
    before = storage.get_metadata("equities/daily/SPY")
    with pytest.raises(ValueError, match="Unknown calendar"):
        GapDetector(calendar="invalid-calendar").detect_gaps_in_storage(
            storage, "equities/daily/SPY", datetime(2025, 7, 11), datetime(2025, 7, 14)
        )
    assert storage.get_metadata("equities/daily/SPY") == before
    assert_frame_equal(storage.read("equities/daily/SPY").collect(), frame)


def test_backfill_does_not_insert_provider_rows_for_a_closed_date_inside_gap(storage) -> None:
    complete = provider_frame("2025-07-02", "2025-07-08")
    existing = complete.filter(
        pl.col("timestamp").dt.date().is_in([date(2025, 7, 2), date(2025, 7, 8)])
    )
    storage.write(existing, "equities/daily/SPY", {"calendar": "NYSE"})
    result = IncrementalUpdater().apply_strategy(
        storage, "equities/daily/SPY", complete, UpdateStrategy.BACKFILL
    )
    assert result.rows_added == 2
    assert storage.read("equities/daily/SPY").collect()["timestamp"].dt.date().to_list() == [
        date(2025, 7, 2),
        date(2025, 7, 3),
        date(2025, 7, 7),
        date(2025, 7, 8),
    ]


def test_timezone_annotated_session_labels_do_not_become_previous_utc_dates(storage) -> None:
    frame = provider_frame("2025-07-15", "2025-07-17").filter(
        pl.col("timestamp").dt.date() != date(2025, 7, 16)
    )
    frame = frame.with_columns(
        pl.col("timestamp")
        .dt.replace_time_zone(None)
        .dt.truncate("1d")
        .dt.replace_time_zone("Asia/Tokyo")
    )
    feed = FeedSpec(
        timezone="Asia/Tokyo", timestamp_semantics="session_label", time_contract_version=2
    )
    storage.write(frame, "equities/daily/SPY", feed.to_dict())
    detector = GapDetector()
    direct = detector.detect_gaps(frame, feed_spec=feed)
    stored = detector.detect_gaps_in_storage(
        storage, "equities/daily/SPY", datetime(2025, 7, 15), datetime(2025, 7, 17)
    )
    assert (
        direct
        == stored
        == [
            {
                "start": datetime(2025, 7, 16, tzinfo=UTC),
                "end": datetime(2025, 7, 16, tzinfo=UTC),
                "size_days": 1,
            }
        ]
    )


@pytest.mark.parametrize("version, expected", [(1, 1), (2, 0)])
def test_persisted_null_disables_a_real_missing_session_only_in_version_two(
    storage, version, expected
) -> None:
    frame = provider_frame("2025-07-15", "2025-07-17")
    frame = frame.filter(pl.col("timestamp").dt.date() != date(2025, 7, 16))
    storage.write(
        frame,
        "equities/daily/SPY",
        FeedSpec(calendar=None, time_contract_version=version).to_dict(),
    )
    gaps = GapDetector().detect_gaps_in_storage(
        storage, "equities/daily/SPY", datetime(2025, 7, 15), datetime(2025, 7, 17)
    )
    assert sum(gap["size_days"] for gap in gaps) == expected


def test_per_call_calendar_overrides_the_configured_selection() -> None:
    frame = provider_frame("2025-07-11", "2025-07-14")
    detector = GapDetector(calendar="24/7")
    assert detector.detect_gaps(frame, calendar="NYSE") == []
    assert detector.detect_gaps(frame, calendar=None) == []
