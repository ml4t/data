"""Calendar comparison reads real stored producer data without altering it."""

from datetime import date, datetime
from pathlib import Path

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from ml4t.data.providers import SyntheticProvider
from ml4t.data.storage.config import StorageConfig
from ml4t.data.storage.flat import FlatStorage
from ml4t.data.storage.hive import HiveStorage
from ml4t.data.update_manager import GapDetector, IncrementalUpdater
from ml4t.specs import FeedSpec


def declaration(calendar="NYSE", frequency="minute", **kwargs):
    return FeedSpec(
        time_contract_version=2,
        calendar=calendar,
        data_frequency=frequency,
        timestamp_semantics=kwargs.pop("timestamp_semantics", "event_time"),
        bar_grid_anchor="session_open",
        bar_label=kwargs.pop("bar_label", "left"),
        partial_bar_policy=kwargs.pop("partial_bar_policy", "include"),
        **kwargs,
    )


def rows(start, end, clocks, frequency="minute"):
    with SyntheticProvider(seed=42, calendar_mode="continuous") as provider:
        frame = provider.fetch_ohlcv("SPY", start, end, frequency)
    return frame.filter(pl.col("timestamp").dt.strftime("%H:%M").is_in(clocks))


def instant(day, clock):
    return datetime.fromisoformat(f"{day}T{clock}:00+00:00")


@pytest.fixture(params=[FlatStorage, HiveStorage])
def storage(request, tmp_path):
    return request.param(StorageConfig(base_path=tmp_path))


def test_disk_report_finds_inclusive_leading_and_trailing_minute_absences(storage):
    frame = rows("2025-07-07", "2025-07-07", ["13:31", "13:32"])
    key = "equities/minute/SPY"
    storage.write(frame, key, declaration().to_dict())
    before = storage.get_metadata(key)
    gaps = GapDetector().detect_gaps_in_storage(
        storage, key, instant("2025-07-07", "13:30"), instant("2025-07-07", "13:33")
    )
    assert gaps == [
        {"start": instant("2025-07-07", clock), "end": instant("2025-07-07", clock), "size_bars": 1}
        for clock in ["13:30", "13:33"]
    ]
    assert storage.get_metadata(key) == before
    assert_frame_equal(storage.read(key).collect(), frame)


def test_disk_report_applies_stored_continuous_calendar_and_hour_frequency(storage):
    frame = rows("2025-07-11", "2025-07-14", ["00:00"], "hourly").filter(
        pl.col("timestamp").dt.date().is_in([date(2025, 7, 11), date(2025, 7, 14)])
    )
    key = "crypto/hourly/SPY"
    storage.write(frame, key, declaration("24/7", "hourly").to_dict())
    report = GapDetector().detect_gaps_in_storage(
        storage, key, instant("2025-07-11", "00:00"), instant("2025-07-14", "00:00")
    )
    assert report == [
        {
            "start": instant("2025-07-11", "01:00"),
            "end": instant("2025-07-13", "23:00"),
            "size_bars": 71,
        }
    ]
    assert (
        GapDetector().detect_gaps_in_storage(
            storage,
            key,
            instant("2025-07-11", "00:00"),
            instant("2025-07-14", "00:00"),
            calendar=None,
        )
        == []
    )


def test_direct_report_does_not_invent_a_weekend_between_adjacent_nyse_bars():
    frame = rows("2025-07-11", "2025-07-14", ["19:59", "13:30"]).filter(
        (
            (pl.col("timestamp").dt.date() == date(2025, 7, 11))
            & (pl.col("timestamp").dt.hour() == 19)
        )
        | (
            (pl.col("timestamp").dt.date() == date(2025, 7, 14))
            & (pl.col("timestamp").dt.hour() == 13)
        )
    )
    assert frame.height == 2
    assert GapDetector(feed_spec=declaration()).detect_gaps(frame) == []


@pytest.mark.parametrize("calendar,expected", [("NYSE", 210), ("24/7", 2880), (None, 0)])
def test_completely_absent_disk_dataset_uses_schedule_not_observed_dates(
    storage, calendar, expected
):
    report = GapDetector(feed_spec=declaration(calendar)).detect_gaps_in_storage(
        storage,
        "equities/minute/ABSENT",
        instant("2025-07-03", "00:00"),
        instant("2025-07-04", "23:59"),
        frequency="minute",
    )
    assert sum(gap["size_bars"] for gap in report) == expected
    assert not storage.exists("equities/minute/ABSENT")


@pytest.mark.parametrize(
    "day,clocks",
    [
        ("2026-09-05", ["12:00", "12:02"]),
        ("2026-05-29", ["21:30", "21:32"]),
    ],
)
def test_cme_crypto_report_covers_the_following_business_trade_date(storage, day, clocks):
    frame = rows(day, day, clocks)
    key = "futures/minute/BTC"
    storage.write(frame, key, declaration("CME Globex Crypto").to_dict())
    report = GapDetector().detect_gaps_in_storage(
        storage, key, instant(day, clocks[0]), instant(day, clocks[1]), frequency="minute"
    )
    middle = "12:01" if day == "2026-09-05" else "21:31"
    assert report == [{"start": instant(day, middle), "end": instant(day, middle), "size_bars": 1}]


@pytest.mark.parametrize(
    "calendar,day,clocks",
    [
        ("HKEX", "2025-07-07", ["03:59", "05:00"]),
        ("CME Globex Crypto", "2026-10-17", ["06:59", "09:00"]),
    ],
)
def test_disk_report_excludes_selected_exchange_breaks_and_maintenance(
    storage, calendar, day, clocks
):
    frame = rows(day, day, clocks)
    key = "futures/minute/SPY"
    storage.write(frame, key, declaration(calendar).to_dict())
    assert (
        GapDetector().detect_gaps_in_storage(
            storage, key, instant(day, clocks[0]), instant(day, clocks[1]), frequency="minute"
        )
        == []
    )


def test_calendar_report_needs_timestamps_not_ohlcv_values(storage):
    frame = rows("2025-07-07", "2025-07-07", ["13:30", "13:32"]).select(
        "timestamp", pl.col("close").alias("measurement")
    )
    key = "measurements/minute/series"
    storage.write(frame, key, declaration().to_dict())
    report = GapDetector().detect_gaps_in_storage(
        storage, key, instant("2025-07-07", "13:30"), instant("2025-07-07", "13:32")
    )
    assert sum(gap["size_bars"] for gap in report) == 1
    assert_frame_equal(storage.read(key).collect(), frame)


def test_disk_report_rejects_reversed_intraday_bounds_without_rewriting(storage):
    key = "equities/minute/SPY"
    frame = rows("2025-07-07", "2025-07-07", ["13:30", "13:32"])
    storage.write(frame, key, declaration().to_dict())
    before = storage.get_metadata(key)
    with pytest.raises(ValueError, match="start_date"):
        GapDetector().detect_gaps_in_storage(
            storage,
            key,
            instant("2025-07-07", "13:32"),
            instant("2025-07-07", "13:30"),
            frequency="minute",
        )
    assert storage.get_metadata(key) == before
    assert_frame_equal(storage.read(key).collect(), frame)


@pytest.mark.parametrize(
    "side,policy,expected",
    [
        ("left", "include", 1),
        ("left", "exclude", 0),
        ("right", "include", 2),
        ("right", "exclude", 1),
    ],
)
def test_requested_bounds_respect_actual_partial_close_bar_labels(storage, side, policy, expected):
    feed = declaration(
        "NYSE",
        "hourly",
        bar_label=side,
        partial_bar_policy=policy,
        timestamp_semantics="bar_close" if side == "right" else "event_time",
    )
    report = GapDetector(feed_spec=feed).detect_gaps_in_storage(
        storage,
        "equities/hourly/ABSENT",
        instant("2025-07-03", "16:00"),
        instant("2025-07-03", "17:00"),
    )
    assert sum(gap["size_bars"] for gap in report) == expected
    if report:
        assert report[0]["end"] == instant(
            "2025-07-03", "17:00" if side == "right" and policy == "include" else "16:30"
        )


def test_whole_session_missing_for_every_stored_asset_remains_missing(storage):
    frame = rows("2025-07-15", "2025-07-17", ["13:30", "19:59"]).filter(
        pl.col("timestamp").dt.date() != date(2025, 7, 16)
    )
    for symbol in ["SPY", "QQQ"]:
        key = f"equities/minute/{symbol}"
        storage.write(
            frame.with_columns(pl.lit(symbol).alias("symbol")), key, declaration().to_dict()
        )
        report = GapDetector().detect_gaps_in_storage(
            storage, key, instant("2025-07-16", "00:00"), instant("2025-07-16", "23:59")
        )
        assert report == [
            {
                "start": instant("2025-07-16", "13:30"),
                "end": instant("2025-07-16", "19:59"),
                "size_bars": 390,
            }
        ]


def test_requested_frequency_cannot_override_stored_intraday_grid(storage):
    key = "equities/minute/SPY"
    storage.write(rows("2025-07-07", "2025-07-07", ["13:30"]), key, declaration().to_dict())
    with pytest.raises(ValueError, match="frequency"):
        GapDetector().detect_gaps_in_storage(
            storage,
            key,
            instant("2025-07-07", "13:30"),
            instant("2025-07-07", "14:30"),
            frequency="hourly",
        )


def test_missing_intraday_grid_is_rejected_instead_of_assuming_elapsed_days(storage):
    frame = rows("2025-07-07", "2025-07-07", ["13:30", "13:32"])
    key = "equities/minute/SPY"
    storage.write(frame, key, {"frequency": "minute"})
    with pytest.raises(ValueError, match="declared.*grid"):
        GapDetector().detect_gaps_in_storage(
            storage, key, instant("2025-07-07", "13:30"), instant("2025-07-07", "13:32")
        )


def test_one_instrument_cannot_hide_another_instruments_missing_bar(storage):
    complete = rows("2025-07-07", "2025-07-07", ["13:30", "13:31", "13:32"])
    spy = complete.filter(pl.col("timestamp").dt.minute() != 31)
    qqq = complete.with_columns(pl.lit("QQQ").alias("symbol"))
    panel = pl.concat([spy, qqq])
    key = "equities/minute/panel"
    storage.write(panel, key, declaration(entity_col="symbol").to_dict())
    report = GapDetector().detect_gaps_in_storage(
        storage, key, instant("2025-07-07", "13:30"), instant("2025-07-07", "13:32")
    )
    assert report == [
        {
            "entity": "SPY",
            "start": instant("2025-07-07", "13:31"),
            "end": instant("2025-07-07", "13:31"),
            "size_bars": 1,
        }
    ]
    assert_frame_equal(storage.read(key).collect(), panel)


def test_stored_instrument_absent_from_the_entire_query_is_still_inspected(storage):
    complete = rows("2025-07-07", "2025-07-07", ["13:29", "13:30", "13:31", "13:32"])
    spy = complete.filter(pl.col("timestamp").dt.minute() == 29)
    qqq = complete.with_columns(pl.lit("QQQ").alias("symbol"))
    key = "equities/minute/panel"
    storage.write(pl.concat([spy, qqq]), key, declaration(entity_col=["symbol"]).to_dict())
    report = GapDetector().detect_gaps_in_storage(
        storage, key, instant("2025-07-07", "13:30"), instant("2025-07-07", "13:32")
    )
    assert report == [
        {
            "entity": "SPY",
            "start": instant("2025-07-07", "13:30"),
            "end": instant("2025-07-07", "13:32"),
            "size_bars": 3,
        }
    ]


@pytest.mark.parametrize("frequency", ["", "unknown", "ticks"])
def test_unknown_direct_frequency_cannot_silently_become_daily(frequency):
    frame = rows("2025-07-07", "2025-07-07", ["13:30", "13:32"])
    with pytest.raises(ValueError, match="frequency"):
        GapDetector().detect_gaps(frame, frequency=frequency)


def test_documented_disk_inspection_reports_absences_without_changing_observations(
    tmp_path, monkeypatch
):
    from ml4t.data.storage import create_storage

    monkeypatch.chdir(tmp_path)
    storage = create_storage("./data", strategy="hive")
    key = "crypto/minute/coinbase-btc-usd-spot"
    frame = rows("2026-10-16", "2026-10-16", ["00:00", "00:02", "23:59"])
    storage.write(frame, key)
    before = storage.get_metadata(key)
    guide = Path(__file__).resolve().parents[1] / "docs/user-guide/incremental-updates.md"
    section = guide.read_text().split("### Inspect Missing Observations on Disk", 1)[1]
    code = section.split("```python\n", 1)[1].split("```", 1)[0]
    namespace = {}
    exec(compile(code, str(guide), "exec"), namespace)
    assert sum(gap["size_bars"] for gap in namespace["report"]) == 1437
    assert storage.get_metadata(key) == before
    assert_frame_equal(storage.read(key).collect(), frame)
    frame.write_parquet("./prices.parquet")
    original_file = Path("./prices.parquet").read_bytes()
    raw_code = section.split("```python\n", 2)[2].split("```", 1)[0]
    exec(compile(raw_code, str(guide), "exec"), namespace)
    assert sum(gap["size_bars"] for gap in namespace["report"]) == 1437
    assert Path("./prices.parquet").read_bytes() == original_file


@pytest.mark.parametrize(
    "calendar,day,clock,end,expected",
    [
        ("24/7", "2026-09-05", "12:00", "2026-09-05T12:02", "2026-09-05T12:01"),
        ("CME Globex Crypto", "2026-09-05", "12:00", "2026-09-05T12:02", "2026-09-05T12:01"),
        ("NYSE", "2025-07-11", "19:59", "2025-07-14T13:31", "2025-07-14T13:30"),
        ("NYSE", "2025-07-03", "16:59", "2025-07-07T13:31", "2025-07-07T13:30"),
        ("NYSE", "2025-07-11", "19:59", "2025-07-13T20:00", None),
    ],
)
def test_incremental_range_starts_at_next_scheduled_bar_not_a_day_later(
    storage, calendar, day, clock, end, expected
):
    frame = rows(day, day, [clock])
    storage.write(frame, "SPY", declaration(calendar).to_dict())
    before = storage.get_metadata("SPY")
    end = datetime.fromisoformat(end + ":00+00:00")
    expected = datetime.fromisoformat(expected + ":00+00:00") if expected else end
    actual = IncrementalUpdater().determine_update_range(storage, "SPY", instant(day, clock), end)
    assert actual == (expected, end, "none" if expected == end else "incremental")
    assert storage.get_metadata("SPY") == before
    assert_frame_equal(storage.read("SPY").collect(), frame)


def test_incremental_disabled_calendar_keeps_row_mode_without_skipping_a_day(storage):
    frame = rows("2026-09-05", "2026-09-05", ["12:00"])
    storage.write(frame, "SPY", declaration(None).to_dict())
    start = instant("2026-09-05", "12:00")
    end = instant("2026-09-05", "12:02")
    assert IncrementalUpdater().determine_update_range(storage, "SPY", start, end) == (
        start,
        end,
        "incremental",
    )


@pytest.mark.parametrize("calendar,expected_day", [("NYSE", "2025-07-14"), ("24/7", "2025-07-12")])
def test_incremental_daily_labels_inherit_the_stored_calendar(storage, calendar, expected_day):
    frame = rows("2025-07-11", "2025-07-11", ["00:00"], "daily")
    storage.write(
        frame,
        "SPY",
        FeedSpec(
            time_contract_version=2,
            calendar=calendar,
            data_frequency="daily",
            timestamp_semantics="session_label",
        ).to_dict(),
    )
    end = instant("2025-07-14", "00:00")
    assert IncrementalUpdater().determine_update_range(
        storage, "SPY", instant("2025-07-11", "00:00"), end
    ) == (instant(expected_day, "00:00"), end, "incremental")


def test_incremental_explicit_calendar_overrides_stored_schedule(storage):
    frame = rows("2026-09-05", "2026-09-05", ["12:00"])
    storage.write(frame, "SPY", declaration("NYSE").to_dict())
    start = instant("2026-09-05", "12:00")
    end = instant("2026-09-05", "12:02")
    updater = IncrementalUpdater()
    assert updater.determine_update_range(storage, "SPY", start, end, calendar="24/7") == (
        instant("2026-09-05", "12:01"),
        end,
        "incremental",
    )
    assert updater.determine_update_range(storage, "SPY", start, end, calendar=None) == (
        start,
        end,
        "incremental",
    )
    with pytest.raises(ValueError, match="requested_start"):
        updater.determine_update_range(storage, "SPY", end, start)
