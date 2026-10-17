"""Existing files need bounded reports without migration or a provider connection."""

from datetime import date, datetime

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from ml4t.data.providers import SyntheticProvider
from ml4t.data.sessions import SessionCompleter
from ml4t.data.update_manager import GapDetector
from ml4t.data.utils.gaps import GapDetector as ObservationGapDetector
from ml4t.specs import FeedSpec


def instant(day, clock):
    return datetime.fromisoformat(f"{day}T{clock}:00+00:00")


def feed(calendar="NYSE", **kwargs):
    return FeedSpec(
        time_contract_version=2,
        calendar=calendar,
        data_frequency="minute",
        timestamp_semantics="event_time",
        bar_grid_anchor="session_open",
        bar_label="left",
        partial_bar_policy="include",
        **kwargs,
    )


def rows(day="2025-07-07"):
    with SyntheticProvider(calendar_mode="continuous") as provider:
        return provider.fetch_ohlcv("SPY", day, day, "minute")


def test_frame_report_ignores_off_session_observations_outside_requested_range():
    frame = rows().filter(pl.col("timestamp") != instant("2025-07-07", "13:31"))
    original = frame.clone()
    report = GapDetector(feed_spec=feed()).detect_gaps(
        frame,
        start_date=instant("2025-07-07", "13:30"),
        end_date=instant("2025-07-07", "13:33"),
    )
    assert report == [
        {
            "start": instant("2025-07-07", "13:31"),
            "end": instant("2025-07-07", "13:31"),
            "size_bars": 1,
        }
    ]
    assert_frame_equal(frame, original)


@pytest.mark.parametrize(
    "calendar,day,start,end,count",
    [
        ("NYSE", "2025-07-07", "13:30", "13:32", 3),
        ("NYSE", "2025-07-12", "12:00", "12:02", 0),
        ("24/7", "2025-07-12", "12:00", "12:02", 3),
        ("CME Globex Crypto", "2026-09-05", "12:00", "12:02", 3),
        ("CME Globex Crypto", "2026-10-17", "06:59", "09:00", 2),
        (None, "2025-07-07", "13:30", "13:32", 0),
    ],
)
def test_empty_file_interval_is_compared_with_authoritative_schedule(
    calendar, day, start, end, count
):
    empty = rows(day).head(0)
    report = GapDetector(feed_spec=feed(calendar)).detect_gaps(
        empty,
        start_date=instant(day, start),
        end_date=instant(day, end),
    )
    assert sum(gap["size_bars"] for gap in report) == count


def test_frame_bounds_retain_a_wholly_missing_daily_session():
    with SyntheticProvider() as provider:
        frame = provider.fetch_ohlcv("SPY", "2025-07-15", "2025-07-17", "daily")
    frame = frame.filter(pl.col("timestamp").dt.date() != date(2025, 7, 16))
    report = GapDetector().detect_gaps(
        frame,
        start_date=instant("2025-07-16", "00:00"),
        end_date=instant("2025-07-16", "23:59"),
    )
    assert report == [
        {
            "start": instant("2025-07-16", "00:00"),
            "end": instant("2025-07-16", "00:00"),
            "size_days": 1,
        }
    ]


def test_frame_report_honors_explicit_timestamp_column_without_changing_input():
    frame = rows().filter(pl.col("timestamp").dt.strftime("%H:%M").is_in(["13:31", "13:32"]))
    frame = frame.rename({"timestamp": "time"})
    original = frame.clone()
    report = GapDetector(feed_spec=feed(timestamp_col="time")).detect_gaps(
        frame,
        start_date=instant("2025-07-07", "13:30"),
        end_date=instant("2025-07-07", "13:33"),
    )
    assert sum(gap["size_bars"] for gap in report) == 2
    assert_frame_equal(frame, original)


def test_panel_instrument_absent_inside_bounds_is_not_omitted():
    complete = rows().filter(
        pl.col("timestamp")
        .dt.strftime("%H:%M")
        .is_in(
            [
                "13:29",
                "13:30",
                "13:31",
                "13:32",
            ]
        )
    )
    spy = complete.filter(pl.col("timestamp").dt.minute() == 29)
    qqq = complete.with_columns(pl.lit("QQQ").alias("symbol"))
    report = GapDetector(feed_spec=feed(entity_col="symbol")).detect_gaps(
        pl.concat([spy, qqq]),
        start_date=instant("2025-07-07", "13:30"),
        end_date=instant("2025-07-07", "13:32"),
    )
    assert report == [
        {
            "entity": "SPY",
            "start": instant("2025-07-07", "13:30"),
            "end": instant("2025-07-07", "13:32"),
            "size_bars": 3,
        }
    ]


def test_a_single_bound_cannot_silently_choose_a_different_coverage_interval():
    with pytest.raises(ValueError, match="together|both"):
        GapDetector(feed_spec=feed()).detect_gaps(rows(), start_date=instant("2025-07-07", "13:30"))


@pytest.mark.parametrize("mode", ["range", "observation"])
def test_real_native_placeholder_is_not_a_recovered_source_observation(mode):
    actual = rows().filter(pl.col("timestamp").dt.strftime("%H:%M").is_in(["13:30", "13:32"]))
    expanded = SessionCompleter("NYSE").complete_sessions(actual)
    assert expanded["is_imputed"].to_list() == [False, True, False]
    before = expanded.clone()
    if mode == "range":
        report = GapDetector(feed_spec=feed()).detect_gaps(expanded)
        assert sum(gap["size_bars"] for gap in report) == 1
    else:
        report = ObservationGapDetector().detect_gaps(expanded, feed_spec=feed())
        assert sum(gap.missing_periods for gap in report) == 1
    assert_frame_equal(expanded, before)


def test_raw_parquet_uses_the_same_bounded_report_without_storage_migration(tmp_path):
    path = tmp_path / "prices.parquet"
    frame = rows().filter(pl.col("timestamp").dt.strftime("%H:%M").is_in(["13:31", "13:32"]))
    frame.write_parquet(path)
    original = path.read_bytes()
    report = GapDetector(feed_spec=feed()).detect_gaps(
        pl.read_parquet(path),
        start_date=instant("2025-07-07", "13:30"),
        end_date=instant("2025-07-07", "13:33"),
    )
    assert sum(gap["size_bars"] for gap in report) == 2
    assert path.read_bytes() == original


def test_wholly_placeholder_interval_is_not_reported_as_complete():
    actual = rows().filter(pl.col("timestamp").dt.strftime("%H:%M").is_in(["13:30", "13:33"]))
    expanded = SessionCompleter("NYSE").complete_sessions(actual)
    placeholders = expanded.filter(pl.col("is_imputed"))
    report = GapDetector(feed_spec=feed()).detect_gaps(placeholders)
    assert report == [
        {
            "start": instant("2025-07-07", "13:31"),
            "end": instant("2025-07-07", "13:32"),
            "size_bars": 2,
        }
    ]


def test_daily_known_placeholder_does_not_satisfy_a_session():
    with SyntheticProvider() as provider:
        frame = provider.fetch_ohlcv("SPY", "2025-07-15", "2025-07-17", "daily")
    frame = frame.with_columns(
        (pl.col("timestamp").dt.date() == date(2025, 7, 16)).alias("is_imputed")
    )
    report = GapDetector().detect_gaps(frame)
    assert len(report) == 1
    assert report[0]["start"].date() == date(2025, 7, 16)
    assert report[0]["size_days"] == 1


def test_zero_volume_alone_is_not_treated_as_a_missing_observation():
    frame = rows().filter(pl.col("timestamp").dt.strftime("%H:%M").is_in(["13:30", "13:31"]))
    frame = frame.with_columns(pl.lit(0.0).alias("volume"))
    assert GapDetector(feed_spec=feed()).detect_gaps(frame) == []


def test_invalid_placeholder_marker_cannot_silently_hide_source_rows():
    frame = rows().filter(pl.col("timestamp").dt.strftime("%H:%M").is_in(["13:30", "13:31"]))
    frame = frame.with_columns(pl.lit("False").alias("is_imputed"))
    with pytest.raises(TypeError, match="Boolean"):
        GapDetector(feed_spec=feed()).detect_gaps(frame)
