"""Real provider outages must not be hidden by fixed US clocks or observed-session inference."""

from datetime import UTC, date, datetime, time
from zoneinfo import ZoneInfo

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from ml4t.data import DataManager
from ml4t.data.providers.synthetic import SyntheticProvider
from ml4t.data.storage.config import StorageConfig
from ml4t.data.storage.flat import FlatStorage
from ml4t.data.storage.hive import HiveStorage
from ml4t.data.utils.gaps import GapDetector
from ml4t.specs import FeedSpec


def provider_frame(start="2025-07-07", end="2025-07-07", frequency="minute"):
    with SyntheticProvider(calendar_mode="continuous") as provider:
        return provider.fetch_ohlcv("SPY", start, end, frequency)


def declaration(calendar="NYSE", frequency="minute", **overrides):
    return FeedSpec(
        time_contract_version=2,
        calendar=calendar,
        data_frequency=frequency,
        timestamp_semantics=overrides.pop("timestamp_semantics", "event_time"),
        bar_grid_anchor="session_open",
        bar_label=overrides.pop("bar_label", "left"),
        partial_bar_policy=overrides.pop("partial_bar_policy", "include"),
        **overrides,
    )


def gaps(frame, feed, **kwargs):
    return GapDetector().detect_gaps(frame, frequency=feed.data_frequency, feed_spec=feed, **kwargs)


@pytest.mark.parametrize("backend", [FlatStorage, HiveStorage])
@pytest.mark.parametrize("disabled", [False, True])
def test_stored_intraday_grid_reaches_real_provider_updates(
    tmp_path, monkeypatch, backend, disabled
):
    class Clock(datetime):
        @classmethod
        def now(cls, tz=UTC):
            return cls(2025, 7, 7, 23, tzinfo=tz)

    monkeypatch.setattr("ml4t.data.managers.storage_manager.datetime", Clock)
    with SyntheticProvider(seed=42, calendar_mode="continuous") as provider:
        complete = provider.fetch_ohlcv("SPY", "2025-07-07", "2025-07-07", "hourly")
    storage = backend(StorageConfig(base_path=tmp_path))
    with DataManager(
        storage=storage,
        providers={"synthetic": {"extra": {"seed": 42, "calendar_mode": "continuous"}}},
    ) as manager:
        key = manager.import_data(
            complete.head(3),
            "SPY",
            "synthetic",
            frequency="hourly",
            feed_spec=declaration("24/7", "hourly"),
        )
        overrides = {"calendar": None} if disabled else {}
        assert manager.update("SPY", frequency="hourly", lookback_days=0, **overrides) == key
        assert_frame_equal(storage.read(key).collect(), complete)
        feed = FeedSpec.from_any(manager.get_metadata_for_key(key))
        assert feed.resolve_calendar() == (None if disabled else "24/7")
        assert GapDetector().detect_gaps(storage.read(key).collect(), feed_spec=feed) == []


def test_unqualified_summer_outage_cannot_silently_return_no_gap():
    frame = provider_frame().filter(
        (pl.col("timestamp").dt.hour() == 13) & pl.col("timestamp").dt.minute().is_in([30, 32])
    )
    with pytest.raises(ValueError, match="declared.*grid"):
        GapDetector().detect_gaps(frame, frequency="minute")


def test_declared_intraday_frequency_cannot_silently_fall_back_to_daily_labels():
    frame = provider_frame().filter(
        (pl.col("timestamp").dt.hour() == 13) & pl.col("timestamp").dt.minute().is_in([30, 32])
    )
    found = GapDetector().detect_gaps(frame, feed_spec=declaration())
    assert sum(g.missing_periods for g in found) == 1
    with pytest.raises(ValueError, match="frequency"):
        GapDetector().detect_gaps(frame, frequency="daily", feed_spec=declaration())


@pytest.mark.parametrize("day,hour,minute", [("2025-07-07", 13, 30), ("2025-01-07", 14, 30)])
def test_declared_nyse_dst_and_winter_outage_each_count_one_bar(day, hour, minute):
    frame = provider_frame(day, day).filter(
        (pl.col("timestamp").dt.hour() == hour)
        & pl.col("timestamp").dt.minute().is_in([minute, minute + 2])
    )
    original = frame.clone()
    assert sum(g.missing_periods for g in gaps(frame, declaration())) == 1
    assert_frame_equal(frame, original)


def test_london_morning_outage_is_not_discarded_by_the_us_clock():
    frame = provider_frame(frequency="hourly").filter(pl.col("timestamp").dt.hour().is_in([8, 10]))
    found = gaps(frame, declaration("LSE", "hourly"))
    assert len(found) == 1
    assert found[0].missing_periods == 1


def test_holiday_and_early_close_are_not_missing_expected_bars():
    frame = provider_frame("2025-07-03", "2025-07-07").filter(
        (
            (pl.col("timestamp").dt.date() == date(2025, 7, 3))
            & (pl.col("timestamp").dt.hour() == 16)
            & (pl.col("timestamp").dt.minute() == 59)
        )
        | (
            (pl.col("timestamp").dt.date() == date(2025, 7, 7))
            & (pl.col("timestamp").dt.hour() == 13)
            & (pl.col("timestamp").dt.minute() == 30)
        )
    )
    assert gaps(frame, declaration()) == []


def test_missing_shortened_session_counts_its_actual_210_minutes():
    frame = provider_frame("2025-07-02", "2025-07-07").filter(
        (
            (pl.col("timestamp").dt.date() == date(2025, 7, 2))
            & (pl.col("timestamp").dt.hour() == 19)
            & (pl.col("timestamp").dt.minute() == 59)
        )
        | (
            (pl.col("timestamp").dt.date() == date(2025, 7, 7))
            & (pl.col("timestamp").dt.hour() == 13)
            & (pl.col("timestamp").dt.minute() == 30)
        )
    )
    assert sum(g.missing_periods for g in gaps(frame, declaration())) == 210


def test_all_asset_missing_session_keeps_390_expected_minutes():
    frame = provider_frame("2025-07-15", "2025-07-17").filter(
        (
            (pl.col("timestamp").dt.date() == date(2025, 7, 15))
            & (pl.col("timestamp").dt.hour() == 19)
            & (pl.col("timestamp").dt.minute() == 59)
        )
        | (
            (pl.col("timestamp").dt.date() == date(2025, 7, 17))
            & (pl.col("timestamp").dt.hour() == 13)
            & (pl.col("timestamp").dt.minute() == 30)
        )
    )
    panel = pl.concat([frame.with_columns(pl.lit(s).alias("symbol")) for s in ("SPY", "QQQ")])
    for asset in panel.partition_by("symbol"):
        assert sum(g.missing_periods for g in gaps(asset, declaration())) == 390


def test_hong_kong_break_does_not_count_as_missing_minutes():
    frame = provider_frame().filter(
        ((pl.col("timestamp").dt.hour() == 3) & (pl.col("timestamp").dt.minute() == 59))
        | ((pl.col("timestamp").dt.hour() == 5) & (pl.col("timestamp").dt.minute() == 0))
    )
    assert gaps(frame, declaration("HKEX")) == []


def test_cme_overnight_calendar_keeps_a_missing_midnight_minute():
    frame = provider_frame("2025-07-06", "2025-07-07").filter(
        (
            (pl.col("timestamp").dt.date() == date(2025, 7, 6))
            & (pl.col("timestamp").dt.hour() == 23)
            & (pl.col("timestamp").dt.minute() == 59)
        )
        | (
            (pl.col("timestamp").dt.date() == date(2025, 7, 7))
            & (pl.col("timestamp").dt.hour() == 0)
            & (pl.col("timestamp").dt.minute() == 1)
        )
    )
    assert sum(g.missing_periods for g in gaps(frame, declaration("CME_Equity"))) == 1


def test_continuous_weekend_is_a_real_grid_but_explicit_none_is_not():
    frame = provider_frame("2025-07-11", "2025-07-14", frequency="hourly").filter(
        pl.col("timestamp").dt.hour().eq(0)
        & pl.col("timestamp").dt.date().is_in([date(2025, 7, 11), date(2025, 7, 14)])
    )
    feed = declaration("24/7", "hourly")
    assert sum(g.missing_periods for g in gaps(frame, feed)) == 71
    assert gaps(frame, feed, calendar=None) == []
    assert gaps(frame, declaration(None, "hourly")) == []


def test_right_labeled_close_bar_is_valid_and_missing_predecessor_is_counted():
    frame = provider_frame().filter(
        ((pl.col("timestamp").dt.hour() == 19) & (pl.col("timestamp").dt.minute() == 58))
        | ((pl.col("timestamp").dt.hour() == 20) & (pl.col("timestamp").dt.minute() == 0))
    )
    feed = declaration(timestamp_semantics="bar_close", bar_label="right")
    assert sum(g.missing_periods for g in gaps(frame, feed)) == 1


def test_naive_local_timestamps_require_and_use_the_declared_timezone():
    frame = (
        provider_frame()
        .filter(
            (pl.col("timestamp").dt.hour() == 13) & pl.col("timestamp").dt.minute().is_in([30, 32])
        )
        .with_columns(
            pl.col("timestamp").dt.convert_time_zone("America/New_York").dt.replace_time_zone(None)
        )
    )
    assert (
        sum(g.missing_periods for g in gaps(frame, declaration(timezone="America/New_York"))) == 1
    )
    with pytest.raises(ValueError, match="timezone"):
        gaps(frame, declaration())


@pytest.mark.parametrize(
    "missing",
    ["bar_grid_anchor", "bar_label", "partial_bar_policy", "timestamp_semantics", "data_frequency"],
)
def test_incomplete_grid_metadata_cannot_silently_choose_a_bar_definition(missing):
    frame = provider_frame().head(2)
    values = declaration("24/7").to_dict()
    values.pop(missing)
    with pytest.raises(ValueError, match="declared.*grid"):
        GapDetector().detect_gaps(frame, frequency="minute", feed_spec=values)


def test_off_grid_provider_rows_fail_without_mutating_input():
    frame = provider_frame(frequency="hourly").filter(pl.col("timestamp").dt.hour().is_in([14, 15]))
    original = frame.clone()
    with pytest.raises(ValueError, match="off.grid"):
        gaps(frame, declaration("NYSE", "hourly"))
    assert_frame_equal(frame, original)


@pytest.mark.parametrize("policy,expected", [("include", 1), ("exclude", 0)])
def test_registered_custom_break_grid_restarts_and_honors_partial_bar_policy(policy, expected):
    from pandas_market_calendars.market_calendar import MarketCalendar

    class DataGridRegressionCalendar(MarketCalendar):
        name = "DataGridRegression"
        aliases = ["DataGridRegression"]
        tz = ZoneInfo("UTC")
        regular_market_times = {
            "market_open": ((None, time(9)),),
            "break_start": ((None, time(11)),),
            "break_end": ((None, time(12)),),
            "market_close": ((None, time(14, 30)),),
        }

    frame = provider_frame("2025-07-07", "2025-07-08", frequency="hourly").filter(
        (
            (pl.col("timestamp").dt.date() == date(2025, 7, 7))
            & (pl.col("timestamp").dt.hour() == 13)
        )
        | (
            (pl.col("timestamp").dt.date() == date(2025, 7, 8))
            & (pl.col("timestamp").dt.hour() == 9)
        )
    )
    feed = declaration("DataGridRegression", "hourly", partial_bar_policy=policy)
    assert sum(g.missing_periods for g in gaps(frame, feed)) == expected


def test_naive_ambiguous_local_clock_is_rejected_without_a_guessed_dst_fold():
    frame = (
        provider_frame("2025-11-02", "2025-11-02")
        .filter(
            (pl.col("timestamp").dt.hour() == 5) & pl.col("timestamp").dt.minute().is_in([30, 32])
        )
        .with_columns(
            pl.col("timestamp").dt.convert_time_zone("America/New_York").dt.replace_time_zone(None)
        )
    )
    with pytest.raises(pl.exceptions.ComputeError, match="ambiguous"):
        gaps(frame, declaration("24/7", timezone="America/New_York"))


def test_a_frequency_conflict_and_event_count_feed_cannot_define_a_time_grid():
    frame = provider_frame().head(2)
    with pytest.raises(ValueError, match="frequency"):
        GapDetector().detect_gaps(frame, frequency="hourly", feed_spec=declaration("24/7"))
    with pytest.raises(ValueError, match="fixed time bars"):
        gaps(frame, declaration("24/7", bar_type="volume"))


def test_right_labeled_break_start_is_the_last_valid_morning_bar():
    frame = provider_frame().filter(
        ((pl.col("timestamp").dt.hour() == 3) & (pl.col("timestamp").dt.minute() == 59))
        | ((pl.col("timestamp").dt.hour() == 4) & (pl.col("timestamp").dt.minute() == 0))
    )
    assert (
        gaps(frame, declaration("HKEX", timestamp_semantics="bar_close", bar_label="right")) == []
    )


def test_sub_microsecond_offset_cannot_be_truncated_into_a_valid_grid_label():
    frame = (
        provider_frame()
        .head(2)
        .with_columns(
            pl.col("timestamp").cast(pl.Datetime("ns", "UTC")) + pl.duration(nanoseconds=1)
        )
    )
    with pytest.raises(ValueError, match="off.grid"):
        gaps(frame, declaration("24/7"))


@pytest.mark.parametrize("policy,expected", [("include", 1), ("exclude", 0)])
def test_right_labels_count_the_partial_shortened_close_at_its_actual_end(policy, expected):
    frame = provider_frame("2025-07-03", "2025-07-07").filter(
        (
            (pl.col("timestamp").dt.date() == date(2025, 7, 3))
            & (pl.col("timestamp").dt.hour() == 16)
            & (pl.col("timestamp").dt.minute() == 30)
        )
        | (
            (pl.col("timestamp").dt.date() == date(2025, 7, 7))
            & (pl.col("timestamp").dt.hour() == 14)
            & (pl.col("timestamp").dt.minute() == 30)
        )
    )
    feed = declaration(
        frequency="hourly",
        timestamp_semantics="bar_close",
        bar_label="right",
        partial_bar_policy=policy,
    )
    assert sum(g.missing_periods for g in gaps(frame, feed)) == expected


def test_excluded_partial_close_is_not_accepted_as_the_last_full_bar_label():
    frame = provider_frame("2025-07-03", "2025-07-03").filter(
        ((pl.col("timestamp").dt.hour() == 16) & (pl.col("timestamp").dt.minute() == 30))
        | ((pl.col("timestamp").dt.hour() == 17) & (pl.col("timestamp").dt.minute() == 0))
    )
    assert (
        gaps(
            frame,
            declaration(frequency="hourly", timestamp_semantics="bar_close", bar_label="right"),
        )
        == []
    )
    with pytest.raises(ValueError, match="off.grid"):
        gaps(
            frame,
            declaration(
                frequency="hourly",
                timestamp_semantics="bar_close",
                bar_label="right",
                partial_bar_policy="exclude",
            ),
        )


def test_registered_calendar_halt_is_not_missing_observations():
    from pandas_market_calendars.market_calendar import MarketCalendar

    class DataGridHaltCalendar(MarketCalendar):
        name = "DataGridHalt"
        aliases = ["DataGridHalt"]
        tz = ZoneInfo("UTC")
        regular_market_times = {
            "market_open": ((None, time(9)),),
            "market_close": ((None, time(14)),),
        }

        @property
        def interruptions(self):
            return [("2025-07-07", time(10), time(10, 30))]

    frame = provider_frame().filter(
        ((pl.col("timestamp").dt.hour() == 9) & (pl.col("timestamp").dt.minute() == 59))
        | ((pl.col("timestamp").dt.hour() == 10) & (pl.col("timestamp").dt.minute() == 30))
    )
    assert gaps(frame, declaration("DataGridHalt")) == []
