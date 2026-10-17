"""Real provider rows must survive the CME crypto schedule transition."""

from datetime import date, datetime

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from ml4t.data import DataManager
from ml4t.data.providers import SyntheticProvider
from ml4t.data.sessions import SessionAssigner, SessionCompleter
from ml4t.data.utils.gaps import GapDetector
from ml4t.specs import FeedSpec


def provider_rows(day: str, clocks: list[str]) -> pl.DataFrame:
    with SyntheticProvider(calendar_mode="continuous") as provider:
        frame = provider.fetch_ohlcv("BTCUSD", day, day, "minute")
    selected = frame.filter(pl.col("timestamp").dt.strftime("%H:%M").is_in(clocks))
    assert selected.height == len(clocks)
    return selected


@pytest.mark.parametrize(
    "day,clock,expected",
    [
        ("2026-05-23", "12:00", None),
        ("2026-05-30", "12:00", date(2026, 6, 1)),
        ("2026-10-17", "12:00", date(2026, 10, 19)),
        ("2026-10-16", "21:03", date(2026, 10, 19)),
        ("2026-10-16", "21:00", None),
        ("2026-10-16", "21:01", None),
        ("2026-10-17", "07:59", None),
        ("2026-10-17", "09:00", date(2026, 10, 19)),
        ("2026-12-05", "08:00", None),
        ("2026-12-05", "10:00", date(2026, 12, 7)),
        ("2026-05-29", "21:15", None),
        ("2026-05-29", "21:30", date(2026, 6, 1)),
        ("2026-09-05", "12:00", date(2026, 9, 8)),
        ("2026-07-02", "21:03", date(2026, 7, 6)),
        ("2026-06-18", "21:03", date(2026, 6, 22)),
        ("2026-11-25", "22:03", date(2026, 11, 27)),
        ("2026-12-24", "22:03", date(2026, 12, 28)),
        ("2026-10-12", "12:00", date(2026, 10, 12)),
        ("2026-11-11", "12:00", date(2026, 11, 11)),
        ("2026-05-29", "20:59", date(2026, 5, 29)),
        ("2026-05-28", "21:30", None),
        ("2026-05-28", "22:00", date(2026, 5, 29)),
        ("2026-11-01", "07:30", date(2026, 11, 2)),
    ],
)
def test_cme_crypto_uses_cutover_maintenance_and_business_dates(day, clock, expected):
    frame = provider_rows(day, [clock])
    assigned = SessionAssigner.from_exchange("CME").assign_sessions(frame, bar_frequency="intraday")
    assert assigned["session_date"].to_list() == [expected]
    assert_frame_equal(assigned.select(frame.columns), frame)


def test_cme_completion_preserves_weekend_observations_and_fills_only_open_gap():
    frame = provider_rows("2026-10-17", ["12:00", "12:02"])
    completed = SessionCompleter("CME Globex Crypto").complete_sessions(frame)
    assert completed.height == 3
    assert completed["is_imputed"].to_list() == [False, True, False]
    assert completed["session_date"].to_list() == [date(2026, 10, 19)] * 3
    observed = completed.filter(~pl.col("is_imputed")).select(frame.columns).cast(frame.schema)
    assert_frame_equal(observed, frame)


def test_spot_continuous_calendar_keeps_civil_dates_not_cme_trade_dates():
    frame = provider_rows("2026-10-17", ["12:00", "12:02"])
    assigned = SessionAssigner("24/7").assign_sessions(frame, bar_frequency="intraday")
    assert assigned["session_date"].to_list() == [date(2026, 10, 17)] * 2
    completed = SessionCompleter("24/7").complete_sessions(frame)
    assert completed.height == 3
    assert completed["session_date"].to_list() == [date(2026, 10, 17)] * 3


def test_calendar_none_never_assigns_or_manufactures_cme_or_spot_rows():
    frame = provider_rows("2026-10-17", ["12:00", "12:02"])
    assert_frame_equal(SessionAssigner(None).assign_sessions(frame), frame)
    assert_frame_equal(SessionCompleter(None).complete_sessions(frame), frame)


@pytest.mark.parametrize(
    "day,clocks,expected",
    [
        ("2026-10-17", ["06:59", "09:00"], [date(2026, 10, 19)] * 2),
        ("2026-09-07", ["20:59", "21:02"], [date(2026, 9, 8)] * 2),
        ("2026-05-29", ["20:59", "21:30"], [date(2026, 5, 29), date(2026, 6, 1)]),
    ],
)
def test_completion_does_not_fill_weekly_holiday_or_launch_maintenance(day, clocks, expected):
    frame = provider_rows(day, clocks)
    completed = SessionCompleter("CME Globex Crypto").complete_sessions(frame)
    assert completed.height == 2
    assert completed["is_imputed"].to_list() == [False, False]
    assert completed["session_date"].to_list() == expected
    assert_frame_equal(completed.select(frame.columns).cast(frame.schema), frame)


@pytest.mark.parametrize("alias", ["CME Globex Crypto", "CME Globex Cryptocurrencies"])
def test_cme_aliases_and_facade_honor_shared_override_and_disable(alias):
    frame = provider_rows("2026-10-17", ["12:00"])
    feed = FeedSpec(calendar=alias, timestamp_semantics="event_time")
    with DataManager() as manager:
        assert manager.assign_sessions(frame, feed_spec=feed)["session_date"].to_list() == [
            date(2026, 10, 19)
        ]
        assert manager.assign_sessions(frame, calendar="24/7", feed_spec=feed)[
            "session_date"
        ].to_list() == [date(2026, 10, 17)]
        assert_frame_equal(manager.assign_sessions(frame, calendar=None, feed_spec=feed), frame)


def test_selected_crypto_schedule_does_not_modify_upstream_or_other_product_calendars():
    import pandas_market_calendars as mcal
    from pandas_market_calendars.calendars.cme_globex_crypto import CMEGlobexCryptoExchangeCalendar

    SessionAssigner("CME Globex Crypto")
    legacy = mcal.get_calendar("CME Globex Crypto")
    assert type(legacy) is CMEGlobexCryptoExchangeCalendar
    assert legacy.schedule("2026-10-17", "2026-10-17").empty
    other = SessionAssigner("CME_Equity")
    frame = provider_rows("2026-10-17", ["12:00"])
    assert other.assign_sessions(frame, bar_frequency="intraday")["session_date"].to_list() == [
        None
    ]


def test_cme_daily_labels_are_business_dates_not_each_open_weekend_civil_date():
    with SyntheticProvider(calendar_mode="continuous") as provider:
        daily = provider.fetch_ohlcv("BTCUSD", "2026-10-17", "2026-10-19", "daily")
    feed = FeedSpec(calendar="CME Globex Crypto", timestamp_semantics="session_label")
    assigned = SessionAssigner(feed_spec=feed).assign_sessions(daily)
    assert assigned["session_date"].to_list() == [None, None, date(2026, 10, 19)]
    info = SessionCompleter("CME Globex Crypto").get_session_info(datetime(2026, 10, 19))
    assert info["market_open"].isoformat() == "2026-10-16T21:02:00+00:00"
    assert info["market_close"].isoformat() == "2026-10-19T21:00:00+00:00"


@pytest.mark.parametrize(
    "day,clocks,missing",
    [
        ("2026-09-05", ["12:00", "12:02"], 1),
        ("2026-10-17", ["06:59", "09:00"], 0),
        ("2026-09-07", ["20:59", "21:02"], 0),
        ("2026-05-29", ["20:59", "21:30"], 0),
    ],
)
def test_intraday_grid_uses_business_date_coverage_and_excludes_cme_maintenance(
    day, clocks, missing
):
    frame = provider_rows(day, clocks)
    feed = FeedSpec(
        time_contract_version=2,
        calendar="CME Globex Crypto",
        data_frequency="minute",
        timestamp_semantics="event_time",
        bar_grid_anchor="session_open",
        bar_label="left",
        partial_bar_policy="exclude",
    )
    gaps = GapDetector().detect_gaps(frame, feed_spec=feed)
    assert sum(gap.missing_periods for gap in gaps) == missing
    if day == "2026-10-17":
        spot = GapDetector().detect_gaps(frame, calendar="24/7", feed_spec=feed)
        assert sum(gap.missing_periods for gap in spot) == 120
    assert GapDetector().detect_gaps(frame, calendar=None, feed_spec=feed) == []
