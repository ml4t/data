"""Daily gaps must distinguish absent observations from scheduled closures."""

from datetime import date

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from ml4t.data.providers.synthetic import SyntheticProvider
from ml4t.data.utils.gaps import GapDetector


def provider_frame(start: str, end: str, *, continuous: bool = False) -> pl.DataFrame:
    with SyntheticProvider(calendar_mode="continuous" if continuous else "equity") as provider:
        return provider.fetch_ohlcv("SPY", start, end, "daily")


def test_provider_weekend_is_not_a_missing_equity_session() -> None:
    frame = provider_frame("2025-07-11", "2025-07-14")
    original = frame.clone()
    assert GapDetector().detect_gaps(frame) == []
    assert_frame_equal(frame, original)


def test_holiday_and_early_close_do_not_create_missing_daily_sessions() -> None:
    frame = provider_frame("2025-07-02", "2025-07-07")
    # The synthetic equity producer excludes weekends, but not exchange holidays.
    frame = frame.filter(pl.col("timestamp").dt.date() != date(2025, 7, 4))
    assert GapDetector().detect_gaps(frame) == []


def test_all_assets_missing_a_session_does_not_redefine_the_calendar() -> None:
    frame = provider_frame("2025-07-14", "2025-07-18")
    panel = pl.concat(
        [frame.with_columns(pl.lit(symbol).alias("symbol")) for symbol in ("SPY", "QQQ")]
    )
    panel = panel.filter(pl.col("timestamp").dt.date() != date(2025, 7, 16))
    for asset in panel.partition_by("symbol"):
        gaps = GapDetector().detect_gaps(asset)
        assert len(gaps) == 1
        assert gaps[0].missing_periods == 1
        assert gaps[0].start.date() == date(2025, 7, 15)
        assert gaps[0].end.date() == date(2025, 7, 17)


@pytest.mark.parametrize("calendar, expected", [("NYSE", 0), ("24/7", 2), (None, 0)])
def test_explicit_calendar_controls_provider_weekend_classification(calendar, expected) -> None:
    frame = provider_frame("2025-07-11", "2025-07-14")
    gaps = GapDetector().detect_gaps(frame, calendar=calendar)
    assert sum(gap.missing_periods for gap in gaps) == expected


@pytest.mark.parametrize(
    "metadata, expected",
    [
        ({}, 0),
        ({"calendar": None}, 0),
        ({"calendar": "24/7"}, 2),
        ({"time_contract_version": 2, "calendar": None}, 0),
    ],
)
def test_feed_declaration_round_trip_preserves_calendar_selection(metadata, expected) -> None:
    from ml4t.specs import FeedSpec

    frame = provider_frame("2025-07-11", "2025-07-14")
    feed = FeedSpec.from_mapping(metadata)
    feed = FeedSpec.from_mapping(feed.to_dict())
    gaps = GapDetector().detect_gaps(frame, feed_spec=feed)
    assert sum(gap.missing_periods for gap in gaps) == expected


def test_explicit_calendar_beats_declared_metadata_and_crypto_hint() -> None:
    frame = provider_frame("2025-07-11", "2025-07-14")
    detector = GapDetector()
    assert (
        detector.detect_gaps(frame, calendar="NYSE", feed_spec={"calendar": "24/7"}, is_crypto=True)
        == []
    )
    assert detector.detect_gaps(frame, calendar=None, feed_spec={"calendar": "24/7"}) == []
    assert detector.detect_gaps(frame, feed_spec={"calendar": "NYSE"}, is_crypto=True) == []
    assert sum(g.missing_periods for g in detector.detect_gaps(frame, is_crypto=True)) == 2


def test_continuous_provider_outage_remains_missing_on_a_real_schedule() -> None:
    frame = provider_frame("2025-07-11", "2025-07-14", continuous=True)
    frame = frame.filter(
        ~pl.col("timestamp").dt.date().is_in([date(2025, 7, 12), date(2025, 7, 13)])
    )
    gaps = GapDetector().detect_gaps(frame, calendar="24/7")
    assert len(gaps) == 1
    assert gaps[0].missing_periods == 2


def test_calendar_none_does_not_classify_irregular_spacing_as_missing() -> None:
    frame = provider_frame("2025-07-01", "2025-07-31").gather_every(5)
    assert GapDetector().detect_gaps(frame, calendar=None) == []


def test_bad_calendar_fails_without_mutating_provider_input() -> None:
    frame = provider_frame("2025-07-11", "2025-07-14")
    original = frame.clone()
    with pytest.raises(ValueError, match="Unknown calendar"):
        GapDetector().detect_gaps(frame, calendar="not-an-exchange")
    assert_frame_equal(frame, original)
