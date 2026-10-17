"""Provider-generated contract files must use scheduled, not observed, expiry offsets."""

from datetime import date

import polars as pl
import pytest

from ml4t.data.futures.adjustment import NoAdjustment
from ml4t.data.futures.continuous import ContinuousContractBuilder
from ml4t.data.futures.roll import FirstNoticeDateRoll, TimeBasedRoll
from ml4t.data.providers.synthetic import SyntheticProvider
from ml4t.specs import FeedSpec


def write_contract_files(root, expiration, *, missing=None):
    with SyntheticProvider() as provider:
        frame = provider.fetch_ohlcv("TEST", "2025-07-01", "2025-07-11", "daily")
    frame = frame.filter(pl.col("timestamp").dt.date() != date(2025, 7, 4))
    if missing is not None:
        frame = frame.filter(pl.col("timestamp").dt.date() != missing)
    frames = []
    for symbol in ("TESTN2025", "TESTU2025"):
        frames.append(frame.with_columns(pl.lit(symbol).alias("symbol")))
    ohlcv = root / "ohlcv_1d" / "product=TEST"
    definitions = root / "definition" / "product=TEST"
    ohlcv.mkdir(parents=True)
    definitions.mkdir(parents=True)
    pl.concat(frames).write_parquet(ohlcv / "ohlcv_1d.parquet")
    pl.DataFrame(
        {
            "raw_symbol": ["TESTN2025", "TESTU2025"],
            "expiration": [expiration, date(2025, 9, 19)],
        }
    ).write_parquet(definitions / "definition.parquet")


def build(root, strategy=None):
    return ContinuousContractBuilder(
        roll_strategy=strategy or TimeBasedRoll(5),
        adjustment_method=NoAdjustment(),
    ).build("TEST", data_source="databento", storage_path=root)


def test_continuous_workflow_counts_the_july4_closure_in_the_expiry_offset(tmp_path):
    write_contract_files(tmp_path, date(2025, 7, 10))
    result = build(tmp_path)
    assert result.filter(pl.col("date") == date(2025, 7, 2))["symbol"].to_list() == ["TESTU2025"]
    assert result.filter(pl.col("is_roll_date"))["date"].to_list() == [date(2025, 7, 2)]


def test_missing_scheduled_roll_session_cannot_move_the_roll_before_the_outage(tmp_path):
    write_contract_files(tmp_path, date(2025, 7, 11), missing=date(2025, 7, 3))
    result = build(tmp_path)
    assert result.filter(pl.col("date") == date(2025, 7, 2))["symbol"].to_list() == ["TESTN2025"]
    assert result.filter(pl.col("is_roll_date"))["date"].to_list() == [date(2025, 7, 7)]


@pytest.mark.parametrize("strategy", [TimeBasedRoll, FirstNoticeDateRoll])
@pytest.mark.parametrize("explicit,expected", [("NYSE", "TESTU2025"), ("24/7", "TESTN2025")])
def test_roll_and_first_notice_delegate_preserve_calendar_precedence(
    tmp_path, strategy, explicit, expected
):
    write_contract_files(tmp_path, date(2025, 7, 10))
    feed = FeedSpec(time_contract_version=2, calendar="24/7")
    result = build(tmp_path, strategy(calendar=explicit, feed_spec=feed))
    assert result.filter(pl.col("date") == date(2025, 7, 3))["symbol"].to_list() == [expected]


@pytest.mark.parametrize("strategy", [TimeBasedRoll, FirstNoticeDateRoll])
def test_declared_continuous_schedule_is_not_replaced_by_default_nyse(tmp_path, strategy):
    write_contract_files(tmp_path, date(2025, 7, 10))
    result = build(tmp_path, strategy(feed_spec=FeedSpec(time_contract_version=2, calendar="24/7")))
    assert result.filter(pl.col("is_roll_date"))["date"].to_list() == [date(2025, 7, 7)]


def test_elapsed_day_mode_does_not_snap_the_trigger_before_a_missing_weekend(tmp_path):
    write_contract_files(tmp_path, date(2025, 7, 10))
    result = build(tmp_path, TimeBasedRoll(5, False))
    assert result.filter(pl.col("date") == date(2025, 7, 3))["symbol"].to_list() == ["TESTN2025"]
    assert result.filter(pl.col("is_roll_date"))["date"].to_list() == [date(2025, 7, 7)]


def test_explicit_elapsed_units_need_no_calendar_but_do_not_become_row_counts(tmp_path):
    write_contract_files(tmp_path, date(2025, 7, 10))
    result = build(tmp_path, TimeBasedRoll(5, False, calendar=None))
    assert result.filter(pl.col("is_roll_date"))["date"].to_list() == [date(2025, 7, 7)]


@pytest.mark.parametrize("strategy", [TimeBasedRoll, FirstNoticeDateRoll])
def test_disabled_session_count_roll_fails_instead_of_substituting_a_continuous_schedule(strategy):
    with pytest.raises(ValueError, match="row-based"):
        strategy(calendar=None)
    with pytest.raises(ValueError, match="row-based"):
        strategy(feed_spec=FeedSpec(time_contract_version=2, calendar=None))


def test_invalid_calendar_does_not_change_provider_contract_files(tmp_path):
    write_contract_files(tmp_path, date(2025, 7, 10))
    source = tmp_path / "ohlcv_1d" / "product=TEST" / "ohlcv_1d.parquet"
    before = source.read_bytes()
    with pytest.raises(ValueError, match="Unknown calendar"):
        build(tmp_path, TimeBasedRoll(calendar="not-a-calendar"))
    assert source.read_bytes() == before
