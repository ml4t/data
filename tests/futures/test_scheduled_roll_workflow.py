"""Provider-generated contract files must use scheduled, not observed, expiry offsets."""

from datetime import date

import polars as pl

from ml4t.data.futures.adjustment import NoAdjustment
from ml4t.data.futures.continuous import ContinuousContractBuilder
from ml4t.data.futures.roll import TimeBasedRoll
from ml4t.data.providers.synthetic import SyntheticProvider


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
