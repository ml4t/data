"""Offline individual-contract preparation beside the existing continuous research builder."""

from datetime import UTC, datetime
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory

import polars as pl
from polars.testing import assert_frame_equal

from ml4t.data.futures import (
    ContinuousContractBuilder,
    FuturesInputs,
    HighestOpenInterestRoll,
    HighestVolumeRoll,
    NoAdjustment,
    load_databento_definitions,
    load_databento_ohlcv,
    prepare_futures_inputs,
)
from ml4t.specs import FeedSpec


def migrate_futures(directory: Path) -> tuple[FuturesInputs, FuturesInputs, FuturesInputs]:
    bars = pl.read_csv(
        StringIO(
            "timestamp,asset,symbol,open,high,low,close,volume,open_interest,currency,price_quote_unit,mark_kind,available_at,open_interest_available_at,source,revision_id\n"
            "2025-07-07,17,TESTN2025,4000,4001,3999,4000,100,100,USD,index_points,trade_close,2025-07-07T21:01:00Z,2025-07-07T21:01:00Z,capture,r1\n"
            "2025-07-07,23,TESTU2025,4010,4011,4009,4010,50,50,USD,index_points,trade_close,2025-07-07T21:01:00Z,2025-07-07T21:01:00Z,capture,r1\n"
            "2025-07-08,17,TESTN2025,4010,4011,4009,4010,40,40,USD,index_points,trade_close,2025-07-08T21:01:00Z,2025-07-08T21:01:00Z,capture,r1\n"
            "2025-07-08,23,TESTU2025,4020,4021,4019,4020,80,80,USD,index_points,trade_close,2025-07-08T21:01:00Z,2025-07-10T21:01:00Z,capture,r1\n"
            "2025-07-09,17,TESTN2025,4020,4021,4019,4020,30,30,USD,index_points,trade_close,2025-07-09T21:01:00Z,2025-07-09T21:01:00Z,capture,r1\n"
            "2025-07-09,23,TESTU2025,4030,4031,4029,4030,90,90,USD,index_points,trade_close,2025-07-09T21:01:00Z,2025-07-10T21:01:00Z,capture,r1\n"
        ),
        try_parse_dates=True,
    )
    definitions = pl.read_csv(
        StringIO(
            "asset,symbol,root,exchange,instrument_type,valid_from,available_at,source,revision_id,expiration,first_notice_date,last_trade_date,auto_close_date,currency,multiplier,tick_size,price_quote_unit,settlement_type,volume_unit,open_interest_unit\n"
            "17,TESTN2025,TEST,CME,future,2025-07-01,2025-06-01T00:00:00Z,capture,r1,2025-07-15,2025-07-10,2025-07-14,2025-07-10,USD,50,0.25,index_points,physical,contracts,contracts\n"
            "23,TESTU2025,TEST,CME,future,2025-07-01,2025-06-01T00:00:00Z,capture,r1,2025-09-15,2025-09-10,2025-09-14,2025-09-10,USD,50,0.25,index_points,physical,contracts,contracts\n"
        ),
        try_parse_dates=True,
    )
    for relative, frame in (
        ("ohlcv_1d/product=TEST/ohlcv_1d.parquet", bars),
        ("definition/product=TEST/definition.parquet", definitions),
    ):
        path = directory / relative
        path.parent.mkdir(parents=True)
        frame.write_parquet(path)
    research = ContinuousContractBuilder(
        roll_strategy=HighestVolumeRoll(), adjustment_method=NoAdjustment()
    ).build("TEST", data_source="databento", storage_path=directory)
    assert research["symbol"].to_list() == ["TESTN2025", "TESTU2025"]
    restored = load_databento_ohlcv("TEST", directory)
    facts = load_databento_definitions("TEST", directory)
    feed = FeedSpec(
        entity_col="asset",
        time_contract_version=2,
        calendar="CME_Equity",
        timestamp_semantics="session_label",
        data_frequency="daily",
    )
    first = prepare_futures_inputs(
        restored,
        facts,
        as_of=datetime(2025, 7, 8, 21, 30, tzinfo=UTC),
        roll_strategy=HighestVolumeRoll(),
        feed_spec=feed,
    )
    second = prepare_futures_inputs(
        restored,
        facts,
        as_of=datetime(2025, 7, 9, 21, 30, tzinfo=UTC),
        roll_strategy=HighestVolumeRoll(),
        feed_spec=feed,
    )
    delayed = prepare_futures_inputs(
        restored,
        facts,
        as_of=datetime(2025, 7, 9, 21, 30, tzinfo=UTC),
        roll_strategy=HighestOpenInterestRoll(),
        feed_spec=feed,
    )
    assert first.selections["selected_instrument_id"].to_list() == [17]
    assert second.selections["selected_instrument_id"].to_list() == [23]
    assert delayed.selections["selected_instrument_id"].to_list() == [17]
    assert second.rolls["old_mark"].to_list() == [4020]
    assert second.rolls["new_mark"].to_list() == [4030]
    assert second.rolls["old_mark_source"].to_list() == ["capture"]
    assert_frame_equal(restored, load_databento_ohlcv("TEST", directory))
    assert_frame_equal(facts, load_databento_definitions("TEST", directory))
    return first, second, delayed


if __name__ == "__main__":
    with TemporaryDirectory(prefix="ml4t-futures-inputs-") as temporary:
        _, prepared, delayed = migrate_futures(Path(temporary))
        print(prepared.selections)
        print(prepared.rolls)
        print(delayed.selections)
