"""Known-at futures preparation must retain source identity,metric timing and raw leg units."""

from datetime import UTC, date, datetime
from io import StringIO

import polars as pl
import pytest
from polars.testing import assert_frame_equal

import ml4t.data.futures as futures
from ml4t.data.futures import (
    ContinuousContractBuilder,
    ContractSpec,
    FirstNoticeDateRoll,
    FuturesAssetClass,
    HighestOpenInterestRoll,
    HighestVolumeRoll,
    NoAdjustment,
    SettlementType,
    TimeBasedRoll,
    load_databento_definitions,
    load_databento_ohlcv,
)
from ml4t.specs import FeedSpec


def captured_contract_files(directory):
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
    price_path = directory / "ohlcv_1d/product=TEST/ohlcv_1d.parquet"
    definition_path = directory / "definition/product=TEST/definition.parquet"
    price_path.parent.mkdir(parents=True)
    definition_path.parent.mkdir(parents=True)
    bars.write_parquet(price_path)
    definitions.write_parquet(definition_path)
    return (
        load_databento_ohlcv("TEST", directory),
        load_databento_definitions("TEST", directory),
    )


def contract_feed():
    return FeedSpec(
        time_contract_version=2,
        entity_col="asset",
        calendar="CME_Equity",
        timestamp_semantics="session_label",
        timezone="America/New_York",
        data_frequency="daily",
    )


def test_source_file_selection_matches_lagged_builder_and_retains_actual_paired_marks(tmp_path):
    bars, definitions = captured_contract_files(tmp_path)
    legacy = ContinuousContractBuilder(
        roll_strategy=HighestVolumeRoll(),
        adjustment_method=NoAdjustment(),
    ).build("TEST", data_source="databento", storage_path=tmp_path)
    assert legacy["symbol"].to_list() == ["TESTN2025", "TESTU2025"]
    first = futures.prepare_futures_inputs(
        bars,
        definitions,
        as_of=datetime(2025, 7, 8, 21, 30, tzinfo=UTC),
        roll_strategy=HighestVolumeRoll(),
        feed_spec=contract_feed(),
    )
    second = futures.prepare_futures_inputs(
        bars,
        definitions,
        as_of=datetime(2025, 7, 9, 21, 30, tzinfo=UTC),
        roll_strategy=HighestVolumeRoll(),
        feed_spec=contract_feed(),
    )
    assert first.selections["selected_instrument_id"].to_list() == [17]
    assert second.selections["selected_instrument_id"].to_list() == [23]
    assert second.selections["metric_observed_at"].to_list() == [date(2025, 7, 8)]
    assert second.selections["metric_available_at"].to_list() == [
        datetime(2025, 7, 8, 21, 1, tzinfo=UTC)
    ]
    assert second.rolls["old_instrument_id"].to_list() == [17]
    assert second.rolls["new_instrument_id"].to_list() == [23]
    assert second.rolls["old_mark"].to_list() == [4020]
    assert second.rolls["new_mark"].to_list() == [4030]
    assert second.rolls["mark_kind"].to_list() == ["trade_close"]
    assert second.definitions.filter(pl.col("asset") == 17)["first_notice_date"].to_list() == [
        date(2025, 7, 10)
    ]
    assert second.bars["close"].to_list() == [4000, 4010, 4010, 4020, 4020, 4030]
    assert second.metadata["calendar"] == "CME_Equity"
    assert second.metadata["selection_policy"]["parameters"] == {"rank": 0, "min_volume": 0}


def test_delayed_open_interest_does_not_enter_an_earlier_current_selection(tmp_path):
    bars, definitions = captured_contract_files(tmp_path)
    before = futures.prepare_futures_inputs(
        bars,
        definitions,
        as_of=datetime(2025, 7, 9, 21, 30, tzinfo=UTC),
        roll_strategy=HighestOpenInterestRoll(),
        feed_spec=contract_feed(),
    )
    after = futures.prepare_futures_inputs(
        bars,
        definitions,
        as_of=datetime(2025, 7, 11, 21, 30, tzinfo=UTC),
        roll_strategy=HighestOpenInterestRoll(),
        feed_spec=contract_feed(),
    )
    assert before.selections["selected_instrument_id"].to_list() == [17]
    assert after.selections["selected_instrument_id"].to_list() == [23]
    assert (
        before.metrics.filter(
            (pl.col("asset") == 23)
            & (pl.col("timestamp") >= date(2025, 7, 8))
            & (pl.col("metric") == "open_interest")
        ).height
        == 0
    )
    assert_frame_equal(bars, load_databento_ohlcv("TEST", tmp_path))


def test_future_price_and_definition_revisions_do_not_rewrite_prior_selection(tmp_path):
    bars, definitions = captured_contract_files(tmp_path)
    cutoff = datetime(2025, 7, 9, 21, 30, tzinfo=UTC)
    original = futures.prepare_futures_inputs(
        bars,
        definitions,
        as_of=cutoff,
        roll_strategy=HighestVolumeRoll(),
        feed_spec=contract_feed(),
    )
    revision = bars.filter(
        (pl.col("asset") == 17) & (pl.col("timestamp") == date(2025, 7, 8))
    ).with_columns(
        pl.lit(900, dtype=bars.schema["volume"]).alias("volume"),
        pl.lit("r2").alias("revision_id"),
        pl.lit(datetime(2025, 7, 20, tzinfo=UTC)).alias("available_at"),
    )
    definition_revision = definitions.head(1).with_columns(
        pl.lit(date(2025, 8, 15)).alias("expiration"),
        pl.lit("r2").alias("revision_id"),
        pl.lit(datetime(2025, 7, 20, tzinfo=UTC)).alias("available_at"),
    )
    changed = futures.prepare_futures_inputs(
        pl.concat([bars, revision]),
        pl.concat([definitions, definition_revision]),
        as_of=cutoff,
        roll_strategy=HighestVolumeRoll(),
        feed_spec=contract_feed(),
    )
    assert_frame_equal(original.selections, changed.selections)
    assert_frame_equal(original.rolls, changed.rolls)


def test_a_missing_old_leg_is_not_replaced_with_a_continuous_price_or_a_synthetic_mark(tmp_path):
    bars, definitions = captured_contract_files(tmp_path)
    missing = bars.filter(~((pl.col("asset") == 17) & (pl.col("timestamp") == date(2025, 7, 9))))
    with pytest.raises(ValueError, match="exactly one close"):
        futures.prepare_futures_inputs(
            missing,
            definitions,
            as_of=datetime(2025, 7, 9, 21, 30, tzinfo=UTC),
            roll_strategy=HighestVolumeRoll(),
            feed_spec=contract_feed(),
        )


def test_actual_dated_notice_is_not_replaced_with_an_expiry_offset_estimate(tmp_path):
    bars, definitions = captured_contract_files(tmp_path)
    first = futures.prepare_futures_inputs(
        bars,
        definitions,
        as_of=datetime(2025, 7, 8, 21, 30, tzinfo=UTC),
        roll_strategy=FirstNoticeDateRoll(1),
        feed_spec=contract_feed(),
    )
    second = futures.prepare_futures_inputs(
        bars,
        definitions,
        as_of=datetime(2025, 7, 9, 21, 30, tzinfo=UTC),
        roll_strategy=FirstNoticeDateRoll(1),
        feed_spec=contract_feed(),
    )
    assert first.selections["selected_instrument_id"].to_list() == [17]
    assert second.selections["selected_instrument_id"].to_list() == [23]
    assert second.metadata["notice_method"] == "source_dates"


@pytest.mark.parametrize("calendar,expected", [("NYSE", 23), ("24/7", 17)])
def test_calendar_argument_binds_the_strategy_without_mutating_its_original_configuration(
    tmp_path, calendar, expected
):
    bars, definitions = captured_contract_files(tmp_path)
    strategy = TimeBasedRoll(5)
    result = futures.prepare_futures_inputs(
        bars,
        definitions,
        as_of=datetime(2025, 7, 8, 21, 30, tzinfo=UTC),
        roll_strategy=strategy,
        feed_spec=contract_feed(),
        calendar=calendar,
    )
    assert result.selections["selected_instrument_id"].to_list() == [expected]
    assert result.metadata["calendar"] == calendar
    assert strategy.calendar == "NYSE"


def test_calendar_none_does_not_silently_leave_a_scheduled_expiry_strategy_enabled(tmp_path):
    bars, definitions = captured_contract_files(tmp_path)
    with pytest.raises(ValueError, match="row-based"):
        futures.prepare_futures_inputs(
            bars,
            definitions,
            as_of=datetime(2025, 7, 8, 21, 30, tzinfo=UTC),
            roll_strategy=TimeBasedRoll(5),
            feed_spec=contract_feed(),
            calendar=None,
        )


def test_cents_normalization_preserves_raw_quotes_and_currency_per_declared_quote_unit(tmp_path):
    bars, definitions = captured_contract_files(tmp_path)
    raw = bars.with_columns(pl.lit("cents").alias("price_quote_unit"))
    terms = definitions.with_columns(pl.lit("cents").alias("price_quote_unit"))
    result = futures.prepare_futures_inputs(
        raw,
        terms,
        as_of=datetime(2025, 7, 9, 21, 30, tzinfo=UTC),
        roll_strategy=HighestVolumeRoll(),
        feed_spec=contract_feed(),
        price_quote_unit="dollars",
    )
    a = result.bars.filter(pl.col("asset") == 17)
    assert a["close"].to_list() == [4000, 4010, 4020]
    assert a["normalized_close"].to_list() == pytest.approx([40, 40.1, 40.2])
    assert a["normalized_multiplier"].to_list() == [5000] * 3
    assert a["normalized_tick_size"].to_list() == [0.0025] * 3
    pnl = (a["normalized_close"][1] - a["normalized_close"][0]) * a["normalized_multiplier"][0]
    assert pnl == pytest.approx(500)
    assert result.rolls["old_mark"].to_list() == [4020]
    assert result.rolls["old_price_quote_unit"].to_list() == ["cents"]
    assert_frame_equal(raw, bars.with_columns(pl.lit("cents").alias("price_quote_unit")))


def test_an_explicit_zero_legacy_notice_offset_is_not_replaced_with_twenty_five_days(tmp_path):
    bars, definitions = captured_contract_files(tmp_path)
    data = bars.rename({"timestamp": "date"}).join(
        definitions.select("asset", "expiration"), on="asset", how="left"
    )
    spec = ContractSpec(
        ticker="TEST",
        name="Captured contract",
        exchange="CME",
        asset_class=FuturesAssetClass.EQUITY_INDEX,
        multiplier=50,
        tick_size=0.25,
        tick_value=12.5,
        price_quote_unit="index_points",
        settlement_type=SettlementType.PHYSICAL,
        contract_months="NU",
        first_notice_days=0,
    )
    selected = FirstNoticeDateRoll(1, feed_spec=contract_feed()).select_contracts(data, spec)
    assert selected["symbol"].to_list() == ["TESTN2025"] * 3
