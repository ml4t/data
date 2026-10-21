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


def test_a_new_price_capture_cannot_hide_an_older_open_interest_fact_that_was_known(tmp_path):
    bars, definitions = captured_contract_files(tmp_path)
    b = (pl.col("asset") == 23) & (pl.col("timestamp") == date(2025, 7, 8))
    original = bars.with_columns(
        pl.when(b)
        .then(datetime(2025, 7, 8, 21, 1, tzinfo=UTC))
        .otherwise(pl.col("open_interest_available_at"))
        .alias("open_interest_available_at")
    )
    correction = original.filter(b).with_columns(
        pl.lit(4022, dtype=bars.schema["close"]).alias("close"),
        pl.lit(4022, dtype=bars.schema["open"]).alias("open"),
        pl.lit(4023, dtype=bars.schema["high"]).alias("high"),
        pl.lit(4021, dtype=bars.schema["low"]).alias("low"),
        pl.lit(5, dtype=bars.schema["open_interest"]).alias("open_interest"),
        pl.lit("r2").alias("revision_id"),
        pl.lit(datetime(2025, 7, 9, 12, tzinfo=UTC)).alias("available_at"),
        pl.lit(datetime(2025, 7, 10, 21, 1, tzinfo=UTC)).alias("open_interest_available_at"),
    )
    captured = pl.concat([original, correction])
    before = futures.prepare_futures_inputs(
        captured,
        definitions,
        as_of=datetime(2025, 7, 9, 21, 30, tzinfo=UTC),
        roll_strategy=HighestOpenInterestRoll(),
        feed_spec=contract_feed(),
    )
    assert before.selections["selected_instrument_id"].to_list() == [23]
    assert before.bars.filter(b)["close"].to_list() == [4022]
    oi = before.metrics.filter(b & (pl.col("metric") == "open_interest"))
    assert oi["open_interest"].to_list() == [80]
    assert oi["revision_id"].to_list() == ["r1"]
    after = futures.prepare_futures_inputs(
        captured,
        definitions,
        as_of=datetime(2025, 7, 11, 21, 30, tzinfo=UTC),
        roll_strategy=HighestOpenInterestRoll(),
        feed_spec=contract_feed(),
    )
    assert after.selections["selected_instrument_id"].to_list() == [17]
    assert_frame_equal(captured, pl.concat([original, correction]))


def test_repeated_unchanged_oi_on_price_revisions_retains_the_original_metric_capture(tmp_path):
    bars, definitions = captured_contract_files(tmp_path)
    b = (pl.col("asset") == 23) & (pl.col("timestamp") == date(2025, 7, 8))
    original = bars.with_columns(
        pl.when(b)
        .then(datetime(2025, 7, 8, 21, 1, tzinfo=UTC))
        .otherwise(pl.col("open_interest_available_at"))
        .alias("open_interest_available_at")
    )
    correction = original.filter(b).with_columns(
        pl.lit("r2").alias("revision_id"),
        pl.lit(datetime(2025, 7, 9, 12, tzinfo=UTC)).alias("available_at"),
    )
    result = futures.prepare_futures_inputs(
        pl.concat([original, correction]),
        definitions,
        as_of=datetime(2025, 7, 9, 21, 30, tzinfo=UTC),
        roll_strategy=HighestOpenInterestRoll(),
        feed_spec=contract_feed(),
    )
    oi = result.metrics.filter(b & (pl.col("metric") == "open_interest"))
    assert oi["open_interest"].to_list() == [80]
    assert oi["revision_id"].to_list() == ["r1"]


@pytest.mark.parametrize("strategy,expected", [(TimeBasedRoll(0), 17), (HighestVolumeRoll(), 23)])
def test_an_all_unknown_optional_oi_column_does_not_block_a_price_or_volume_policy(
    tmp_path, strategy, expected
):
    bars, definitions = captured_contract_files(tmp_path)
    unknown = bars.with_columns(
        pl.lit(None).alias("open_interest"), pl.lit(None).alias("open_interest_available_at")
    )
    result = futures.prepare_futures_inputs(
        unknown,
        definitions,
        as_of=datetime(2025, 7, 9, 21, 30, tzinfo=UTC),
        roll_strategy=strategy,
        feed_spec=contract_feed(),
    )
    assert result.selections["selected_instrument_id"].to_list() == [expected]
    assert result.metrics.filter(pl.col("metric") == "open_interest").height == 0
    assert result.bars["open_interest"].null_count() == 6


def test_empty_source_observations_return_typed_empty_inputs_without_inventing_a_contract(tmp_path):
    bars, definitions = captured_contract_files(tmp_path)
    result = futures.prepare_futures_inputs(
        bars.head(0),
        definitions,
        as_of=datetime(2025, 7, 9, 21, 30, tzinfo=UTC),
        roll_strategy=HighestVolumeRoll(),
        feed_spec=contract_feed(),
    )
    assert result.bars.height == result.selections.height == result.rolls.height == 0
    assert (
        result.bars.schema["asset"]
        == result.selections.schema["selected_instrument_id"]
        == pl.Int64
    )
    assert result.bars.schema["timestamp"] == pl.Date
    assert result.metrics.schema["volume"] == pl.Int64


def test_explicit_metric_revisions_are_not_deduplicated_as_repeated_price_captures(tmp_path):
    bars, definitions = captured_contract_files(tmp_path)
    b = (pl.col("asset") == 23) & (pl.col("timestamp") == date(2025, 7, 8))
    original = bars.with_columns(
        pl.lit("oi-r1").alias("open_interest_revision_id"),
        pl.when(b)
        .then(datetime(2025, 7, 8, 21, 1, tzinfo=UTC))
        .otherwise(pl.col("open_interest_available_at"))
        .alias("open_interest_available_at"),
    )
    ambiguous = original.filter(b).with_columns(
        pl.lit("r2").alias("revision_id"),
        pl.lit("oi-r2").alias("open_interest_revision_id"),
        pl.lit(datetime(2025, 7, 9, 12, tzinfo=UTC)).alias("available_at"),
    )
    with pytest.raises(ValueError, match="indistinguishable"):
        futures.prepare_futures_inputs(
            pl.concat([original, ambiguous]),
            definitions,
            as_of=datetime(2025, 7, 9, 21, 30, tzinfo=UTC),
            roll_strategy=HighestOpenInterestRoll(),
            feed_spec=contract_feed(),
        )


def test_cme_daily_period_open_instants_map_to_the_same_notice_session_as_date_labels(tmp_path):
    bars, definitions = captured_contract_files(tmp_path)
    instants = bars.with_columns(
        (
            pl.col("timestamp").cast(pl.Datetime("ns", "UTC"))
            - pl.duration(days=1, time_unit="ns")
            + pl.duration(hours=22, nanoseconds=1, time_unit="ns")
        ).alias("timestamp")
    )
    result = futures.prepare_futures_inputs(
        instants,
        definitions,
        as_of=datetime(2025, 7, 9, 21, 30, tzinfo=UTC),
        roll_strategy=FirstNoticeDateRoll(1),
        feed_spec=contract_feed().with_overrides(timestamp_semantics="event_time", timezone="UTC"),
        definitions_feed_spec=contract_feed(),
    )
    assert result.selections["selected_instrument_id"].to_list() == [23]
    assert result.selections["observation_label"].to_list() == [date(2025, 7, 9)]
    assert result.bars["timestamp"].equals(instants["timestamp"])
    assert result.bars.schema["timestamp"] == pl.Datetime("ns", "UTC")


def test_aware_session_labels_keep_their_intrinsic_civil_date_when_timezone_is_omitted(tmp_path):
    bars, definitions = captured_contract_files(tmp_path)
    labels = bars.with_columns(
        pl.col("timestamp").cast(pl.Datetime("ns")).dt.replace_time_zone("Asia/Tokyo")
    )
    result = futures.prepare_futures_inputs(
        labels,
        definitions,
        as_of=datetime(2025, 7, 9, 21, 30, tzinfo=UTC),
        roll_strategy=FirstNoticeDateRoll(1),
        feed_spec={**contract_feed().to_dict(), "timezone": None},
    )
    assert result.selections["selected_instrument_id"].to_list() == [23]
    assert result.selections["observation_label"].to_list() == [date(2025, 7, 9)]
    assert result.bars["timestamp"].equals(labels["timestamp"])


@pytest.mark.parametrize("limit", ["expiration", "last_trade_date", "valid_to"])
def test_stale_observations_cannot_qualify_a_current_contract_past_its_declared_lifecycle(
    tmp_path, limit
):
    bars, definitions = captured_contract_files(tmp_path)
    dated = definitions.with_columns(pl.lit(date(2025, 7, 10)).alias(limit))
    with pytest.raises(ValueError, match="current.*(identity|lifecycle)"):
        futures.prepare_futures_inputs(
            bars,
            dated,
            as_of=datetime(2025, 7, 11, 21, 30, tzinfo=UTC),
            roll_strategy=TimeBasedRoll(0),
            feed_spec=contract_feed(),
        )
    assert_frame_equal(bars, load_databento_ohlcv("TEST", tmp_path))


def test_current_definition_provenance_is_distinct_from_the_stale_observation_definition(tmp_path):
    bars, definitions = captured_contract_files(tmp_path)
    prior = definitions.with_columns(pl.lit(date(2025, 7, 10)).alias("valid_to"))
    current = definitions.with_columns(
        pl.lit(date(2025, 7, 10)).alias("valid_from"),
        pl.lit("r2").alias("revision_id"),
        pl.lit(100, dtype=definitions.schema["multiplier"]).alias("multiplier"),
    )
    result = futures.prepare_futures_inputs(
        bars,
        pl.concat([prior, current], how="diagonal"),
        as_of=datetime(2025, 7, 11, 21, 30, tzinfo=UTC),
        roll_strategy=TimeBasedRoll(0),
        feed_spec=contract_feed(),
    )
    assert result.bars["instrument_revision_id"].unique().to_list() == ["r1"]
    assert result.selections["definition_revision_id"].to_list() == ["r2"]
    assert set(result.definitions["revision_id"]) == {"r1", "r2"}


def test_unpublished_invalid_definition_values_do_not_change_an_earlier_causal_selection(tmp_path):
    bars, definitions = captured_contract_files(tmp_path)
    future = definitions.head(1).with_columns(
        pl.lit("r2").alias("revision_id"),
        pl.lit(datetime(2025, 7, 20, tzinfo=UTC)).alias("available_at"),
        pl.lit(0, dtype=definitions.schema["multiplier"]).alias("multiplier"),
        pl.lit("").alias("currency"),
    )
    result = futures.prepare_futures_inputs(
        bars,
        pl.concat([definitions, future]),
        as_of=datetime(2025, 7, 9, 21, 30, tzinfo=UTC),
        roll_strategy=HighestVolumeRoll(),
        feed_spec=contract_feed(),
    )
    assert result.selections["selected_instrument_id"].to_list() == [23]
    assert result.definitions["revision_id"].unique().to_list() == ["r1"]


def test_normalized_feed_and_roll_provenance_point_to_the_actual_leg_observations(tmp_path):
    bars, definitions = captured_contract_files(tmp_path)
    cents = bars.with_columns(pl.lit("cents").alias("price_quote_unit"))
    facts = definitions.with_columns(pl.lit("cents").alias("price_quote_unit"))
    result = futures.prepare_futures_inputs(
        cents,
        facts,
        as_of=datetime(2025, 7, 9, 21, 30, tzinfo=UTC),
        roll_strategy=HighestVolumeRoll(),
        price_quote_unit="dollars",
        feed_spec=contract_feed(),
    )
    normalized = FeedSpec.from_any(result.metadata["normalized_feed_spec"])
    assert normalized.price_col == "normalized_close"
    assert normalized.volume_col == "volume"
    assert result.rolls["old_normalized_mark"].to_list() == [40.2]
    assert result.rolls["new_normalized_mark"].to_list() == pytest.approx([40.3], rel=0, abs=1e-12)
    assert result.rolls["old_normalized_multiplier"].to_list() == [5000.0]
    assert result.rolls["old_mark_source"].to_list() == ["capture"]
    assert result.rolls["new_mark_revision_id"].to_list() == ["r1"]
    assert result.rolls["old_currency"].to_list() == ["USD"]
    assert result.rolls["new_normalized_price_quote_unit"].to_list() == ["dollars"]


def test_empty_and_populated_preparations_have_the_same_output_table_schemas(tmp_path):
    bars, definitions = captured_contract_files(tmp_path)
    kwargs = {
        "as_of": datetime(2025, 7, 9, 21, 30, tzinfo=UTC),
        "roll_strategy": HighestVolumeRoll(),
        "feed_spec": contract_feed(),
    }
    populated = futures.prepare_futures_inputs(bars, definitions, **kwargs)
    empty = futures.prepare_futures_inputs(bars.head(0), definitions, **kwargs)
    for name in ("bars", "definitions", "chain", "metrics", "selections", "rolls", "coverage"):
        assert getattr(empty, name).schema == getattr(populated, name).schema, name
        assert getattr(empty, name).height == 0


def test_declared_custom_columns_and_symbol_identity_preserve_the_source_tables(tmp_path):
    bars, definitions = captured_contract_files(tmp_path)
    physical = bars.drop("asset").rename(
        {"timestamp": "observed_on", "symbol": "contract_code", "close": "mark", "volume": "qty"}
    )
    feed = FeedSpec(
        timestamp_col="observed_on",
        entity_col="contract_code",
        price_col="mark",
        close_col="mark",
        volume_col="qty",
        time_contract_version=2,
        calendar="CME_Equity",
        timestamp_semantics="session_label",
        data_frequency="daily",
    )
    result = futures.prepare_futures_inputs(
        physical,
        definitions,
        as_of=datetime(2025, 7, 9, 21, 30, tzinfo=UTC),
        roll_strategy=HighestVolumeRoll(),
        feed_spec=feed,
        definitions_feed_spec=contract_feed(),
        identity_mode="symbol",
    )
    assert result.selections["selected_instrument_id"].to_list() == [23]
    assert result.bars["contract_code"].equals(physical["contract_code"])
    assert result.bars["mark"].equals(physical["mark"])
    assert result.rolls["old_mark"].to_list() == [4020]
    assert (
        result.metrics.filter(pl.col("metric") == "volume")["volume"].to_list()
        == physical["qty"].to_list()
    )


def test_unknown_mark_meaning_cannot_be_presented_as_a_qualified_roll_mark(tmp_path):
    bars, definitions = captured_contract_files(tmp_path)
    with pytest.raises(ValueError, match="mark_kind"):
        futures.prepare_futures_inputs(
            bars.with_columns(pl.lit("").alias("mark_kind")),
            definitions,
            as_of=datetime(2025, 7, 9, 21, 30, tzinfo=UTC),
            roll_strategy=HighestVolumeRoll(),
            feed_spec=contract_feed(),
        )


def test_identity_unit_mapping_keeps_integer_marks_beyond_float64_exact_range(tmp_path):
    bars, definitions = captured_contract_files(tmp_path)
    captured = bars.with_columns(
        (pl.col(name) + 2**53 + 1).alias(name) for name in ("open", "high", "low", "close")
    )
    result = futures.prepare_futures_inputs(
        captured,
        definitions,
        as_of=datetime(2025, 7, 9, 21, 30, tzinfo=UTC),
        roll_strategy=HighestVolumeRoll(),
        feed_spec=contract_feed(),
    )
    assert result.bars["normalized_close"].to_list() == [
        9007199254744993,
        9007199254745003,
        9007199254745003,
        9007199254745013,
        9007199254745013,
        9007199254745023,
    ]
    assert result.bars["normalized_close"].dtype == pl.Int64


def test_declared_local_naive_open_instants_use_the_same_session_for_identity_and_selection(
    tmp_path,
):
    bars, definitions = captured_contract_files(tmp_path)
    local = bars.with_columns(
        (
            pl.col("timestamp").cast(pl.Datetime("ns"))
            - pl.duration(days=1, time_unit="ns")
            + pl.duration(hours=18, time_unit="ns")
        ).alias("timestamp")
    )
    dated = definitions.with_columns(pl.lit(date(2025, 7, 7)).alias("valid_from"))
    result = futures.prepare_futures_inputs(
        local,
        dated,
        as_of=datetime(2025, 7, 9, 21, 30, tzinfo=UTC),
        roll_strategy=FirstNoticeDateRoll(1),
        feed_spec=contract_feed().with_overrides(timestamp_semantics="event_time"),
        definitions_feed_spec=contract_feed(),
    )
    assert result.selections["selected_instrument_id"].to_list() == [23]
    assert result.bars["timestamp"].equals(
        local["timestamp"].dt.replace_time_zone("America/New_York").dt.convert_time_zone("UTC")
    )
