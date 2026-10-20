"""Known-at history must use available source revisions and explicit adjustment definitions."""

from datetime import UTC, datetime
from io import StringIO

import pandas as pd
import polars as pl
import pytest
from polars.testing import assert_frame_equal

import ml4t.data.adjustments as adjustments
from ml4t.data.storage import FlatStorage, StorageConfig
from ml4t.specs import FeedSpec


def history_feed():
    return FeedSpec(
        time_contract_version=2,
        entity_col="asset",
        calendar="NYSE",
        timestamp_semantics="session_label",
        timezone="America/New_York",
        data_frequency="daily",
    )


def captured_history():
    prices = pl.read_csv(
        StringIO(
            "timestamp,asset,close,volume,currency,available_at,source,revision_id\n"
            "2025-07-02,17,200,1000,USD,2025-07-02T20:00:00Z,captured,r1\n"
            "2025-07-03,17,98,7,USD,2025-07-03T17:00:00Z,captured,r1\n"
        ),
        try_parse_dates=True,
    )
    actions = pl.read_csv(
        StringIO(
            "asset,event_id,revision_id,action_kind,source,source_record_id,effective_at,available_at,ex_date,payment_date,split_ratio,cash_per_share,currency,share_basis\n"
            "17,s1,r1,split,captured,101,2025-07-03T00:00:00Z,2025-07-01T12:00:00Z,2025-07-03,,2,,,\n"
            "17,c1,r1,cash_dividend,captured,102,2025-07-03T00:00:00Z,2025-07-01T12:00:00Z,2025-07-03,2025-07-15,,3,USD,post_event\n"
        ),
        try_parse_dates=True,
        schema_overrides={"split_ratio": pl.Float64, "cash_per_share": pl.Float64},
    )
    return prices, actions


def test_native_known_at_history_preserves_raw_values_and_the_independent_shareholder_return():
    prices, actions = captured_history()
    before = prices.clone()
    prepared = adjustments.prepare_adjusted_history(
        prices,
        actions,
        as_of=datetime(2025, 7, 3, 18, tzinfo=UTC),
        feed_spec=history_feed(),
        action_coverage="complete",
    )
    assert prepared.prices["close"].to_list() == [200, 98]
    assert prepared.prices["volume"].to_list() == [1000, 7]
    assert prepared.prices["price_adjustment_factor"].to_list() == pytest.approx([98 / 202, 1])
    assert prepared.prices["adj_volume"].to_list() == [2000, 7]
    assert prepared.prices["adj_close"][1] / prepared.prices["adj_close"][0] == pytest.approx(1.01)
    assert prepared.actions["event_id"].to_list() == ["s1", "c1"]
    assert prepared.metadata["perspective"] == "causal"
    assert prepared.metadata["price_basis"] == "total_return"
    assert_frame_equal(prices, before)


def test_prepared_feed_declares_the_adjusted_price_and_volume_fields_to_consumers():
    prices, actions = captured_history()
    result = adjustments.prepare_adjusted_history(
        prices,
        actions,
        as_of=datetime(2025, 7, 3, 18, tzinfo=UTC),
        feed_spec=history_feed(),
        action_coverage="complete",
    )
    consumer = FeedSpec.from_any(result.metadata["feed_spec"])
    values = result.prices[consumer.price_col]
    assert values[1] / values[0] == pytest.approx(1.01)
    assert result.prices[consumer.volume_col].to_list() == [2000, 7]
    source = FeedSpec.from_any(result.metadata["source_feed_spec"])
    assert result.prices[source.price_col].to_list() == [200, 98]


def test_cash_terms_per_pre_event_share_convert_without_overwriting_the_source_basis():
    prices, actions = captured_history()
    actions = actions.with_columns(
        pl.when(pl.col("action_kind") == "cash_dividend")
        .then(6.0)
        .otherwise(pl.col("cash_per_share"))
        .alias("cash_per_share"),
        pl.when(pl.col("action_kind") == "cash_dividend")
        .then(pl.lit("pre_event"))
        .otherwise(pl.col("share_basis"))
        .alias("share_basis"),
    )
    result = adjustments.prepare_adjusted_history(
        prices,
        actions,
        as_of=datetime(2025, 7, 3, 18, tzinfo=UTC),
        feed_spec=history_feed(),
        action_coverage="complete",
    )
    assert result.prices["adj_close"][1] / result.prices["adj_close"][0] == pytest.approx(1.01)
    cash = result.actions.filter(pl.col("action_kind") == "cash_dividend")
    assert cash["cash_per_share"].to_list() == [6]
    assert cash["share_basis"].to_list() == ["pre_event"]
    assert result.prices["adj_volume"].to_list() == [2000, 7]


def test_intraday_cash_factors_require_a_captured_actual_close_in_the_same_currency():
    daily, actions = captured_history()
    observed = pl.col("timestamp").cast(pl.Datetime("ns", "UTC")) + pl.duration(
        hours=14, minutes=30
    )
    prices = daily.with_columns(
        observed.alias("timestamp"),
        observed.alias("available_at"),
    )
    closes = pl.read_csv(
        StringIO(
            "asset,ex_date,close,currency,available_at,source,revision_id\n"
            "17,2025-07-03,98,EUR,2025-07-03T17:00:00Z,capture,r1\n"
        ),
        try_parse_dates=True,
    )
    feed = history_feed().with_overrides(
        timestamp_semantics="event_time",
        data_frequency="1h",
        bar_grid_anchor="session_open",
        bar_label="left",
        partial_bar_policy="include",
    )
    with pytest.raises(ValueError, match="currency"):
        adjustments.prepare_adjusted_history(
            prices,
            actions,
            ex_date_closes=closes,
            as_of=datetime(2025, 7, 3, 18, tzinfo=UTC),
            feed_spec=feed,
            action_coverage="complete",
        )
    valid = adjustments.prepare_adjusted_history(
        prices,
        actions,
        ex_date_closes=closes.with_columns(pl.lit("USD").alias("currency")),
        as_of=datetime(2025, 7, 3, 18, tzinfo=UTC),
        feed_spec=feed,
        action_coverage="complete",
    )
    assert valid.prices["close"].to_list() == [200, 98]
    assert valid.prices["adj_close"][0] == pytest.approx(200 * 98 / 202)
    assert valid.prices["adj_close"][1] == 98


def test_retrospective_latest_observation_basis_does_not_apply_an_announced_future_split():
    prices, actions = captured_history()
    future = pl.read_csv(
        StringIO(
            "asset,event_id,revision_id,action_kind,source,source_record_id,effective_at,available_at,ex_date,payment_date,split_ratio,cash_per_share,currency,share_basis\n"
            "17,f1,r1,split,captured,201,2025-08-01T00:00:00Z,2025-07-01T12:00:00Z,2025-08-01,,10,,,\n"
        ),
        try_parse_dates=True,
        schema_overrides=actions.schema,
    )
    result = adjustments.prepare_adjusted_history(
        prices,
        pl.concat([actions, future]),
        as_of=None,
        feed_spec=history_feed(),
        action_coverage="complete",
    )
    assert result.prices["adj_close"][1] == 98
    assert result.prices["adj_volume"][1] == 7
    assert result.actions["event_id"].to_list() == ["s1", "c1"]


def test_native_cash_factor_cannot_use_an_ex_date_close_before_its_actual_availability():
    prices, actions = captured_history()
    with pytest.raises(ValueError, match="ex-date close"):
        adjustments.prepare_adjusted_history(
            prices,
            actions,
            as_of=datetime(2025, 7, 3, 16, tzinfo=UTC),
            feed_spec=history_feed(),
            action_coverage="complete",
        )
    splits = adjustments.prepare_adjusted_history(
        prices,
        actions,
        as_of=datetime(2025, 7, 3, 16, tzinfo=UTC),
        feed_spec=history_feed(),
        action_coverage="complete",
        basis="split_adjusted",
    )
    assert splits.prices["close"].to_list() == [200]
    assert splits.prices["adj_close"].to_list() == [100]
    assert splits.prices["adj_volume"].to_list() == [2000]


def test_a_known_provisional_daily_quote_is_not_the_available_ex_date_close():
    prices, actions = captured_history()
    preliminary = pl.read_csv(
        StringIO(
            "timestamp,asset,close,volume,currency,available_at,source,revision_id\n"
            "2025-07-03,17,99,5,USD,2025-07-03T16:00:00Z,captured,r0\n"
        ),
        try_parse_dates=True,
    )
    source = pl.concat([prices, preliminary])
    with pytest.raises(ValueError, match="ex-date close"):
        adjustments.prepare_adjusted_history(
            source,
            actions,
            as_of=datetime(2025, 7, 3, 16, 30, tzinfo=UTC),
            feed_spec=history_feed(),
            action_coverage="complete",
        )
    final = adjustments.prepare_adjusted_history(
        source,
        actions,
        as_of=datetime(2025, 7, 3, 18, tzinfo=UTC),
        feed_spec=history_feed(),
        action_coverage="complete",
    )
    assert final.prices["close"].to_list() == [200, 98]
    assert final.prices["adj_close"][1] / final.prices["adj_close"][0] == pytest.approx(1.01)


def test_future_source_revisions_cannot_change_prior_prices_or_a_price_sensitive_decision():
    prices, actions = captured_history()
    cutoff = datetime(2025, 7, 2, 21, tzinfo=UTC)
    original = adjustments.prepare_adjusted_history(
        prices,
        actions,
        as_of=cutoff,
        feed_spec=history_feed(),
        action_coverage="complete",
    )
    revised = pl.read_csv(
        StringIO(
            "timestamp,asset,close,volume,currency,available_at,source,revision_id\n"
            "2025-07-02,17,900,1000,USD,2025-07-20T00:00:00Z,captured,r2\n"
        ),
        try_parse_dates=True,
    )
    changed = adjustments.prepare_adjusted_history(
        pl.concat([prices, revised]),
        actions,
        as_of=cutoff,
        feed_spec=history_feed(),
        action_coverage="complete",
    )
    assert_frame_equal(original.prices, changed.prices)
    assert original.prices["adj_close"].to_list() == [200]
    assert original.prices["adj_close"][0] > 150
    assert changed.prices["adj_close"][0] > 150
    retrospective = adjustments.prepare_adjusted_history(
        pl.concat([prices, revised]),
        actions,
        as_of=None,
        feed_spec=history_feed(),
        action_coverage="complete",
    )
    assert retrospective.prices["close"].to_list() == [900, 98]
    assert retrospective.metadata["perspective"] == "retrospective"


def test_raw_history_does_not_require_unneeded_action_availability_or_fabricate_no_action_facts():
    prices, actions = captured_history()
    unavailable = actions.with_columns(pl.lit(None).alias("effective_at"))
    result = adjustments.prepare_adjusted_history(
        prices,
        unavailable,
        as_of=datetime(2025, 7, 3, 18, tzinfo=UTC),
        feed_spec=history_feed(),
        basis="raw",
    )
    assert result.prices["adj_close"].to_list() == [200, 98]
    assert result.actions.is_empty()
    assert result.metadata["actions_used"] is False
    assert result.metadata["action_coverage"] == "unknown"


def test_empty_source_history_preserves_native_output_schema_without_inventing_rows():
    prices, actions = captured_history()
    prepared = adjustments.prepare_adjusted_history(
        prices.head(0),
        actions.head(0),
        as_of=datetime(2025, 7, 3, 18, tzinfo=UTC),
        feed_spec=history_feed(),
        action_coverage="complete",
    )
    assert prepared.prices.is_empty() and prepared.actions.is_empty()
    assert prepared.prices.schema["timestamp"] == pl.Date
    assert prepared.prices.schema["close"] == pl.Int64
    assert prepared.prices.schema["adj_close"] == pl.Float64
    assert prepared.coverage.schema["asset"] == pl.Int64
    assert prepared.coverage.schema["missing_periods"] == pl.UInt64


def test_missing_quote_and_whole_missing_session_remain_missing_after_adjustment():
    prices, actions = captured_history()
    later = pl.read_csv(
        StringIO(
            "timestamp,asset,close,volume,currency,available_at,source,revision_id\n"
            "2025-07-07,17,,,USD,2025-07-07T20:00:00Z,captured,r1\n"
            "2025-07-09,17,105,100,USD,2025-07-09T20:00:00Z,captured,r1\n"
        ),
        try_parse_dates=True,
    )
    source = pl.concat([prices, later])
    prepared = adjustments.prepare_adjusted_history(
        source,
        actions,
        as_of=datetime(2025, 7, 9, 21, tzinfo=UTC),
        feed_spec=history_feed(),
        action_coverage="complete",
    )
    assert prepared.prices["timestamp"].equals(source["timestamp"])
    assert prepared.prices["close"].to_list() == [200, 98, None, 105]
    assert prepared.prices["adj_close"][2] is None
    assert prepared.prices["volume"].to_list() == [1000, 7, None, 100]
    assert prepared.prices["adj_volume"][2] is None
    assert prepared.coverage["null_prices"].to_list() == [1]
    assert prepared.coverage["missing_periods"].to_list() == [1]


def test_explicit_calendar_override_and_none_preserve_crypto_gap_versus_row_mode():
    source = pl.read_csv(
        StringIO(
            "timestamp,asset,close,volume,currency,available_at,source,revision_id\n"
            "2025-07-11,17,100,10,USD,2025-07-11T20:00:00Z,captured,r1\n"
            "2025-07-13,17,101,20,USD,2025-07-13T20:00:00Z,captured,r1\n"
        ),
        try_parse_dates=True,
    )
    scheduled = adjustments.prepare_adjusted_history(
        source,
        None,
        as_of=datetime(2025, 7, 14, tzinfo=UTC),
        feed_spec=history_feed(),
        calendar="24/7",
        basis="raw",
    )
    rows = adjustments.prepare_adjusted_history(
        source,
        None,
        as_of=datetime(2025, 7, 14, tzinfo=UTC),
        feed_spec=history_feed(),
        calendar=None,
        basis="raw",
    )
    assert scheduled.coverage["missing_periods"].to_list() == [1]
    assert rows.coverage["missing_periods"].to_list() == [0]
    assert rows.metadata["feed_spec"]["calendar"] is None
    assert scheduled.metadata["feed_spec"]["calendar"] == "24/7"
    assert_frame_equal(scheduled.prices, rows.prices)


def test_observation_and_availability_cutoffs_do_not_round_adjacent_nanoseconds():
    source = pl.read_csv(
        StringIO(
            "timestamp,asset,close,available_at,source,revision_id\n"
            "2025-07-02T00:00:00.000000000Z,17,100,2025-07-02T00:00:00.000000000Z,captured,r1\n"
            "2025-07-02T00:00:00.000000001Z,17,101,2025-07-02T00:00:00.000000001Z,captured,r1\n"
        )
    ).with_columns(
        pl.col("timestamp", "available_at").str.to_datetime(time_unit="ns", time_zone="UTC")
    )
    feed = FeedSpec(time_contract_version=2, entity_col="asset", calendar=None, timezone="UTC")
    first = adjustments.prepare_adjusted_history(
        source,
        None,
        as_of=pd.Timestamp("2025-07-02T00:00:00.000000000Z"),
        feed_spec=feed,
        basis="raw",
    )
    second = adjustments.prepare_adjusted_history(
        source,
        None,
        as_of=pd.Timestamp("2025-07-02T00:00:00.000000001Z"),
        feed_spec=feed,
        basis="raw",
    )
    assert first.prices["close"].to_list() == [100]
    assert second.prices["close"].to_list() == [100, 101]
    assert second.prices.schema["timestamp"] == pl.Datetime("ns", "UTC")


@pytest.mark.parametrize("representation", ["date", "declared_local", "undeclared_local"])
def test_current_cme_session_label_is_not_discarded_using_the_previous_civil_date(
    tmp_path, representation
):
    source = pl.read_csv(
        StringIO(
            "timestamp,asset,close,available_at,source,revision_id\n"
            "2025-07-07,17,100,2025-07-06T23:30:00Z,captured,r1\n"
        ),
        try_parse_dates=True,
    )
    feed = FeedSpec(
        time_contract_version=2,
        entity_col="asset",
        calendar="CME_Equity",
        timestamp_semantics="session_label",
        timezone="America/New_York",
        data_frequency="daily",
    )
    if representation != "date":
        source = source.with_columns(
            pl.col("timestamp").cast(pl.Datetime("ns")).dt.replace_time_zone("America/New_York")
        )
    if representation == "undeclared_local":
        feed = feed.with_overrides(timezone=None)
    storage = FlatStorage(StorageConfig(base_path=tmp_path))
    storage.write(source, "captured/cme", feed.to_dict())
    restored = storage.read("captured/cme").collect()
    result = adjustments.prepare_adjusted_history(
        restored,
        None,
        as_of=datetime(2025, 7, 6, 23, 45, tzinfo=UTC),
        feed_spec=feed,
        basis="raw",
    )
    assert result.prices["close"].to_list() == [100]
    assert result.prices.schema["timestamp"] == restored.schema["timestamp"]
