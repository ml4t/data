"""Imported source factor definitions must remain explicit and actually known at the cutoff."""

from datetime import UTC, datetime
from io import StringIO

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from ml4t.data.adjustments import prepare_adjusted_history
from ml4t.specs import FeedSpec


def captured_factor_inputs():
    prices = pl.read_csv(
        StringIO(
            "timestamp,asset,close,volume,currency,available_at,source,revision_id\n"
            "2025-07-02,17,200,1000,USD,2025-07-02T20:00:00Z,captured,r1\n"
            "2025-07-03,17,99,5,USD,2025-07-03T16:00:00Z,captured,r0\n"
        ),
        try_parse_dates=True,
    )
    actions = pl.read_csv(
        StringIO(
            "asset,event_id,revision_id,action_kind,source,effective_at,available_at,ex_date,split_ratio,cash_per_share,currency,share_basis\n"
            "17,s1,r1,split,captured,2025-07-03T00:00:00Z,2025-07-01T12:00:00Z,2025-07-03,2,,,\n"
            "17,c1,r1,cash_dividend,captured,2025-07-03T00:00:00Z,2025-07-01T12:00:00Z,2025-07-03,,3,USD,post_event\n"
        ),
        try_parse_dates=True,
        schema_overrides={"split_ratio": pl.Float64, "cash_per_share": pl.Float64},
    )
    factors = pl.read_csv(
        StringIO(
            "timestamp,asset,price_adjustment_factor,volume_adjustment_factor,basis_at,available_at,source,revision_id,currency\n"
            "2025-07-02,17,0.485,2,2025-07-03T13:30:00Z,2025-07-03T13:30:00Z,vendor,r1,USD\n"
            "2025-07-03,17,1,1,2025-07-03T13:30:00Z,2025-07-03T13:30:00Z,vendor,r1,USD\n"
        ),
        try_parse_dates=True,
    )
    feed = FeedSpec(
        time_contract_version=2,
        entity_col="asset",
        calendar="NYSE",
        data_frequency="daily",
        timezone="America/New_York",
        timestamp_semantics="session_label",
    )
    return prices, actions, factors, feed


def test_known_prior_close_factors_work_before_ex_date_close_without_changing_their_definition():
    prices, actions, factors, feed = captured_factor_inputs()
    result = prepare_adjusted_history(
        prices,
        actions,
        factors=factors,
        factor_method="vendor_prior_close",
        basis="total_return",
        as_of=datetime(2025, 7, 3, 16, 30, tzinfo=UTC),
        feed_spec=feed,
        action_coverage="complete",
    )
    assert result.prices["close"].to_list() == [200, 99]
    assert result.prices["adj_close"].to_list() == [97, 99]
    assert result.prices["adj_volume"].to_list() == [2000, 5]
    assert result.prices["factor_source"].to_list() == ["vendor", "vendor"]
    assert result.prices["adjustment_method"].to_list() == ["vendor_prior_close"] * 2
    assert result.prices["adj_close"][1] / result.prices["adj_close"][0] - 1 == pytest.approx(
        2 / 97
    )


def test_a_supplied_factor_history_requires_a_declared_method_and_share_price_basis():
    prices, actions, factors, feed = captured_factor_inputs()
    with pytest.raises(ValueError, match="basis"):
        prepare_adjusted_history(
            prices,
            actions,
            factors=factors,
            factor_method="vendor_prior_close",
            as_of=None,
            feed_spec=feed,
            action_coverage="complete",
        )
    with pytest.raises(ValueError, match="method"):
        prepare_adjusted_history(
            prices,
            actions,
            factors=factors,
            basis="total_return",
            as_of=None,
            feed_spec=feed,
            action_coverage="complete",
        )


def test_a_factor_published_later_cannot_fill_an_earlier_causal_history():
    prices, actions, factors, feed = captured_factor_inputs()
    later = factors.with_columns(pl.lit(datetime(2025, 7, 4, tzinfo=UTC)).alias("available_at"))
    with pytest.raises(ValueError, match="coverage"):
        prepare_adjusted_history(
            prices,
            actions,
            factors=later,
            factor_method="vendor_prior_close",
            basis="total_return",
            as_of=datetime(2025, 7, 3, 16, 30, tzinfo=UTC),
            feed_spec=feed,
            action_coverage="complete",
        )


def test_imported_merger_price_factors_do_not_invent_missing_account_consideration():
    prices, actions, factors, feed = captured_factor_inputs()
    merger = actions.head(1).with_columns(pl.lit("merger").alias("action_kind"))
    result = prepare_adjusted_history(
        prices,
        merger,
        factors=factors,
        factor_method="captured_merger",
        basis="total_return",
        as_of=datetime(2025, 7, 3, 16, 30, tzinfo=UTC),
        feed_spec=feed,
        action_coverage="complete",
    )
    assert result.prices["adj_close"].to_list() == [97, 99]
    assert result.actions["quantity_ratio"].null_count() == 1
    assert "cash_component" in result.actions["missing_terms"][0]
    assert "positions" not in result.prices.columns and "cash" not in result.prices.columns


def test_unknown_imported_availability_is_retained_for_research_but_cannot_claim_causality():
    prices, actions, factors, feed = captured_factor_inputs()
    unknown = factors.drop("source", "revision_id", "available_at", "basis_at")
    result = prepare_adjusted_history(
        prices,
        actions,
        factors=unknown,
        factor_method="legacy_capture",
        basis="total_return",
        as_of=None,
        feed_spec=feed,
    )
    assert result.prices["adj_close"].to_list() == [97, 99]
    assert result.metadata["factor_revision_identity"] == "unknown unversioned capture"
    assert result.metadata["factor_basis_known"] is False
    assert result.metadata["factor_availability_known"] is False
    with pytest.raises(ValueError, match="revision identity"):
        prepare_adjusted_history(
            prices,
            actions,
            factors=unknown,
            factor_method="legacy_capture",
            basis="total_return",
            as_of=datetime(2025, 7, 3, 16, 30, tzinfo=UTC),
            feed_spec=feed,
            action_coverage="complete",
        )


def test_a_source_factor_method_cannot_be_silently_relabelled_with_another_definition():
    prices, actions, factors, feed = captured_factor_inputs()
    declared = factors.with_columns(pl.lit("vendor_prior_close").alias("adjustment_method"))
    with pytest.raises(ValueError, match="method"):
        prepare_adjusted_history(
            prices,
            actions,
            factors=declared,
            factor_method="native_ex_date_close",
            basis="total_return",
            as_of=None,
            feed_spec=feed,
        )


@pytest.mark.parametrize(
    "defect,match",
    [
        ("future_basis", "later than"),
        ("stale_basis", "predates"),
        ("mixed_basis", "one supplied share basis"),
        ("currency", "currency"),
        ("missing", "coverage"),
        ("duplicate", "Duplicate"),
        ("negative", "positive"),
        ("infinite", "finite"),
    ],
)
def test_invalid_causal_factor_definitions_fail_without_changing_source_inputs(defect, match):
    prices, actions, original, feed = captured_factor_inputs()
    factors = original.clone()
    if defect == "future_basis":
        factors = factors.with_columns(
            pl.lit(datetime(2025, 7, 3, 17, tzinfo=UTC)).alias("basis_at")
        )
    elif defect == "stale_basis":
        factors = factors.with_columns(pl.lit(datetime(2025, 7, 1, tzinfo=UTC)).alias("basis_at"))
    elif defect == "mixed_basis":
        factors = factors.with_columns(
            pl.when(pl.col("price_adjustment_factor") == 1)
            .then(datetime(2025, 7, 3, 14, tzinfo=UTC))
            .otherwise(pl.col("basis_at"))
            .alias("basis_at")
        )
    elif defect == "currency":
        factors = factors.with_columns(pl.lit("EUR").alias("currency"))
    elif defect == "missing":
        factors = factors.head(1)
    elif defect == "duplicate":
        factors = pl.concat([factors, factors.head(1)])
    else:
        factors = factors.with_columns(
            pl.lit(-1.0 if defect == "negative" else float("inf")).alias("price_adjustment_factor")
        )
    before = factors.clone()
    with pytest.raises(ValueError, match=match):
        prepare_adjusted_history(
            prices,
            actions,
            factors=factors,
            basis="total_return",
            factor_method="captured",
            as_of=datetime(2025, 7, 3, 16, 30, tzinfo=UTC),
            feed_spec=feed,
            action_coverage="complete",
        )
    assert_frame_equal(factors, before)
    assert_frame_equal(original, captured_factor_inputs()[2])


def test_future_factor_revisions_cannot_change_prior_causal_history_or_raw_source_provenance():
    prices, actions, factors, feed = captured_factor_inputs()
    later = factors.with_columns(
        pl.lit(9.0).alias("price_adjustment_factor"),
        pl.lit("r2").alias("revision_id"),
        pl.lit(datetime(2025, 7, 20, tzinfo=UTC)).alias("available_at"),
    )
    results = [
        prepare_adjusted_history(
            prices,
            actions,
            factors=table,
            basis="total_return",
            factor_method="captured",
            as_of=datetime(2025, 7, 3, 16, 30, tzinfo=UTC),
            feed_spec=feed,
            action_coverage="complete",
        )
        for table in (factors, pl.concat([factors, later]))
    ]
    assert_frame_equal(results[0].prices, results[1].prices)
    assert results[1].prices["revision_id"].to_list() == ["r1", "r0"]
    assert results[1].prices["factor_revision_id"].to_list() == ["r1", "r1"]


def test_supplied_price_only_factors_do_not_fabricate_adjusted_volume_and_keep_typed_empties():
    prices, actions, factors, feed = captured_factor_inputs()
    for count in (2, 0):
        result = prepare_adjusted_history(
            prices.head(count),
            actions,
            factors=factors.head(count).drop("volume_adjustment_factor"),
            basis="total_return",
            factor_method="captured",
            action_coverage="complete",
            as_of=datetime(2025, 7, 3, 16, 30, tzinfo=UTC),
            feed_spec=feed,
        )
        assert "adj_volume" not in result.prices.columns
        assert result.prices["volume"].to_list() == [1000, 5][:count]
        assert result.metadata["volume_factor_supplied"] is False
        assert result.prices.schema["adj_close"] == pl.Float64
