"""Imported source factor definitions must remain explicit and actually known at the cutoff."""

from datetime import UTC, datetime
from io import StringIO

import polars as pl
import pytest

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
