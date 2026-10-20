"""Captured corporate-action tables must retain facts and explicit adjustment definitions."""

from datetime import UTC, date, datetime
from io import StringIO
from pathlib import Path
from runpy import run_path

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from ml4t.data.adjustments import (
    apply_adjustment_factors,
    apply_corporate_actions,
    prepare_action_inputs,
)
from ml4t.specs import FeedSpec


def captured_actions():
    return pl.read_csv(
        StringIO(
            "asset,event_id,revision_id,action_kind,source,source_record_id,effective_at,available_at,ex_date,payment_date,split_ratio,cash_per_share,currency,share_basis\n"
            "17,s1,r1,split,captured,101,2025-07-03T00:00:00Z,2025-07-01T12:00:00Z,2025-07-03,,2,,,\n"
            "17,c1,r1,cash_dividend,captured,102,2025-07-03T00:00:00Z,2025-07-01T12:00:00Z,2025-07-03,2025-07-15,,3,USD,post_event\n"
        ),
        schema_overrides={"split_ratio": pl.Float64, "cash_per_share": pl.Float64},
        try_parse_dates=True,
    )


def captured_prices():
    return pl.read_csv(
        StringIO(
            "timestamp,asset,close,volume,action_state\n"
            "2025-07-02,17,200,1000,no_action\n2025-07-03,17,98,7,actions\n"
        ),
        try_parse_dates=True,
    )


def test_captured_event_terms_feed_the_existing_native_split_dividend_definition():
    source = captured_actions()
    facts = prepare_action_inputs(source, pl.DataFrame({"asset": [17]}))
    assert facts["event_id"].to_list() == ["s1", "c1"]
    assert facts["available_at"].to_list() == [datetime(2025, 7, 1, 12, tzinfo=UTC)] * 2
    assert facts["payment_date"].to_list() == [None, date(2025, 7, 15)]
    actions = facts.group_by("ex_date").agg(
        pl.col("split_ratio").fill_null(1).product(),
        pl.col("cash_per_share").fill_null(0).sum().alias("ex-dividend"),
    )
    prices = (
        captured_prices()
        .rename({"timestamp": "date"})
        .join(actions, left_on="date", right_on="ex_date", how="left")
        .with_columns(
            pl.when(pl.col("action_state") == "no_action")
            .then(1)
            .otherwise(pl.col("split_ratio"))
            .alias("split_ratio"),
            pl.when(pl.col("action_state") == "no_action")
            .then(0)
            .otherwise(pl.col("ex-dividend"))
            .alias("ex-dividend"),
        )
    )
    adjusted = apply_corporate_actions(prices)
    assert adjusted["close"].to_list() == [200, 98]
    assert adjusted["volume"].to_list() == [1000, 7]
    assert adjusted["price_adjustment_factor"].to_list() == pytest.approx([98 / 202, 1])
    assert adjusted["adj_volume"].to_list() == [2000, 7]
    assert adjusted["adj_close"][1] / adjusted["adj_close"][0] == pytest.approx(1.01)
    assert_frame_equal(source, captured_actions())


def test_legacy_price_factor_direction_is_explicit_and_raw_source_terms_survive():
    source = (
        captured_actions()
        .filter(pl.col("action_kind") == "split")
        .with_columns(pl.lit(0.5).alias("split_ratio"))
    )
    facts = prepare_action_inputs(source, split_convention="price_factor")
    assert facts["split_ratio"].to_list() == [2]
    assert facts["raw_split_ratio"].to_list() == [0.5]
    assert facts["source_split_convention"].to_list() == ["price_factor"]
    assert_frame_equal(prepare_action_inputs(facts), facts)
    with pytest.raises(ValueError, match="convention"):
        prepare_action_inputs(facts, split_convention="price_factor")


def test_supplied_prior_close_factors_are_not_recomputed_using_ex_date_close():
    prices = captured_prices()
    factors = prices.select("timestamp", "asset").with_columns(
        pl.Series("price_adjustment_factor", [0.485, 1.0]),
        pl.Series("volume_adjustment_factor", [2.0, 1.0]),
    )
    result = apply_adjustment_factors(prices, factors, method="vendor_prior_close")
    assert result["adj_close"].to_list() == [97, 98]
    assert result["adj_volume"].to_list() == [2000, 7]
    assert result["adjustment_method"].to_list() == ["vendor_prior_close"] * 2
    assert_frame_equal(result.select(prices.columns), prices)


def test_unknown_availability_and_merger_terms_remain_unknown_not_invented():
    source = (
        captured_actions()
        .head(1)
        .with_columns(pl.lit("merger").alias("action_kind"), pl.lit(None).alias("available_at"))
    )
    result = prepare_action_inputs(source)
    assert result["available_at"].null_count() == 1
    assert {"available_at", "destination_asset", "quantity_ratio", "cash_component"} <= set(
        result["missing_terms"][0]
    )
    with pytest.raises(ValueError, match="available_at"):
        prepare_action_inputs(source, required_terms=["available_at"])


@pytest.mark.parametrize("ratio", [0.0, -1.0, float("inf"), float("nan")])
def test_invalid_known_split_ratio_fails_without_mutating_the_source(ratio):
    source = captured_actions().with_columns(
        pl.when(pl.col("action_kind") == "split")
        .then(ratio)
        .otherwise(pl.col("split_ratio"))
        .alias("split_ratio")
    )
    before = source.clone()
    with pytest.raises(ValueError, match="ratio"):
        prepare_action_inputs(source)
    assert_frame_equal(source, before)


@pytest.mark.parametrize(
    "source_terms,convention",
    [
        ("split_ratio\n5e-324", "price_factor"),
        ("split_ratio,split_numerator,split_denominator\n,1e308,1e-308", "new_per_old"),
        ("split_ratio,split_numerator,split_denominator\n,1e-308,1e308", "new_per_old"),
    ],
)
def test_normalizing_finite_csv_terms_cannot_publish_infinite_or_zero_share_ratios(
    source_terms, convention
):
    header, values = source_terms.split("\n")
    source = pl.read_csv(
        StringIO(
            f"asset,source,event_id,revision_id,action_kind,{header}\n"
            f"17,captured,s1,r1,split,{values}\n"
        ),
        schema_overrides={"split_ratio": pl.Float64},
    )
    before = source.clone()
    with pytest.raises(ValueError, match="positive.*finite"):
        prepare_action_inputs(source, split_convention=convention, required_terms=["split_ratio"])
    assert_frame_equal(source, before)


def test_duplicate_revisions_and_unknown_stable_ids_are_not_silently_selected():
    source = captured_actions()
    with pytest.raises(ValueError, match="Duplicate"):
        prepare_action_inputs(pl.concat([source, source.head(1)]))
    with pytest.raises(ValueError, match="identity"):
        prepare_action_inputs(source, pl.DataFrame({"asset": [23]}))


def test_declared_event_columns_and_calendar_presence_are_preserved():
    source = captured_actions().rename({"asset": "sid", "effective_at": "event_clock"})
    feed = FeedSpec(
        time_contract_version=2, calendar=None, entity_col="sid", timestamp_col="event_clock"
    )
    facts = prepare_action_inputs(source, feed_spec=feed)
    assert facts["sid"].to_list() == [17, 17]
    assert facts["event_clock"].equals(source["event_clock"])
    assert facts["preparation_calendar"].to_list() == [None, None]
    assert "preparation_calendar" not in prepare_action_inputs(captured_actions()).columns


@pytest.mark.parametrize("factor", [0.0, -1.0, None, float("inf")])
def test_unusable_supplied_factors_cannot_partially_adjust_raw_prices(factor):
    prices = captured_prices()
    factors = prices.select("timestamp", "asset").with_columns(
        pl.Series("price_adjustment_factor", [factor, 1.0], dtype=pl.Float64)
    )
    with pytest.raises(ValueError, match="factor"):
        apply_adjustment_factors(prices, factors)
    assert_frame_equal(prices, captured_prices())


def test_factor_key_coverage_and_known_currency_units_must_match_prices():
    prices = captured_prices()
    factors = prices.select("timestamp", "asset").with_columns(
        pl.lit(1.0).alias("price_adjustment_factor")
    )
    with pytest.raises(ValueError, match="coverage"):
        apply_adjustment_factors(prices, factors.head(1))
    with pytest.raises(ValueError, match="Duplicate"):
        apply_adjustment_factors(prices, pl.concat([factors, factors.head(1)]))
    with pytest.raises(ValueError, match="currency"):
        apply_adjustment_factors(
            prices.with_columns(pl.lit("USD").alias("currency")),
            factors.with_columns(pl.lit("EUR").alias("currency")),
        )


def test_source_rational_share_terms_can_supply_an_unknown_ratio_without_losing_raw_facts():
    source = (
        captured_actions()
        .head(1)
        .with_columns(
            pl.lit(None, dtype=pl.Float64).alias("split_ratio"),
            pl.lit(3).alias("split_numerator"),
            pl.lit(2).alias("split_denominator"),
        )
    )
    result = prepare_action_inputs(source)
    assert result["split_ratio"].to_list() == [1.5]
    assert result["raw_split_ratio"].to_list() == [None]
    assert result["split_numerator"].to_list() == [3]
    assert result["split_denominator"].to_list() == [2]
    with pytest.raises(ValueError, match="fraction"):
        prepare_action_inputs(source.with_columns(pl.lit(0).alias("split_denominator")))
    with pytest.raises(ValueError, match="conflicts"):
        prepare_action_inputs(source.with_columns(pl.lit(7).alias("split_ratio")))


def test_stock_distribution_and_cash_only_merger_are_facts_not_invented_positions():
    source = (
        captured_actions()
        .head(1)
        .with_columns(
            pl.lit("stock_distribution").alias("action_kind"),
            pl.lit(23).alias("destination_asset"),
            pl.lit(0.1).alias("quantity_ratio"),
        )
    )
    stock = prepare_action_inputs(source)
    assert stock["destination_asset"].to_list() == [23]
    assert stock["quantity_ratio"].to_list() == [0.1]
    assert stock["available_at"].equals(source["available_at"])
    cash = prepare_action_inputs(
        source.with_columns(
            pl.lit("merger").alias("action_kind"),
            pl.lit(None).alias("destination_asset"),
            pl.lit(0.0).alias("quantity_ratio"),
            pl.lit(50.0).alias("cash_component"),
            pl.lit("USD").alias("currency"),
        )
    )
    assert "destination_asset" not in cash["missing_terms"][0]
    assert cash["cash_component"].to_list() == [50]
    assert "cash" not in cash.columns and "positions" not in cash.columns


def test_explicit_calendar_and_mapping_conflicts_follow_the_shared_declaration_contract():
    feed = FeedSpec(
        time_contract_version=2, calendar="NYSE", entity_col="asset", timestamp_col="effective_at"
    )
    result = prepare_action_inputs(captured_actions(), feed_spec=feed, calendar=None)
    assert result["preparation_calendar"].to_list() == [None, None]
    with pytest.raises(ValueError, match="Conflicting"):
        prepare_action_inputs(
            captured_actions(), feed_spec={"entity_col": "asset", "symbol_col": "symbol"}
        )


def test_custom_price_fields_and_unknown_volume_factors_do_not_invent_volume_adjustments():
    prices = captured_prices().rename({"timestamp": "stamp", "asset": "sid", "close": "raw"})
    factors = prices.select("stamp", "sid").with_columns(
        pl.Series("price_adjustment_factor", [0.5, 1.0])
    )
    result = apply_adjustment_factors(
        prices,
        factors,
        feed_spec=FeedSpec(timestamp_col="stamp", entity_col="sid", price_col="raw"),
    )
    assert result["adj_raw"].to_list() == [100, 98]
    assert "adj_volume" not in result.columns
    assert result["volume"].to_list() == [1000, 7]


def test_produced_adjustment_method_cannot_be_silently_overwritten_as_factor_provenance():
    prices = captured_prices()
    factors = prices.select("timestamp", "asset").with_columns(
        pl.lit(1.0).alias("price_adjustment_factor")
    )
    produced = apply_adjustment_factors(prices, factors, method="captured_vendor")
    imported = produced.select("timestamp", "asset", "price_adjustment_factor", "adjustment_method")
    before = imported.clone()
    with pytest.raises(ValueError, match="provenance"):
        apply_adjustment_factors(prices, imported)
    assert_frame_equal(imported, before)


def test_declared_instrument_membership_includes_a_stock_distribution_destination():
    source = (
        captured_actions()
        .head(1)
        .with_columns(
            pl.lit("stock_distribution").alias("action_kind"),
            pl.lit(23).alias("destination_asset"),
            pl.lit(0.1).alias("quantity_ratio"),
        )
    )
    with pytest.raises(ValueError, match="Destination identity"):
        prepare_action_inputs(source, pl.DataFrame({"asset": [17]}))
    result = prepare_action_inputs(source, pl.DataFrame({"asset": [17, 23]}))
    assert result["destination_asset"].to_list() == [23]


def test_whitespace_source_terms_are_retained_as_unknown_coverage_not_usable_currency():
    source = captured_actions().with_columns(pl.lit(" ").alias("currency"))
    result = prepare_action_inputs(source)
    assert result["currency"].to_list() == [" ", " "]
    assert "currency" in result["missing_terms"][1]
    with pytest.raises(ValueError, match="currency"):
        prepare_action_inputs(source, required_terms=["currency"])


@pytest.mark.parametrize("identifier", [float("nan"), float("inf")])
def test_nonfinite_source_ids_cannot_become_stable_action_identities(identifier):
    source = captured_actions().with_columns(pl.lit(identifier).alias("asset"))
    with pytest.raises(ValueError, match="finite"):
        prepare_action_inputs(source)


def test_offline_csv_to_disk_action_recipe_retains_source_facts_and_method_difference(tmp_path):
    recipe = run_path(str(Path(__file__).parents[1] / "examples/action_input_migration.py"))
    facts, native, supplied = recipe["migrate_actions"](tmp_path)
    assert facts["available_at"].to_list() == [datetime(2025, 7, 1, 12, tzinfo=UTC)] * 2
    assert native["close"].to_list() == supplied["close"].to_list() == [200, 98]
    assert native["price_adjustment_factor"][0] == pytest.approx(98 / 202)
    assert supplied["price_adjustment_factor"][0] == 0.485
