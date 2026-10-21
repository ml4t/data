"""Real optional bundle migration through standard readers and ingestion writers."""

import hashlib
from uuid import uuid4

import numpy as np
import pandas as pd
import polars as pl
import pytest
from polars.testing import assert_frame_equal

pytest.importorskip(
    "zipline", reason="Legacy bundle exchange is an optional conversion environment"
)

from zipline.data import bundles

from ml4t.data.adjustments import prepare_adjusted_history
from ml4t.data.compat.zipline import export_bundle, make_bundle_ingest
from ml4t.specs import FeedSpec
from tests._zipline_bundle_fixture import produce_bundle


def _hashes(root):
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in root.rglob("*")
        if path.is_file()
    }


@pytest.fixture
def source_bundle(tmp_path):
    name = "source-" + uuid4().hex
    bundle, original, _ = produce_bundle(tmp_path / "source", name)
    try:
        yield bundle, original
    finally:
        bundle.close()
        bundles.unregister(name)


@pytest.mark.parametrize("target_scale", [1000, 10000])
def test_minute_precision_is_rejected_before_writes_or_preserved_by_configured_writer(
    tmp_path, target_scale
):
    from functools import partial
    from unittest.mock import patch

    from zipline.data.bcolz_minute_bars import BcolzMinuteBarWriter

    source_name, target_name = "precise-source-" + uuid4().hex, "precise-target-" + uuid4().hex
    bundle, original, _ = produce_bundle(tmp_path / "source", source_name, minute_scale=10000)
    writes = []
    try:
        minutes = original["minute_dates"]
        export_bundle(
            bundle,
            tmp_path / "snapshot",
            namespace="precise-source",
            source_revision="precision-v1",
            sids=[1],
            start=minutes[0],
            end=minutes[-1],
            frequencies=("minute",),
        )
        callback = make_bundle_ingest(tmp_path / "snapshot")

        def ingest(*args):
            args = list(args)
            asset_writer = args[1]
            original_write = asset_writer.write_direct

            def record_write(**kwargs):
                writes.append("assets")
                return original_write(**kwargs)

            asset_writer.write_direct = record_write
            callback(*args)

        bundles.register(
            target_name,
            ingest,
            calendar_name="XNYS",
            start_session=pd.Timestamp("2024-01-02"),
            end_session=pd.Timestamp("2024-01-04"),
        )
        environment = {"ZIPLINE_ROOT": str(tmp_path / "target")}
        with patch(
            "zipline.data.bundles.core.BcolzMinuteBarWriter",
            partial(BcolzMinuteBarWriter, ohlc_ratios_per_sid={1: target_scale}),
        ):
            if target_scale == 1000:
                with pytest.raises(ValueError, match="price precision"):
                    bundles.ingest(
                        target_name,
                        environ=environment,
                        timestamp=pd.Timestamp("2024-02-03", tz="UTC"),
                    )
                assert writes == []
            else:
                bundles.ingest(
                    target_name, environ=environment, timestamp=pd.Timestamp("2024-02-03", tz="UTC")
                )
        if target_scale != 1000:
            with bundles.load(
                target_name, environ=environment, timestamp=pd.Timestamp("2024-02-04", tz="UTC")
            ) as restored:
                for before, after in zip(
                    bundle.equity_minute_bar_reader.load_raw_arrays(
                        ["open", "high", "low", "close", "volume"], minutes[0], minutes[-1], [1]
                    ),
                    restored.equity_minute_bar_reader.load_raw_arrays(
                        ["open", "high", "low", "close", "volume"], minutes[0], minutes[-1], [1]
                    ),
                    strict=True,
                ):
                    np.testing.assert_array_equal(after, before)
    finally:
        bundle.close()
        bundles.unregister(source_name)
        if target_name in bundles.bundles:
            bundles.unregister(target_name)


def test_real_bundle_export_preserves_source_and_reingests_daily_minute_actions_and_contracts(
    tmp_path,
):
    source_name, target_name = "source-" + uuid4().hex, "target-" + uuid4().hex
    source_root = tmp_path / "source"
    bundle, original, _ = produce_bundle(source_root, source_name)
    try:
        before = _hashes(source_root)
        minutes = original["minute_dates"]
        options = {
            "namespace": "permanent-source",
            "source_revision": "original-ingestion-20240201",
            "sids": [1, 2, 101, 102],
            "start": "2024-01-02",
            "end": "2024-01-04",
            "frequencies": ("daily", "minute"),
            "frequency_bounds": {"minute": (minutes[0], minutes[-1])},
        }
        snapshot = export_bundle(bundle, tmp_path / "snapshot", **options)
        assert _hashes(source_root) == before
        assert snapshot.tables["zipline_equities"]["exchange"].to_list() == ["NYSE", "NASDAQ"]
        assert snapshot.tables["zipline_futures"]["exchange"].to_list() == ["CME", "CME"]
        entry = snapshot.manifest["content"]["tables"]["minute_bars"]
        assert entry["clock"]["feed_spec"]["timestamp_semantics"] == "bar_close"
        assert entry["clock"]["feed_spec"]["bar_label"] == "right"
        assert entry["metadata"]["source_schedule_provider"] == "exchange-calendars"
        assert entry["metadata"]["source_schedule_provider_version"]
        daily = snapshot.tables["daily_bars"]
        factors = snapshot.tables["daily_factors"]
        first = factors.filter(factors["source_sid"] == 1)
        np.testing.assert_allclose(
            first["price_adjustment_factor"].to_numpy(),
            [0.5 * (1 - 1 / 51), 1 - 1 / 51, 1],
            rtol=0,
            atol=2e-16,
        )
        assert first["volume_adjustment_factor"].to_list() == [2, 1, 1]
        assert daily["available_at"].null_count() == daily.height
        action = snapshot.tables["actions"]
        assert action.filter(action["action_kind"] == "split")["split_ratio"].to_list() == [2]
        assert action.filter(action["action_kind"] == "stock_dividend")[
            "destination_asset"
        ].to_list() == ["permanent-source:2"]
        assert action["available_at"].null_count() == action.height
        feed = FeedSpec.from_mapping(
            snapshot.manifest["content"]["tables"]["daily_bars"]["clock"]["feed_spec"]
        )
        prepared = prepare_adjusted_history(
            daily,
            None,
            as_of=None,
            factors=factors,
            factor_method="zipline history reader capture",
            basis="total_return",
            feed_spec=feed,
        )
        equity = prepared.prices.filter(prepared.prices["source_sid"] == 1)
        np.testing.assert_allclose(
            equity["adj_close"].to_numpy(), [100 * 0.5 * (1 - 1 / 51), 50, 52], rtol=0, atol=1e-12
        )
        callback = make_bundle_ingest(tmp_path / "snapshot")
        bundles.register(
            target_name,
            callback,
            calendar_name="XNYS",
            start_session=pd.Timestamp("2024-01-02"),
            end_session=pd.Timestamp("2024-01-04"),
        )
        target_environ = {"ZIPLINE_ROOT": str(tmp_path / "target")}
        bundles.ingest(
            target_name, environ=target_environ, timestamp=pd.Timestamp("2024-02-03", tz="UTC")
        )
        with bundles.load(
            target_name, environ=target_environ, timestamp=pd.Timestamp("2024-02-04", tz="UTC")
        ) as rebuilt:
            for name, lower, upper in (
                ("equity_daily_bar_reader", original["sessions"][0], original["sessions"][-1]),
                ("equity_minute_bar_reader", minutes[0], minutes[-1]),
            ):
                before_arrays = getattr(bundle, name).load_raw_arrays(
                    ["open", "high", "low", "close", "volume"], lower, upper, [1, 2, 101, 102]
                )
                after_arrays = getattr(rebuilt, name).load_raw_arrays(
                    ["open", "high", "low", "close", "volume"], lower, upper, [1, 2, 101, 102]
                )
                for before_array, after_array in zip(before_arrays, after_arrays, strict=True):
                    np.testing.assert_array_equal(after_array, before_array)
            assert rebuilt.asset_finder.lookup_symbol("OLD", original["sessions"][0]).sid == 1
            assert rebuilt.asset_finder.lookup_symbol("NEW", original["sessions"][1]).sid == 1
            for sid in (1, 2, 101, 102):
                before_asset = bundle.asset_finder.retrieve_asset(sid)
                after_asset = rebuilt.asset_finder.retrieve_asset(sid)
                for name in (
                    "symbol",
                    "start_date",
                    "end_date",
                    "exchange",
                    "exchange_full",
                    "price_multiplier",
                    "tick_size",
                ):
                    assert getattr(after_asset, name) == getattr(before_asset, name)
                if sid >= 100:
                    for name in ("notice_date", "expiration_date", "auto_close_date"):
                        assert getattr(after_asset, name) == getattr(before_asset, name)
            for name, before_frame in bundle.adjustment_reader.unpack_db_to_component_dfs().items():
                pd.testing.assert_frame_equal(
                    rebuilt.adjustment_reader.get_df_from_table(name), before_frame, obj=name
                )
            second = export_bundle(rebuilt, tmp_path / "second", **options)
            for name in ("daily_bars", "minute_bars", "daily_factors", "minute_factors", "actions"):
                assert_frame_equal(second.tables[name], snapshot.tables[name])
        assert _hashes(source_root) == before
    finally:
        bundle.close()
        bundles.unregister(source_name)
        if target_name in bundles.bundles:
            bundles.unregister(target_name)


def test_empty_asset_selection_exports_typed_empty_daily_and_minute_inputs_and_reingests(
    tmp_path, source_bundle
):
    bundle, original = source_bundle
    minutes = original["minute_dates"]
    snapshot = export_bundle(
        bundle,
        tmp_path / "empty",
        namespace="empty-selection",
        source_revision="captured-v1",
        sids=[],
        start="2024-01-02",
        end="2024-01-04",
        frequencies=("daily", "minute"),
        frequency_bounds={"minute": (minutes[0], minutes[-1])},
    )
    assert snapshot.tables["daily_bars"].height == 0
    assert snapshot.tables["minute_bars"].height == 0
    assert snapshot.tables["daily_factors"].height == 0
    assert snapshot.tables["minute_factors"].height == 0
    assert snapshot.tables["daily_bars"].schema["timestamp"] == pl.Date
    name = "empty-target-" + uuid4().hex
    bundles.register(
        name,
        make_bundle_ingest(tmp_path / "empty"),
        calendar_name="XNYS",
        start_session=pd.Timestamp("2024-01-02"),
        end_session=pd.Timestamp("2024-01-04"),
    )
    environment = {"ZIPLINE_ROOT": str(tmp_path / "target")}
    try:
        bundles.ingest(name, environ=environment, timestamp=pd.Timestamp("2024-02-03", tz="UTC"))
        with bundles.load(
            name, environ=environment, timestamp=pd.Timestamp("2024-02-04", tz="UTC")
        ) as restored:
            assert restored.asset_finder.sids == ()
            arrays = restored.equity_minute_bar_reader.load_raw_arrays(
                ["close"], minutes[0], minutes[-1], []
            )
            assert arrays[0].shape == (3, 0)
    finally:
        bundles.unregister(name)


def test_bundle_calendar_none_disables_native_schedule_without_relabeling_source_clock(
    tmp_path, source_bundle
):
    bundle, _ = source_bundle
    snapshot = export_bundle(
        bundle,
        tmp_path / "row-mode",
        namespace="explicit-source",
        source_revision="captured-v1",
        sids=[1],
        start="2024-01-02",
        end="2024-01-04",
        calendar=None,
    )
    entry = snapshot.manifest["content"]["tables"]["daily_bars"]
    assert entry["clock"]["resolved_calendar"] is None
    assert entry["clock"]["schedule_provider"] is None
    assert entry["metadata"]["source_reader_calendar"] == "XNYS"
    assert snapshot.tables["daily_bars"]["close"].to_list() == [100, 51, 52]


@pytest.mark.parametrize(
    "overrides",
    [
        {"sids": [1, 1]},
        {"sids": [True]},
        {"namespace": ""},
        {"source_revision": ""},
        {"frequencies": ["weekly"]},
        {"end": "2024-01-06"},
        {"start": "2024-01-02T12:00:00"},
    ],
)
def test_invalid_bounded_bundle_requests_do_not_publish_an_archive(
    tmp_path, source_bundle, overrides
):
    bundle, _ = source_bundle
    options = {
        "namespace": "permanent",
        "source_revision": "captured-v1",
        "sids": [1],
        "start": "2024-01-02",
        "end": "2024-01-04",
        **overrides,
    }
    with pytest.raises(ValueError):
        export_bundle(bundle, tmp_path / "invalid", **options)
    assert not (tmp_path / "invalid").exists()


def test_selected_bundle_export_retains_stock_payment_identity_without_inventing_its_prices(
    tmp_path, source_bundle
):
    bundle, _ = source_bundle
    snapshot = export_bundle(
        bundle,
        tmp_path / "selected",
        namespace="permanent",
        source_revision="capture-v1",
        sids=[1],
        start="2024-01-02",
        end="2024-01-04",
    )
    assert snapshot.tables["zipline_equities"]["source_sid"].to_list() == [1, 2]
    assert snapshot.tables["daily_bars"]["source_sid"].to_list() == [1, 1, 1]
    name = "selected-target-" + uuid4().hex
    bundles.register(
        name,
        make_bundle_ingest(tmp_path / "selected"),
        calendar_name="XNYS",
        start_session=pd.Timestamp("2024-01-02"),
        end_session=pd.Timestamp("2024-01-04"),
    )
    environment = {"ZIPLINE_ROOT": str(tmp_path / "target")}
    try:
        bundles.ingest(name, environ=environment, timestamp=pd.Timestamp("2024-02-03", tz="UTC"))
        with bundles.load(
            name, environ=environment, timestamp=pd.Timestamp("2024-02-04", tz="UTC")
        ) as restored:
            payouts = restored.adjustment_reader.get_stock_dividends_with_ex_date(
                [1], pd.Timestamp("2024-01-04", tz="UTC"), restored.asset_finder
            )
            assert payouts[0].payment_asset.sid == 2
    finally:
        bundles.unregister(name)
