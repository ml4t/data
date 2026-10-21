"""Real optional bundle migration through standard readers and ingestion writers."""

import hashlib
from uuid import uuid4

import numpy as np
import pandas as pd
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
