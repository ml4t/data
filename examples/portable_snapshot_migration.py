"""Rebuild native equity and individual-contract preparation from relocated input copies."""

import hashlib
import json
from pathlib import Path
from runpy import run_path
from tempfile import TemporaryDirectory

import polars as pl
from polars.testing import assert_frame_equal

from ml4t.data.adjustments import prepare_adjusted_history
from ml4t.data.assets import InstrumentRegistry
from ml4t.data.futures import HighestVolumeRoll, prepare_futures_inputs
from ml4t.data.storage import (
    FlatStorage,
    SnapshotTable,
    StorageConfig,
    export_snapshot,
    read_snapshot,
)
from ml4t.specs import FeedSpec


def _declaration(namespace: str, source_revision: str, units: dict, as_of: str) -> dict:
    return {
        "source_revision": source_revision,
        "identity_namespace": namespace,
        "units": units,
        "as_of": as_of,
        "price_basis": None,
        "share_basis": None,
        "revisions": {"capture": "r1"},
    }


def _rebuild(snapshot):
    tables = snapshot.tables
    content = snapshot.manifest["content"]
    entries = content["tables"]
    policy = content["effective_config"]

    def feed(name):
        return FeedSpec.from_mapping(entries[name]["clock"]["feed_spec"])

    registry = InstrumentRegistry.from_frame(
        tables["equity_identities"],
        tables["equity_membership"],
        feed_spec=feed("equity_identities"),
    )
    equity = prepare_adjusted_history(
        tables["equity_prices"],
        tables["equity_actions"],
        as_of=policy["equity"]["as_of"],
        basis=policy["equity"]["basis"],
        action_coverage=policy["equity"]["action_coverage"],
        instruments=registry,
        identity_mode="symbol",
        feed_spec=feed("equity_prices"),
        calendar=entries["equity_prices"]["clock"]["resolved_calendar"],
    )
    future = prepare_futures_inputs(
        tables["futures_prices"],
        tables["futures_definitions"],
        as_of=policy["futures"]["as_of"],
        roll_strategy=HighestVolumeRoll(
            rank=policy["futures"]["roll_strategy"]["rank"],
            min_volume=policy["futures"]["roll_strategy"]["min_volume"],
        ),
        feed_spec=feed("futures_prices"),
        calendar=entries["futures_prices"]["clock"]["resolved_calendar"],
    )
    return equity, future


def migrate_snapshot(directory: Path) -> dict:
    history_path = Path(__file__).with_name("dated_history_migration.py")
    futures_path = Path(__file__).with_name("futures_input_migration.py")
    history_recipe = run_path(history_path)["migrate_history"]
    futures_recipe = run_path(futures_path)["migrate_futures"]
    equity_root, futures_root = directory / "equity-source", directory / "futures-source"
    _, _, original_equity = history_recipe(equity_root)
    _, original_future, _ = futures_recipe(futures_root)
    equity_storage = FlatStorage(StorageConfig(base_path=equity_root))
    equity_revision = hashlib.sha256(history_path.read_bytes()).hexdigest()
    futures_revision = hashlib.sha256(futures_path.read_bytes()).hexdigest()
    cutoff = "2025-07-07T21:00:00Z"
    sources = {}
    equity_units = {
        "equity_prices": {"close": "USD/share", "volume": "shares"},
        "equity_actions": {"cash_per_share": "USD/post_event_share", "split_ratio": "new_per_old"},
        "equity_identities": {},
        "equity_membership": {},
    }
    for name, key in (
        ("equity_prices", "capture/prices"),
        ("equity_identities", "capture/identities"),
        ("equity_membership", "capture/membership"),
        ("equity_actions", "capture/actions"),
    ):
        declaration = _declaration("equity-fixture", equity_revision, equity_units[name], cutoff)
        if name == "equity_prices":
            declaration.update(price_basis="raw", share_basis="source")
        sources[name] = SnapshotTable(
            equity_storage,
            name,
            declaration,
            key=key,
        )
    futures_feed = FeedSpec(
        time_contract_version=2,
        calendar="CME_Equity",
        entity_col="asset",
        timestamp_semantics="session_label",
        data_frequency="daily",
    )
    futures_cutoff = "2025-07-09T21:30:00Z"
    prices_path = futures_root / "ohlcv_1d/product=TEST/ohlcv_1d.parquet"
    definitions_path = futures_root / "definition/product=TEST/definition.parquet"
    for name, path, feed in (
        ("futures_prices", prices_path, futures_feed),
        (
            "futures_definitions",
            definitions_path,
            futures_feed.with_overrides(timestamp_col="valid_from"),
        ),
    ):
        units = (
            {"close": "index_points", "volume": "contracts"}
            if name == "futures_prices"
            else {"multiplier": "USD/index_point/contract", "tick_size": "index_points"}
        )
        declaration = _declaration("futures-fixture", futures_revision, units, futures_cutoff)
        if name == "futures_prices":
            declaration.update(price_basis="raw", share_basis="individual_contracts")
        sources[name] = SnapshotTable(
            pl.scan_parquet(path),
            name,
            declaration,
            feed_spec=feed,
        )
    configuration = {
        "workflow": "native input preparation; no broker/account execution",
        "equity": {"as_of": cutoff, "basis": "total_return", "action_coverage": "complete"},
        "futures": {
            "as_of": futures_cutoff,
            "roll_strategy": {"type": "HighestVolumeRoll", "rank": 0, "min_volume": 0},
        },
    }
    captured = export_snapshot(
        directory / "portable",
        sources,
        effective_config=configuration,
        artifacts={"equity-example": equity_revision, "futures-example": futures_revision},
    )
    before = _rebuild(captured)
    assert_frame_equal(before[0].prices, original_equity.prices)
    assert_frame_equal(before[1].selections, original_future.selections)
    original_generation = equity_storage._current_commit("capture/prices").generation_path
    changed = (
        equity_storage.read("capture/prices").collect().with_columns(pl.lit(999999).alias("close"))
    )
    for _ in range(5):
        equity_storage.write(changed, "capture/prices", sources["equity_prices"].metadata)
    assert not original_generation.exists()
    pl.read_parquet(prices_path).with_columns(pl.lit(999999).alias("close")).write_parquet(
        prices_path
    )
    (directory / "portable").rename(directory / "relocated")
    restored = read_snapshot(directory / "relocated")
    equity, future = _rebuild(restored)
    assert restored.snapshot_id == captured.snapshot_id
    assert_frame_equal(equity.prices, before[0].prices)
    assert_frame_equal(equity.actions, before[0].actions)
    assert_frame_equal(future.bars, before[1].bars)
    assert_frame_equal(future.selections, before[1].selections)
    assert_frame_equal(future.rolls, before[1].rolls)
    assert equity.prices["asset"].to_list() == [17, 17, 17, 17]
    assert equity.prices["ticker"].to_list() == ["OLD", "OLD", "OLD", "NEW"]
    assert equity.prices["close"].to_list() == [198, 200, 98, 101]
    assert equity.prices["adj_volume"].to_list() == [2000, 2000, 7, 9]
    assert future.selections["selected_instrument_id"].to_list() == [23]
    assert future.rolls["old_mark"].to_list() == [4020]
    assert future.rolls["new_mark"].to_list() == [4030]
    return {
        "snapshot_id": restored.snapshot_id,
        "tables": sorted(restored.tables),
        "equity_stable_ids": equity.prices["asset"].to_list(),
        "futures_selected_ids": future.selections["selected_instrument_id"].to_list(),
        "assertions": "passed",
        "account_execution": "not certified by this input recipe",
    }


if __name__ == "__main__":
    with TemporaryDirectory(prefix="ml4t-portable-inputs-") as temporary:
        print(json.dumps(migrate_snapshot(Path(temporary)), sort_keys=True))
