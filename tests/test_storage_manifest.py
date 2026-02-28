"""Tests for canonical storage metadata manifest contract."""

from __future__ import annotations

from datetime import datetime

import pytest

from ml4t.data.storage.manifest import (
    MANIFEST_VERSION,
    ManifestV1,
    manifest_from_write,
    manifest_with_incremental_update,
    normalize_manifest_payload,
    upgrade_manifest_payload,
)


def test_manifest_v1_from_payload_backfills_key_components() -> None:
    """Manifest parser should derive symbol/asset_class/frequency from storage key."""
    manifest = ManifestV1.from_payload(
        "equities/daily/AAPL",
        {
            "row_count": 25,
            "custom": {
                "provider": "yahoo",
                "start_date": "2024-01-01T00:00:00",
                "end_date": "2024-01-31T00:00:00",
            },
        },
    )

    assert manifest.manifest_version == MANIFEST_VERSION
    assert manifest.key == "equities/daily/AAPL"
    assert manifest.provider == "yahoo"
    assert manifest.symbol == "AAPL"
    assert manifest.asset_class == "equities"
    assert manifest.frequency == "daily"
    assert manifest.row_count == 25
    assert manifest.data_range["start"] == "2024-01-01T00:00:00"
    assert manifest.data_range["end"] == "2024-01-31T00:00:00"


def test_manifest_from_write_sets_canonical_fields() -> None:
    """Write manifest builder should populate required contract fields."""
    payload = manifest_from_write(
        "crypto/hourly/BTCUSD",
        row_count=10,
        schema=["timestamp", "close"],
        custom={"provider": "cryptocompare"},
        range_start=datetime(2024, 1, 1),
        range_end=datetime(2024, 1, 2),
    )

    assert payload["manifest_version"] == MANIFEST_VERSION
    assert payload["key"] == "crypto/hourly/BTCUSD"
    assert payload["provider"] == "cryptocompare"
    assert payload["symbol"] == "BTCUSD"
    assert payload["asset_class"] == "crypto"
    assert payload["frequency"] == "hourly"
    assert payload["row_count"] == 10
    assert payload["total_rows"] == 10
    assert payload["data_range"]["start"].startswith("2024-01-01T00:00:00")
    assert payload["data_range"]["end"].startswith("2024-01-02T00:00:00")


def test_normalize_manifest_payload_adds_compatibility_aliases() -> None:
    """Normalizer should provide stable aliases expected by existing managers."""
    normalized = normalize_manifest_payload(
        "futures/daily/ESZ4",
        {
            "provider": "databento",
            "row_count": 3,
            "update_history": [{"timestamp": "2024-01-01T00:00:00"}],
            "last_update": "2024-01-05T00:00:00",
        },
    )

    assert normalized["update_count"] == 1
    assert normalized["attributes"]["last_update"] == "2024-01-05T00:00:00"
    assert normalized["manifest_version"] == MANIFEST_VERSION


def test_manifest_with_incremental_update_keeps_bounded_history() -> None:
    """Incremental metadata updates should append and cap history safely."""
    payload = {"provider": "yahoo", "symbol": "AAPL", "update_history": []}
    for i in range(105):
        payload = manifest_with_incremental_update(
            "equities/daily/AAPL",
            payload,
            last_update=datetime(2024, 1, 1, 0, 0, i % 60),
            records_added=i + 1,
            chunk_file=f"chunk_{i}.parquet",
            max_history=100,
        )

    assert len(payload["update_history"]) == 100
    assert payload["update_history"][-1]["records_added"] == 105


def test_upgrade_manifest_payload_emits_deprecation_for_unversioned() -> None:
    """Legacy unversioned manifests should emit deprecation warnings on upgrade."""
    with pytest.deprecated_call(match="without 'manifest_version'"):
        upgraded, changed = upgrade_manifest_payload(
            "equities/daily/AAPL",
            {
                "row_count": 5,
                "custom": {"provider": "yahoo"},
            },
            emit_deprecation_warning=True,
        )

    assert changed is True
    assert upgraded["manifest_version"] == MANIFEST_VERSION
    assert upgraded["provider"] == "yahoo"


def test_upgrade_manifest_payload_noop_for_current_v1() -> None:
    """Already normalized v1 manifests should not be marked as changed."""
    payload = manifest_from_write(
        "equities/daily/MSFT",
        row_count=7,
        schema=["timestamp", "close"],
        custom={"provider": "yahoo"},
    )
    upgraded, changed = upgrade_manifest_payload("equities/daily/MSFT", payload)

    assert changed is False
    assert upgraded == payload
