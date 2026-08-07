"""Tests for migration from the ambiguous pre-0.1 storage layout."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import polars as pl
import pytest
from polars.testing import assert_frame_equal

import ml4t.data.storage.legacy_migration as legacy_migration
from ml4t.data.storage import (
    FlatStorage,
    HiveStorage,
    LegacyStorageMigrationError,
    StorageConfig,
    find_legacy_storage_entries,
    migrate_legacy_storage,
)
from ml4t.data.storage.chunked import ChunkedStorage


def _market_frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "timestamp": [datetime(2024, 1, 2), datetime(2024, 2, 2)],
            "open": [100.0, 101.0],
            "high": [102.0, 103.0],
            "low": [99.0, 100.0],
            "close": [101.0, 102.0],
            "volume": [1000.0, 1100.0],
        }
    )


def test_flat_migration_requires_explicit_complete_mapping(tmp_path: Path) -> None:
    data_path = tmp_path / "equities_daily_BRK_B.parquet"
    _market_frame().write_parquet(data_path)

    entries = find_legacy_storage_entries(tmp_path, "flat")

    assert [entry.physical_key for entry in entries] == ["equities_daily_BRK_B"]
    with pytest.raises(LegacyStorageMigrationError, match="missing=.*equities_daily_BRK_B"):
        migrate_legacy_storage(tmp_path, "flat", {})
    assert data_path.is_file()


def test_flat_migration_verifies_data_and_preserves_legacy_backup(tmp_path: Path) -> None:
    expected = _market_frame()
    source = tmp_path / "equities_daily_BRK_B.parquet"
    source_metadata = tmp_path / ".metadata" / "equities_daily_BRK_B.json"
    source_metadata.parent.mkdir()
    expected.write_parquet(source)
    source_metadata.write_text(
        json.dumps(
            {
                "custom": {"source": "legacy"},
                "first_update": "2024-01-02T00:00:00",
                "update_history": [{"records_added": 2}],
            }
        )
    )

    results = migrate_legacy_storage(
        tmp_path,
        "flat",
        {"equities_daily_BRK_B": "equities/daily/BRK_B"},
    )

    storage = FlatStorage(StorageConfig(tmp_path, strategy="flat"))
    actual = storage.read("equities/daily/BRK_B").collect()
    migrated_metadata = storage.get_metadata("equities/daily/BRK_B")
    assert_frame_equal(actual, expected)
    assert migrated_metadata is not None
    assert migrated_metadata["custom"] == {
        "source": "legacy",
        "first_update": "2024-01-02T00:00:00",
        "update_history": [{"records_added": 2}],
        "migrated_from_legacy_layout": True,
    }
    assert [result.logical_key for result in results] == ["equities/daily/BRK_B"]
    assert not source.exists()
    assert not source_metadata.exists()
    assert (tmp_path / ".legacy-v0-backup/flat/equities_daily_BRK_B.parquet").is_file()
    assert (tmp_path / ".legacy-v0-backup/flat/.metadata/equities_daily_BRK_B.json").is_file()
    assert find_legacy_storage_entries(tmp_path, "flat") == []
    assert migrate_legacy_storage(tmp_path, "flat", {}) == []


def test_migration_rejects_duplicate_destinations_before_writing(tmp_path: Path) -> None:
    frame = _market_frame()
    frame.write_parquet(tmp_path / "first.parquet")
    frame.write_parquet(tmp_path / "second.parquet")

    with pytest.raises(LegacyStorageMigrationError, match="same logical key"):
        migrate_legacy_storage(
            tmp_path,
            "flat",
            {"first": "equities/daily/AAPL", "second": "equities/daily/AAPL"},
        )

    assert (tmp_path / "first.parquet").is_file()
    assert (tmp_path / "second.parquet").is_file()
    assert not any(tmp_path.glob("k1_*"))


def test_migration_rejects_invalid_destination_before_writing(tmp_path: Path) -> None:
    source = tmp_path / "ambiguous_key.parquet"
    _market_frame().write_parquet(source)

    with pytest.raises(LegacyStorageMigrationError, match="Invalid destination"):
        migrate_legacy_storage(tmp_path, "flat", {"ambiguous_key": "../escape"})

    assert source.is_file()
    assert not (tmp_path.parent / "escape").exists()


def test_hive_migration_uses_caller_partition_granularity(tmp_path: Path) -> None:
    expected = _market_frame()
    legacy_root = tmp_path / "equities_daily_BRK_B"
    january = legacy_root / "year=2024/month=1"
    february = legacy_root / "year=2024/month=2"
    january.mkdir(parents=True)
    february.mkdir(parents=True)
    expected.head(1).write_parquet(january / "data.parquet")
    expected.tail(1).write_parquet(february / "data.parquet")

    config = StorageConfig(tmp_path, strategy="hive", partition_granularity="day")
    migrate_legacy_storage(
        tmp_path,
        "hive",
        {"equities_daily_BRK_B": "equities/daily/BRK_B"},
        storage_config=config,
    )

    storage = HiveStorage(config)
    assert_frame_equal(storage.read("equities/daily/BRK_B").collect(), expected)
    assert not legacy_root.exists()
    assert (
        tmp_path / ".legacy-v0-backup/hive/equities_daily_BRK_B/year=2024/month=1/data.parquet"
    ).is_file()
    assert find_legacy_storage_entries(tmp_path, "hive") == []
    assert migrate_legacy_storage(tmp_path, "hive", {}, storage_config=config) == []


def test_chunked_migration_uses_index_files_without_guessing_keys(tmp_path: Path) -> None:
    expected = _market_frame()
    chunks = tmp_path / "chunks"
    metadata = tmp_path / "metadata"
    chunks.mkdir()
    metadata.mkdir()
    chunk_paths = [
        chunks / "equities_daily_BRK_B_BRK_B_daily_2024_01.parquet",
        chunks / "equities_daily_BRK_B_BRK_B_daily_2024_02.parquet",
    ]
    expected.head(1).write_parquet(chunk_paths[0])
    expected.tail(1).write_parquet(chunk_paths[1])
    index = {
        "BRK_B_daily_2024_01": {
            "file_path": str(chunk_paths[0]),
            "start_date": "2024-01-01T00:00:00",
            "end_date": "2024-01-31T23:59:59",
            "row_count": 1,
            "size_bytes": chunk_paths[0].stat().st_size,
        },
        "BRK_B_daily_2024_02": {
            "file_path": str(chunk_paths[1]),
            "start_date": "2024-02-01T00:00:00",
            "end_date": "2024-02-29T23:59:59",
            "row_count": 1,
            "size_bytes": chunk_paths[1].stat().st_size,
        },
    }
    index_path = metadata / "equities_daily_BRK_B_index.json"
    index_path.write_text(json.dumps(index))

    migrate_legacy_storage(
        tmp_path,
        "chunked",
        {"equities_daily_BRK_B": "equities/daily/BRK_B"},
    )

    actual = ChunkedStorage(tmp_path).read("equities/daily/BRK_B").data
    assert_frame_equal(actual, expected)
    assert not index_path.exists()
    assert not any(path.exists() for path in chunk_paths)
    assert (
        tmp_path / ".legacy-v0-backup/chunked/metadata/equities_daily_BRK_B_index.json"
    ).is_file()
    assert find_legacy_storage_entries(tmp_path, "chunked") == []
    assert migrate_legacy_storage(tmp_path, "chunked", {}) == []


def test_verification_failure_removes_destination_and_preserves_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "ambiguous_key.parquet"
    _market_frame().write_parquet(source)

    def fail_verification(*_args: object) -> None:
        raise LegacyStorageMigrationError("injected verification failure")

    monkeypatch.setattr(legacy_migration, "_verify_frame", fail_verification)

    with pytest.raises(LegacyStorageMigrationError, match="injected verification failure"):
        migrate_legacy_storage(
            tmp_path,
            "flat",
            {"ambiguous_key": "equities/daily/AAPL"},
        )

    assert source.is_file()
    assert not FlatStorage(StorageConfig(tmp_path, strategy="flat")).exists("equities/daily/AAPL")


def test_migration_refuses_to_overwrite_existing_destination(tmp_path: Path) -> None:
    destination = "equities/daily/AAPL"
    storage = FlatStorage(StorageConfig(tmp_path, strategy="flat"))
    storage.write(_market_frame(), destination)
    source = tmp_path / "ambiguous_key.parquet"
    _market_frame().write_parquet(source)

    with pytest.raises(LegacyStorageMigrationError, match="Refusing to overwrite"):
        migrate_legacy_storage(tmp_path, "flat", {"ambiguous_key": destination})

    assert source.is_file()
    assert storage.exists(destination)


def test_backup_move_failure_restores_sources_and_removes_destinations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = tmp_path / "first.parquet"
    second = tmp_path / "second.parquet"
    _market_frame().write_parquet(first)
    _market_frame().write_parquet(second)
    real_move = legacy_migration._move_to_backup
    moves = 0

    def fail_second_move(source: Path, destination: Path) -> None:
        nonlocal moves
        moves += 1
        if moves == 2:
            raise OSError("injected backup failure")
        real_move(source, destination)

    monkeypatch.setattr(legacy_migration, "_move_to_backup", fail_second_move)

    with pytest.raises(OSError, match="injected backup failure"):
        migrate_legacy_storage(
            tmp_path,
            "flat",
            {"first": "equities/daily/AAPL", "second": "equities/daily/MSFT"},
        )

    assert first.is_file()
    assert second.is_file()
    storage = FlatStorage(StorageConfig(tmp_path, strategy="flat"))
    assert not storage.exists("equities/daily/AAPL")
    assert not storage.exists("equities/daily/MSFT")


def test_chunked_schema_failure_uses_migration_error(tmp_path: Path) -> None:
    chunks = tmp_path / "chunks"
    metadata = tmp_path / "metadata"
    chunks.mkdir()
    metadata.mkdir()
    chunk_path = chunks / "fundamentals_annual_AAPL_2024.parquet"
    pl.DataFrame({"timestamp": [datetime(2024, 1, 2)], "value": [1.0]}).write_parquet(chunk_path)
    (metadata / "fundamentals_annual_AAPL_index.json").write_text(
        json.dumps(
            {
                "AAPL_annual_2024": {
                    "file_path": str(chunk_path),
                    "start_date": "2024-01-01T00:00:00",
                    "end_date": "2024-12-31T23:59:59",
                    "row_count": 1,
                    "size_bytes": chunk_path.stat().st_size,
                }
            }
        )
    )

    with pytest.raises(
        LegacyStorageMigrationError,
        match="Cannot migrate legacy chunked entry 'fundamentals_annual_AAPL'",
    ):
        migrate_legacy_storage(
            tmp_path,
            "chunked",
            {"fundamentals_annual_AAPL": "fundamentals/annual/AAPL"},
        )


def test_migration_refuses_to_overwrite_a_previous_backup(tmp_path: Path) -> None:
    source = tmp_path / "ambiguous_key.parquet"
    backup = tmp_path / ".legacy-v0-backup/flat/ambiguous_key.parquet"
    backup.parent.mkdir(parents=True)
    _market_frame().write_parquet(source)
    _market_frame().write_parquet(backup)

    with pytest.raises(LegacyStorageMigrationError, match="Backup destination already exists"):
        migrate_legacy_storage(
            tmp_path,
            "flat",
            {"ambiguous_key": "equities/daily/AAPL"},
        )

    assert source.is_file()
