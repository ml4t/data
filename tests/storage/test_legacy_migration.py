"""Tests for migration from the ambiguous pre-0.1 storage layout."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import polars as pl
import pytest
from polars.testing import assert_frame_equal

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
    source_metadata.write_text('{"custom": {"source": "legacy"}}')

    results = migrate_legacy_storage(
        tmp_path,
        "flat",
        {"equities_daily_BRK_B": "equities/daily/BRK_B"},
    )

    storage = FlatStorage(StorageConfig(tmp_path, strategy="flat"))
    actual = storage.read("equities/daily/BRK_B").collect()
    assert_frame_equal(actual, expected)
    assert [result.logical_key for result in results] == ["equities/daily/BRK_B"]
    assert not source.exists()
    assert not source_metadata.exists()
    assert (tmp_path / ".legacy-v0-backup/flat/equities_daily_BRK_B.parquet").is_file()
    assert (tmp_path / ".legacy-v0-backup/flat/.metadata/equities_daily_BRK_B.json").is_file()


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


def test_hive_migration_reads_all_partitions_and_preserves_backup(tmp_path: Path) -> None:
    expected = _market_frame()
    legacy_root = tmp_path / "equities_daily_BRK_B"
    january = legacy_root / "year=2024/month=1"
    february = legacy_root / "year=2024/month=2"
    january.mkdir(parents=True)
    february.mkdir(parents=True)
    expected.head(1).write_parquet(january / "data.parquet")
    expected.tail(1).write_parquet(february / "data.parquet")

    migrate_legacy_storage(
        tmp_path,
        "hive",
        {"equities_daily_BRK_B": "equities/daily/BRK_B"},
    )

    storage = HiveStorage(StorageConfig(tmp_path, strategy="hive"))
    assert_frame_equal(storage.read("equities/daily/BRK_B").collect(), expected)
    assert not legacy_root.exists()
    assert (
        tmp_path / ".legacy-v0-backup/hive/equities_daily_BRK_B/year=2024/month=1/data.parquet"
    ).is_file()


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
