"""Explicit migration from the ambiguous pre-0.1 storage layout."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import polars as pl
from polars.testing import assert_frame_equal

from ml4t.data.core.models import DataObject, Metadata

from .backend import StorageConfig
from .chunked import ChunkedStorage
from .flat import FlatStorage
from .hive import HiveStorage
from .keys import KEY_ENCODING_PREFIX, contained_path, validate_storage_key

LegacyStrategy = Literal["flat", "hive", "chunked"]


class LegacyStorageMigrationError(RuntimeError):
    """Raised when legacy storage cannot be migrated without ambiguity or data loss."""


@dataclass(frozen=True)
class LegacyStorageEntry:
    """One ambiguous physical key and the files or directories that belong to it."""

    physical_key: str
    paths: tuple[Path, ...]


@dataclass(frozen=True)
class LegacyStorageMigration:
    """The result of one verified legacy-key migration."""

    physical_key: str
    logical_key: str
    backup_paths: tuple[Path, ...]


def find_legacy_storage_entries(
    base_path: str | Path, strategy: LegacyStrategy
) -> list[LegacyStorageEntry]:
    """Inventory pre-0.1 storage entries without guessing their logical keys."""
    root = Path(base_path)
    if not root.is_dir():
        return []
    if strategy == "flat":
        return _find_flat_entries(root)
    if strategy == "hive":
        return _find_hive_entries(root)
    if strategy == "chunked":
        return _find_chunked_entries(root)
    raise ValueError(f"Unknown legacy storage strategy: {strategy}")


def migrate_legacy_storage(
    base_path: str | Path,
    strategy: LegacyStrategy,
    key_mapping: Mapping[str, str],
) -> list[LegacyStorageMigration]:
    """Migrate every legacy entry using an explicit physical-to-logical key mapping.

    The mapping must cover the inventory exactly. New data is written and verified
    before legacy sources are moved under ``.legacy-v0-backup``.
    """
    root = Path(base_path)
    entries = find_legacy_storage_entries(root, strategy)
    _validate_mapping(entries, key_mapping)
    backup_root = contained_path(root, ".legacy-v0-backup", strategy)
    backup_plan = _build_backup_plan(root, backup_root, entries)

    loaded = {entry.physical_key: _read_legacy_entry(root, strategy, entry) for entry in entries}
    backend = _create_backend(root, strategy)
    destinations = [key_mapping[entry.physical_key] for entry in entries]
    existing = [key for key in destinations if backend.exists(key)]
    if existing:
        raise LegacyStorageMigrationError(
            f"Refusing to overwrite existing destination keys: {sorted(existing)}"
        )

    written: list[str] = []
    try:
        for entry in entries:
            logical_key = key_mapping[entry.physical_key]
            expected = loaded[entry.physical_key]
            _write_migrated(backend, strategy, logical_key, expected)
            written.append(logical_key)
            actual = _read_migrated(backend, strategy, logical_key)
            _verify_frame(expected, actual, logical_key)
    except BaseException:
        for logical_key in written:
            backend.delete(logical_key)
        raise

    results = []
    for entry in entries:
        moved = []
        for source in entry.paths:
            destination = backup_plan[source]
            destination.parent.mkdir(parents=True, exist_ok=True)
            source.replace(destination)
            moved.append(destination)
        results.append(
            LegacyStorageMigration(
                physical_key=entry.physical_key,
                logical_key=key_mapping[entry.physical_key],
                backup_paths=tuple(moved),
            )
        )
    return results


def _find_flat_entries(root: Path) -> list[LegacyStorageEntry]:
    entries = []
    for data_path in sorted(root.glob("*.parquet")):
        paths = [data_path]
        metadata_path = root / ".metadata" / f"{data_path.stem}.json"
        if metadata_path.is_file():
            paths.append(metadata_path)
        entries.append(LegacyStorageEntry(data_path.stem, tuple(paths)))
    return entries


def _find_hive_entries(root: Path) -> list[LegacyStorageEntry]:
    entries = []
    for data_path in sorted(root.iterdir()):
        if (
            not data_path.is_dir()
            or data_path.name.startswith(".")
            or data_path.name.startswith(KEY_ENCODING_PREFIX)
            or not any(data_path.glob("year=*/**/data.parquet"))
        ):
            continue
        paths = [data_path]
        metadata_path = root / ".metadata" / f"{data_path.name}.json"
        if metadata_path.is_file():
            paths.append(metadata_path)
        entries.append(LegacyStorageEntry(data_path.name, tuple(paths)))
    return entries


def _find_chunked_entries(root: Path) -> list[LegacyStorageEntry]:
    metadata_root = root / "metadata"
    if not metadata_root.is_dir():
        return []
    entries = []
    for index_path in sorted(metadata_root.glob("*_index.json")):
        physical_key = index_path.name.removesuffix("_index.json")
        paths = [index_path, *_legacy_chunk_paths(root, index_path)]
        entries.append(LegacyStorageEntry(physical_key, tuple(paths)))
    return entries


def _legacy_chunk_paths(root: Path, index_path: Path) -> list[Path]:
    try:
        index = json.loads(index_path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise LegacyStorageMigrationError(
            f"Cannot read legacy chunk index: {index_path}"
        ) from error
    if not isinstance(index, dict):
        raise LegacyStorageMigrationError(f"Legacy chunk index is not an object: {index_path}")

    chunk_paths = []
    for info in index.values():
        if not isinstance(info, dict) or not isinstance(info.get("file_path"), str):
            raise LegacyStorageMigrationError(f"Invalid chunk entry in {index_path}")
        chunk_path = root / "chunks" / Path(info["file_path"]).name
        if not chunk_path.is_file():
            raise LegacyStorageMigrationError(f"Legacy chunk file is missing: {chunk_path}")
        chunk_paths.append(chunk_path)
    if len(set(chunk_paths)) != len(chunk_paths):
        raise LegacyStorageMigrationError(f"Legacy chunk index repeats a file: {index_path}")
    return sorted(chunk_paths)


def _validate_mapping(entries: list[LegacyStorageEntry], key_mapping: Mapping[str, str]) -> None:
    inventory = {entry.physical_key for entry in entries}
    supplied = set(key_mapping)
    if inventory != supplied:
        missing = sorted(inventory - supplied)
        extra = sorted(supplied - inventory)
        raise LegacyStorageMigrationError(
            f"Mapping must cover the legacy inventory exactly; missing={missing}, extra={extra}"
        )
    destinations = []
    for physical_key in sorted(inventory):
        try:
            destinations.append(validate_storage_key(key_mapping[physical_key]))
        except ValueError as error:
            raise LegacyStorageMigrationError(
                f"Invalid destination for physical key '{physical_key}': {error}"
            ) from error
    if len(set(destinations)) != len(destinations):
        raise LegacyStorageMigrationError("Multiple legacy entries map to the same logical key")


def _build_backup_plan(
    root: Path,
    backup_root: Path,
    entries: list[LegacyStorageEntry],
) -> dict[Path, Path]:
    plan = {}
    for entry in entries:
        for source in entry.paths:
            if source.is_symlink() or not source.resolve().is_relative_to(root.resolve()):
                raise LegacyStorageMigrationError(f"Legacy source escapes storage root: {source}")
            relative = source.relative_to(root)
            destination = contained_path(backup_root, *relative.parts)
            if destination.exists():
                raise LegacyStorageMigrationError(
                    f"Backup destination already exists: {destination}"
                )
            plan[source] = destination
    return plan


def _read_legacy_entry(
    root: Path,
    strategy: LegacyStrategy,
    entry: LegacyStorageEntry,
) -> pl.DataFrame:
    if strategy == "flat":
        return pl.read_parquet(entry.paths[0])
    if strategy == "hive":
        files = sorted(entry.paths[0].glob("**/data.parquet"))
        if not files:
            raise LegacyStorageMigrationError(
                f"No Parquet partitions found for {entry.physical_key}"
            )
        return pl.concat([pl.read_parquet(path) for path in files], how="vertical_relaxed")

    index_path = root / "metadata" / f"{entry.physical_key}_index.json"
    return pl.concat(
        [pl.read_parquet(path) for path in _legacy_chunk_paths(root, index_path)],
        how="vertical_relaxed",
    )


def _create_backend(
    root: Path, strategy: LegacyStrategy
) -> FlatStorage | HiveStorage | ChunkedStorage:
    if strategy == "flat":
        return FlatStorage(StorageConfig(base_path=root, strategy="flat"))
    if strategy == "hive":
        return HiveStorage(StorageConfig(base_path=root, strategy="hive"))
    return ChunkedStorage(root)


def _write_migrated(
    backend: FlatStorage | HiveStorage | ChunkedStorage,
    strategy: LegacyStrategy,
    logical_key: str,
    frame: pl.DataFrame,
) -> None:
    if strategy != "chunked":
        assert isinstance(backend, FlatStorage | HiveStorage)
        backend.write(frame, logical_key, {"migrated_from_legacy_layout": True})
        return

    parts = logical_key.split("/")
    if len(parts) != 3:
        raise LegacyStorageMigrationError(
            "Chunked storage destination keys must have asset_class/frequency/symbol form"
        )
    asset_class, frequency, symbol = parts
    assert isinstance(backend, ChunkedStorage)
    backend.write(
        DataObject(
            data=frame,
            metadata=Metadata(
                provider="legacy",
                symbol=symbol,
                asset_class=asset_class,
                bar_params={"frequency": frequency},
            ),
        )
    )


def _read_migrated(
    backend: FlatStorage | HiveStorage | ChunkedStorage,
    strategy: LegacyStrategy,
    logical_key: str,
) -> pl.DataFrame:
    if strategy == "chunked":
        assert isinstance(backend, ChunkedStorage)
        return backend.read(logical_key).data
    assert isinstance(backend, FlatStorage | HiveStorage)
    return backend.read(logical_key).collect()


def _verify_frame(expected: pl.DataFrame, actual: pl.DataFrame, logical_key: str) -> None:
    try:
        assert_frame_equal(expected, actual, check_row_order=False)
    except AssertionError as error:
        raise LegacyStorageMigrationError(
            f"Verification failed for migrated key '{logical_key}': {error}"
        ) from error
