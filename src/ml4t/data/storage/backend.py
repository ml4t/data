"""Storage backend interface and implementations for ML4T Data.

This module provides the abstract interface for storage backends and concrete
implementations for Hive partitioned and flat file storage strategies.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import polars as pl
from filelock import FileLock

from ml4t.data.storage.keys import contained_path, storage_key_path

# Type alias for partition granularity
PartitionGranularityType = Literal["year", "month", "day", "hour"]


def normalize_storage_metadata(metadata: Any, key: str | None = None) -> dict[str, Any] | None:
    """Return domain metadata from a canonical or legacy storage record."""
    if not isinstance(metadata, dict) or not metadata:
        return None

    custom = metadata.get("custom")
    normalized = {**metadata, **custom} if isinstance(custom, dict) else metadata.copy()

    if key is not None:
        parts = key.split("/", 2)
        if len(parts) == 3:
            asset_class, frequency, symbol = parts
            normalized.setdefault("asset_class", asset_class)
            normalized.setdefault("frequency", frequency)
            normalized.setdefault("symbol", symbol)

    if "frequency" not in normalized:
        bar_params = normalized.get("bar_params")
        if isinstance(bar_params, dict) and isinstance(bar_params.get("frequency"), str):
            normalized["frequency"] = bar_params["frequency"]

    return normalized


@dataclass
class StorageConfig:
    """Configuration for storage backends.

    Attributes:
        base_path: Base directory for storage.
        strategy: Storage strategy ("hive" or "flat").
        compression: Compression type for Parquet files.
        partition_granularity: Time-based partition granularity for Hive storage.
            - "year": Best for daily data (~252 rows/partition for stocks)
            - "month": Best for hourly data (~720 rows/partition)
            - "day": Best for minute data (~1,440 rows/partition)
            - "hour": Best for second/tick data (~3,600 rows/partition)
        partition_cols: Deprecated. Use partition_granularity instead.
        atomic_writes: Use atomic writes with temp file rename.
        enable_locking: Enable file locking for concurrent access.
        metadata_tracking: Track metadata in manifest files.
    """

    base_path: Path
    strategy: str = "hive"  # "hive" or "flat"
    compression: str | None = "zstd"  # "zstd", "lz4", "snappy", None
    partition_granularity: PartitionGranularityType = "month"
    partition_cols: list[str] | None = None  # Deprecated, kept for backward compat
    atomic_writes: bool = True
    enable_locking: bool = True
    metadata_tracking: bool = True
    generate_profile: bool = True  # Generate column-level statistics on write

    def __post_init__(self) -> None:
        """Validate and set defaults."""
        self.base_path = Path(self.base_path)
        # Set partition_cols based on granularity for backward compatibility
        if self.partition_cols is None:
            if self.strategy == "hive":
                self.partition_cols = self._get_partition_cols_from_granularity()
            else:
                self.partition_cols = []

    def _get_partition_cols_from_granularity(self) -> list[str]:
        """Get partition columns based on granularity setting."""
        granularity_to_cols = {
            "year": ["year"],
            "month": ["year", "month"],
            "day": ["year", "month", "day"],
            "hour": ["year", "month", "day", "hour"],
        }
        return granularity_to_cols.get(self.partition_granularity, ["year", "month"])


@dataclass(frozen=True)
class CommitState:
    """One published immutable data generation and its metadata."""

    commit_id: str
    generation_id: str
    generation_path: Path
    metadata: dict[str, Any]


class StorageBackend(ABC):
    """Abstract base class for storage backends."""

    def __init__(self, config: StorageConfig) -> None:
        """Initialize storage backend with configuration.

        Args:
            config: Storage configuration
        """
        config.base_path = config.base_path.expanduser().resolve()
        self.config = config
        self.base_path = config.base_path
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.metadata_dir = self.base_path / ".metadata"
        self.metadata_dir.mkdir(exist_ok=True)
        self._recover_unpublished_staging()

    def _recover_unpublished_staging(self) -> None:
        """Remove generations that were never made visible by a commit pointer."""
        for key_path in self.base_path.glob("k1_*"):
            if key_path.is_symlink() or not key_path.is_dir():
                continue
            for staging_path in key_path.glob(".staging-*"):
                if staging_path.is_symlink():
                    staging_path.unlink()
                elif staging_path.is_dir():
                    shutil.rmtree(staging_path)

    @abstractmethod
    def write(self, data: pl.LazyFrame, key: str, metadata: dict[str, Any] | None = None) -> Path:
        """Write data to storage.

        Args:
            data: Polars LazyFrame to write
            key: Storage key (e.g., "BTC-USD", "SPY")
            metadata: Optional metadata to store alongside data

        Returns:
            Path to written file
        """

    @abstractmethod
    def read(
        self,
        key: str,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        columns: list[str] | None = None,
    ) -> pl.LazyFrame:
        """Read data from storage.

        Args:
            key: Storage key
            start_date: Optional start date filter
            end_date: Optional end date filter
            columns: Optional columns to select

        Returns:
            Polars LazyFrame with requested data
        """

    @abstractmethod
    def list_keys(self) -> list[str]:
        """List all available keys in storage.

        Returns:
            List of storage keys
        """

    @abstractmethod
    def exists(self, key: str) -> bool:
        """Check if a key exists in storage.

        Args:
            key: Storage key to check

        Returns:
            True if key exists
        """

    @abstractmethod
    def delete(self, key: str) -> bool:
        """Delete data for a key.

        Args:
            key: Storage key to delete

        Returns:
            True if deletion was successful
        """

    def get_metadata(self, key: str) -> dict[str, Any] | None:
        """Get metadata for a key.

        Args:
            key: Storage key

        Returns:
            Metadata dict or None
        """
        try:
            return self._current_commit(key).metadata
        except KeyError:
            return None

    def _key_path(self, key: str) -> Path:
        """Return the versioned dataset directory for a logical key."""
        return storage_key_path(self.base_path, key)

    def _key_lock(self, key: str) -> FileLock:
        """Return the lock covering a complete logical-key mutation."""
        return FileLock(storage_key_path(self.metadata_dir, key, ".lock"), timeout=30)

    def _prepare_generation(self, key: str) -> tuple[Path, str]:
        """Create an unpublished staging directory for a new generation."""
        key_path = self._key_path(key)
        key_path.mkdir(exist_ok=True)
        contained_path(key_path, "generations").mkdir(exist_ok=True)
        contained_path(key_path, "commits").mkdir(exist_ok=True)
        generation_id = uuid.uuid4().hex
        staging_path = contained_path(key_path, f".staging-{generation_id}")
        staging_path.mkdir()
        return staging_path, generation_id

    def _publish_generation(
        self,
        key: str,
        staging_path: Path,
        generation_id: str,
        metadata: dict[str, Any],
    ) -> CommitState:
        """Publish a complete staged generation through one atomic pointer."""
        key_path = self._key_path(key)
        generations_path = contained_path(key_path, "generations")
        generation_path = contained_path(generations_path, generation_id)
        staging_path.replace(generation_path)
        return self._publish_commit(key, generation_id, metadata)

    def _publish_commit(
        self,
        key: str,
        generation_id: str,
        metadata: dict[str, Any],
    ) -> CommitState:
        """Publish metadata for an existing immutable generation."""
        key_path = self._key_path(key)
        generations_path = contained_path(key_path, "generations")
        generation_path = contained_path(generations_path, generation_id)
        if not generation_path.is_dir():
            raise RuntimeError(f"Storage generation is missing: {generation_path}")

        commit_id = uuid.uuid4().hex
        commits_path = contained_path(key_path, "commits")
        commit_path = contained_path(commits_path, f"{commit_id}.json")
        self._write_metadata_file(
            commit_path,
            {
                "format_version": 1,
                "generation": generation_id,
                "metadata": metadata,
            },
        )
        self._atomic_write_text(contained_path(key_path, "CURRENT"), f"{commit_id}\n")
        return CommitState(commit_id, generation_id, generation_path, metadata)

    def _current_commit(self, key: str) -> CommitState:
        """Resolve the single commit visible to readers."""
        key_path = self._key_path(key)
        pointer_path = contained_path(key_path, "CURRENT")
        if not pointer_path.is_file():
            raise KeyError(f"Key '{key}' not found in storage")

        commit_id = pointer_path.read_text(encoding="utf-8").strip()
        if len(commit_id) != 32 or any(
            character not in "0123456789abcdef" for character in commit_id
        ):
            raise RuntimeError(f"Invalid CURRENT pointer for key '{key}'")
        commits_path = contained_path(key_path, "commits")
        commit_path = contained_path(commits_path, f"{commit_id}.json")
        try:
            with open(commit_path, encoding="utf-8") as commit_file:
                manifest = json.load(commit_file)
            generation_id = manifest["generation"]
            metadata = manifest["metadata"]
        except (FileNotFoundError, KeyError, TypeError, json.JSONDecodeError) as error:
            raise RuntimeError(f"Invalid commit manifest for key '{key}'") from error
        if (
            not isinstance(generation_id, str)
            or len(generation_id) != 32
            or any(character not in "0123456789abcdef" for character in generation_id)
            or not isinstance(metadata, dict)
        ):
            raise RuntimeError(f"Invalid commit manifest for key '{key}'")

        generations_path = contained_path(key_path, "generations")
        generation_path = contained_path(generations_path, generation_id)
        if not generation_path.is_dir():
            raise RuntimeError(f"Published generation is missing for key '{key}'")
        return CommitState(commit_id, generation_id, generation_path, metadata)

    def _delete_key(self, key: str) -> bool:
        """Atomically make a key inaccessible, then remove its old generations."""
        key_path = self._key_path(key)
        with self._key_lock(key):
            if not (key_path / "CURRENT").is_file():
                return False
            trash_dir = self.base_path / ".trash"
            trash_dir.mkdir(exist_ok=True)
            trash_path = trash_dir / f"{key_path.name}-{uuid.uuid4().hex}"
            key_path.replace(trash_path)
        shutil.rmtree(trash_path)
        return True

    @staticmethod
    def _cleanup_staging(staging_path: Path) -> None:
        """Remove an unpublished generation after a failed write."""
        shutil.rmtree(staging_path, ignore_errors=True)

    def _atomic_write(self, df: pl.DataFrame, target_path: Path) -> None:
        """Write DataFrame atomically using temp file pattern.

        Args:
            df: DataFrame to write
            target_path: Target file path
        """
        fd, tmp_name = tempfile.mkstemp(dir=target_path.parent, suffix=".parquet.tmp")
        os.close(fd)
        tmp_path = Path(tmp_name)

        try:
            df.write_parquet(tmp_path, compression=self.config.compression or "zstd")
            tmp_path.replace(target_path)
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

    def _update_metadata(self, key: str, metadata: dict[str, Any]) -> None:
        """Update metadata for a key.

        Args:
            key: Storage key
            metadata: Metadata to store
        """
        with self._key_lock(key):
            current = self._current_commit(key)
            self._publish_commit(key, current.generation_id, metadata)

    def _atomic_write_text(self, path: Path, content: str) -> None:
        """Atomically replace a small text file and flush its contents."""
        fd, tmp_name = tempfile.mkstemp(dir=path.parent, suffix=".txt.tmp", text=True)
        tmp_path = Path(tmp_name)
        try:
            with os.fdopen(fd, mode="w", encoding="utf-8") as tmp_file:
                tmp_file.write(content)
                tmp_file.flush()
                os.fsync(tmp_file.fileno())
            tmp_path.replace(path)
        finally:
            tmp_path.unlink(missing_ok=True)

    def _write_metadata_file(self, path: Path, metadata: dict[str, Any]) -> None:
        """Write metadata to file.

        Args:
            path: Metadata file path
            metadata: Metadata to write
        """
        fd, tmp_name = tempfile.mkstemp(dir=path.parent, suffix=".json.tmp", text=True)
        tmp_path = Path(tmp_name)

        try:
            with os.fdopen(fd, mode="w") as tmp_file:
                json.dump(metadata, tmp_file, indent=2, default=str)
            tmp_path.replace(path)
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

    def _ensure_lazy(self, data: pl.DataFrame | pl.LazyFrame) -> pl.LazyFrame:
        """Ensure data is a LazyFrame for efficient processing.

        Args:
            data: DataFrame or LazyFrame

        Returns:
            LazyFrame
        """
        if isinstance(data, pl.DataFrame):
            return data.lazy()
        return data
