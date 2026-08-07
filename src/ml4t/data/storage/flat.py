"""Flat file storage implementation.

Simple storage backend that stores each key as a single parquet file.
Suitable for smaller datasets or when partitioning is not beneficial.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import polars as pl

from .backend import StorageBackend, StorageConfig
from .keys import KEY_ENCODING_PREFIX, decode_storage_key


class FlatStorage(StorageBackend):
    """Flat file storage without partitioning.

    This implementation provides:
    - Simple single-file storage per key
    - Atomic writes with temp file pattern
    - Metadata tracking in JSON manifests
    - File locking for concurrent access safety
    - Polars lazy evaluation throughout
    """

    def __init__(self, config: StorageConfig):
        """Initialize flat storage backend.

        Args:
            config: Storage configuration
        """
        super().__init__(config)

    def write(
        self, data: pl.LazyFrame | pl.DataFrame, key: str, metadata: dict[str, Any] | None = None
    ) -> Path:
        """Write data as a single file.

        Args:
            data: Data to write
            key: Storage key (e.g., "BTC-USD")
            metadata: Optional metadata

        Returns:
            Path to written file
        """
        # Ensure LazyFrame for efficiency
        lazy_data = self._ensure_lazy(data)

        df = lazy_data.collect()
        with self._key_lock(key):
            staging_path, generation_id = self._prepare_generation(key)
            try:
                staged_file = staging_path / "data.parquet"
                self._atomic_write(df, staged_file)
                commit_metadata = (
                    {
                        "last_updated": datetime.now().isoformat(),
                        "file_path": "data.parquet",
                        "row_count": len(df),
                        "schema": list(df.columns),
                        "file_size_mb": staged_file.stat().st_size / (1024 * 1024),
                        "custom": metadata or {},
                    }
                    if self.config.metadata_tracking
                    else {}
                )
                commit = self._publish_generation(key, staging_path, generation_id, commit_metadata)
            except BaseException:
                self._cleanup_staging(staging_path)
                raise

        return commit.generation_path / "data.parquet"

    def read(
        self,
        key: str,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        columns: list[str] | None = None,
    ) -> pl.LazyFrame:
        """Read data from flat file.

        Args:
            key: Storage key
            start_date: Optional start date filter
            end_date: Optional end date filter
            columns: Optional columns to select

        Returns:
            LazyFrame with requested data
        """
        file_path = self._current_commit(key).generation_path / "data.parquet"
        if not file_path.is_file():
            raise RuntimeError(f"Published data file is missing for key '{key}'")

        # Use lazy reading
        lf = pl.scan_parquet(file_path)

        # Apply column selection
        if columns:
            lf = lf.select(columns)

        # Apply date filters if timestamp column exists
        schema = lf.collect_schema()
        if "timestamp" in schema:
            if start_date:
                lf = lf.filter(pl.col("timestamp") >= start_date)
            if end_date:
                lf = lf.filter(pl.col("timestamp") < end_date)

        return lf

    def list_keys(self) -> list[str]:
        """List all keys in storage.

        Returns:
            List of storage keys
        """
        keys = []
        for path in self.base_path.glob(f"{KEY_ENCODING_PREFIX}*"):
            if path.is_dir() and (path / "CURRENT").is_file():
                keys.append(decode_storage_key(path.name))
        return sorted(keys)

    def exists(self, key: str) -> bool:
        """Check if key exists.

        Args:
            key: Storage key

        Returns:
            True if key exists
        """
        try:
            self._current_commit(key)
        except KeyError:
            return False
        return True

    def delete(self, key: str) -> bool:
        """Delete data for a key.

        Args:
            key: Storage key

        Returns:
            True if successful
        """
        return self._delete_key(key)
