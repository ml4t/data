"""Flat file storage implementation.

Simple storage backend that stores each key as a single parquet file.
Suitable for smaller datasets or when partitioning is not beneficial.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl
import structlog

from .backend import StorageBackend, StorageConfig
from .keys import KEY_ENCODING_PREFIX, decode_storage_key

logger = structlog.get_logger()


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

        # Apply date filters if timestamp column exists
        schema = lf.collect_schema()
        if "timestamp" in schema:
            timestamp_type = schema["timestamp"]
            if isinstance(timestamp_type, pl.Datetime):
                timestamp = pl.col("timestamp")
                if timestamp_type.time_zone is None:
                    timestamp = timestamp.dt.replace_time_zone("UTC")
                else:
                    timestamp = timestamp.dt.convert_time_zone("UTC")
                lf = lf.with_columns(timestamp.cast(pl.Datetime("us", "UTC")))

            normalized_start = (
                start_date.replace(tzinfo=UTC)
                if start_date is not None and start_date.tzinfo is None
                else start_date.astimezone(UTC)
                if start_date is not None
                else None
            )
            normalized_end = (
                end_date.replace(tzinfo=UTC)
                if end_date is not None and end_date.tzinfo is None
                else end_date.astimezone(UTC)
                if end_date is not None
                else None
            )
            if normalized_start:
                lf = lf.filter(pl.col("timestamp") >= normalized_start)
            if normalized_end:
                lf = lf.filter(pl.col("timestamp") < normalized_end)

        # Select after timestamp filtering so callers can omit the filter column.
        if columns:
            lf = lf.select(columns)

        return lf

    def list_keys(self) -> list[str]:
        """List all keys in storage.

        Returns:
            List of storage keys
        """
        keys = []
        for path in self.base_path.glob(f"{KEY_ENCODING_PREFIX}*"):
            if path.is_dir() and (path / "CURRENT").is_file():
                try:
                    keys.append(decode_storage_key(path.name))
                except ValueError as error:
                    logger.warning("Ignoring invalid storage entry", path=path, error=str(error))
        if any(self.base_path.glob("*.parquet")):
            logger.warning(
                "Legacy flat storage entries require explicit migration",
                base_path=self.base_path,
            )
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
