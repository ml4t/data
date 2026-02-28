"""Flat file storage implementation.

Simple storage backend that stores each key as a single parquet file.
Suitable for smaller datasets or when partitioning is not beneficial.
"""

from __future__ import annotations

import json
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

import polars as pl
from filelock import FileLock

from .backend import StorageBackend, StorageConfig
from .key_codec import decode_storage_key, encode_storage_key
from .manifest import manifest_from_write, normalize_manifest_payload, upgrade_manifest_payload


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
        self.metadata_dir = self.base_path / ".metadata"
        self.metadata_dir.mkdir(exist_ok=True)

    def _encoded_data_path(self, key: str) -> Path:
        """Return encoded data file path."""
        return self.base_path / f"{encode_storage_key(key)}.parquet"

    def _legacy_data_path(self, key: str) -> Path:
        """Return legacy data file path."""
        return self.base_path / f"{key.replace('/', '_')}.parquet"

    def _resolve_existing_data_path(self, key: str) -> Path | None:
        """Resolve existing data file path, preferring encoded format."""
        encoded = self._encoded_data_path(key)
        if encoded.exists():
            return encoded

        legacy = self._legacy_data_path(key)
        if legacy.exists():
            return legacy

        return None

    def _encoded_metadata_path(self, key: str) -> Path:
        """Return encoded metadata file path."""
        return self.metadata_dir / f"{encode_storage_key(key)}.json"

    def _legacy_metadata_path(self, key: str) -> Path:
        """Return legacy metadata file path."""
        return self.metadata_dir / f"{key.replace('/', '_')}.json"

    def _resolve_existing_metadata_path(self, key: str) -> Path | None:
        """Resolve existing metadata file path, preferring encoded format."""
        encoded = self._encoded_metadata_path(key)
        if encoded.exists():
            return encoded

        legacy = self._legacy_metadata_path(key)
        if legacy.exists():
            return legacy

        return None

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

        # Create file path
        file_path = self._encoded_data_path(key)

        # Collect and write atomically
        df = lazy_data.collect()
        self._atomic_write(df, file_path)

        # Update metadata
        if self.config.metadata_tracking:
            self._update_metadata(
                key,
                manifest_from_write(
                    key,
                    row_count=len(df),
                    schema=list(df.columns),
                    custom=metadata or {},
                    range_start=df["timestamp"].min() if "timestamp" in df.columns else None,
                    range_end=df["timestamp"].max() if "timestamp" in df.columns else None,
                    file_path=str(file_path.relative_to(self.base_path)),
                    file_size_mb=file_path.stat().st_size / (1024 * 1024),
                ),
            )

        return file_path

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
        file_path = self._resolve_existing_data_path(key)
        if file_path is None:
            raise KeyError(f"Key '{key}' not found in storage")

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
        for path in self.base_path.glob("*.parquet"):
            keys.append(decode_storage_key(path.stem))
        return sorted(keys)

    def exists(self, key: str) -> bool:
        """Check if key exists.

        Args:
            key: Storage key

        Returns:
            True if key exists
        """
        return self._resolve_existing_data_path(key) is not None

    def delete(self, key: str) -> bool:
        """Delete data for a key.

        Args:
            key: Storage key

        Returns:
            True if successful
        """
        deleted = False
        file_path = self._resolve_existing_data_path(key)
        if file_path is not None:
            file_path.unlink()
            deleted = True

        for metadata_file in (self._encoded_metadata_path(key), self._legacy_metadata_path(key)):
            if metadata_file.exists():
                metadata_file.unlink()
                deleted = True

        return deleted

    def get_metadata(self, key: str) -> dict[str, Any] | None:
        """Get metadata for a key.

        Args:
            key: Storage key

        Returns:
            Metadata dict or None
        """
        metadata_file = self._resolve_existing_metadata_path(key)
        if metadata_file is None:
            return None
        with open(metadata_file) as f:
            raw = json.load(f)
        upgraded, changed = upgrade_manifest_payload(
            key,
            raw,
            emit_deprecation_warning=True,
        )
        if changed:
            self._update_metadata(key, upgraded)
        return normalize_manifest_payload(key, upgraded)

    def _atomic_write(self, df: pl.DataFrame, target_path: Path) -> None:
        """Write DataFrame atomically using temp file pattern.

        Args:
            df: DataFrame to write
            target_path: Target file path
        """
        # Write to temp file first
        tmp_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                dir=target_path.parent, suffix=".parquet.tmp", delete=False
            ) as tmp_file:
                tmp_path = Path(tmp_file.name)
                df.write_parquet(tmp_path, compression=self.config.compression or "zstd")

            if tmp_path is not None:
                tmp_path.replace(target_path)
        except Exception:
            if tmp_path is not None and tmp_path.exists():
                tmp_path.unlink()
            raise

    def _update_metadata(self, key: str, metadata: dict[str, Any]) -> None:
        """Update metadata for a key.

        Args:
            key: Storage key
            metadata: Metadata to store
        """
        metadata_file = self._encoded_metadata_path(key)

        if self.config.enable_locking:
            lock_file = self.metadata_dir / f"{encode_storage_key(key)}.lock"
            lock = FileLock(lock_file, timeout=10)

            with lock:
                self._write_metadata_file(metadata_file, metadata)
        else:
            self._write_metadata_file(metadata_file, metadata)

    def _write_metadata_file(self, path: Path, metadata: dict[str, Any]) -> None:
        """Write metadata to file.

        Args:
            path: Metadata file path
            metadata: Metadata to write
        """
        tmp_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                dir=path.parent, mode="w", suffix=".json.tmp", delete=False
            ) as tmp_file:
                tmp_path = Path(tmp_file.name)
                json.dump(metadata, tmp_file, indent=2, default=str)

            if tmp_path is not None:
                tmp_path.replace(path)
        except Exception:
            if tmp_path is not None and tmp_path.exists():
                tmp_path.unlink()
            raise
