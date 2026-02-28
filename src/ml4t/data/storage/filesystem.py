"""Filesystem storage backend with improved locking and security.

.. deprecated:: 0.2.0
    FileSystemBackend is deprecated. Use HiveStorage for production workloads.
    HiveStorage provides better performance with Hive-partitioned Parquet files.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import filelock
import polars as pl
import structlog

from ml4t.data.core.exceptions import LockError, StorageError
from ml4t.data.core.models import DataObject, Metadata
from ml4t.data.security import PathTraversalError, PathValidator
from ml4t.data.storage.base import StorageBackend

logger = structlog.get_logger()


# Storage exceptions imported from core.exceptions
# Legacy aliases for backward compatibility
LockAcquisitionError = LockError


class FileSystemBackend(StorageBackend):
    """Filesystem-based storage backend with race-condition-free locking.

    .. deprecated:: 0.2.0
        Use :class:`~ml4t.data.storage.hive.HiveStorage` instead for better performance
        and Hive-partitioned Parquet storage.
    """

    def __init__(
        self,
        data_root: Path,
        lock_timeout: float = 30.0,
        max_retries: int = 3,
        retry_base_delay: float = 0.1,
    ) -> None:
        """
        Initialize filesystem backend with improved locking.

        Args:
            data_root: Root directory for data storage
            lock_timeout: Maximum time to wait for lock acquisition (seconds)
            max_retries: Maximum number of lock acquisition retries
            retry_base_delay: Base delay for exponential backoff (seconds)
        """
        self.data_root = Path(data_root)
        self.data_root.mkdir(parents=True, exist_ok=True)

        # Create locks directory
        self.locks_dir = self.data_root / ".locks"
        self.locks_dir.mkdir(parents=True, exist_ok=True)

        # Lock configuration
        self.lock_timeout = lock_timeout
        self.max_retries = max_retries
        self.retry_base_delay = retry_base_delay

    def _get_storage_path(self, metadata: Metadata) -> tuple[Path, str]:
        """
        Get storage path for data based on metadata.

        Args:
            metadata: Data metadata

        Returns:
            Tuple of (directory path, storage key)

        Raises:
            PathTraversalError: If metadata contains dangerous patterns
        """
        # Validate components to prevent path traversal
        PathValidator._validate_component(metadata.asset_class, "asset_class")
        PathValidator._validate_component(metadata.frequency, "frequency")
        PathValidator._validate_component(metadata.symbol, "symbol")

        # Build hierarchical path: asset_class/frequency/symbol
        dir_path = self.data_root / metadata.asset_class / metadata.frequency
        key = f"{metadata.asset_class}/{metadata.frequency}/{metadata.symbol}"

        # Ensure the path doesn't escape our data root
        dir_path = PathValidator.sanitize_path(
            self.data_root, f"{metadata.asset_class}/{metadata.frequency}"
        )

        return dir_path, key

    def _get_lock_path(self, key: str) -> Path:
        """Get the file path for the lock file."""
        # Replace slashes with underscores for flat lock directory
        lock_name = key.replace("/", "_") + ".lock"
        return self.locks_dir / lock_name

    def _acquire_lock_with_retry(
        self, lock_path: Path, operation: str, key: str
    ) -> filelock.FileLock:
        """
        Acquire lock with retry mechanism and exponential backoff.

        Args:
            lock_path: Path to lock file
            operation: Operation type (read/write/delete)
            key: Storage key for logging

        Returns:
            Acquired lock context manager

        Raises:
            LockAcquisitionError: If lock cannot be acquired after all retries
        """
        lock = filelock.FileLock(lock_path, timeout=self.lock_timeout)

        for attempt in range(self.max_retries):
            try:
                lock.acquire(timeout=self.lock_timeout)
                logger.info(
                    f"Acquired lock for {operation}",
                    key=key,
                    attempt=attempt + 1,
                    operation=operation,
                )
                return lock

            except filelock.Timeout:
                if attempt < self.max_retries - 1:
                    # Exponential backoff: 0.1s, 0.2s, 0.4s...
                    wait_time = self.retry_base_delay * (2**attempt)
                    logger.warning(
                        f"Lock acquisition failed for {operation}, retrying",
                        key=key,
                        attempt=attempt + 1,
                        wait_time=wait_time,
                        operation=operation,
                    )
                    time.sleep(wait_time)
                else:
                    logger.error(
                        f"Failed to acquire lock for {operation} after all retries",
                        key=key,
                        attempts=self.max_retries,
                        operation=operation,
                    )
                    raise LockAcquisitionError(key, self.lock_timeout) from None

        # Should never reach here
        raise LockAcquisitionError(key, self.lock_timeout)

    def write(self, data: DataObject) -> str:
        """
        Write data to filesystem with file locking and atomic writes.

        Args:
            data: DataObject to write

        Returns:
            Storage key for the written data

        Raises:
            LockAcquisitionError: If write lock cannot be acquired
            StorageError: If write operation fails
        """
        dir_path, key = self._get_storage_path(data.metadata)
        dir_path.mkdir(parents=True, exist_ok=True)

        # Get lock for this key with retry mechanism
        lock_path = self._get_lock_path(key)
        lock = self._acquire_lock_with_retry(lock_path, "write", key)

        try:
            # File paths
            parquet_path = dir_path / f"{data.metadata.symbol}.parquet"
            manifest_path = dir_path / f"{data.metadata.symbol}.manifest.json"

            # Temporary file paths for atomic writes
            parquet_tmp = parquet_path.with_suffix(".parquet.tmp")
            manifest_tmp = manifest_path.with_suffix(".json.tmp")

            # Convert to canonical schema before writing
            canonical_df = data.to_canonical_schema()

            try:
                # Write to temporary files first
                canonical_df.write_parquet(parquet_tmp)

                # Prepare manifest
                manifest = {
                    "provider": data.metadata.provider,
                    "symbol": data.metadata.symbol,
                    "asset_class": data.metadata.asset_class,
                    "bar_type": data.metadata.bar_type,
                    "bar_params": data.metadata.bar_params,
                    "schema_version": data.metadata.schema_version,
                    "download_utc_timestamp": data.metadata.download_utc_timestamp.isoformat(),
                    "data_range": data.metadata.data_range or {},
                    "provider_params": data.metadata.provider_params,
                }

                with open(manifest_tmp, "w") as f:
                    json.dump(manifest, f, indent=2)

                # Atomic rename operations
                os.replace(str(parquet_tmp), str(parquet_path))
                os.replace(str(manifest_tmp), str(manifest_path))

                logger.info("Successfully wrote data", key=key)

            except Exception as e:
                # Clean up temporary files on error
                if parquet_tmp.exists():
                    parquet_tmp.unlink()
                if manifest_tmp.exists():
                    manifest_tmp.unlink()
                logger.error("Failed to write data", key=key, error=str(e))
                raise StorageError(f"Write operation failed for key {key}: {e}") from e

        finally:
            if lock.is_locked:
                lock.release()
                logger.debug("Released write lock", key=key)

        return key

    def read(self, key: str) -> DataObject:
        """
        Read data from filesystem with read locking.

        Args:
            key: Storage key (e.g., "equities/daily/AAPL")

        Returns:
            DataObject from storage

        Raises:
            LockAcquisitionError: If read lock cannot be acquired
            FileNotFoundError: If data files not found
            ValueError: If storage key is invalid
            PathTraversalError: If key contains dangerous patterns
        """
        # Validate and parse the storage key
        try:
            asset_class, frequency, symbol = PathValidator.validate_storage_key(key)
        except (PathTraversalError, ValueError) as e:
            logger.error("Invalid storage key", key=key, error=str(e))
            raise

        # Get lock for this key - ALWAYS require lock for consistency
        lock_path = self._get_lock_path(key)
        lock = self._acquire_lock_with_retry(lock_path, "read", key)

        try:
            # File paths
            dir_path = self.data_root / asset_class / frequency
            parquet_path = dir_path / f"{symbol}.parquet"
            manifest_path = dir_path / f"{symbol}.manifest.json"

            if not parquet_path.exists():
                raise FileNotFoundError(f"Data file not found: {parquet_path}")
            if not manifest_path.exists():
                raise FileNotFoundError(f"Manifest file not found: {manifest_path}")

            # Read Parquet file
            df = pl.read_parquet(parquet_path)

            # Read manifest
            with open(manifest_path) as f:
                manifest = json.load(f)

            # Reconstruct metadata
            # Handle both old (frequency) and new (bar_type/bar_params) formats
            if "bar_type" in manifest:
                # New format
                metadata = Metadata(
                    provider=manifest["provider"],
                    symbol=manifest["symbol"],
                    asset_class=manifest["asset_class"],
                    bar_type=manifest["bar_type"],
                    bar_params=manifest.get("bar_params", {}),
                    schema_version=manifest["schema_version"],
                    data_range=manifest.get("data_range"),
                    provider_params=manifest.get("provider_params", {}),
                )
            else:
                # Old format - convert frequency to new format
                frequency = manifest.get("frequency", "unknown")
                metadata = Metadata(
                    provider=manifest["provider"],
                    symbol=manifest["symbol"],
                    asset_class=manifest["asset_class"],
                    bar_type="time",
                    bar_params={"frequency": frequency},
                    schema_version=manifest["schema_version"],
                    data_range=manifest.get("data_range"),
                    provider_params=manifest.get("provider_params", {}),
                )

            return DataObject(data=df, metadata=metadata)

        finally:
            if lock.is_locked:
                lock.release()
                logger.debug("Released read lock", key=key)

    def exists(self, key: str) -> bool:
        """
        Check if data exists in filesystem.

        Args:
            key: Storage key

        Returns:
            True if data exists, False otherwise
        """
        try:
            asset_class, frequency, symbol = PathValidator.validate_storage_key(key)
        except (PathTraversalError, ValueError):
            # Invalid keys don't exist
            return False

        # Use sanitized path to check existence
        safe_path = PathValidator.sanitize_path(
            self.data_root, f"{asset_class}/{frequency}/{symbol}.parquet"
        )
        return safe_path.exists()

    def delete(self, key: str) -> None:
        """
        Delete data from filesystem with locking.

        Args:
            key: Storage key

        Raises:
            LockAcquisitionError: If delete lock cannot be acquired
            ValueError: If storage key is invalid
            PathTraversalError: If key contains dangerous patterns
        """
        # Validate and parse the storage key
        try:
            asset_class, frequency, symbol = PathValidator.validate_storage_key(key)
        except (PathTraversalError, ValueError) as e:
            logger.error("Invalid storage key for delete", key=key, error=str(e))
            raise

        # Get lock for this key with retry mechanism
        lock_path = self._get_lock_path(key)
        lock = self._acquire_lock_with_retry(lock_path, "delete", key)

        try:
            dir_path = self.data_root / asset_class / frequency

            # Delete both files
            parquet_path = dir_path / f"{symbol}.parquet"
            manifest_path = dir_path / f"{symbol}.manifest.json"

            if parquet_path.exists():
                parquet_path.unlink()
            if manifest_path.exists():
                manifest_path.unlink()

            logger.info("Successfully deleted data", key=key)

        finally:
            if lock.is_locked:
                lock.release()
                logger.debug("Released delete lock", key=key)

    def list_keys(self, prefix: str = "") -> list[str]:
        """
        List all keys in filesystem storage.

        Args:
            prefix: Optional prefix to filter keys

        Returns:
            List of storage keys
        """
        keys = []

        # Walk through the directory structure
        for parquet_file in self.data_root.rglob("*.parquet"):
            # Build key from path
            relative_path = parquet_file.relative_to(self.data_root)
            parts = relative_path.parts[:-1]  # Remove filename
            symbol = parquet_file.stem  # Remove .parquet extension

            if len(parts) == 2:  # asset_class/frequency
                key = f"{parts[0]}/{parts[1]}/{symbol}"
                if not prefix or key.startswith(prefix):
                    keys.append(key)

        return sorted(keys)
