"""Async filesystem storage backend with aiofiles."""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any

import aiofiles
import aiofiles.os
import polars as pl
import structlog

from ml4t.data.core.exceptions import LockError, StorageError
from ml4t.data.core.models import DataObject, Metadata
from ml4t.data.security import PathTraversalError, PathValidator
from ml4t.data.storage.async_base import AsyncStorageBackend

logger = structlog.get_logger()


# Storage exceptions imported from core.exceptions
# Legacy aliases for backward compatibility
LockAcquisitionError = LockError


class AsyncFileLock:
    """Async file lock implementation using asyncio locks."""

    def __init__(self, lock_path: Path, timeout: float = 30.0):
        """Initialize async file lock.

        Args:
            lock_path: Path to lock file
            timeout: Maximum time to wait for lock
        """
        self.lock_path = lock_path
        self.timeout = timeout
        # Create a new lock for each instance
        self._lock = asyncio.Lock()
        self._file_handle: Any | None = None

    async def acquire(self) -> bool:
        """Acquire the lock asynchronously.

        Returns:
            True if lock acquired successfully

        Raises:
            LockAcquisitionError: If lock cannot be acquired within timeout
        """
        try:
            await asyncio.wait_for(self._lock.acquire(), timeout=self.timeout)

            # Create lock file to indicate lock is held
            self.lock_path.parent.mkdir(parents=True, exist_ok=True)
            self._file_handle = await aiofiles.open(self.lock_path, mode="w")
            await self._file_handle.write(str(time.time()))
            await self._file_handle.flush()

            return True

        except TimeoutError as e:
            raise LockAcquisitionError(str(self.lock_path), self.timeout) from e

    async def release(self) -> None:
        """Release the lock."""
        try:
            if self._file_handle:
                await self._file_handle.close()
                self._file_handle = None

            # Remove lock file
            if self.lock_path.exists():
                await aiofiles.os.remove(self.lock_path)

        finally:
            if self._lock.locked():
                self._lock.release()

    async def __aenter__(self):
        """Async context manager entry."""
        await self.acquire()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.release()


class AsyncFileSystemBackend(AsyncStorageBackend):
    """Async filesystem-based storage backend with race-condition-free locking."""

    def __init__(
        self,
        data_root: Path,
        lock_timeout: float = 30.0,
        max_retries: int = 3,
        retry_base_delay: float = 0.1,
    ) -> None:
        """
        Initialize async filesystem backend.

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

        # Store locks for reuse - shared at backend level
        self._lock_registry: dict[str, asyncio.Lock] = {}
        self._registry_lock = asyncio.Lock()

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

    async def _get_or_create_lock(self, key: str) -> asyncio.Lock:
        """Get or create a lock for the given key.

        Args:
            key: Storage key

        Returns:
            asyncio.Lock for the key
        """
        async with self._registry_lock:
            if key not in self._lock_registry:
                self._lock_registry[key] = asyncio.Lock()
            return self._lock_registry[key]

    async def _acquire_lock_with_retry(self, key: str, operation: str) -> AsyncFileLock:
        """
        Acquire lock with exponential backoff retry.

        Args:
            key: Storage key
            operation: Operation type for logging

        Returns:
            Acquired lock

        Raises:
            LockAcquisitionError: If lock cannot be acquired after retries
        """
        # Get the shared asyncio.Lock for this key
        shared_lock = await self._get_or_create_lock(key)

        # Create file lock that uses the shared asyncio.Lock
        lock_path = self._get_lock_path(key)
        lock = AsyncFileLock(lock_path, self.lock_timeout)

        # Replace the lock's internal lock with the shared one
        lock._lock = shared_lock

        for attempt in range(self.max_retries):
            try:
                await lock.acquire()
                logger.info(
                    "Acquired lock for async operation",
                    key=key,
                    operation=operation,
                    attempt=attempt + 1,
                )
                return lock

            except LockAcquisitionError:
                if attempt == self.max_retries - 1:
                    raise

                # Exponential backoff
                delay = self.retry_base_delay * (2**attempt)
                logger.warning(
                    "Lock acquisition failed, retrying",
                    key=key,
                    operation=operation,
                    attempt=attempt + 1,
                    delay=delay,
                )
                await asyncio.sleep(delay)

        raise LockAcquisitionError(key, self.lock_timeout)

    async def write(self, data: DataObject) -> str:
        """
        Write data to filesystem asynchronously.

        Args:
            data: DataObject to write

        Returns:
            Storage key for the written data

        Raises:
            StorageError: If write operation fails
            LockAcquisitionError: If lock cannot be acquired
        """
        dir_path, key = self._get_storage_path(data.metadata)

        lock = await self._acquire_lock_with_retry(key, "write")
        try:
            # Create directory if it doesn't exist
            dir_path.mkdir(parents=True, exist_ok=True)

            # Save data
            data_file = dir_path / f"{data.metadata.symbol}.parquet"
            metadata_file = dir_path / f"{data.metadata.symbol}.json"

            # Write Parquet file (polars doesn't have async write yet)
            await asyncio.to_thread(data.data.write_parquet, data_file)

            # Write metadata file asynchronously
            # Use model_dump_json to handle datetime serialization
            metadata_json = data.metadata.model_dump_json(indent=2)
            async with aiofiles.open(metadata_file, mode="w") as f:
                await f.write(metadata_json)

            logger.info("Successfully wrote data asynchronously", key=key)
            return key

        except Exception as e:
            logger.error(
                "Failed to write data asynchronously",
                key=key,
                error=str(e),
            )
            raise StorageError(f"Failed to write {key}: {e}") from e
        finally:
            await lock.release()

    async def read(self, key: str) -> DataObject:
        """
        Read data from filesystem asynchronously.

        Args:
            key: Storage key

        Returns:
            DataObject from storage

        Raises:
            StorageError: If read operation fails or key doesn't exist
            LockAcquisitionError: If lock cannot be acquired
        """
        parts = key.split("/")
        if len(parts) != 3:
            raise StorageError(f"Invalid key format: {key}")

        asset_class, frequency, symbol = parts

        # Validate path components
        PathValidator._validate_component(asset_class, "asset_class")
        PathValidator._validate_component(frequency, "frequency")
        PathValidator._validate_component(symbol, "symbol")

        data_file = PathValidator.sanitize_path(
            self.data_root, f"{asset_class}/{frequency}/{symbol}.parquet"
        )
        metadata_file = PathValidator.sanitize_path(
            self.data_root, f"{asset_class}/{frequency}/{symbol}.json"
        )

        lock = await self._acquire_lock_with_retry(key, "read")
        try:
            if not data_file.exists() or not metadata_file.exists():
                raise StorageError(f"Data not found for key: {key}")

            # Read Parquet file (polars doesn't have async read yet)
            df = await asyncio.to_thread(pl.read_parquet, data_file)

            # Read metadata file asynchronously
            async with aiofiles.open(metadata_file) as f:
                metadata_json = await f.read()
                metadata_dict = json.loads(metadata_json)

            metadata = Metadata(**metadata_dict)

            logger.debug("Successfully read data asynchronously", key=key)
            return DataObject(data=df, metadata=metadata)

        except Exception as e:
            logger.error(
                "Failed to read data asynchronously",
                key=key,
                error=str(e),
            )
            raise StorageError(f"Failed to read {key}: {e}") from e
        finally:
            await lock.release()

    async def exists(self, key: str) -> bool:
        """
        Check if data exists in filesystem asynchronously.

        Args:
            key: Storage key

        Returns:
            True if data exists, False otherwise
        """
        parts = key.split("/")
        if len(parts) != 3:
            return False

        try:
            asset_class, frequency, symbol = parts

            # Validate components
            PathValidator._validate_component(asset_class, "asset_class")
            PathValidator._validate_component(frequency, "frequency")
            PathValidator._validate_component(symbol, "symbol")

            data_file = PathValidator.sanitize_path(
                self.data_root, f"{asset_class}/{frequency}/{symbol}.parquet"
            )

            # Use aiofiles.os.path.exists for async check
            return await asyncio.to_thread(data_file.exists)

        except (PathTraversalError, Exception):
            return False

    async def delete(self, key: str) -> None:
        """
        Delete data from filesystem asynchronously.

        Args:
            key: Storage key

        Raises:
            StorageError: If delete operation fails
            LockAcquisitionError: If lock cannot be acquired
        """
        parts = key.split("/")
        if len(parts) != 3:
            raise StorageError(f"Invalid key format: {key}")

        asset_class, frequency, symbol = parts

        # Validate components
        PathValidator._validate_component(asset_class, "asset_class")
        PathValidator._validate_component(frequency, "frequency")
        PathValidator._validate_component(symbol, "symbol")

        data_file = PathValidator.sanitize_path(
            self.data_root, f"{asset_class}/{frequency}/{symbol}.parquet"
        )
        metadata_file = PathValidator.sanitize_path(
            self.data_root, f"{asset_class}/{frequency}/{symbol}.json"
        )

        lock = await self._acquire_lock_with_retry(key, "delete")
        try:
            # Delete files if they exist
            if data_file.exists():
                await aiofiles.os.remove(data_file)
            if metadata_file.exists():
                await aiofiles.os.remove(metadata_file)

            logger.info("Successfully deleted data asynchronously", key=key)

        except Exception as e:
            logger.error(
                "Failed to delete data asynchronously",
                key=key,
                error=str(e),
            )
            raise StorageError(f"Failed to delete {key}: {e}") from e
        finally:
            await lock.release()

    async def list_keys(self, prefix: str = "") -> list[str]:
        """
        List all keys in filesystem storage asynchronously.

        Args:
            prefix: Optional prefix to filter keys

        Returns:
            List of storage keys
        """
        keys = []

        # Use asyncio.to_thread for filesystem traversal
        def _collect_keys():
            result = []
            for parquet_file in self.data_root.rglob("*.parquet"):
                # Extract key from path
                rel_path = parquet_file.relative_to(self.data_root)
                parts = rel_path.parts[:-1]  # Remove filename

                if len(parts) >= 2:
                    asset_class = parts[0]
                    frequency = parts[1] if len(parts) > 1 else ""
                    symbol = parquet_file.stem

                    key = f"{asset_class}/{frequency}/{symbol}"

                    if not prefix or key.startswith(prefix):
                        result.append(key)
            return result

        keys = await asyncio.to_thread(_collect_keys)
        return sorted(keys)
