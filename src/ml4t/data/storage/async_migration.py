"""Migration utilities for transitioning between sync and async storage."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from pathlib import Path

import structlog

from ml4t.data.storage.async_base import AsyncStorageBackend
from ml4t.data.storage.async_filesystem import AsyncFileSystemBackend
from ml4t.data.storage.base import StorageBackend

logger = structlog.get_logger()


class StorageMigrator:
    """Utilities for migrating between sync and async storage backends."""

    @staticmethod
    async def sync_to_async(
        sync_backend: StorageBackend,
        async_backend: AsyncStorageBackend,
        prefix: str = "",
        batch_size: int = 10,
        progress_callback: Callable | None = None,
    ) -> tuple[int, int]:
        """
        Migrate data from sync to async storage backend.

        Args:
            sync_backend: Source sync storage backend
            async_backend: Target async storage backend
            prefix: Optional prefix to filter keys
            batch_size: Number of items to process concurrently
            progress_callback: Optional callback for progress updates

        Returns:
            Tuple of (successful_count, failed_count)
        """
        keys = sync_backend.list_keys(prefix)
        total = len(keys)
        successful = 0
        failed = 0

        logger.info(
            "Starting sync to async migration",
            total_keys=total,
            prefix=prefix,
            batch_size=batch_size,
        )

        for i in range(0, total, batch_size):
            batch = keys[i : i + batch_size]
            tasks = []

            for key in batch:
                tasks.append(
                    StorageMigrator._migrate_single_item(
                        key, sync_backend, async_backend, "sync_to_async"
                    )
                )

            # Process batch concurrently
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Count results
            for result in results:
                if isinstance(result, Exception):
                    failed += 1
                    logger.error("Migration failed for item", error=str(result))
                else:
                    successful += 1

            # Report progress
            if progress_callback:
                progress = (i + len(batch)) / total
                await progress_callback(f"Migrated {i + len(batch)}/{total} items", progress)

        logger.info(
            "Sync to async migration complete",
            successful=successful,
            failed=failed,
            total=total,
        )

        return successful, failed

    @staticmethod
    async def async_to_sync(
        async_backend: AsyncStorageBackend,
        sync_backend: StorageBackend,
        prefix: str = "",
        batch_size: int = 10,
        progress_callback: Callable | None = None,
    ) -> tuple[int, int]:
        """
        Migrate data from async to sync storage backend.

        Args:
            async_backend: Source async storage backend
            sync_backend: Target sync storage backend
            prefix: Optional prefix to filter keys
            batch_size: Number of items to process concurrently
            progress_callback: Optional callback for progress updates

        Returns:
            Tuple of (successful_count, failed_count)
        """
        keys = await async_backend.list_keys(prefix)
        total = len(keys)
        successful = 0
        failed = 0

        logger.info(
            "Starting async to sync migration",
            total_keys=total,
            prefix=prefix,
            batch_size=batch_size,
        )

        for i in range(0, total, batch_size):
            batch = keys[i : i + batch_size]
            tasks = []

            for key in batch:
                tasks.append(
                    StorageMigrator._migrate_single_item(
                        key, async_backend, sync_backend, "async_to_sync"
                    )
                )

            # Process batch concurrently
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Count results
            for result in results:
                if isinstance(result, Exception):
                    failed += 1
                    logger.error("Migration failed for item", error=str(result))
                else:
                    successful += 1

            # Report progress
            if progress_callback:
                progress = (i + len(batch)) / total
                await progress_callback(f"Migrated {i + len(batch)}/{total} items", progress)

        logger.info(
            "Async to sync migration complete",
            successful=successful,
            failed=failed,
            total=total,
        )

        return successful, failed

    @staticmethod
    async def _migrate_single_item(
        key: str,
        source_backend: StorageBackend | AsyncStorageBackend,
        target_backend: StorageBackend | AsyncStorageBackend,
        direction: str,
    ) -> str:
        """
        Migrate a single item between backends.

        Args:
            key: Storage key to migrate
            source_backend: Source backend
            target_backend: Target backend
            direction: Migration direction for logging

        Returns:
            Storage key on success

        Raises:
            Exception: If migration fails
        """
        try:
            # Read from source
            if isinstance(source_backend, AsyncStorageBackend):
                data = await source_backend.read(key)
            else:
                data = await asyncio.to_thread(source_backend.read, key)

            # Write to target
            if isinstance(target_backend, AsyncStorageBackend):
                result_key = await target_backend.write(data)
            else:
                result_key = await asyncio.to_thread(target_backend.write, data)

            logger.debug(
                "Successfully migrated item",
                key=key,
                direction=direction,
            )
            return result_key

        except Exception as e:
            logger.error(
                "Failed to migrate item",
                key=key,
                direction=direction,
                error=str(e),
            )
            raise


class AsyncStorageAdapter:
    """Adapter to use async storage in sync contexts."""

    def __init__(self, async_backend: AsyncStorageBackend):
        """
        Initialize adapter.

        Args:
            async_backend: Async storage backend to wrap
        """
        self.async_backend = async_backend
        self._loop: asyncio.AbstractEventLoop | None = None

    def _get_loop(self) -> asyncio.AbstractEventLoop:
        """Get or create event loop."""
        if self._loop is None or self._loop.is_closed():
            try:
                self._loop = asyncio.get_running_loop()
            except RuntimeError:
                self._loop = asyncio.new_event_loop()
                asyncio.set_event_loop(self._loop)
        return self._loop

    def write(self, data):
        """Sync wrapper for async write."""
        loop = self._get_loop()
        return loop.run_until_complete(self.async_backend.write(data))

    def read(self, key):
        """Sync wrapper for async read."""
        loop = self._get_loop()
        return loop.run_until_complete(self.async_backend.read(key))

    def exists(self, key):
        """Sync wrapper for async exists."""
        loop = self._get_loop()
        return loop.run_until_complete(self.async_backend.exists(key))

    def delete(self, key):
        """Sync wrapper for async delete."""
        loop = self._get_loop()
        return loop.run_until_complete(self.async_backend.delete(key))

    def list_keys(self, prefix=""):
        """Sync wrapper for async list_keys."""
        loop = self._get_loop()
        return loop.run_until_complete(self.async_backend.list_keys(prefix))


def create_async_backend(data_root: Path, **kwargs) -> AsyncFileSystemBackend:
    """
    Factory function to create async filesystem backend.

    Args:
        data_root: Root directory for data storage
        **kwargs: Additional arguments for backend

    Returns:
        Configured async filesystem backend
    """
    return AsyncFileSystemBackend(data_root=data_root, **kwargs)


def create_sync_adapter(async_backend: AsyncStorageBackend) -> AsyncStorageAdapter:
    """
    Create a sync adapter for async storage backend.

    Args:
        async_backend: Async storage backend to wrap

    Returns:
        Sync adapter for the async backend
    """
    return AsyncStorageAdapter(async_backend)
