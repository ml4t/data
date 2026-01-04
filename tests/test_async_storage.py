"""Tests for async storage backend."""

import asyncio
import importlib.util
import tempfile
from datetime import datetime
from pathlib import Path

import polars as pl
import pytest

# Skip if aiofiles not installed
if importlib.util.find_spec("aiofiles") is None:
    pytestmark = pytest.mark.skip(
        reason="aiofiles not installed - reinstall with: pip install -e ."
    )

from ml4t.data.core.models import DataObject, Metadata
from ml4t.data.storage.async_filesystem import (
    AsyncFileLock,
    AsyncFileSystemBackend,
    LockAcquisitionError,
    StorageError,
)
from ml4t.data.storage.async_migration import (
    AsyncStorageAdapter,
    StorageMigrator,
    create_async_backend,
    create_sync_adapter,
)
from ml4t.data.storage.filesystem import FileSystemBackend


@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    df = pl.DataFrame(
        {
            "timestamp": [
                datetime(2024, 1, 1),
                datetime(2024, 1, 2),
                datetime(2024, 1, 3),
            ],
            "open": [100.0, 101.0, 102.0],
            "high": [105.0, 106.0, 107.0],
            "low": [99.0, 100.0, 101.0],
            "close": [104.0, 105.0, 106.0],
            "volume": [1000000, 1100000, 1200000],
        }
    )

    metadata = Metadata(
        provider="test",
        symbol="TEST",
        asset_class="equities",
        bar_type="time",
        bar_params={"frequency": "daily"},
        schema_version="1.0",
    )

    return DataObject(data=df, metadata=metadata)


@pytest.fixture
def sample_data_2():
    """Create another sample data for testing."""
    df = pl.DataFrame(
        {
            "timestamp": [
                datetime(2024, 1, 1),
                datetime(2024, 1, 2),
            ],
            "open": [200.0, 201.0],
            "high": [205.0, 206.0],
            "low": [199.0, 200.0],
            "close": [204.0, 205.0],
            "volume": [2000000, 2100000],
        }
    )

    metadata = Metadata(
        provider="test",
        symbol="TEST2",
        asset_class="equities",
        bar_type="time",
        bar_params={"frequency": "daily"},
        schema_version="1.0",
    )

    return DataObject(data=df, metadata=metadata)


class TestAsyncFileSystemBackend:
    """Test async filesystem storage backend."""

    @pytest.mark.asyncio
    async def test_write_and_read(self, sample_data):
        """Test writing and reading data asynchronously."""
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = AsyncFileSystemBackend(data_root=Path(tmpdir))

            # Write data
            key = await backend.write(sample_data)
            assert key == "equities/daily/TEST"

            # Read data
            loaded_data = await backend.read(key)
            assert len(loaded_data.data) == 3
            assert loaded_data.metadata.symbol == "TEST"
            assert loaded_data.data.equals(sample_data.data)

    @pytest.mark.asyncio
    async def test_exists(self, sample_data):
        """Test checking if data exists asynchronously."""
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = AsyncFileSystemBackend(data_root=Path(tmpdir))

            key = "equities/daily/TEST"

            # Should not exist initially
            assert not await backend.exists(key)

            # Write data
            await backend.write(sample_data)

            # Should exist now
            assert await backend.exists(key)

    @pytest.mark.asyncio
    async def test_delete(self, sample_data):
        """Test deleting data asynchronously."""
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = AsyncFileSystemBackend(data_root=Path(tmpdir))

            # Write data
            key = await backend.write(sample_data)
            assert await backend.exists(key)

            # Delete data
            await backend.delete(key)

            # Should not exist anymore
            assert not await backend.exists(key)

    @pytest.mark.asyncio
    async def test_list_keys(self, sample_data, sample_data_2):
        """Test listing keys asynchronously."""
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = AsyncFileSystemBackend(data_root=Path(tmpdir))

            # Initially empty
            keys = await backend.list_keys()
            assert len(keys) == 0

            # Write multiple data objects
            await backend.write(sample_data)
            await backend.write(sample_data_2)

            # List all keys
            keys = await backend.list_keys()
            assert len(keys) == 2
            assert "equities/daily/TEST" in keys
            assert "equities/daily/TEST2" in keys

            # List with prefix
            keys = await backend.list_keys("equities/daily/TEST2")
            assert len(keys) == 1
            assert keys[0] == "equities/daily/TEST2"

    @pytest.mark.asyncio
    async def test_update(self, sample_data):
        """Test updating data asynchronously."""
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = AsyncFileSystemBackend(data_root=Path(tmpdir))

            # Write initial data
            key = await backend.write(sample_data)

            # Modify data
            sample_data.data = sample_data.data.with_columns(pl.col("volume") * 2)

            # Update data
            updated_key = await backend.update(key, sample_data)
            assert updated_key == key

            # Read and verify
            loaded_data = await backend.read(key)
            assert loaded_data.data["volume"][0] == 2000000

    @pytest.mark.asyncio
    async def test_concurrent_writes(self, sample_data):
        """Test concurrent write operations."""
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = AsyncFileSystemBackend(data_root=Path(tmpdir))

            # Create multiple data objects with different symbols
            data_objects = []
            for i in range(5):
                data = DataObject(
                    data=sample_data.data,
                    metadata=Metadata(
                        provider="test",
                        symbol=f"TEST{i}",
                        asset_class="equities",
                        bar_type="time",
                        bar_params={"frequency": "daily"},
                        schema_version="1.0",
                    ),
                )
                data_objects.append(data)

            # Write concurrently
            tasks = [backend.write(data) for data in data_objects]
            keys = await asyncio.gather(*tasks)

            # Verify all written
            assert len(keys) == 5
            for i, key in enumerate(keys):
                assert key == f"equities/daily/TEST{i}"
                assert await backend.exists(key)

    @pytest.mark.asyncio
    async def test_concurrent_reads(self, sample_data):
        """Test concurrent read operations."""
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = AsyncFileSystemBackend(data_root=Path(tmpdir))

            # Write data
            key = await backend.write(sample_data)

            # Read concurrently
            tasks = [backend.read(key) for _ in range(10)]
            results = await asyncio.gather(*tasks)

            # Verify all reads successful
            assert len(results) == 10
            for result in results:
                assert result.metadata.symbol == "TEST"
                assert len(result.data) == 3

    @pytest.mark.asyncio
    async def test_lock_timeout(self, sample_data):
        """Test lock acquisition timeout."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create backend with short timeout
            backend = AsyncFileSystemBackend(
                data_root=Path(tmpdir),
                lock_timeout=0.1,
                max_retries=1,
            )

            key = "equities/daily/TEST"

            # Acquire the shared lock directly from the backend's registry
            shared_lock = await backend._get_or_create_lock(key)
            await shared_lock.acquire()

            try:
                # Try to write (should timeout)
                with pytest.raises(LockAcquisitionError):
                    await backend.write(sample_data)
            finally:
                shared_lock.release()

    @pytest.mark.asyncio
    async def test_invalid_key_format(self):
        """Test handling of invalid key formats."""
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = AsyncFileSystemBackend(data_root=Path(tmpdir))

            # Invalid key format
            with pytest.raises(StorageError, match="Invalid key format"):
                await backend.read("invalid_key")

            with pytest.raises(StorageError, match="Invalid key format"):
                await backend.delete("invalid/key")

    @pytest.mark.asyncio
    async def test_read_nonexistent_key(self):
        """Test reading non-existent key."""
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = AsyncFileSystemBackend(data_root=Path(tmpdir))

            with pytest.raises(StorageError, match="Data not found"):
                await backend.read("equities/daily/NONEXISTENT")


class TestAsyncFileLock:
    """Test async file lock implementation."""

    @pytest.mark.asyncio
    async def test_lock_acquire_release(self):
        """Test lock acquisition and release."""
        with tempfile.TemporaryDirectory() as tmpdir:
            lock_path = Path(tmpdir) / "test.lock"
            lock = AsyncFileLock(lock_path)

            # Acquire lock
            assert await lock.acquire()
            assert lock_path.exists()

            # Release lock
            await lock.release()
            assert not lock_path.exists()

    @pytest.mark.asyncio
    async def test_lock_context_manager(self):
        """Test lock as context manager."""
        with tempfile.TemporaryDirectory() as tmpdir:
            lock_path = Path(tmpdir) / "test.lock"
            lock = AsyncFileLock(lock_path)

            async with lock:
                assert lock_path.exists()

            assert not lock_path.exists()

    @pytest.mark.asyncio
    async def test_lock_timeout(self):
        """Test lock timeout."""
        with tempfile.TemporaryDirectory() as tmpdir:
            lock_path = Path(tmpdir) / "test.lock"

            # Create a shared lock (simulating backend behavior)
            shared_lock = asyncio.Lock()

            lock1 = AsyncFileLock(lock_path, timeout=0.1)
            lock2 = AsyncFileLock(lock_path, timeout=0.1)

            # Replace their internal locks with the shared one
            lock1._lock = shared_lock
            lock2._lock = shared_lock

            # First lock succeeds
            await lock1.acquire()

            # Second lock should timeout
            with pytest.raises(LockAcquisitionError):
                await lock2.acquire()

            # Cleanup
            await lock1.release()


class TestStorageMigration:
    """Test storage migration utilities."""

    @pytest.mark.asyncio
    async def test_sync_to_async_migration(self, sample_data, sample_data_2):
        """Test migrating from sync to async storage."""
        with tempfile.TemporaryDirectory() as sync_dir, tempfile.TemporaryDirectory() as async_dir:
            # Create and populate sync backend
            sync_backend = FileSystemBackend(data_root=Path(sync_dir))
            sync_backend.write(sample_data)
            sync_backend.write(sample_data_2)

            # Create async backend
            async_backend = AsyncFileSystemBackend(data_root=Path(async_dir))

            # Migrate
            successful, failed = await StorageMigrator.sync_to_async(sync_backend, async_backend)

            assert successful == 2
            assert failed == 0

            # Verify data migrated
            assert await async_backend.exists("equities/daily/TEST")
            assert await async_backend.exists("equities/daily/TEST2")

            # Verify data integrity
            data1 = await async_backend.read("equities/daily/TEST")
            assert data1.metadata.symbol == "TEST"
            assert len(data1.data) == 3

    @pytest.mark.asyncio
    async def test_async_to_sync_migration(self, sample_data, sample_data_2):
        """Test migrating from async to sync storage."""
        with tempfile.TemporaryDirectory() as async_dir, tempfile.TemporaryDirectory() as sync_dir:
            # Create and populate async backend
            async_backend = AsyncFileSystemBackend(data_root=Path(async_dir))
            await async_backend.write(sample_data)
            await async_backend.write(sample_data_2)

            # Create sync backend
            sync_backend = FileSystemBackend(data_root=Path(sync_dir))

            # Migrate
            successful, failed = await StorageMigrator.async_to_sync(async_backend, sync_backend)

            assert successful == 2
            assert failed == 0

            # Verify data migrated
            assert sync_backend.exists("equities/daily/TEST")
            assert sync_backend.exists("equities/daily/TEST2")

            # Verify data integrity
            data1 = sync_backend.read("equities/daily/TEST")
            assert data1.metadata.symbol == "TEST"
            assert len(data1.data) == 3

    @pytest.mark.asyncio
    async def test_migration_with_prefix(self, sample_data, sample_data_2):
        """Test migration with prefix filter."""
        with tempfile.TemporaryDirectory() as sync_dir, tempfile.TemporaryDirectory() as async_dir:
            # Create and populate sync backend
            sync_backend = FileSystemBackend(data_root=Path(sync_dir))
            sync_backend.write(sample_data)
            sync_backend.write(sample_data_2)

            # Create async backend
            async_backend = AsyncFileSystemBackend(data_root=Path(async_dir))

            # Migrate only TEST2
            successful, failed = await StorageMigrator.sync_to_async(
                sync_backend, async_backend, prefix="equities/daily/TEST2"
            )

            assert successful == 1
            assert failed == 0

            # Verify only TEST2 migrated
            assert not await async_backend.exists("equities/daily/TEST")
            assert await async_backend.exists("equities/daily/TEST2")


class TestAsyncStorageAdapter:
    """Test async storage adapter for sync contexts."""

    def test_adapter_write_read(self, sample_data):
        """Test adapter write and read in sync context."""
        with tempfile.TemporaryDirectory() as tmpdir:
            async_backend = AsyncFileSystemBackend(data_root=Path(tmpdir))
            adapter = AsyncStorageAdapter(async_backend)

            # Write data (sync)
            key = adapter.write(sample_data)
            assert key == "equities/daily/TEST"

            # Read data (sync)
            loaded_data = adapter.read(key)
            assert loaded_data.metadata.symbol == "TEST"
            assert len(loaded_data.data) == 3

    def test_adapter_exists_delete(self, sample_data):
        """Test adapter exists and delete in sync context."""
        with tempfile.TemporaryDirectory() as tmpdir:
            async_backend = AsyncFileSystemBackend(data_root=Path(tmpdir))
            adapter = AsyncStorageAdapter(async_backend)

            key = "equities/daily/TEST"

            # Check exists
            assert not adapter.exists(key)

            # Write data
            adapter.write(sample_data)
            assert adapter.exists(key)

            # Delete data
            adapter.delete(key)
            assert not adapter.exists(key)

    def test_adapter_list_keys(self, sample_data, sample_data_2):
        """Test adapter list_keys in sync context."""
        with tempfile.TemporaryDirectory() as tmpdir:
            async_backend = AsyncFileSystemBackend(data_root=Path(tmpdir))
            adapter = AsyncStorageAdapter(async_backend)

            # Write data
            adapter.write(sample_data)
            adapter.write(sample_data_2)

            # List keys
            keys = adapter.list_keys()
            assert len(keys) == 2
            assert "equities/daily/TEST" in keys
            assert "equities/daily/TEST2" in keys


class TestFactoryFunctions:
    """Test factory functions."""

    @pytest.mark.asyncio
    async def test_create_async_backend(self, sample_data):
        """Test async backend factory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = create_async_backend(
                data_root=Path(tmpdir),
                lock_timeout=20.0,
            )

            assert isinstance(backend, AsyncFileSystemBackend)
            assert backend.lock_timeout == 20.0

            # Test it works
            key = await backend.write(sample_data)
            assert await backend.exists(key)

    def test_create_sync_adapter(self, sample_data):
        """Test sync adapter factory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            async_backend = AsyncFileSystemBackend(data_root=Path(tmpdir))
            adapter = create_sync_adapter(async_backend)

            assert isinstance(adapter, AsyncStorageAdapter)

            # Test it works
            key = adapter.write(sample_data)
            assert adapter.exists(key)
