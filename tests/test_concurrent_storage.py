"""Tests for concurrent storage operations and file locking."""

import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import polars as pl
import pytest

from ml4t.data.core.models import DataObject, Metadata
from ml4t.data.storage.filesystem import FileSystemBackend, LockAcquisitionError


@pytest.fixture
def temp_storage():
    """Create a temporary storage backend for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield FileSystemBackend(
            data_root=Path(tmpdir),
            lock_timeout=2.0,  # Shorter timeout for tests
            max_retries=3,
            retry_base_delay=0.05,  # Shorter delays for tests
        )


@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    from datetime import datetime

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
        frequency="daily",
        schema_version="1.0",
    )

    return DataObject(data=df, metadata=metadata)


class TestConcurrentAccess:
    """Test concurrent access scenarios."""

    def test_concurrent_writes_same_key(self, temp_storage, sample_data):
        """Test that concurrent writes to the same key are serialized."""
        key = None
        write_times = []
        errors = []

        def write_data(index):
            try:
                # Modify data slightly to distinguish writes
                # Change the volume to identify which writer succeeded
                data_copy = DataObject(
                    data=sample_data.data.with_columns(pl.col("volume") * (index + 1)),
                    metadata=sample_data.metadata,
                )

                start = time.time()
                nonlocal key
                key = temp_storage.write(data_copy)
                write_times.append((index, time.time() - start))
            except Exception as e:
                errors.append((index, str(e)))

        # Launch 5 concurrent write attempts
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(write_data, i) for i in range(5)]
            for future in futures:
                future.result()

        # All writes should complete without errors (serialized by locks)
        assert len(errors) == 0, f"Errors occurred: {errors}"
        assert len(write_times) == 5

        # Verify the data was written (last writer wins)
        result = temp_storage.read(key)
        assert result is not None
        assert len(result.data) == 3  # Original row count preserved

        # Check that writes were serialized (took some time)
        total_time = sum(t for _, t in write_times)
        assert total_time > 0.01  # Should take measurable time due to serialization

    def test_concurrent_read_write(self, temp_storage, sample_data):
        """Test concurrent reads and writes don't corrupt data."""
        key = temp_storage.write(sample_data)

        read_results = []
        write_results = []
        errors = []

        def read_data(index):
            try:
                for _ in range(3):  # Multiple reads
                    result = temp_storage.read(key)
                    read_results.append((index, len(result.data)))
                    time.sleep(0.01)  # Small delay
            except Exception as e:
                errors.append(("read", index, str(e)))

        def write_data(index):
            try:
                # Modify and write - use volume to track changes
                data_copy = DataObject(
                    data=sample_data.data.with_columns(pl.col("volume") * (index + 10)),
                    metadata=sample_data.metadata,
                )
                temp_storage.write(data_copy)
                write_results.append(index)
            except Exception as e:
                errors.append(("write", index, str(e)))

        # Mix reads and writes
        with ThreadPoolExecutor(max_workers=8) as executor:
            # 5 readers, 3 writers
            read_futures = [executor.submit(read_data, i) for i in range(5)]
            write_futures = [executor.submit(write_data, i) for i in range(3)]

            for future in read_futures + write_futures:
                future.result()

        # All operations should complete
        assert len(errors) == 0, f"Errors occurred: {errors}"
        assert len(read_results) == 15  # 5 readers * 3 reads each
        assert len(write_results) == 3

        # Verify data integrity
        final_data = temp_storage.read(key)
        assert final_data is not None
        assert len(final_data.data) == 3  # Original row count preserved

    def test_lock_timeout_handling(self, temp_storage, sample_data):
        """Test proper handling of lock timeout."""
        # Create a backend with very short timeout
        storage = FileSystemBackend(
            data_root=temp_storage.data_root,
            lock_timeout=0.1,  # Very short timeout
            max_retries=1,  # Single retry
            retry_base_delay=0.05,
        )

        key = storage.write(sample_data)
        lock_path = storage._get_lock_path(key)

        # Manually acquire lock to block operations
        import filelock

        blocking_lock = filelock.FileLock(lock_path)
        blocking_lock.acquire()

        try:
            # This should timeout and raise LockAcquisitionError
            with pytest.raises(LockAcquisitionError) as exc_info:
                storage.read(key)

            # Check the new LockError message format
            assert "Could not acquire lock for key" in str(exc_info.value)
            assert "within 0.1 seconds" in str(exc_info.value)

        finally:
            blocking_lock.release()

    def test_retry_mechanism(self, temp_storage, sample_data):
        """Test that retry mechanism works with temporary lock contention."""
        key = temp_storage.write(sample_data)
        lock_path = temp_storage._get_lock_path(key)

        # Track retry attempts

        def temporary_block():
            """Hold lock briefly then release."""
            import filelock

            lock = filelock.FileLock(lock_path)
            lock.acquire()
            time.sleep(0.2)  # Hold for 200ms
            lock.release()

        # Start blocking thread
        blocker = threading.Thread(target=temporary_block)
        blocker.start()

        # Give blocker time to acquire lock
        time.sleep(0.05)

        # This should retry and eventually succeed
        start = time.time()
        result = temp_storage.read(key)
        elapsed = time.time() - start

        blocker.join()

        # Should have succeeded after retry
        assert result is not None
        assert len(result.data) == 3
        # Should have taken more than initial attempt but less than full timeout
        # Allow for timing variance in CI environments
        assert 0.05 < elapsed < temp_storage.lock_timeout

    def test_delete_with_concurrent_reads(self, temp_storage, sample_data):
        """Test deletion while reads are in progress."""
        key = temp_storage.write(sample_data)

        read_count = 0
        delete_done = False
        errors = []

        def read_loop():
            nonlocal read_count
            try:
                while not delete_done:
                    try:
                        temp_storage.read(key)
                        read_count += 1
                    except FileNotFoundError:
                        # Expected after deletion
                        break
                    time.sleep(0.01)
            except Exception as e:
                errors.append(("read", str(e)))

        def delete_data():
            nonlocal delete_done
            try:
                time.sleep(0.1)  # Let some reads happen first
                temp_storage.delete(key)
                delete_done = True
            except Exception as e:
                errors.append(("delete", str(e)))

        # Start readers and deleter
        with ThreadPoolExecutor(max_workers=4) as executor:
            read_futures = [executor.submit(read_loop) for _ in range(3)]
            delete_future = executor.submit(delete_data)

            delete_future.result()
            time.sleep(0.1)  # Give readers time to notice
            delete_done = True  # Ensure readers stop

            for future in read_futures:
                future.result()

        # Delete should succeed without errors
        assert not any(e[0] == "delete" for e in errors)
        assert read_count > 0  # Some reads should have succeeded

        # Verify deletion
        assert not temp_storage.exists(key)


class TestAtomicOperations:
    """Test atomic write operations."""

    def test_atomic_write_on_failure(self, temp_storage):
        """Test that failed writes don't leave partial data."""
        import os
        from datetime import datetime

        from ml4t.data.storage.filesystem import StorageError  # Use the filesystem StorageError

        # Create test data
        df = pl.DataFrame(
            {
                "timestamp": [datetime(2024, 1, 1)],
                "open": [100.0],
                "high": [105.0],
                "low": [99.0],
                "close": [104.0],
                "volume": [1000000],
            }
        )
        metadata = Metadata(
            provider="test",
            symbol="FAIL",
            asset_class="equities",
            frequency="daily",
            schema_version="1.0",
        )
        data = DataObject(data=df, metadata=metadata)

        # Monkey-patch os.replace to fail
        original_replace = os.replace
        os.replace = lambda src, dst: exec('raise OSError("Simulated write failure")')

        # Test the write fails appropriately
        exception_raised = False
        try:
            temp_storage.write(data)
        except StorageError:
            exception_raised = True
        except Exception as e:
            pytest.fail(f"Wrong exception type raised: {type(e).__name__}: {e}")
        finally:
            os.replace = original_replace

        # Verify exception was raised
        assert exception_raised, "Expected StorageError to be raised"

        # Verify no partial files remain
        key = f"{metadata.asset_class}/{metadata.frequency}/{metadata.symbol}"
        assert not temp_storage.exists(key)
        temp_files = list(temp_storage.data_root.rglob("*.tmp"))
        assert len(temp_files) == 0


class TestLockCleanup:
    """Test proper lock cleanup."""

    def test_lock_cleanup_on_exception(self, temp_storage, sample_data):
        """Test that locks are released even when operations fail."""
        key = temp_storage.write(sample_data)

        # Monkey-patch to cause exception during read
        original_read = pl.read_parquet

        def failing_read(path):
            raise ValueError("Simulated read failure")

        pl.read_parquet = failing_read

        try:
            with pytest.raises(ValueError):
                temp_storage.read(key)

            # Lock should be released, so another operation should work
            pl.read_parquet = original_read  # Restore for next operation
            result = temp_storage.read(key)
            assert result is not None

        finally:
            pl.read_parquet = original_read

    def test_no_orphaned_locks(self, temp_storage, sample_data):
        """Test that operations don't leave orphaned lock files."""
        # Perform various operations
        key1 = temp_storage.write(sample_data)
        temp_storage.read(key1)

        sample_data.metadata.symbol = "TEST2"
        key2 = temp_storage.write(sample_data)
        temp_storage.delete(key2)

        # Check for orphaned locks (locked files)
        lock_files = list(temp_storage.locks_dir.glob("*.lock"))

        # Lock files may exist but shouldn't be locked
        import filelock

        for lock_file in lock_files:
            lock = filelock.FileLock(lock_file)
            # Should be able to acquire immediately
            try:
                lock.acquire(timeout=0)
                lock.release()
            except filelock.Timeout:
                pytest.fail(f"Lock file {lock_file} is still locked")
