"""Tests for file locking utilities."""

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from filelock import Timeout

from ml4t.data.utils.locking import FileLock, file_lock, get_lock_manager


class TestFileLock:
    """Test file locking functionality."""

    def test_basic_lock_acquisition(self, tmp_path: Path) -> None:
        """Test basic lock acquisition and release."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("test data")

        lock_manager = FileLock()

        # Acquire lock
        with lock_manager.acquire(test_file):
            # Check lock is held
            assert lock_manager.is_locked(test_file)

        # Check lock is released
        assert not lock_manager.is_locked(test_file)

    def test_exclusive_lock_blocks_other_locks(self, tmp_path: Path) -> None:
        """Test that exclusive lock blocks other lock attempts."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("test data")

        lock_manager1 = FileLock(timeout=0.5)
        lock_manager2 = FileLock(timeout=0.5)

        # Acquire first lock
        with lock_manager1.acquire(test_file):
            # Try to acquire second lock with different manager (should timeout)
            with pytest.raises(Timeout):
                with lock_manager2.acquire(test_file, blocking=True):
                    pass

    def test_non_blocking_lock(self, tmp_path: Path) -> None:
        """Test non-blocking lock acquisition."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("test data")

        lock_manager1 = FileLock()
        lock_manager2 = FileLock()

        # Acquire first lock
        with lock_manager1.acquire(test_file):
            # Try non-blocking acquisition with different manager (should fail immediately)
            with pytest.raises(Timeout):
                with lock_manager2.acquire(test_file, blocking=False):
                    pass

    def test_concurrent_access_with_locks(self, tmp_path: Path) -> None:
        """Test concurrent file access with proper locking."""
        test_file = tmp_path / "counter.txt"
        test_file.write_text("0")

        lock_manager = FileLock()
        results = []

        def increment_counter() -> int:
            """Read, increment, and write counter with locking."""
            with lock_manager.acquire(test_file):
                # Read current value
                current = int(test_file.read_text())
                # Simulate some processing
                time.sleep(0.01)
                # Write incremented value
                new_value = current + 1
                test_file.write_text(str(new_value))
                results.append(new_value)
                return new_value

        # Run concurrent increments
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(increment_counter) for _ in range(10)]
            [f.result() for f in futures]

        # Check final value is correct
        final_value = int(test_file.read_text())
        assert final_value == 10

        # Check all increments were sequential
        results.sort()
        assert results == list(range(1, 11))

    def test_lock_cleanup(self, tmp_path: Path) -> None:
        """Test lock cleanup removes lock files."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("test data")

        lock_manager = FileLock()

        # Acquire and release lock
        with lock_manager.acquire(test_file):
            pass

        # Check lock file exists
        lock_file = test_file.with_suffix(".txt.lock")
        assert lock_file.exists()

        # Clean up
        lock_manager.cleanup()

        # Check lock file is removed
        assert not lock_file.exists()

    def test_global_lock_manager(self, tmp_path: Path) -> None:
        """Test global lock manager singleton."""
        manager1 = get_lock_manager()
        manager2 = get_lock_manager()

        # Should be the same instance
        assert manager1 is manager2

    def test_file_lock_convenience_function(self, tmp_path: Path) -> None:
        """Test the convenience file_lock function."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("test data")

        # Use convenience function
        with file_lock(test_file):
            # Check lock is held
            manager = get_lock_manager()
            assert manager.is_locked(test_file)

    def test_lock_timeout_configuration(self, tmp_path: Path) -> None:
        """Test configurable lock timeout."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("test data")

        # Create manager with short timeout
        lock_manager = FileLock(timeout=0.1)

        # Hold lock in thread
        def hold_lock() -> None:
            with lock_manager.acquire(test_file):
                time.sleep(0.5)

        thread = threading.Thread(target=hold_lock)
        thread.start()

        # Wait a bit for thread to acquire lock
        time.sleep(0.05)

        # Try to acquire (should timeout quickly)
        start = time.time()
        with pytest.raises(Timeout), lock_manager.acquire(test_file):
            pass
        elapsed = time.time() - start

        # Should timeout around 0.1 seconds
        assert 0.05 < elapsed < 0.2

        thread.join()

    def test_lock_with_missing_directory(self, tmp_path: Path) -> None:
        """Test lock creation when directory doesn't exist."""
        test_file = tmp_path / "subdir" / "nested" / "test.txt"

        lock_manager = FileLock()

        # Should create lock directory automatically
        with lock_manager.acquire(test_file):
            lock_file = test_file.with_suffix(".txt.lock")
            assert lock_file.exists()

    def test_reentrant_lock_behavior(self, tmp_path: Path) -> None:
        """Test reentrant locking behavior."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("test data")

        lock_manager = FileLock(timeout=0.1)

        # Acquire lock
        with lock_manager.acquire(test_file):
            # filelock supports reentrant locking in same thread
            # This should succeed (not timeout)
            with lock_manager.acquire(test_file):
                assert lock_manager.is_locked(test_file)
