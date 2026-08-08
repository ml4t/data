"""File locking utilities for concurrent access safety."""

from __future__ import annotations

import time
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path

import structlog
from filelock import FileLock as FileSystemLock
from filelock import Timeout

logger = structlog.get_logger()


class FileLock:
    """
    Thread-safe and process-safe file locking mechanism.

    Uses filelock library for cross-platform compatibility.
    Ensures safe concurrent access to data files.
    """

    def __init__(
        self,
        timeout: float = 30.0,
        retry_delay: float = 0.1,
    ) -> None:
        """
        Initialize file lock manager.

        Args:
            timeout: Maximum time to wait for lock acquisition (seconds)
            retry_delay: Delay between lock acquisition attempts (seconds)
        """
        self.timeout = timeout
        self.retry_delay = retry_delay
        self._locks: dict[str, FileSystemLock] = {}

    def _get_lock_path(self, file_path: Path) -> Path:
        """Get the lock file path for a given file."""
        return file_path.with_suffix(file_path.suffix + ".lock")

    def _get_lock(self, file_path: Path) -> FileSystemLock:
        """Get or create a lock object for a file."""
        lock_path = self._get_lock_path(file_path)
        lock_key = str(lock_path)

        if lock_key not in self._locks:
            # Ensure lock directory exists
            lock_path.parent.mkdir(parents=True, exist_ok=True)
            self._locks[lock_key] = FileSystemLock(
                lock_path,
                timeout=self.timeout,
            )

        return self._locks[lock_key]

    @contextmanager
    def acquire(
        self,
        file_path: Path,
        exclusive: bool = True,
        blocking: bool = True,
    ) -> Generator[None, None, None]:
        """
        Acquire a file lock.

        Args:
            file_path: Path to the file to lock
            exclusive: If True, acquire exclusive lock; if False, shared lock
            blocking: If True, wait for lock; if False, raise immediately if locked

        Yields:
            None when lock is acquired

        Raises:
            Timeout: If lock cannot be acquired within timeout
        """
        lock = self._get_lock(file_path)
        lock_type = "exclusive" if exclusive else "shared"

        logger.debug(
            f"Attempting to acquire {lock_type} lock",
            file=str(file_path),
            timeout=self.timeout,
        )

        start_time = time.time()

        try:
            # filelock doesn't support shared locks directly,
            # but we can use it for exclusive locking
            if blocking:
                lock.acquire(timeout=self.timeout, poll_interval=self.retry_delay)
            else:
                lock.acquire(timeout=0)

            elapsed = time.time() - start_time
            logger.debug(
                f"Acquired {lock_type} lock",
                file=str(file_path),
                elapsed=f"{elapsed:.2f}s",
            )

            yield

        except Timeout:
            logger.error(
                f"Failed to acquire {lock_type} lock within timeout",
                file=str(file_path),
                timeout=self.timeout,
            )
            raise

        finally:
            if lock.is_locked:
                lock.release()
                logger.debug(
                    f"Released {lock_type} lock",
                    file=str(file_path),
                )

    def is_locked(self, file_path: Path) -> bool:
        """
        Check if a file is currently locked.

        Args:
            file_path: Path to check

        Returns:
            True if file is locked, False otherwise
        """
        lock = self._get_lock(file_path)
        return bool(lock.is_locked)

    def cleanup(self) -> None:
        """Clean up all lock files."""
        for _lock_key, lock in self._locks.items():
            if lock.is_locked:
                lock.release()
            # Try to remove lock file
            lock_path = Path(lock.lock_file)
            try:
                lock_path.unlink()
            except FileNotFoundError:
                pass
            except Exception as e:
                logger.warning(
                    "Failed to remove lock file",
                    lock_file=str(lock_path),
                    error=str(e),
                )

        self._locks.clear()


# Global lock manager instance
_lock_manager: FileLock | None = None


def get_lock_manager() -> FileLock:
    """Get the global lock manager instance."""
    global _lock_manager
    if _lock_manager is None:
        _lock_manager = FileLock()
    return _lock_manager


@contextmanager
def file_lock(
    file_path: Path,
    exclusive: bool = True,
    blocking: bool = True,
    timeout: float = 30.0,
) -> Generator[None, None, None]:
    """
    Convenience function to acquire a file lock.

    Args:
        file_path: Path to the file to lock
        exclusive: If True, acquire exclusive lock
        blocking: If True, wait for lock
        timeout: Maximum time to wait for lock

    Yields:
        None when lock is acquired
    """
    manager = get_lock_manager()
    manager.timeout = timeout
    with manager.acquire(file_path, exclusive=exclusive, blocking=blocking):
        yield
