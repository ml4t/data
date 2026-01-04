"""Test simple utility functions."""


class TestSimpleUtils:
    """Test simple utility functionality."""

    def test_rate_limit_basic(self):
        """Test rate limiter basic functionality."""
        from ml4t.data.utils.rate_limit import RateLimiter

        # Test creation
        limiter = RateLimiter(max_calls=10, period=1.0)
        assert limiter.max_calls == 10
        assert limiter.period == 1.0

        # Test initial acquisition
        acquired = limiter.acquire()
        assert isinstance(acquired, bool)

        # Test reset
        limiter.reset()
        # Should still work after reset
        acquired = limiter.acquire()
        assert isinstance(acquired, bool)

    def test_rate_limit_non_blocking(self):
        """Test rate limiter non-blocking functionality."""
        from ml4t.data.utils.rate_limit import RateLimiter

        # Test non-blocking acquisition
        limiter = RateLimiter(max_calls=1, period=1.0)

        # First acquisition should succeed
        acquired = limiter.acquire(blocking=False)
        assert acquired is True

        # Second acquisition should fail (non-blocking)
        acquired = limiter.acquire(blocking=False)
        assert acquired is False

    def test_locking_basic(self):
        """Test file locking basic functionality."""
        from ml4t.data.utils.locking import FileLock

        # Test basic creation
        lock = FileLock(timeout=1.0)
        assert lock.timeout == 1.0

        # Test lock exists
        assert hasattr(lock, "acquire")
        assert callable(lock.acquire)
