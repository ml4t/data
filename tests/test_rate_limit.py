"""Tests for rate limiting utilities."""

import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from ml4t.data.utils.rate_limit import RateLimiter

# Mark all tests in this module as slow (they use actual sleep/waits)
pytestmark = pytest.mark.slow


class TestRateLimiter:
    """Test rate limiter functionality."""

    def test_basic_rate_limiting(self) -> None:
        """Test basic rate limiting behavior."""
        # Allow 3 calls per second
        limiter = RateLimiter(max_calls=3, period=1.0)

        # First 3 calls should succeed immediately
        start = time.time()
        for _ in range(3):
            assert limiter.acquire(blocking=False)

        # 4th call should fail without blocking
        assert not limiter.acquire(blocking=False)

        # Should complete in less than 0.1 seconds
        assert time.time() - start < 0.1

    def test_blocking_acquire(self) -> None:
        """Test blocking acquire waits appropriately."""
        # Allow 2 calls per 0.5 seconds
        limiter = RateLimiter(max_calls=2, period=0.5)

        start = time.time()

        # Make 2 immediate calls
        assert limiter.acquire()
        assert limiter.acquire()

        # 3rd call should block for ~0.5 seconds
        assert limiter.acquire(blocking=True)

        elapsed = time.time() - start
        # Should have waited approximately 0.5 seconds
        assert 0.4 < elapsed < 0.7

    def test_burst_allowance(self) -> None:
        """Test burst allowance configuration."""
        # Allow 5 calls per 2 seconds, with burst of 3
        limiter = RateLimiter(max_calls=5, period=2.0, burst=3)

        # Can only store 3 calls at once due to burst limit
        for _ in range(3):
            assert limiter.acquire(blocking=False)

        # 4th call fails even though we're under max_calls
        assert not limiter.acquire(blocking=False)

    def test_thread_safety(self) -> None:
        """Test rate limiter is thread-safe."""
        # Allow 10 calls per second
        limiter = RateLimiter(max_calls=10, period=1.0)

        successful_calls = []

        def make_call() -> bool:
            if limiter.acquire(blocking=False):
                successful_calls.append(1)
                return True
            return False

        # Try to make 20 calls from 4 threads
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(make_call) for _ in range(20)]
            results = [f.result() for f in futures]

        # Should have exactly 10 successful calls
        assert len(successful_calls) == 10
        assert sum(results) == 10

    def test_reset(self) -> None:
        """Test resetting the rate limiter."""
        limiter = RateLimiter(max_calls=2, period=1.0)

        # Use up the limit
        assert limiter.acquire()
        assert limiter.acquire()
        assert not limiter.acquire(blocking=False)

        # Reset should clear the history
        limiter.reset()

        # Should be able to make calls again
        assert limiter.acquire()
        assert limiter.acquire()

    def test_old_calls_cleanup(self) -> None:
        """Test that old calls are cleaned up properly."""
        limiter = RateLimiter(max_calls=2, period=0.2)

        # Make 2 calls
        assert limiter.acquire()
        assert limiter.acquire()

        # Wait for period to expire
        time.sleep(0.25)

        # Should be able to make new calls
        assert limiter.acquire(blocking=False)
        assert limiter.acquire(blocking=False)
