"""Tests for async rate limiting utilities."""

import asyncio
import time

import pytest

from ml4t.data.utils.async_rate_limit import (
    AsyncAdaptiveRateLimiter,
    AsyncRateLimiter,
    MultiProviderRateLimiter,
)

# Mark all tests in this module as asyncio
pytestmark = [pytest.mark.asyncio]


@pytest.mark.asyncio
async def test_async_rate_limiter_basic():
    """Test basic async rate limiter functionality."""
    limiter = AsyncRateLimiter(max_calls=3, period=1.0)

    # Should allow 3 calls immediately
    assert await limiter.acquire(blocking=False) is True
    assert await limiter.acquire(blocking=False) is True
    assert await limiter.acquire(blocking=False) is True

    # 4th call should be blocked
    assert await limiter.acquire(blocking=False) is False

    # Wait for window to expire
    await asyncio.sleep(1.1)

    # Should allow calls again
    assert await limiter.acquire(blocking=False) is True


@pytest.mark.asyncio
async def test_async_rate_limiter_blocking():
    """Test blocking behavior of async rate limiter."""
    limiter = AsyncRateLimiter(max_calls=2, period=0.5)

    # Fill up the limiter
    await limiter.acquire()
    await limiter.acquire()

    # This should block for ~0.5 seconds
    start = time.time()
    result = await limiter.acquire(blocking=True)
    elapsed = time.time() - start

    assert result is True
    assert 0.4 < elapsed < 0.7  # Allow some timing variance


@pytest.mark.asyncio
async def test_async_rate_limiter_burst():
    """Test burst functionality."""
    limiter = AsyncRateLimiter(max_calls=5, period=2.0, burst=3)

    # Burst limit is 3, even though max_calls is 5
    assert await limiter.acquire(blocking=False) is True
    assert await limiter.acquire(blocking=False) is True
    assert await limiter.acquire(blocking=False) is True
    assert await limiter.acquire(blocking=False) is False

    # Remaining calls should be 0
    assert limiter.remaining_calls() == 0


@pytest.mark.asyncio
async def test_async_rate_limiter_reset():
    """Test reset functionality."""
    limiter = AsyncRateLimiter(max_calls=2, period=1.0)

    # Use up calls
    await limiter.acquire()
    await limiter.acquire()
    assert await limiter.acquire(blocking=False) is False

    # Reset should clear the calls
    await limiter.reset()
    assert await limiter.acquire(blocking=False) is True
    assert await limiter.acquire(blocking=False) is True


@pytest.mark.asyncio
async def test_adaptive_rate_limiter():
    """Test adaptive rate limiter."""
    limiter = AsyncAdaptiveRateLimiter(max_calls=10, period=1.0, backoff_factor=2.0)

    initial_max = limiter.max_calls
    assert initial_max == 10

    # Simulate rate limit response
    await limiter.handle_rate_limit_response(retry_after=0.1)

    # Should have reduced max_calls
    assert limiter.max_calls == 5
    assert limiter.is_backed_off is True

    # Restore rate
    await limiter.restore_rate()
    assert limiter.max_calls == initial_max
    assert limiter.is_backed_off is False


@pytest.mark.asyncio
async def test_multi_provider_rate_limiter():
    """Test multi-provider rate limiter."""
    multi_limiter = MultiProviderRateLimiter()

    # Add providers with different limits
    await multi_limiter.add_provider("yahoo", max_calls=100, period=60)
    await multi_limiter.add_provider("binance", max_calls=1200, period=60)
    await multi_limiter.add_provider("cryptocompare", max_calls=50, period=60, adaptive=True)

    # Test acquiring for different providers
    assert await multi_limiter.acquire("yahoo", blocking=False) is True
    assert await multi_limiter.acquire("binance", blocking=False) is True
    assert await multi_limiter.acquire("cryptocompare", blocking=False) is True

    # Test unknown provider
    with pytest.raises(KeyError):
        await multi_limiter.acquire("unknown", blocking=False)

    # Test status
    status = multi_limiter.get_status()
    assert "yahoo" in status
    assert "binance" in status
    assert "cryptocompare" in status

    assert status["yahoo"]["remaining_calls"] == 99
    assert status["binance"]["remaining_calls"] == 1199
    assert status["cryptocompare"]["remaining_calls"] == 49

    # Test reset
    await multi_limiter.reset("yahoo")
    status = multi_limiter.get_status()
    assert status["yahoo"]["remaining_calls"] == 100

    # Test reset all
    await multi_limiter.reset_all()
    status = multi_limiter.get_status()
    assert status["binance"]["remaining_calls"] == 1200


@pytest.mark.asyncio
async def test_concurrent_access():
    """Test concurrent access to rate limiter."""
    limiter = AsyncRateLimiter(max_calls=5, period=1.0)

    async def make_call():
        return await limiter.acquire(blocking=False)

    # Launch 10 concurrent calls
    results = await asyncio.gather(*[make_call() for _ in range(10)])

    # Only 5 should succeed
    successful = sum(1 for r in results if r)
    assert successful == 5

    # Remaining should be 0
    assert limiter.remaining_calls() == 0


@pytest.mark.asyncio
async def test_time_until_reset():
    """Test time until reset calculation."""
    limiter = AsyncRateLimiter(max_calls=2, period=1.0)

    # Initially should be 0
    assert limiter.time_until_reset() == 0.0

    # After acquiring
    await limiter.acquire()
    time_remaining = limiter.time_until_reset()
    assert 0.9 < time_remaining <= 1.0

    # Wait a bit
    await asyncio.sleep(0.5)
    time_remaining = limiter.time_until_reset()
    assert 0.4 < time_remaining <= 0.5


@pytest.mark.asyncio
async def test_async_rate_limiter_remaining_calls():
    """Test remaining_calls method."""
    limiter = AsyncRateLimiter(max_calls=5, period=1.0)

    # Initially all calls available
    assert limiter.remaining_calls() == 5

    # After one call
    await limiter.acquire()
    assert limiter.remaining_calls() == 4

    # After more calls
    await limiter.acquire()
    await limiter.acquire()
    assert limiter.remaining_calls() == 2


@pytest.mark.asyncio
async def test_adaptive_rate_limiter_remaining_zero():
    """Test adaptive limiter handles remaining=0."""
    limiter = AsyncAdaptiveRateLimiter(max_calls=10, period=1.0)

    # Simulate exhausted API limit
    await limiter.handle_rate_limit_response(remaining=0)

    # Calls should be cleared
    assert limiter.remaining_calls() == 10  # Cleared so all available


@pytest.mark.asyncio
async def test_adaptive_rate_limiter_no_double_backoff():
    """Test that adaptive limiter doesn't back off twice."""
    limiter = AsyncAdaptiveRateLimiter(max_calls=10, period=1.0, backoff_factor=2.0)

    # First rate limit
    await limiter.handle_rate_limit_response(retry_after=0.1)
    assert limiter.max_calls == 5
    assert limiter.is_backed_off is True

    # Second rate limit should not reduce further
    await limiter.handle_rate_limit_response(retry_after=0.1)
    assert limiter.max_calls == 5  # Unchanged because already backed off


@pytest.mark.asyncio
async def test_multi_provider_add_adaptive():
    """Test adding adaptive provider."""
    multi_limiter = MultiProviderRateLimiter()

    await multi_limiter.add_provider(
        "test_provider",
        max_calls=10,
        period=1.0,
        adaptive=True,
    )

    # Verify it's an adaptive limiter
    assert isinstance(
        multi_limiter.limiters["test_provider"],
        AsyncAdaptiveRateLimiter,
    )


@pytest.mark.asyncio
async def test_multi_provider_reset_nonexistent():
    """Test reset with non-existent provider does nothing."""
    multi_limiter = MultiProviderRateLimiter()

    # Should not raise
    await multi_limiter.reset("nonexistent")


@pytest.mark.asyncio
async def test_rate_limiter_expired_calls_removed():
    """Test that expired calls are properly removed from window."""
    limiter = AsyncRateLimiter(max_calls=2, period=0.3)

    # Fill up
    await limiter.acquire()
    await limiter.acquire()
    assert await limiter.acquire(blocking=False) is False

    # Wait for calls to expire
    await asyncio.sleep(0.4)

    # Now we should have capacity again
    assert limiter.remaining_calls() == 2
    assert await limiter.acquire(blocking=False) is True


@pytest.mark.asyncio
async def test_rate_limiter_with_default_burst():
    """Test that burst defaults to max_calls."""
    limiter = AsyncRateLimiter(max_calls=5, period=1.0)
    assert limiter.burst == 5


@pytest.mark.asyncio
async def test_adaptive_limiter_restore_when_not_backed_off():
    """Test restore_rate when not backed off."""
    limiter = AsyncAdaptiveRateLimiter(max_calls=10, period=1.0)

    # Should do nothing if not backed off
    assert limiter.is_backed_off is False
    await limiter.restore_rate()
    assert limiter.max_calls == 10  # Unchanged


@pytest.mark.asyncio
async def test_multi_provider_get_status_includes_all_info():
    """Test that status includes all required fields."""
    multi_limiter = MultiProviderRateLimiter()

    await multi_limiter.add_provider("provider1", max_calls=100, period=60)
    await multi_limiter.acquire("provider1")

    status = multi_limiter.get_status()

    assert "provider1" in status
    assert "remaining_calls" in status["provider1"]
    assert "time_until_reset" in status["provider1"]
    assert "max_calls" in status["provider1"]
    assert "period" in status["provider1"]
    assert status["provider1"]["max_calls"] == 100
    assert status["provider1"]["period"] == 60


@pytest.mark.asyncio
async def test_rate_limiter_min_calculation():
    """Test rate limiter uses min of max_calls and burst."""
    # burst < max_calls
    limiter1 = AsyncRateLimiter(max_calls=10, period=1.0, burst=3)
    for _ in range(3):
        assert await limiter1.acquire(blocking=False) is True
    assert await limiter1.acquire(blocking=False) is False

    # max_calls < burst
    limiter2 = AsyncRateLimiter(max_calls=2, period=1.0, burst=10)
    for _ in range(2):
        assert await limiter2.acquire(blocking=False) is True
    assert await limiter2.acquire(blocking=False) is False
