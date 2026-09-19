"""Async rate limiting utilities for ml4t-data."""

from __future__ import annotations

import asyncio
import time
from collections import deque
from typing import Any

import structlog

logger = structlog.get_logger()


class AsyncRateLimiter:
    """
    Async-friendly rate limiter using token bucket algorithm.

    This implementation is designed for use in async contexts,
    using asyncio.sleep() instead of blocking time.sleep().
    """

    def __init__(
        self,
        max_calls: int,
        period: float,
        burst: int | None = None,
    ) -> None:
        """
        Initialize async rate limiter.

        Args:
            max_calls: Maximum number of calls allowed
            period: Time period in seconds
            burst: Optional burst allowance (defaults to max_calls)
        """
        self.max_calls = max_calls
        self.period = period
        self.burst = burst or max_calls
        self.calls: deque[float] = deque(maxlen=self.burst)
        self.lock = asyncio.Lock()

    async def acquire(self, blocking: bool = True) -> bool:
        """
        Acquire permission to make a call (async).

        Args:
            blocking: If True, wait until permission is available

        Returns:
            True if permission was acquired, False otherwise
        """
        while True:
            async with self.lock:
                now = time.time()

                # Remove old calls outside the time window
                cutoff = now - self.period
                while self.calls and self.calls[0] <= cutoff:
                    self.calls.popleft()

                # Check if we can make a call
                # Limited by both max_calls and burst size
                if len(self.calls) < min(self.max_calls, self.burst):
                    self.calls.append(now)
                    return True

                if not blocking:
                    return False

                # Calculate wait time
                wait_time = self.period - (now - self.calls[0]) if self.calls else 0

            # Wait outside the lock using async sleep
            if wait_time > 0:
                logger.debug(
                    "Rate limit reached, waiting",
                    wait_seconds=f"{wait_time:.2f}",
                    calls_in_window=len(self.calls),
                )
                await asyncio.sleep(wait_time)
            else:
                # Yield control to allow other tasks to run
                await asyncio.sleep(0)

    async def reset(self) -> None:
        """Reset the rate limiter."""
        async with self.lock:
            self.calls.clear()

    def remaining_calls(self) -> int:
        """
        Get the number of remaining calls available.

        Returns:
            Number of calls that can be made immediately
        """
        now = time.time()
        cutoff = now - self.period

        # Count valid calls in the window
        valid_calls = sum(1 for call_time in self.calls if call_time > cutoff)
        return max(0, min(self.max_calls, self.burst) - valid_calls)

    def time_until_reset(self) -> float:
        """
        Get time until the oldest call expires.

        Returns:
            Seconds until a slot becomes available (0 if slots available)
        """
        if not self.calls:
            return 0.0

        now = time.time()
        cutoff = now - self.period

        # Find oldest call still in window
        for call_time in self.calls:
            if call_time > cutoff:
                return max(0, self.period - (now - call_time))

        return 0.0


class AsyncAdaptiveRateLimiter(AsyncRateLimiter):
    """
    Adaptive async rate limiter that adjusts based on response headers.

    This version can dynamically adjust its limits based on rate limit
    headers returned by APIs (e.g., X-RateLimit-Remaining).
    """

    def __init__(
        self,
        max_calls: int,
        period: float,
        burst: int | None = None,
        backoff_factor: float = 2.0,
    ) -> None:
        """
        Initialize adaptive async rate limiter.

        Args:
            max_calls: Initial maximum number of calls allowed
            period: Time period in seconds
            burst: Optional burst allowance (defaults to max_calls)
            backoff_factor: Factor to reduce rate when limited
        """
        super().__init__(max_calls, period, burst)
        self.initial_max_calls = max_calls
        self.backoff_factor = backoff_factor
        self.is_backed_off = False

    async def handle_rate_limit_response(
        self,
        retry_after: float | None = None,
        remaining: int | None = None,
    ) -> None:
        """
        Handle rate limit response from API.

        Args:
            retry_after: Seconds to wait before retrying (from Retry-After header)
            remaining: Number of remaining calls (from X-RateLimit-Remaining header)
        """
        async with self.lock:
            if retry_after is not None:
                # Clear current calls and wait
                self.calls.clear()
                logger.warning(
                    "Rate limited by API",
                    retry_after=retry_after,
                    current_max_calls=self.max_calls,
                )

                # Reduce rate temporarily
                if not self.is_backed_off:
                    self.max_calls = max(1, int(self.max_calls / self.backoff_factor))
                    self.is_backed_off = True
                    logger.info(
                        "Reduced rate limit",
                        new_max_calls=self.max_calls,
                    )

                # Wait for the specified time
                await asyncio.sleep(retry_after)

            elif remaining is not None and remaining == 0:
                # We've exhausted our limit
                self.calls.clear()
                logger.warning("API rate limit exhausted", remaining=0)

    async def restore_rate(self) -> None:
        """Restore the rate limit to its original value."""
        async with self.lock:
            if self.is_backed_off:
                self.max_calls = self.initial_max_calls
                self.is_backed_off = False
                logger.info(
                    "Restored rate limit",
                    max_calls=self.max_calls,
                )


class MultiProviderRateLimiter:
    """
    Manage multiple rate limiters for different providers.

    This allows each provider to have its own rate limits while
    sharing a common interface.
    """

    def __init__(self) -> None:
        """Initialize multi-provider rate limiter."""
        self.limiters: dict[str, AsyncRateLimiter] = {}
        self.lock = asyncio.Lock()

    async def add_provider(
        self,
        provider: str,
        max_calls: int,
        period: float,
        burst: int | None = None,
        adaptive: bool = False,
    ) -> None:
        """
        Add a rate limiter for a provider.

        Args:
            provider: Provider name
            max_calls: Maximum calls allowed
            period: Time period in seconds
            burst: Optional burst allowance
            adaptive: Whether to use adaptive rate limiting
        """
        async with self.lock:
            limiter: AsyncRateLimiter
            if adaptive:
                limiter = AsyncAdaptiveRateLimiter(max_calls, period, burst)
            else:
                limiter = AsyncRateLimiter(max_calls, period, burst)

            self.limiters[provider] = limiter
            logger.info(
                "Added rate limiter for provider",
                provider=provider,
                max_calls=max_calls,
                period=period,
                adaptive=adaptive,
            )

    async def acquire(self, provider: str, blocking: bool = True) -> bool:
        """
        Acquire permission for a provider.

        Args:
            provider: Provider name
            blocking: Whether to block until available

        Returns:
            True if acquired, False otherwise

        Raises:
            KeyError: If provider not configured
        """
        if provider not in self.limiters:
            raise KeyError(f"No rate limiter configured for provider: {provider}")

        return await self.limiters[provider].acquire(blocking)

    async def reset(self, provider: str) -> None:
        """
        Reset rate limiter for a provider.

        Args:
            provider: Provider name
        """
        if provider in self.limiters:
            await self.limiters[provider].reset()

    async def reset_all(self) -> None:
        """Reset all rate limiters."""
        for limiter in self.limiters.values():
            await limiter.reset()

    def get_status(self) -> dict[str, dict[str, Any]]:
        """
        Get status of all rate limiters.

        Returns:
            Dictionary with status for each provider
        """
        status = {}
        for provider, limiter in self.limiters.items():
            status[provider] = {
                "remaining_calls": limiter.remaining_calls(),
                "time_until_reset": limiter.time_until_reset(),
                "max_calls": limiter.max_calls,
                "period": limiter.period,
            }
        return status
