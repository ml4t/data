"""Rate limiting utilities for ml4t-data."""

from __future__ import annotations

import time
from collections import deque
from threading import Lock

import structlog

logger = structlog.get_logger()


class RateLimiter:
    """
    Thread-safe rate limiter using token bucket algorithm.

    This implementation ensures we don't exceed the specified rate limit
    for API calls or other rate-limited operations.
    """

    def __init__(
        self,
        max_calls: int,
        period: float,
        burst: int | None = None,
    ) -> None:
        """
        Initialize rate limiter.

        Args:
            max_calls: Maximum number of calls allowed
            period: Time period in seconds
            burst: Optional burst allowance (defaults to max_calls)
        """
        self.max_calls = max_calls
        self.period = period
        self.burst = burst or max_calls
        self.calls: deque[float] = deque(maxlen=self.burst)
        self.lock = Lock()

    def acquire(self, blocking: bool = True) -> bool:
        """
        Acquire permission to make a call.

        Args:
            blocking: If True, wait until permission is available

        Returns:
            True if permission was acquired, False otherwise
        """
        while True:
            with self.lock:
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
                wait_time = self.period - (now - self.calls[0])

            # Wait outside the lock
            if wait_time > 0:
                logger.debug(
                    "Rate limit reached, waiting",
                    wait_seconds=f"{wait_time:.2f}",
                    calls_in_window=len(self.calls),
                )
                time.sleep(wait_time)
            else:
                return False

    def reset(self) -> None:
        """Reset the rate limiter."""
        with self.lock:
            self.calls.clear()
