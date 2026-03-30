"""Global rate limiting system for data providers."""

from __future__ import annotations

import threading
from typing import Any, ClassVar

from ml4t.data.utils.rate_limit import RateLimiter


class GlobalRateLimitManager:
    """
    Manages rate limits globally across all provider instances.

    This ensures that multiple instances of the same provider type
    share the same rate limits, preventing API abuse when multiple
    DataManager instances or provider instances are created.
    """

    _instance: ClassVar[GlobalRateLimitManager | None] = None
    _lock: ClassVar[threading.Lock] = threading.Lock()

    def __new__(cls) -> GlobalRateLimitManager:
        """Singleton pattern to ensure only one global manager exists."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        """Initialize the global rate limit manager."""
        if not hasattr(self, "_initialized"):
            self._rate_limiters: dict[str, RateLimiter] = {}
            self._limiter_lock = threading.Lock()
            self._initialized = True

    def get_rate_limiter(
        self,
        provider_name: str,
        max_calls: int,
        period: float,
        burst: int | None = None,
    ) -> RateLimiter:
        """
        Get or create a rate limiter for a specific provider.

        Args:
            provider_name: Name of the provider (e.g., 'yahoo', 'binance_api')
            max_calls: Maximum number of calls allowed
            period: Time period in seconds
            burst: Optional burst allowance

        Returns:
            RateLimiter instance shared across all instances of this provider
        """
        # Create a unique key based on provider and rate limit config
        key = f"{provider_name}:{max_calls}:{period}:{burst}"

        with self._limiter_lock:
            if key not in self._rate_limiters:
                self._rate_limiters[key] = RateLimiter(
                    max_calls=max_calls,
                    period=period,
                    burst=burst,
                )

        return self._rate_limiters[key]

    def reset_provider_limits(self, provider_name: str) -> None:
        """
        Reset all rate limiters for a specific provider.

        Args:
            provider_name: Name of the provider to reset
        """
        with self._limiter_lock:
            keys_to_reset = [
                key for key in self._rate_limiters if key.startswith(f"{provider_name}:")
            ]
            for key in keys_to_reset:
                self._rate_limiters[key].reset()

    def reset_all_limits(self) -> None:
        """Reset all rate limiters."""
        with self._limiter_lock:
            for limiter in self._rate_limiters.values():
                limiter.reset()

    def get_limiter_status(self) -> dict[str, dict[str, Any]]:
        """
        Get status of all active rate limiters.

        Returns:
            Dictionary with limiter status information
        """
        status = {}
        with self._limiter_lock:
            for key, limiter in self._rate_limiters.items():
                provider_name = key.split(":")[0]
                if provider_name not in status:
                    status[provider_name] = []

                status[provider_name].append(
                    {
                        "key": key,
                        "max_calls": limiter.max_calls,
                        "period": limiter.period,
                        "burst": limiter.burst,
                        "current_calls": len(limiter.calls),
                    }
                )

        return status


# Global singleton instance
global_rate_limit_manager = GlobalRateLimitManager()
