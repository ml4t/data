"""Enhanced base provider architecture with Protocol + Mixins pattern.

This module provides the base provider implementation using a composition
of mixins for reusable functionality:
- RateLimitMixin: API rate limiting
- RetryMixin: Automatic retry with backoff
- CircuitBreakerMixin: Circuit breaker pattern
- ValidationMixin: OHLCV data validation
- SessionMixin: HTTP session management

Providers can either:
1. Extend BaseProvider (gets all functionality automatically)
2. Compose mixins directly for custom behavior
3. Implement OHLCVProvider protocol without inheritance

Example (extending BaseProvider):
    class MyProvider(BaseProvider):
        @property
        def name(self) -> str:
            return "my_provider"

        def _fetch_and_transform_data(self, symbol, start, end, frequency):
            # Fetch and return data
            pass

Example (composing mixins):
    class LightweightProvider(RateLimitMixin, ValidationMixin):
        # Custom implementation with just rate limiting and validation
        pass
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Any, ClassVar

import polars as pl
import structlog
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from ml4t.data.core.exceptions import NetworkError, RateLimitError
from ml4t.data.providers.mixins.circuit_breaker import CircuitBreaker, CircuitBreakerMixin
from ml4t.data.providers.mixins.rate_limit import RateLimitMixin
from ml4t.data.providers.mixins.session import SessionMixin
from ml4t.data.providers.mixins.validation import ValidationMixin
from ml4t.data.providers.protocols import OHLCVProvider, ProviderCapabilities

logger = structlog.get_logger()


# Re-export for backward compatibility
__all__ = [
    "BaseProvider",
    "Provider",
    "CircuitBreaker",
    "circuit_breaker",
    "OHLCVProvider",
    "ProviderCapabilities",
]


def circuit_breaker(
    failure_threshold: int = 5,
    reset_timeout: float = 300.0,
    expected_exception: type[Exception] = Exception,
):
    """Decorator for circuit breaker pattern (backward compatibility).

    Args:
        failure_threshold: Failures before opening circuit
        reset_timeout: Seconds before attempting reset
        expected_exception: Exception type that counts as failure

    Returns:
        Decorated function with circuit breaker protection
    """
    from functools import wraps

    def decorator(func):
        breaker = CircuitBreaker(failure_threshold, reset_timeout, expected_exception)

        @wraps(func)
        def wrapper(*args, **kwargs):
            return breaker.call(func, *args, **kwargs)

        return wrapper

    return decorator


class BaseProvider(
    RateLimitMixin,
    CircuitBreakerMixin,
    ValidationMixin,
    SessionMixin,
    ABC,
):
    """Enhanced base provider composing all mixins.

    All providers must return OHLCV data in the canonical schema with columns
    in standard order: [timestamp, symbol, open, high, low, close, volume].

    Each provider must implement either:
    - _fetch_and_transform_data() for single-step implementation
    - _fetch_raw_data() + _transform_data() for two-step implementation

    Class Variables:
        DEFAULT_RATE_LIMIT: Default (calls, period_seconds) for rate limiting
        FREQUENCY_MAP: Mapping of frequency names to provider-specific values
        CIRCUIT_BREAKER_CONFIG: Circuit breaker failure threshold and reset timeout

    Key Contracts:
        - Columns always in order: timestamp, symbol, open, high, low, close, volume
        - Timestamps are Datetime type
        - OHLCV values are Float64
        - Symbol is uppercase String
        - Data sorted by timestamp ascending
        - No duplicate timestamps
    """

    # Default rate limiting settings (can be overridden)
    DEFAULT_RATE_LIMIT: ClassVar[tuple[int, float]] = (60, 60.0)  # 60 calls per minute
    CIRCUIT_BREAKER_CONFIG: ClassVar[dict[str, Any]] = {
        "failure_threshold": 5,
        "reset_timeout": 300.0,
    }

    def __init__(
        self,
        rate_limit: tuple[int, float] | None = None,
        session_config: dict[str, Any] | None = None,
        circuit_breaker_config: dict[str, Any] | None = None,
    ):
        """Initialize base provider with common infrastructure.

        Args:
            rate_limit: Tuple of (calls, period_seconds) for rate limiting
            session_config: HTTP session configuration
            circuit_breaker_config: Circuit breaker configuration
        """
        self.logger = structlog.get_logger(name=self.__class__.__name__)
        self._log_close_events = os.getenv("ML4T_LOG_PROVIDER_CLOSE", "0") == "1"

        # Initialize mixins
        self.init_rate_limit(self.name, rate_limit)
        self.init_circuit_breaker(circuit_breaker_config)
        self.init_session(**(session_config or {}))

    def close(self):
        """Clean up resources."""
        self.close_session()

    def _log_close_event(self, message: str) -> None:
        """Log noisy provider close events only when explicitly enabled."""
        if self._log_close_events:
            self.logger.debug(message)

    def __del__(self) -> None:
        """Best-effort session cleanup for leaked provider instances."""
        try:
            self.close()
        except Exception:
            pass

    def capabilities(self) -> ProviderCapabilities:
        """Return provider capabilities (default implementation).

        Override in subclasses to provide accurate capabilities.
        """
        return ProviderCapabilities(
            rate_limit=self.DEFAULT_RATE_LIMIT,
        )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((NetworkError, RateLimitError)),
        reraise=True,
    )
    def fetch_ohlcv(
        self,
        symbol: str,
        start: str,
        end: str,
        frequency: str = "daily",
    ) -> pl.DataFrame:
        """Template method for fetching OHLCV data.

        This method implements the common workflow:
        1. Validate inputs
        2. Apply rate limiting
        3. Fetch and transform data (provider-specific)
        4. Validate and normalize data

        Providers can implement either:
        - _fetch_and_transform_data() for single-step implementation
        - _fetch_raw_data() + _transform_data() for two-step implementation

        Args:
            symbol: The symbol to fetch data for
            start: Start date in YYYY-MM-DD format (inclusive)
            end: End date in YYYY-MM-DD format (see note below)
            frequency: Data frequency (daily, minute, etc.)

        Returns:
            DataFrame with OHLCV data in canonical schema:
            [timestamp, symbol, open, high, low, close, volume]

        Note:
            Date range semantics vary by provider:
            - Most providers: Both start and end are INCLUSIVE
            - Yahoo Finance: end is EXCLUSIVE (internally adds 1 day)
        """
        self.logger.info(
            "Fetching OHLCV data",
            symbol=symbol,
            start=start,
            end=end,
            frequency=frequency,
            provider=self.name,
        )

        # Step 1: Validate inputs (from ValidationMixin)
        self._validate_inputs(symbol, start, end, frequency)

        # Step 2: Apply rate limiting (from RateLimitMixin)
        self._acquire_rate_limit()

        # Step 3-4: Fetch, transform, and validate with circuit breaker
        def _fetch_and_process():
            # Step 3: Fetch and transform (provider-specific)
            data = self._fetch_and_transform_data(symbol, start, end, frequency)

            # Step 4: Validate and normalize (from ValidationMixin)
            return self._validate_ohlcv(data, self.name)

        # Execute with circuit breaker protection (from CircuitBreakerMixin)
        validated_data = self._with_circuit_breaker(_fetch_and_process)

        self.logger.info(
            "Successfully fetched OHLCV data",
            symbol=symbol,
            rows=len(validated_data),
            provider=self.name,
        )

        return validated_data

    # Abstract methods - must be implemented by concrete providers
    @property
    @abstractmethod
    def name(self) -> str:
        """Return the provider name."""

    # Provider implementation methods - choose your pattern
    def _fetch_and_transform_data(
        self, symbol: str, start: str, end: str, frequency: str
    ) -> pl.DataFrame:
        """Fetch and transform data in one step.

        Providers can either:
        1. Override this method directly (for complex providers)
        2. Implement _fetch_raw_data + _transform_data (for simple providers)

        Args:
            symbol: The symbol to fetch data for
            start: Start date in YYYY-MM-DD format
            end: End date in YYYY-MM-DD format
            frequency: Data frequency

        Returns:
            DataFrame with standardized OHLCV schema
        """
        # Default: call two-step pattern
        raw_data = self._fetch_raw_data(symbol, start, end, frequency)
        return self._transform_data(raw_data, symbol)

    def _fetch_raw_data(self, symbol: str, start: str, end: str, frequency: str) -> Any:
        """Fetch raw data from provider API.

        Override this + _transform_data for simple providers, OR
        override _fetch_and_transform_data directly for complex providers.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement either "
            "_fetch_and_transform_data() or both _fetch_raw_data() + _transform_data()"
        )

    def _transform_data(self, raw_data: Any, symbol: str) -> pl.DataFrame:
        """Transform raw data to standardized schema.

        Override this + _fetch_raw_data for simple providers, OR
        override _fetch_and_transform_data directly for complex providers.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement either "
            "_fetch_and_transform_data() or both _fetch_raw_data() + _transform_data()"
        )


# Keep backward compatibility
class Provider(BaseProvider):
    """Backward compatibility alias for BaseProvider."""
