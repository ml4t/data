"""Circuit breaker mixin for provider resilience."""

from __future__ import annotations

from collections.abc import Callable, Coroutine
from datetime import datetime
from typing import Any, ClassVar, TypeVar

import structlog

from ml4t.data.core.exceptions import CircuitBreakerOpenError

logger = structlog.get_logger()

T = TypeVar("T")


class CircuitBreaker:
    """Circuit breaker implementation for API reliability.

    States:
        - CLOSED: Normal operation, requests pass through
        - OPEN: Too many failures, requests blocked
        - HALF_OPEN: Testing if service recovered

    Attributes:
        failure_threshold: Failures before opening circuit
        reset_timeout: Seconds before attempting recovery
        failure_count: Current failure count
        state: Current circuit state
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        reset_timeout: float = 300.0,
        expected_exception: type[Exception] = Exception,
    ):
        """Initialize circuit breaker.

        Args:
            failure_threshold: Number of failures before opening
            reset_timeout: Seconds before attempting reset
            expected_exception: Exception type that counts as failure
        """
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self.expected_exception = expected_exception
        self.failure_count = 0
        self.last_failure_time: float | None = None
        self.state = "CLOSED"

    def call(self, func: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        """Execute function with circuit breaker protection.

        Args:
            func: Function to execute
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Function result

        Raises:
            CircuitBreakerOpenError: If circuit is open
            Original exception: If function fails
        """
        if self.state == "OPEN":
            if self._should_attempt_reset():
                self.state = "HALF_OPEN"
                logger.info("Circuit breaker entering HALF_OPEN state")
            else:
                raise CircuitBreakerOpenError(
                    f"Circuit breaker is OPEN. Failures: {self.failure_count}"
                )

        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except self.expected_exception:
            self._on_failure()
            raise

    async def call_async(
        self,
        func: Callable[..., Coroutine[Any, Any, T]],
        *args: Any,
        **kwargs: Any,
    ) -> T:
        """Execute an async function with circuit breaker protection.

        Mirrors :meth:`call` for awaited callables, so async fetch paths get
        the same state handling and failure accounting as sync ones.

        Args:
            func: Async function to execute
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Function result

        Raises:
            CircuitBreakerOpenError: If circuit is open
            Original exception: If function fails
        """
        if self.state == "OPEN":
            if self._should_attempt_reset():
                self.state = "HALF_OPEN"
                logger.info("Circuit breaker entering HALF_OPEN state")
            else:
                raise CircuitBreakerOpenError(
                    f"Circuit breaker is OPEN. Failures: {self.failure_count}"
                )

        try:
            result = await func(*args, **kwargs)
            self._on_success()
            return result
        except self.expected_exception:
            self._on_failure()
            raise

    def _should_attempt_reset(self) -> bool:
        """Check if enough time passed to attempt reset."""
        if self.last_failure_time is None:
            return True
        elapsed = datetime.now().timestamp() - self.last_failure_time
        return elapsed >= self.reset_timeout

    def _on_success(self) -> None:
        """Handle successful call."""
        if self.state == "HALF_OPEN":
            logger.info("Circuit breaker recovered, entering CLOSED state")
        self.failure_count = 0
        self.state = "CLOSED"

    def _on_failure(self) -> None:
        """Handle failed call."""
        self.failure_count += 1
        self.last_failure_time = datetime.now().timestamp()

        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"
            logger.warning(
                "Circuit breaker opened",
                failure_count=self.failure_count,
                threshold=self.failure_threshold,
            )

    def reset(self) -> None:
        """Manually reset the circuit breaker."""
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"
        logger.info("Circuit breaker manually reset")

    @property
    def is_open(self) -> bool:
        """Check if circuit is open."""
        return self.state == "OPEN"

    @property
    def is_closed(self) -> bool:
        """Check if circuit is closed."""
        return self.state == "CLOSED"


class CircuitBreakerMixin:
    """Mixin providing circuit breaker functionality.

    Automatically breaks the circuit after repeated failures,
    preventing cascade failures and allowing services to recover.

    Class Variables:
        CIRCUIT_BREAKER_CONFIG: Default circuit breaker settings

    Example:
        class MyProvider(CircuitBreakerMixin):
            def fetch_data(self, symbol):
                return self._with_circuit_breaker(self._do_fetch, symbol)

            def _do_fetch(self, symbol):
                # ... actual fetch logic ...
    """

    CIRCUIT_BREAKER_CONFIG: ClassVar[dict[str, Any]] = {
        "failure_threshold": 5,
        "reset_timeout": 300.0,
    }

    # Instance attribute
    circuit_breaker: CircuitBreaker

    def init_circuit_breaker(
        self,
        config: dict[str, Any] | None = None,
    ) -> None:
        """Initialize circuit breaker.

        Args:
            config: Optional config override
        """
        cb_config = {**self.CIRCUIT_BREAKER_CONFIG, **(config or {})}
        self.circuit_breaker = CircuitBreaker(**cb_config)

    def _with_circuit_breaker(
        self,
        func: Callable[..., T],
        *args: Any,
        **kwargs: Any,
    ) -> T:
        """Execute function with circuit breaker protection.

        Args:
            func: Function to execute
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Function result

        Raises:
            CircuitBreakerOpenError: If circuit is open
        """
        if not hasattr(self, "circuit_breaker"):
            self.init_circuit_breaker()

        return self.circuit_breaker.call(func, *args, **kwargs)

    async def _with_circuit_breaker_async(
        self,
        func: Callable[..., Coroutine[Any, Any, T]],
        *args: Any,
        **kwargs: Any,
    ) -> T:
        """Execute an async function with circuit breaker protection.

        Args:
            func: Async function to execute
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Function result

        Raises:
            CircuitBreakerOpenError: If circuit is open
        """
        if not hasattr(self, "circuit_breaker"):
            self.init_circuit_breaker()

        return await self.circuit_breaker.call_async(func, *args, **kwargs)

    def _get_circuit_status(self) -> dict[str, Any]:
        """Get circuit breaker status.

        Returns:
            Dict with circuit breaker information
        """
        if not hasattr(self, "circuit_breaker"):
            return {"initialized": False}

        return {
            "initialized": True,
            "state": self.circuit_breaker.state,
            "failure_count": self.circuit_breaker.failure_count,
            "failure_threshold": self.circuit_breaker.failure_threshold,
            "is_open": self.circuit_breaker.is_open,
        }

    def _reset_circuit_breaker(self) -> None:
        """Manually reset circuit breaker."""
        if hasattr(self, "circuit_breaker"):
            self.circuit_breaker.reset()
