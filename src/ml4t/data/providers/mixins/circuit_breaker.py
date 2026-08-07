"""Circuit breaker mixin for provider resilience."""

from __future__ import annotations

from collections.abc import Callable, Coroutine
from time import monotonic
from typing import Any, ClassVar, Literal, TypeVar

import structlog

from ml4t.data.core.exceptions import CircuitBreakerOpenError, NetworkError, RateLimitError

logger = structlog.get_logger()

T = TypeVar("T")
type CircuitState = Literal["CLOSED", "OPEN", "HALF_OPEN"]
type ExceptionTypes = type[Exception] | tuple[type[Exception], ...]


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
        expected_exception: ExceptionTypes = NetworkError,
        excluded_exceptions: ExceptionTypes = (RateLimitError,),
        clock: Callable[[], float] = monotonic,
    ) -> None:
        """Initialize circuit breaker.

        Args:
            failure_threshold: Number of failures before opening
            reset_timeout: Seconds before attempting reset
            expected_exception: Exception types that represent transient service failures
            excluded_exceptions: Subtypes that must not affect service health
            clock: Monotonic clock used for reset deadlines
        """
        if failure_threshold < 1:
            raise ValueError("failure_threshold must be at least 1")
        if reset_timeout < 0:
            raise ValueError("reset_timeout cannot be negative")

        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self.expected_exception = expected_exception
        self.excluded_exceptions = excluded_exceptions
        self._clock = clock
        self.failure_count = 0
        self.last_failure_time: float | None = None
        self.state: CircuitState = "CLOSED"
        self.counted_failure_count = 0
        self.ignored_failure_count = 0
        self.open_count = 0
        self.recovery_count = 0
        self.rejected_call_count = 0

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
                logger.info(
                    "Circuit breaker state changed",
                    circuit_event="probe_started",
                    previous_state="OPEN",
                    state=self.state,
                    failure_count=self.failure_count,
                )
            else:
                self.rejected_call_count += 1
                raise CircuitBreakerOpenError(
                    f"Circuit breaker is OPEN. Failures: {self.failure_count}"
                )

        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as error:
            if self._counts_failure(error):
                self._on_failure(error)
            else:
                self._on_ignored_failure(error)
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
                logger.info(
                    "Circuit breaker state changed",
                    circuit_event="probe_started",
                    previous_state="OPEN",
                    state=self.state,
                    failure_count=self.failure_count,
                )
            else:
                self.rejected_call_count += 1
                raise CircuitBreakerOpenError(
                    f"Circuit breaker is OPEN. Failures: {self.failure_count}"
                )

        try:
            result = await func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as error:
            if self._counts_failure(error):
                self._on_failure(error)
            else:
                self._on_ignored_failure(error)
            raise

    def _counts_failure(self, error: Exception) -> bool:
        """Return whether an error is evidence of a transient service outage."""
        return isinstance(error, self.expected_exception) and not isinstance(
            error, self.excluded_exceptions
        )

    def _should_attempt_reset(self) -> bool:
        """Check if enough time passed to attempt reset."""
        if self.last_failure_time is None:
            return True
        elapsed = self._clock() - self.last_failure_time
        return elapsed >= self.reset_timeout

    def _on_success(self) -> None:
        """Handle successful call."""
        if self.state == "HALF_OPEN":
            self.recovery_count += 1
            logger.info(
                "Circuit breaker state changed",
                circuit_event="recovered",
                previous_state="HALF_OPEN",
                state="CLOSED",
                failure_count=0,
            )
        self.failure_count = 0
        self.state = "CLOSED"

    def _on_failure(self, error: Exception) -> None:
        """Handle failed call."""
        self.failure_count += 1
        self.counted_failure_count += 1
        self.last_failure_time = self._clock()

        if self.failure_count >= self.failure_threshold:
            previous_state = self.state
            self.state = "OPEN"
            self.open_count += 1
            logger.warning(
                "Circuit breaker state changed",
                circuit_event="opened",
                previous_state=previous_state,
                state=self.state,
                error_type=type(error).__name__,
                failure_count=self.failure_count,
                threshold=self.failure_threshold,
            )

    def _on_ignored_failure(self, error: Exception) -> None:
        """Record a client or permanent error without changing health counts."""
        self.ignored_failure_count += 1
        logger.debug(
            "Circuit breaker ignored unclassified failure",
            circuit_event="failure_ignored",
            state=self.state,
            error_type=type(error).__name__,
            failure_count=self.failure_count,
        )
        if self.state == "HALF_OPEN":
            self.state = "OPEN"

    def reset(self) -> None:
        """Manually reset the circuit breaker."""
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"
        logger.info(
            "Circuit breaker state changed",
            circuit_event="manual_reset",
            state=self.state,
            failure_count=0,
        )

    @property
    def is_open(self) -> bool:
        """Check if circuit is open."""
        return self.state == "OPEN"

    @property
    def is_closed(self) -> bool:
        """Check if circuit is closed."""
        return self.state == "CLOSED"

    @property
    def metrics(self) -> dict[str, int]:
        """Return process-local circuit transition counters."""
        return {
            "counted_failures": self.counted_failure_count,
            "ignored_failures": self.ignored_failure_count,
            "opened": self.open_count,
            "recovered": self.recovery_count,
            "rejected_calls": self.rejected_call_count,
        }


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
            "metrics": self.circuit_breaker.metrics,
        }

    def _reset_circuit_breaker(self) -> None:
        """Manually reset circuit breaker."""
        if hasattr(self, "circuit_breaker"):
            self.circuit_breaker.reset()
