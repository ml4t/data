"""Custom exceptions for ml4t-data."""

from __future__ import annotations

from typing import Any


class ML4TDataError(Exception):
    """Base exception for all ml4t-data errors."""

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        """
        Initialize ml4t-data error.

        Args:
            message: Error message
            details: Optional dictionary with error details
        """
        super().__init__(message)
        self.message = message
        self.details = details or {}


# Backward compatibility alias
QldmError = ML4TDataError


class ProviderError(ML4TDataError):
    """Base exception for provider-related errors."""

    def __init__(self, provider: str, message: str, details: dict[str, Any] | None = None) -> None:
        """
        Initialize provider error.

        Args:
            provider: Provider name
            message: Error message
            details: Optional error details
        """
        super().__init__(f"{provider}: {message}", details)
        self.provider = provider


class NetworkError(ProviderError):
    """Network-related errors (connection, timeout, etc.)."""

    def __init__(
        self,
        provider: str,
        message: str = "Network error occurred",
        details: dict[str, Any] | None = None,
        retry_after: float | None = None,
    ) -> None:
        """
        Initialize network error.

        Args:
            provider: Provider name
            message: Error message
            details: Optional error details
            retry_after: Seconds to wait before retry
        """
        super().__init__(provider, message, details)
        self.retry_after = retry_after


class RateLimitError(NetworkError):
    """Rate limit exceeded error."""

    def __init__(
        self,
        provider: str,
        retry_after: float | None = None,
        remaining: int | None = None,
        limit: int | None = None,
    ) -> None:
        """
        Initialize rate limit error.

        Args:
            provider: Provider name
            retry_after: Seconds to wait before retry
            remaining: Remaining API calls
            limit: API call limit
        """
        details = {
            "remaining": remaining,
            "limit": limit,
        }
        message = "Rate limit exceeded"
        if retry_after:
            message += f", retry after {retry_after} seconds"

        super().__init__(provider, message, details, retry_after)
        self.remaining = remaining
        self.limit = limit


class AuthenticationError(ProviderError):
    """Authentication/authorization errors."""

    def __init__(
        self,
        provider: str,
        message: str = "Authentication failed",
        details: dict[str, Any] | None = None,
    ) -> None:
        """Initialize authentication error."""
        super().__init__(provider, message, details)


class DataValidationError(ProviderError):
    """Data validation errors."""

    def __init__(
        self,
        provider: str,
        message: str,
        field: str | None = None,
        value: Any | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        """
        Initialize data validation error.

        Args:
            provider: Provider name
            message: Error message
            field: Field that failed validation
            value: Invalid value
            details: Optional error details
        """
        if details is None:
            details = {}
        if field:
            details["field"] = field
        if value is not None:
            details["value"] = str(value)

        super().__init__(provider, message, details)
        self.field = field
        self.value = value


class SymbolNotFoundError(ProviderError):
    """Symbol not found or invalid."""

    def __init__(self, provider: str, symbol: str, details: dict[str, Any] | None = None) -> None:
        """
        Initialize symbol not found error.

        Args:
            provider: Provider name
            symbol: The symbol that was not found
            details: Optional error details
        """
        message = f"Symbol '{symbol}' not found or invalid"
        if details is None:
            details = {}
        details["symbol"] = symbol

        super().__init__(provider, message, details)
        self.symbol = symbol


class DataNotAvailableError(ProviderError):
    """Data not available for the requested period."""

    def __init__(
        self,
        provider: str,
        symbol: str,
        start: str | None = None,
        end: str | None = None,
        frequency: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        """
        Initialize data not available error.

        Args:
            provider: Provider name
            symbol: Symbol requested
            start: Start date
            end: End date
            frequency: Data frequency
            details: Optional error details
        """
        message = f"No data available for {symbol}"
        if start and end:
            message += f" from {start} to {end}"
        if frequency:
            message += f" at {frequency} frequency"

        if details is None:
            details = {}
        details.update(
            {
                "symbol": symbol,
                "start": start,
                "end": end,
                "frequency": frequency,
            }
        )

        super().__init__(provider, message, details)
        self.symbol = symbol
        self.start = start
        self.end = end
        self.frequency = frequency


class StorageError(ML4TDataError):
    """Storage-related errors."""

    def __init__(
        self, message: str, key: str | None = None, details: dict[str, Any] | None = None
    ) -> None:
        """
        Initialize storage error.

        Args:
            message: Error message
            key: Storage key involved
            details: Optional error details
        """
        if details is None:
            details = {}
        if key:
            details["key"] = key

        super().__init__(message, details)
        self.key = key


class LockError(StorageError):
    """File locking errors."""

    def __init__(self, key: str, timeout: float, details: dict[str, Any] | None = None) -> None:
        """
        Initialize lock error.

        Args:
            key: Storage key
            timeout: Lock timeout that was exceeded
            details: Optional error details
        """
        message = f"Could not acquire lock for key {key} within {timeout} seconds"
        if details is None:
            details = {}
        details["timeout"] = timeout

        super().__init__(message, key, details)
        self.timeout = timeout


class ConfigurationError(ML4TDataError):
    """Configuration-related errors."""

    def __init__(
        self,
        message: str,
        parameter: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        """
        Initialize configuration error.

        Args:
            message: Error message
            parameter: Configuration parameter involved
            details: Optional error details
        """
        if details is None:
            details = {}
        if parameter:
            details["parameter"] = parameter

        super().__init__(message, details)
        self.parameter = parameter


class CircuitBreakerOpenError(ML4TDataError):
    """Circuit breaker is open and preventing calls."""

    def __init__(
        self,
        message: str = "Circuit breaker is open",
        failure_count: int | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        """
        Initialize circuit breaker open error.

        Args:
            message: Error message
            failure_count: Number of failures that caused circuit to open
            details: Optional error details
        """
        if details is None:
            details = {}
        if failure_count is not None:
            details["failure_count"] = failure_count

        super().__init__(message, details)
        self.failure_count = failure_count
