"""Retry utilities for ml4t-data."""

from collections.abc import Callable
from typing import Any, TypeVar, cast

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

T = TypeVar("T")


def with_retry(
    max_attempts: int = 3,
    min_wait: float = 1.0,
    max_wait: float = 60.0,
    multiplier: float = 2.0,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """
    Create a retry decorator with exponential backoff.

    Args:
        max_attempts: Maximum number of retry attempts
        min_wait: Minimum wait time between retries (seconds)
        max_wait: Maximum wait time between retries (seconds)
        multiplier: Exponential backoff multiplier

    Returns:
        Decorator function
    """
    return cast(
        Callable[[Callable[..., Any]], Callable[..., Any]],
        retry(
            stop=stop_after_attempt(max_attempts),
            wait=wait_exponential(multiplier=multiplier, min=min_wait, max=max_wait),
            retry=retry_if_exception_type((ConnectionError, TimeoutError, IOError)),
            # Removed before_sleep logging to avoid structlog warnings in tests
        ),
    )


class RetryableError(Exception):
    """Base class for errors that should trigger a retry."""


class ProviderError(RetryableError):
    """Error from data provider that may be transient."""


class NetworkError(RetryableError):
    """Network-related error that may be transient."""
