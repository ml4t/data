"""Pytest configuration and fixtures."""

import os

import pytest
import structlog
from dotenv import load_dotenv

# Load .env file before running tests (module level)
load_dotenv()


def pytest_configure(config):
    """Load environment variables before test collection."""
    # Ensure .env is loaded before any test collection happens
    load_dotenv(override=False)  # Don't override already-set vars


# Set TESTING environment variable for all tests
os.environ["TESTING"] = "true"

# Configure structlog for tests without format_exc_info to avoid warnings
structlog.configure(
    processors=[
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        # format_exc_info removed - ConsoleRenderer handles exceptions
        structlog.dev.ConsoleRenderer(),
    ],
    context_class=dict,
    logger_factory=structlog.PrintLoggerFactory(),
    cache_logger_on_first_use=False,
)


# ===== Rate Limiter Reset Fixtures =====


@pytest.fixture(autouse=True)
def reset_global_rate_limiter():
    """Reset global rate limiter state between tests.

    This prevents test pollution from accumulated rate limit calls.
    The rate limiter tracks calls over time windows, and without reset,
    tests can unexpectedly block when the window is exhausted.
    """
    try:
        from ml4t.data.utils.global_rate_limit import global_rate_limit_manager

        global_rate_limit_manager.reset_all_limits()
    except ImportError:
        pass  # Module not available

    yield

    try:
        from ml4t.data.utils.global_rate_limit import global_rate_limit_manager

        global_rate_limit_manager.reset_all_limits()
    except ImportError:
        pass


@pytest.fixture(autouse=True)
def reset_provider_cache():
    """Reset provider class cache between tests.

    ProviderManager caches discovered provider classes at the class level.
    Without reset, tests can pollute each other's registry state.
    """
    try:
        from ml4t.data.managers.provider_manager import ProviderManager

        ProviderManager._PROVIDER_CLASSES = None
    except (ImportError, AttributeError):
        pass  # Module not available or attribute doesn't exist

    yield

    try:
        from ml4t.data.managers.provider_manager import ProviderManager

        ProviderManager._PROVIDER_CLASSES = None
    except (ImportError, AttributeError):
        pass
