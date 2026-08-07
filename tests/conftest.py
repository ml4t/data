"""Pytest configuration and fixtures."""

import asyncio
import gc
import ipaddress
import os
import socket

import pytest
import structlog

# Set TESTING environment variable for all tests
os.environ["TESTING"] = "true"

NETWORK_MARKERS = ("integration", "real_api", "requires_api_key", "paid_tier")
LOCAL_HOSTNAMES = {"localhost", "localhost.localdomain", "ip6-localhost"}

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


def _close_yfinance_caches() -> None:
    """Close yfinance sqlite caches if available."""
    try:
        import yfinance.cache as yf_cache
    except ImportError:
        return

    for manager_name in ("_TzDBManager", "_CookieDBManager", "_ISINDBManager"):
        manager = getattr(yf_cache, manager_name, None)
        close_db = getattr(manager, "close_db", None) if manager is not None else None
        if callable(close_db):
            close_db()


def _is_loopback_host(host: object) -> bool:
    if host is None:
        return True
    if isinstance(host, bytes):
        try:
            host = host.decode("ascii")
        except UnicodeDecodeError:
            return False
    if not isinstance(host, str) or not host:
        return False
    if host.lower() in LOCAL_HOSTNAMES:
        return True
    try:
        address = ipaddress.ip_address(host.split("%", 1)[0])
    except ValueError:
        return False
    mapped = getattr(address, "ipv4_mapped", None)
    return (mapped or address).is_loopback


def _is_loopback_address(address: object) -> bool:
    if not isinstance(address, tuple):
        return True
    return bool(address) and _is_loopback_host(address[0])


@pytest.fixture(autouse=True)
def block_external_network(request, monkeypatch):
    """Default-lane tests must declare live network access with a marker."""
    if any(request.node.get_closest_marker(marker) for marker in NETWORK_MARKERS):
        yield
        return

    original_connect = socket.socket.connect
    original_connect_ex = socket.socket.connect_ex
    original_getaddrinfo = socket.getaddrinfo
    violations: list[object] = []

    def reject_external(target: object) -> None:
        violations.append(target)
        raise RuntimeError(
            f"Offline test attempted an external network connection to {target!r}; "
            "mark the test as integration or mock the transport"
        )

    def guarded_connect(sock, address):
        if not _is_loopback_address(address):
            reject_external(address)
        return original_connect(sock, address)

    def guarded_connect_ex(sock, address):
        if not _is_loopback_address(address):
            reject_external(address)
        return original_connect_ex(sock, address)

    def guarded_getaddrinfo(host, *args, **kwargs):
        if not _is_loopback_host(host):
            reject_external(host)
        return original_getaddrinfo(host, *args, **kwargs)

    monkeypatch.setattr(socket.socket, "connect", guarded_connect)
    monkeypatch.setattr(socket.socket, "connect_ex", guarded_connect_ex)
    monkeypatch.setattr(socket, "getaddrinfo", guarded_getaddrinfo)
    yield
    if violations and not request.node.get_closest_marker("network_guard_probe"):
        pytest.fail(f"Offline test attempted external network access: {violations!r}")


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


@pytest.fixture(scope="session", autouse=True)
def close_default_event_loop():
    """Close any leaked event loops at session end to avoid ResourceWarning on Python 3.13."""
    yield

    for obj in gc.get_objects():
        if not isinstance(obj, asyncio.AbstractEventLoop):
            continue
        if obj.is_running() or obj.is_closed():
            continue
        try:
            obj.close()
        except (RuntimeError, ValueError):
            # Best-effort cleanup: ignore already-invalid loop internals at teardown.
            continue


@pytest.fixture(scope="module", autouse=True)
def collect_module_resources():
    """Force delayed finalizers to run before pytest leaves each module."""
    yield
    gc.collect()


@pytest.fixture(autouse=True)
def close_yfinance_caches():
    """Close yfinance sqlite caches around each test to prevent leaked sqlite handles."""
    _close_yfinance_caches()
    yield
    _close_yfinance_caches()
