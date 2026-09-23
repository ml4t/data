"""Offline test-lane network policy tests."""

import asyncio
import socket

import pytest


@pytest.mark.network_guard_probe
def test_default_lane_rejects_external_network_connections() -> None:
    """Unmarked tests cannot silently contact an external service."""
    with socket.socket() as client:
        with pytest.raises(RuntimeError, match="Offline test attempted"):
            client.connect(("192.0.2.1", 443))
        with pytest.raises(RuntimeError, match="Offline test attempted"):
            client.connect_ex(("192.0.2.1", 443))


@pytest.mark.network_guard_probe
def test_default_lane_rejects_external_dns_resolution() -> None:
    with pytest.raises(RuntimeError, match="Offline test attempted"):
        socket.getaddrinfo("example.com", 443)


def test_default_lane_allows_loopback_connections() -> None:
    with socket.socket() as server:
        server.bind(("127.0.0.1", 0))
        server.listen()
        with socket.create_connection(server.getsockname(), timeout=1):
            connection, _ = server.accept()
            connection.close()


@pytest.mark.network_guard_probe
def test_default_lane_rejects_curl_cffi_requests() -> None:
    """yfinance's transport is libcurl, which never touches Python's socket module."""
    curl_requests = pytest.importorskip("curl_cffi.requests")

    with curl_requests.Session() as session:
        with pytest.raises(RuntimeError, match="Offline test attempted"):
            session.get("https://query1.finance.yahoo.com/v8/finance/chart/AAPL")


@pytest.mark.network_guard_probe
def test_default_lane_rejects_async_curl_cffi_requests() -> None:
    curl_requests = pytest.importorskip("curl_cffi.requests")

    async def fetch() -> None:
        async with curl_requests.AsyncSession() as session:
            await session.get("https://query1.finance.yahoo.com/v8/finance/chart/AAPL")

    with pytest.raises(RuntimeError, match="Offline test attempted"):
        asyncio.run(fetch())
