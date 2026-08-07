"""Offline test-lane network policy tests."""

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
