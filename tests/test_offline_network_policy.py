"""Offline test-lane network policy tests."""

import socket

import pytest


def test_default_lane_rejects_external_network_connections() -> None:
    """Unmarked tests cannot silently contact an external service."""
    with socket.socket() as client:
        with pytest.raises(RuntimeError, match="Offline test attempted"):
            client.connect(("192.0.2.1", 443))
