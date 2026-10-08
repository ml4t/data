"""Shared fixtures for offline provider adapter tests."""

from __future__ import annotations

from collections.abc import Callable
from unittest.mock import MagicMock

import httpx
import pytest

from ml4t.data.providers.base import BaseProvider

Handler = Callable[[httpx.Request], httpx.Response]


@pytest.fixture
def retry_sleeps(monkeypatch) -> MagicMock:
    """Replace the shared request retry sleep so retry-path tests never wait."""
    sleep = MagicMock()
    monkeypatch.setattr(BaseProvider._request.retry, "sleep", sleep)
    return sleep


@pytest.fixture
def use_transport(retry_sleeps) -> Callable[[BaseProvider, Handler], list[httpx.Request]]:
    """Route a provider's HTTP session through an in-process handler.

    Returns a function that installs ``handler`` on the provider and returns the list that
    records every request the provider sends.
    """

    def install(provider: BaseProvider, handler: Handler) -> list[httpx.Request]:
        requests: list[httpx.Request] = []

        def recording(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            return handler(request)

        provider.session.close()
        provider.session = httpx.Client(transport=httpx.MockTransport(recording))
        return requests

    return install
