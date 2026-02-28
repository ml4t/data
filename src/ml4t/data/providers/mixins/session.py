"""Session mixin for HTTP connection management."""

from __future__ import annotations

from typing import Any, ClassVar

import httpx
import structlog

logger = structlog.get_logger()


class SessionMixin:
    """Mixin providing HTTP session management.

    Manages httpx.Client with connection pooling for efficient
    API communication.

    Class Variables:
        DEFAULT_TIMEOUT: Default request timeout
        DEFAULT_MAX_CONNECTIONS: Max concurrent connections
        DEFAULT_MAX_KEEPALIVE: Max keepalive connections

    Example:
        class MyProvider(SessionMixin):
            def __init__(self):
                self.init_session()

            def fetch_data(self, url):
                response = self.session.get(url)
                return response.json()

            def close(self):
                self.close_session()
    """

    DEFAULT_TIMEOUT: ClassVar[float] = 30.0
    DEFAULT_MAX_CONNECTIONS: ClassVar[int] = 10
    DEFAULT_MAX_KEEPALIVE: ClassVar[int] = 5

    # Instance attribute
    session: httpx.Client

    def init_session(
        self,
        timeout: float | None = None,
        max_connections: int | None = None,
        max_keepalive: int | None = None,
        headers: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize HTTP session.

        Args:
            timeout: Request timeout in seconds
            max_connections: Maximum concurrent connections
            max_keepalive: Maximum keepalive connections
            headers: Default headers for all requests
            **kwargs: Additional httpx.Client arguments
        """
        self.session = httpx.Client(
            timeout=httpx.Timeout(timeout or self.DEFAULT_TIMEOUT),
            limits=httpx.Limits(
                max_connections=max_connections or self.DEFAULT_MAX_CONNECTIONS,
                max_keepalive_connections=max_keepalive or self.DEFAULT_MAX_KEEPALIVE,
            ),
            headers=headers,
            **kwargs,
        )

        logger.debug(
            "HTTP session initialized",
            timeout=timeout or self.DEFAULT_TIMEOUT,
            max_connections=max_connections or self.DEFAULT_MAX_CONNECTIONS,
        )

    def close_session(self) -> None:
        """Close HTTP session and release resources."""
        if hasattr(self, "session"):
            self.session.close()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - close session."""
        self.close_session()

    def _get(self, url: str, **kwargs: Any) -> httpx.Response:
        """Make GET request.

        Args:
            url: Request URL
            **kwargs: Additional request arguments

        Returns:
            HTTP response
        """
        return self.session.get(url, **kwargs)

    def _post(self, url: str, **kwargs: Any) -> httpx.Response:
        """Make POST request.

        Args:
            url: Request URL
            **kwargs: Additional request arguments

        Returns:
            HTTP response
        """
        return self.session.post(url, **kwargs)

    def _get_json(self, url: str, **kwargs: Any) -> Any:
        """Make GET request and parse JSON response.

        Args:
            url: Request URL
            **kwargs: Additional request arguments

        Returns:
            Parsed JSON response
        """
        response = self.session.get(url, **kwargs)
        response.raise_for_status()
        return response.json()
