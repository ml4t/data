"""Async session mixin for HTTP connection management."""

from __future__ import annotations

from typing import Any, ClassVar

import httpx
import structlog

logger = structlog.get_logger()


class AsyncSessionMixin:
    """Mixin providing async HTTP session management.

    Manages httpx.AsyncClient with connection pooling for efficient
    async API communication.

    Class Variables:
        DEFAULT_TIMEOUT: Default request timeout
        DEFAULT_MAX_CONNECTIONS: Max concurrent connections
        DEFAULT_MAX_KEEPALIVE: Max keepalive connections

    Example:
        class MyAsyncProvider(AsyncSessionMixin):
            async def __aenter__(self):
                await self.init_async_session()
                return self

            async def __aexit__(self, *args):
                await self.close_async_session()

            async def fetch_data(self, url):
                response = await self.async_session.get(url)
                return response.json()
    """

    DEFAULT_TIMEOUT: ClassVar[float] = 30.0
    DEFAULT_MAX_CONNECTIONS: ClassVar[int] = 100  # Higher for async
    DEFAULT_MAX_KEEPALIVE: ClassVar[int] = 20

    # Instance attribute
    async_session: httpx.AsyncClient | None = None

    async def init_async_session(
        self,
        timeout: float | None = None,
        max_connections: int | None = None,
        max_keepalive: int | None = None,
        headers: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize async HTTP session.

        Args:
            timeout: Request timeout in seconds
            max_connections: Maximum concurrent connections
            max_keepalive: Maximum keepalive connections
            headers: Default headers for all requests
            **kwargs: Additional httpx.AsyncClient arguments
        """
        if self.async_session is not None:
            return  # Already initialized

        self.async_session = httpx.AsyncClient(
            timeout=httpx.Timeout(timeout or self.DEFAULT_TIMEOUT),
            limits=httpx.Limits(
                max_connections=max_connections or self.DEFAULT_MAX_CONNECTIONS,
                max_keepalive_connections=max_keepalive or self.DEFAULT_MAX_KEEPALIVE,
            ),
            headers=headers,
            **kwargs,
        )

        logger.debug(
            "Async HTTP session initialized",
            timeout=timeout or self.DEFAULT_TIMEOUT,
            max_connections=max_connections or self.DEFAULT_MAX_CONNECTIONS,
        )

    async def close_async_session(self) -> None:
        """Close async HTTP session and release resources."""
        if self.async_session is not None:
            await self.async_session.aclose()
            self.async_session = None
            logger.debug("Async HTTP session closed")

    async def __aenter__(self):
        """Async context manager entry."""
        await self.init_async_session()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit - close session."""
        await self.close_async_session()

    async def _aget(self, url: str, **kwargs: Any) -> httpx.Response:
        """Make async GET request.

        Args:
            url: Request URL
            **kwargs: Additional request arguments

        Returns:
            HTTP response
        """
        if self.async_session is None:
            await self.init_async_session()
        session = self.async_session
        if session is None:
            raise RuntimeError("async session initialization did not create a client")
        return await session.get(url, **kwargs)

    async def _apost(self, url: str, **kwargs: Any) -> httpx.Response:
        """Make async POST request.

        Args:
            url: Request URL
            **kwargs: Additional request arguments

        Returns:
            HTTP response
        """
        if self.async_session is None:
            await self.init_async_session()
        session = self.async_session
        if session is None:
            raise RuntimeError("async session initialization did not create a client")
        return await session.post(url, **kwargs)

    async def _aget_json(self, url: str, **kwargs: Any) -> Any:
        """Make async GET request and parse JSON response.

        Args:
            url: Request URL
            **kwargs: Additional request arguments

        Returns:
            Parsed JSON response
        """
        response = await self._aget(url, **kwargs)
        response.raise_for_status()
        return response.json()
