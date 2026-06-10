"""Alpaca data provider.

Alpaca provides long-history, high-frequency market data across multiple asset
classes including equities and crypto, served over a REST API.

API Documentation: https://docs.alpaca.markets/

Authentication:
- Two credentials are required: an API key id and an API secret, sent as the
  ``APCA-API-KEY-ID`` / ``APCA-API-SECRET-KEY`` request headers.
- Set ``ALPACA_API_KEY`` / ``ALPACA_API_SECRET`` (primary, project convention)
  or pass ``api_key`` / ``api_secret`` to the constructor. Alpaca's own SDK/CLI
  names ``APCA_API_KEY_ID`` / ``APCA_API_SECRET_KEY`` are accepted as a fallback.

Feed selection:
- ``feed`` defaults to ``"iex"`` (free tier, ~15-min delayed, thinner coverage).
  Paid subscribers pass ``feed="sip"`` once at construction.

Rate limiting:
- ``DEFAULT_RATE_LIMIT`` is a conservative client-side throttle of 200 calls per
  minute reflecting the commonly-cited Basic (free) plan figure. The API itself
  enforces limits via HTTP 429 and rate-limit response headers rather than a
  fixed documented number, so this default is tentative and overridable via the
  ``rate_limit`` constructor argument.

Example:
    >>> from ml4t.data.providers.alpaca import AlpacaDataProvider
    >>> provider = AlpacaDataProvider(api_key="k", api_secret="s")
    >>> provider.close()
"""

from __future__ import annotations

import os
from typing import Any, ClassVar

import structlog

from ml4t.data.core.exceptions import AuthenticationError, DataValidationError
from ml4t.data.providers.base import BaseProvider
from ml4t.data.providers.mixins import AsyncSessionMixin
from ml4t.data.providers.protocols import ProviderCapabilities

logger = structlog.get_logger()


class AlpacaDataProvider(AsyncSessionMixin, BaseProvider):
    """Alpaca market data provider.

    Supports equities and crypto with daily, hourly, and minute OHLCV bars over
    Alpaca's historical REST API. Authentication uses two header credentials that
    are wired onto both the sync and async HTTP sessions.

    Supports both sync and async operations:
        # Sync
        provider = AlpacaDataProvider(api_key="k", api_secret="s")

        # Async
        async with AlpacaDataProvider(api_key="k", api_secret="s") as provider:
            ...
    """

    # 200 requests/min — Alpaca Basic (free) plan, per docs.alpaca.markets
    # "About Market Data API" and Alpaca support (usage-limit-api-calls).
    # Tentative: the API enforces via HTTP 429 + rate-limit headers, not a fixed
    # documented number — verify from response headers and adjust if needed.
    DEFAULT_RATE_LIMIT: ClassVar[tuple[int, float]] = (200, 60.0)

    # Map canonical frequency keys and aliases to Alpaca timeframe strings
    FREQUENCY_MAP: ClassVar[dict[str, str]] = {
        "daily": "1Day",
        "day": "1Day",
        "1d": "1Day",
        "1day": "1Day",
        "hourly": "1Hour",
        "hour": "1Hour",
        "1h": "1Hour",
        "1hour": "1Hour",
        "minute": "1Min",
        "1m": "1Min",
        "1minute": "1Min",
    }

    def __init__(
        self,
        api_key: str | None = None,
        api_secret: str | None = None,
        feed: str = "iex",
        rate_limit: tuple[int, float] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the Alpaca data provider.

        Args:
            api_key: Alpaca API key id (or set ALPACA_API_KEY / APCA_API_KEY_ID).
            api_secret: Alpaca API secret (or set ALPACA_API_SECRET /
                APCA_API_SECRET_KEY).
            feed: Data feed, "iex" (free) or "sip" (paid).
            rate_limit: Optional custom (calls, period_seconds) override.
            **kwargs: Additional arguments forwarded to BaseProvider.

        Raises:
            AuthenticationError: If either credential is missing.
        """
        self.api_key = api_key or os.getenv("ALPACA_API_KEY") or os.getenv("APCA_API_KEY_ID")
        self.api_secret = (
            api_secret or os.getenv("ALPACA_API_SECRET") or os.getenv("APCA_API_SECRET_KEY")
        )
        if not self.api_key or not self.api_secret:
            raise AuthenticationError(
                provider="alpaca",
                message="Alpaca API key and secret required. Set ALPACA_API_KEY "
                "and ALPACA_API_SECRET environment variables or pass api_key and "
                "api_secret parameters. Get a free key at: https://alpaca.markets/",
            )

        self.feed = feed
        self.base_url = "https://data.alpaca.markets"
        self.crypto_base_url = "https://data.alpaca.markets"

        # Built once, reused for both the sync and async sessions so the two
        # credentials accompany every request on either transport.
        self._auth_headers = {
            "APCA-API-KEY-ID": self.api_key,
            "APCA-API-SECRET-KEY": self.api_secret,
        }

        super().__init__(
            rate_limit=rate_limit or self.DEFAULT_RATE_LIMIT,
            session_config={"headers": self._auth_headers},
            **kwargs,
        )

        self.logger.info(
            "Initialized Alpaca provider",
            feed=feed,
            rate_limit=rate_limit or self.DEFAULT_RATE_LIMIT,
        )

    @property
    def name(self) -> str:
        """Return provider name."""
        return "alpaca"

    async def init_async_session(
        self,
        headers: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the async session, defaulting to the auth headers.

        The async session must carry the same two credentials as the sync
        session, so the auth headers are applied unless a caller supplies its own.

        Args:
            headers: Default headers for all requests; falls back to the auth
                headers when not provided.
            **kwargs: Additional arguments forwarded to AsyncSessionMixin.
        """
        await super().init_async_session(headers=headers or self._auth_headers, **kwargs)

    def capabilities(self) -> ProviderCapabilities:
        """Return provider capabilities.

        Returns:
            Capabilities advertising crypto, intraday, required authentication,
            and the client-side rate limit.
        """
        return ProviderCapabilities(
            supports_intraday=True,
            supports_crypto=True,
            requires_api_key=True,
            rate_limit=self.DEFAULT_RATE_LIMIT,
        )

    def _map_frequency(self, frequency: str) -> str:
        """Map a canonical frequency to an Alpaca timeframe string.

        Args:
            frequency: Canonical frequency key or alias (e.g. "daily", "1h").

        Returns:
            The Alpaca timeframe string (e.g. "1Day", "1Hour", "1Min").

        Raises:
            DataValidationError: If the frequency is not supported.
        """
        try:
            return self.FREQUENCY_MAP[frequency.lower()]
        except KeyError as err:
            raise DataValidationError(
                provider="alpaca",
                message=f"Unsupported frequency '{frequency}'. "
                f"Supported: {list(self.FREQUENCY_MAP.keys())}",
                field="frequency",
                value=frequency,
            ) from err
