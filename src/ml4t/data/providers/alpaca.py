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

import polars as pl
import structlog

from ml4t.data.core.exceptions import (
    AuthenticationError,
    DataNotAvailableError,
    DataValidationError,
    NetworkError,
    ProviderError,
    RateLimitError,
)
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

    def _stock_bars_params(self, frequency: str, start: str, end: str) -> dict[str, Any]:
        """Build the query parameters for a stock bars request.

        Args:
            frequency: Canonical frequency key or alias.
            start: Inclusive start date/datetime in ISO-8601 (RFC-3339) form.
            end: Inclusive end date/datetime in ISO-8601 (RFC-3339) form.

        Returns:
            The query parameter mapping, including the configured data feed.
        """
        return {
            "timeframe": self._map_frequency(frequency),
            "start": start,
            "end": end,
            "limit": 10000,
            "feed": self.feed,
        }

    def _check_response_status(self, status_code: int, symbol: str, response_text: str) -> None:
        """Map an HTTP status code to a typed provider exception.

        The status code is inspected directly rather than delegating to
        ``raise_for_status``, so that each documented failure mode surfaces as a
        specific provider error instead of a generic transport error.

        Args:
            status_code: The HTTP status code from the response.
            symbol: The requested symbol, for error context.
            response_text: The response body, included in network-error messages.

        Raises:
            RateLimitError: On HTTP 429.
            AuthenticationError: On HTTP 401/403.
            DataNotAvailableError: On HTTP 404.
            NetworkError: On any other non-200 status.
        """
        if status_code == 429:
            raise RateLimitError(provider="alpaca", retry_after=60.0)
        if status_code in (401, 403):
            raise AuthenticationError(
                provider="alpaca", message="Invalid API key or unauthorized access"
            )
        if status_code == 404:
            raise DataNotAvailableError(provider="alpaca", symbol=symbol)
        if status_code != 200:
            raise NetworkError(provider="alpaca", message=f"HTTP {status_code}: {response_text}")

    def _parse_bars_response(self, response: Any) -> dict[str, Any]:
        """Parse a validated bars response into a JSON dict.

        Args:
            response: An HTTP response whose status has already been checked.

        Returns:
            The parsed JSON payload.

        Raises:
            NetworkError: If the body cannot be decoded as JSON.
        """
        try:
            return response.json()
        except Exception as err:
            raise NetworkError(provider="alpaca", message="Failed to parse JSON response") from err

    def _fetch_raw_data(
        self,
        symbol: str,
        start: str,
        end: str,
        frequency: str = "daily",
        asset_class: str | None = None,  # noqa: ARG002
    ) -> dict[str, Any]:
        """Fetch a single page of stock bars from Alpaca.

        Args:
            symbol: The equity symbol to fetch (case-insensitive).
            start: Inclusive start date/datetime in ISO-8601 (RFC-3339) form.
            end: Inclusive end date/datetime in ISO-8601 (RFC-3339) form.
            frequency: Canonical frequency key or alias.
            asset_class: Reserved for routing to other asset classes; the stock
                branch is the default.

        Returns:
            The parsed JSON payload containing the ``bars`` data.

        Raises:
            RateLimitError, AuthenticationError, DataNotAvailableError,
            NetworkError: Per the HTTP status of the response.
        """
        endpoint = f"{self.base_url}/v2/stocks/{symbol.upper()}/bars"
        params = self._stock_bars_params(frequency, start, end)

        try:
            self.rate_limiter.acquire(blocking=True)
            response = self.session.get(endpoint, params=params)
            self._check_response_status(response.status_code, symbol, response.text)
            return self._parse_bars_response(response)
        except (
            AuthenticationError,
            RateLimitError,
            NetworkError,
            DataNotAvailableError,
            ProviderError,
        ):
            raise
        except Exception as err:
            raise NetworkError(provider="alpaca", message=f"Request failed: {endpoint}") from err

    async def _fetch_raw_data_async(
        self,
        symbol: str,
        start: str,
        end: str,
        frequency: str = "daily",
        asset_class: str | None = None,  # noqa: ARG002
    ) -> dict[str, Any]:
        """Asynchronously fetch a single page of stock bars from Alpaca.

        Mirrors :meth:`_fetch_raw_data` over the async transport; the async
        session carries the same two-header credentials as the sync session.

        Args:
            symbol: The equity symbol to fetch (case-insensitive).
            start: Inclusive start date/datetime in ISO-8601 (RFC-3339) form.
            end: Inclusive end date/datetime in ISO-8601 (RFC-3339) form.
            frequency: Canonical frequency key or alias.
            asset_class: Reserved for routing to other asset classes; the stock
                branch is the default.

        Returns:
            The parsed JSON payload containing the ``bars`` data.

        Raises:
            RateLimitError, AuthenticationError, DataNotAvailableError,
            NetworkError: Per the HTTP status of the response.
        """
        endpoint = f"{self.base_url}/v2/stocks/{symbol.upper()}/bars"
        params = self._stock_bars_params(frequency, start, end)

        try:
            self.rate_limiter.acquire(blocking=True)
            response = await self._aget(endpoint, params=params)
            self._check_response_status(response.status_code, symbol, response.text)
            return self._parse_bars_response(response)
        except (
            AuthenticationError,
            RateLimitError,
            NetworkError,
            DataNotAvailableError,
            ProviderError,
        ):
            raise
        except Exception as err:
            raise NetworkError(provider="alpaca", message=f"Request failed: {endpoint}") from err

    def _extract_bars(self, raw_data: dict[str, Any], symbol: str) -> list[dict[str, Any]]:
        """Extract the bar list from either response shape.

        Alpaca's single-symbol endpoint returns ``{"bars": [...]}`` while the
        multi-symbol endpoint returns ``{"bars": {"<SYMBOL>": [...]}}``. Both are
        accepted so the endpoint choice does not ripple into the transform.

        Args:
            raw_data: The parsed JSON payload.
            symbol: The requested symbol, used to key into a dict payload.

        Returns:
            The list of bar dicts, or an empty list when none are present.
        """
        bars = raw_data.get("bars")
        if isinstance(bars, dict):
            return bars.get(symbol) or bars.get(symbol.upper()) or []
        return bars or []

    def _transform_data(
        self,
        raw_data: dict[str, Any],
        symbol: str,
        asset_class: str | None = None,
    ) -> pl.DataFrame:
        """Transform a raw bars payload into the canonical OHLCV schema.

        Args:
            raw_data: The parsed JSON payload from a bars request.
            symbol: The requested symbol; the literal is added as a column,
                uppercased for stocks and used verbatim for crypto.
            asset_class: When ``"crypto"``, the symbol literal is preserved
                verbatim; otherwise it is uppercased.

        Returns:
            A DataFrame with columns
            ``[timestamp, symbol, open, high, low, close, volume]`` sorted by
            timestamp, or the canonical empty DataFrame when there are no bars.

        Raises:
            DataValidationError: If the bars cannot be transformed.
        """
        bars = self._extract_bars(raw_data, symbol)
        if not bars:
            return self._create_empty_dataframe()

        symbol_literal = symbol if asset_class == "crypto" else symbol.upper()

        try:
            df = pl.DataFrame(bars)
            df = df.rename(
                {
                    "t": "timestamp",
                    "o": "open",
                    "h": "high",
                    "l": "low",
                    "c": "close",
                    "v": "volume",
                }
            )
            # Alpaca timestamps are RFC-3339 with a UTC ("Z") offset; parse them
            # tz-aware then drop the zone to match the canonical naive schema.
            df = df.with_columns(
                pl.col("timestamp")
                .str.to_datetime(format="%Y-%m-%dT%H:%M:%S%.f%#z")
                .dt.replace_time_zone(None)
            )
            for col in ["open", "high", "low", "close", "volume"]:
                df = df.with_columns(pl.col(col).cast(pl.Float64))
            df = df.with_columns(pl.lit(symbol_literal).alias("symbol"))
            df = df.sort("timestamp")
            return df.select(["timestamp", "symbol", "open", "high", "low", "close", "volume"])
        except Exception as err:
            raise DataValidationError(
                provider="alpaca", message=f"Failed to transform data for {symbol}"
            ) from err
