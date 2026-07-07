"""Alpaca data provider.

Alpaca provides long-history, high-frequency market data across multiple asset
classes including US equities and crypto, served over a historical REST API.
Daily, hourly, and minute OHLCV bars (1/5/15/30-minute) are supported for both
asset classes from a single symbol-routed provider.

API Documentation: https://docs.alpaca.markets/us/docs/about-market-data-api

Symbol Format:
- Stocks use a plain ticker (e.g. "AAPL"); the symbol is uppercased and routed to
  the stock bars endpoint.
- Crypto uses ``BASE/QUOTE`` (e.g. "BTC/USD"); the slash routes the request to the
  crypto bars endpoint, and the symbol is uppercased with the slash preserved so
  the canonical uppercase-symbol output contract holds for both asset classes.

Date bounds:
- ``start``/``end`` accept either a date (``YYYY-MM-DD``) or an RFC-3339 datetime
  (e.g. ``2024-01-01T00:00:00Z``), both inclusive. Datetime bounds matter for
  minute/hour frequencies, where a sub-day window avoids paginating a full day.

Authentication:
- Two credentials are required: an API key id and an API secret, sent as the
  ``APCA-API-KEY-ID`` / ``APCA-API-SECRET-KEY`` request headers on both the sync
  and async sessions.
- Set ``ALPACA_API_KEY`` / ``ALPACA_API_SECRET`` (primary, project convention)
  or pass ``api_key`` / ``api_secret`` to the constructor. Alpaca's own SDK/CLI
  names ``APCA_API_KEY_ID`` / ``APCA_API_SECRET_KEY`` are accepted as a fallback.
- Get a free key at: https://alpaca.markets/

Feed selection (stocks):
- ``feed`` defaults to ``"iex"``: the free, real-time feed served from a single
  exchange (IEX, roughly 2-3% of US market volume), so coverage is thinner than
  the consolidated tape.
- ``feed="sip"`` selects the consolidated tape (100% of US market volume across
  all exchanges). It is real time on paid plans; the free Basic plan can still
  query SIP bars, only not the most recent 15 minutes.
- ``"otc"`` (over-the-counter) and ``"boats"`` (Blue Ocean overnight) are also
  accepted; any other value raises ``DataValidationError`` at construction.
- Crypto has a single consolidated feed, so ``feed`` does not apply to it.

Price adjustment (stocks):
- ``adjustment`` defaults to ``"raw"`` (Alpaca's own default): stock bars are
  NOT adjusted for splits or dividends. Pass ``adjustment="split"``,
  ``"dividend"``, or ``"all"`` at construction for adjusted bars. Crypto bars
  have no adjustment concept.

Rate limiting:
- ``DEFAULT_RATE_LIMIT`` is a conservative client-side throttle of 200 calls per
  minute, reflecting the Basic (free) plan figure documented at
  docs.alpaca.markets "About Market Data API" and Alpaca support
  (usage-limit-api-calls). Alpaca enforces the limit server-side via HTTP 429 and
  rate-limit response headers; this client-side default front-runs that and is
  overridable via the ``rate_limit`` constructor argument.

Sync Example:
    >>> from ml4t.data.providers.alpaca import AlpacaDataProvider
    >>> provider = AlpacaDataProvider(api_key="key", api_secret="secret")
    >>> df = provider.fetch_ohlcv("AAPL", "2024-01-01", "2024-01-31")
    >>> crypto = provider.fetch_ohlcv("BTC/USD", "2024-01-01", "2024-01-02",
    ...                               frequency="minute")
    >>> provider.close()

Async Example:
    >>> async with AlpacaDataProvider(api_key="key", api_secret="secret") as provider:
    ...     df = await provider.fetch_ohlcv_async("AAPL", "2024-01-01", "2024-01-31")
"""

from __future__ import annotations

import asyncio
import os
import time
from datetime import UTC, datetime
from typing import Any, ClassVar

import polars as pl
import structlog
from tenacity import (
    RetryCallState,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from ml4t.data.core.exceptions import (
    AuthenticationError,
    DataNotAvailableError,
    DataValidationError,
    NetworkError,
    RateLimitError,
)
from ml4t.data.providers.base import BaseProvider
from ml4t.data.providers.mixins import AsyncSessionMixin
from ml4t.data.providers.protocols import ProviderCapabilities

logger = structlog.get_logger()

_EXPONENTIAL_WAIT = wait_exponential(multiplier=1, min=4, max=10)


def _retry_wait(retry_state: RetryCallState) -> float:
    """Honor a server-provided retry delay; otherwise back off exponentially.

    A 429 carries the server's own ``retry_after`` (derived from its rate-limit
    headers), which beats a generic exponential guess. The delay is capped so a
    pathological header cannot stall a fetch for hours.

    Args:
        retry_state: The tenacity retry state for the failed attempt.

    Returns:
        Seconds to wait before the next attempt.
    """
    exception = retry_state.outcome.exception() if retry_state.outcome else None
    if isinstance(exception, RateLimitError) and exception.retry_after is not None:
        return min(float(exception.retry_after), 60.0)
    return _EXPONENTIAL_WAIT(retry_state)


# Retries transient failures at the page level so a failure on page N does not
# refetch pages 1..N-1, and a multi-page fetch cannot amplify API load.
_PAGE_RETRY = retry(
    stop=stop_after_attempt(3),
    wait=_retry_wait,
    retry=retry_if_exception_type((NetworkError, RateLimitError)),
    reraise=True,
)


class AlpacaDataProvider(AsyncSessionMixin, BaseProvider):
    """Alpaca market data provider.

    Supports equities and crypto with daily, hourly, and minute OHLCV bars
    (1/5/15/30-minute) over Alpaca's historical REST API. Authentication uses two
    header credentials that are wired onto both the sync and async HTTP sessions.

    Supports both sync and async operations:
        # Sync
        provider = AlpacaDataProvider(api_key="k", api_secret="s")

        # Async
        async with AlpacaDataProvider(api_key="k", api_secret="s") as provider:
            ...
    """

    # 200 requests/min — Alpaca Basic (free) plan, per docs.alpaca.markets
    # "About Market Data API" and Alpaca support (usage-limit-api-calls). The
    # API also enforces this server-side via HTTP 429 + rate-limit headers.
    DEFAULT_RATE_LIMIT: ClassVar[tuple[int, float]] = (200, 60.0)

    # Maximum bars per page accepted by both bars endpoints; pagination follows
    # next_page_token beyond this.
    PAGE_LIMIT: ClassVar[int] = 10000

    # Asset classes a request can be routed to.
    ASSET_CLASSES: ClassVar[frozenset[str]] = frozenset({"stock", "crypto"})

    # Stock data feeds accepted by the bars endpoint. "iex" is the free
    # single-exchange feed; "sip" is the consolidated tape (real-time on paid
    # plans, 15-min-delayed on the free plan); "otc" and "boats" are niche
    # venues. The feed does not apply to crypto (single consolidated feed).
    FEEDS: ClassVar[frozenset[str]] = frozenset({"iex", "sip", "otc", "boats"})

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
        "5m": "5Min",
        "5minute": "5Min",
        "15m": "15Min",
        "15minute": "15Min",
        "30m": "30Min",
        "30minute": "30Min",
    }

    def __init__(
        self,
        api_key: str | None = None,
        api_secret: str | None = None,
        feed: str = "iex",
        adjustment: str = "raw",
        rate_limit: tuple[int, float] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the Alpaca data provider.

        Args:
            api_key: Alpaca API key id (or set ALPACA_API_KEY / APCA_API_KEY_ID).
            api_secret: Alpaca API secret (or set ALPACA_API_SECRET /
                APCA_API_SECRET_KEY).
            feed: Stock data feed, case-insensitive, one of "iex" (free,
                real-time single exchange), "sip" (consolidated tape; real-time
                on paid plans, 15-min-delayed on the free plan), "otc", or
                "boats". Ignored for crypto. Defaults to "iex".
            adjustment: Stock price adjustment, one of "raw" (default, matching
                Alpaca's own default: no split/dividend adjustment), "split",
                "dividend", or "all". Ignored for crypto.
            rate_limit: Optional custom (calls, period_seconds) override.
            **kwargs: Additional arguments forwarded to BaseProvider.

        Raises:
            AuthenticationError: If either credential is missing.
            DataValidationError: If ``feed`` is not a supported value.
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

        normalized_feed = feed.lower()
        if normalized_feed not in self.FEEDS:
            raise DataValidationError(
                provider="alpaca",
                message=f"Invalid feed '{feed}'. Supported: {sorted(self.FEEDS)}",
                field="feed",
                value=feed,
            )
        self.feed = normalized_feed
        self.adjustment = adjustment
        self.base_url = "https://data.alpaca.markets"

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

    def _validate_inputs(
        self,
        symbol: str,
        start: str,
        end: str,
        frequency: str,  # noqa: ARG002
    ) -> None:
        """Validate input parameters, accepting dates or RFC-3339 datetimes.

        Overrides the base date-only validation because Alpaca's bars endpoints
        accept full RFC-3339 datetime bounds, which matter for minute/hour
        frequencies where a sub-day window is the natural request shape.

        Args:
            symbol: Symbol to fetch.
            start: Inclusive start, ``YYYY-MM-DD`` or RFC-3339 datetime.
            end: Inclusive end, ``YYYY-MM-DD`` or RFC-3339 datetime.
            frequency: Data frequency (validated later against FREQUENCY_MAP).

        Raises:
            ValueError: If the symbol is empty, a bound cannot be parsed, or
                start is after end.
        """
        if not symbol or not symbol.strip():
            raise ValueError("Symbol cannot be empty")

        try:
            start_dt = self._parse_time_bound(start)
            end_dt = self._parse_time_bound(end)
        except ValueError as e:
            raise ValueError(
                f"Invalid date format (expected YYYY-MM-DD or RFC-3339 datetime): {e}"
            ) from e

        if start_dt > end_dt:
            raise ValueError("Start date must be before or equal to end date")

    @staticmethod
    def _parse_time_bound(value: str) -> datetime:
        """Parse a date or RFC-3339 datetime bound into a UTC-aware datetime.

        Args:
            value: ``YYYY-MM-DD`` or RFC-3339 datetime string (a trailing ``Z``
                is accepted).

        Returns:
            A timezone-aware datetime; naive inputs are assumed UTC so mixed
            date/datetime bounds stay comparable.

        Raises:
            ValueError: If the value is not a valid ISO-8601 date or datetime.
        """
        parsed = datetime.fromisoformat(value)
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)

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

    def _resolve_asset_class(self, symbol: str, asset_class: str | None) -> str:
        """Resolve and validate the asset class for a request.

        An explicit ``asset_class`` always wins but must name a real asset
        class — a typo would otherwise silently route to the wrong endpoint and
        surface as a misleading 404. Otherwise a ``BASE/QUOTE`` symbol (e.g.
        ``"BTC/USD"``) is treated as crypto and everything else as a stock.

        Args:
            symbol: The requested symbol.
            asset_class: An explicit asset class, or ``None`` to infer it.

        Returns:
            Either ``"crypto"`` or ``"stock"``.

        Raises:
            DataValidationError: If an explicit ``asset_class`` is not one of
                the supported asset classes (case-insensitively).
        """
        if asset_class is not None:
            normalized = asset_class.lower()
            if normalized not in self.ASSET_CLASSES:
                raise DataValidationError(
                    provider="alpaca",
                    message=f"Invalid asset_class '{asset_class}'. "
                    f"Supported: {sorted(self.ASSET_CLASSES)}",
                    field="asset_class",
                    value=asset_class,
                )
            return normalized
        return "crypto" if "/" in symbol else "stock"

    def _stock_bars_params(self, frequency: str, start: str, end: str) -> dict[str, Any]:
        """Build the query parameters for a stock bars request.

        Args:
            frequency: Canonical frequency key or alias.
            start: Inclusive start date/datetime in ISO-8601 (RFC-3339) form.
            end: Inclusive end date/datetime in ISO-8601 (RFC-3339) form.

        Returns:
            The query parameter mapping, including the configured data feed and
            price adjustment.
        """
        return {
            "timeframe": self._map_frequency(frequency),
            "start": start,
            "end": end,
            "limit": self.PAGE_LIMIT,
            "feed": self.feed,
            "adjustment": self.adjustment,
        }

    def _crypto_bars_params(
        self, symbol: str, frequency: str, start: str, end: str
    ) -> dict[str, Any]:
        """Build the query parameters for a crypto bars request.

        The crypto bars endpoint is multi-symbol: the symbol travels in the
        ``symbols`` parameter (uppercased ``BASE/QUOTE`` form, the slash
        preserved) and no ``feed`` is sent, since crypto has a single
        consolidated feed.

        Args:
            symbol: The crypto symbol in ``BASE/QUOTE`` form (e.g. "BTC/USD").
            frequency: Canonical frequency key or alias.
            start: Inclusive start date/datetime in ISO-8601 (RFC-3339) form.
            end: Inclusive end date/datetime in ISO-8601 (RFC-3339) form.

        Returns:
            The query parameter mapping for the crypto bars endpoint.
        """
        return {
            "symbols": symbol.upper(),
            "timeframe": self._map_frequency(frequency),
            "start": start,
            "end": end,
            "limit": self.PAGE_LIMIT,
        }

    def _bars_request(
        self, symbol: str, start: str, end: str, frequency: str, asset_class: str
    ) -> tuple[str, dict[str, Any]]:
        """Resolve the endpoint and query params for a bars request.

        Branches on the resolved asset class so the sync and async fetchers share
        one routing decision. The stock branch uppercases the symbol into the
        path; the crypto branch hits the multi-symbol endpoint with the
        uppercased ``BASE/QUOTE`` symbol (slash preserved).

        Args:
            symbol: The requested symbol.
            start: Inclusive start date/datetime in ISO-8601 (RFC-3339) form.
            end: Inclusive end date/datetime in ISO-8601 (RFC-3339) form.
            frequency: Canonical frequency key or alias.
            asset_class: The resolved asset class, ``"crypto"`` or ``"stock"``.

        Returns:
            A ``(endpoint, params)`` tuple.
        """
        if asset_class == "crypto":
            # The {loc} path segment selects the venue group; "us" is the
            # default consolidated US crypto feed.
            endpoint = f"{self.base_url}/v1beta3/crypto/us/bars"
            return endpoint, self._crypto_bars_params(symbol, frequency, start, end)
        endpoint = f"{self.base_url}/v2/stocks/{symbol.upper()}/bars"
        return endpoint, self._stock_bars_params(frequency, start, end)

    def _merge_bars(self, accumulated: Any, page_bars: Any) -> Any:
        """Merge one page's bars into the accumulator across both response shapes.

        The single-symbol endpoint returns ``bars`` as a list, which is simply
        extended. The multi-symbol crypto endpoint returns ``bars`` as a dict of
        per-symbol lists, whose entries are concatenated by symbol. The seed
        accumulator may be ``None`` on the first page, in which case the page's
        own container type is adopted.

        Args:
            accumulated: The bars merged from prior pages, or ``None`` on page 1.
            page_bars: The ``bars`` value from the current page.

        Returns:
            The accumulator with the current page's bars appended.
        """
        if isinstance(page_bars, dict):
            merged: dict[str, list[Any]] = accumulated if isinstance(accumulated, dict) else {}
            for sym, sym_bars in page_bars.items():
                merged.setdefault(sym, []).extend(sym_bars or [])
            return merged
        merged_list: list[Any] = accumulated if isinstance(accumulated, list) else []
        merged_list.extend(page_bars or [])
        return merged_list

    def _check_response_status(self, response: Any, symbol: str) -> None:
        """Map an HTTP error response to a typed provider exception.

        The status code is inspected directly rather than delegating to
        ``raise_for_status``, so that each documented failure mode surfaces as a
        specific provider error instead of a generic transport error.

        Args:
            response: The HTTP response (status code, headers, and body text
                are consulted).
            symbol: The requested symbol, for error context.

        Raises:
            RateLimitError: On HTTP 429, with ``retry_after`` derived from the
                rate-limit response headers when present.
            AuthenticationError: On HTTP 401/403.
            DataNotAvailableError: On HTTP 404.
            NetworkError: On any other non-200 status.
        """
        status_code = response.status_code
        if status_code == 429:
            raise RateLimitError(provider="alpaca", retry_after=self._retry_after_seconds(response))
        if status_code in (401, 403):
            raise AuthenticationError(
                provider="alpaca", message="Invalid API key or unauthorized access"
            )
        if status_code == 404:
            raise DataNotAvailableError(provider="alpaca", symbol=symbol)
        if status_code != 200:
            raise NetworkError(provider="alpaca", message=f"HTTP {status_code}: {response.text}")

    @staticmethod
    def _retry_after_seconds(response: Any) -> float:
        """Derive a retry delay from a 429 response's rate-limit headers.

        Prefers an explicit ``Retry-After`` (delay in seconds), then
        ``X-RateLimit-Reset`` (epoch seconds of the window reset), and falls
        back to one rate-limit period when neither header is usable.

        Args:
            response: The 429 HTTP response.

        Returns:
            Seconds to wait before retrying (never negative).
        """
        headers = getattr(response, "headers", None) or {}
        retry_after = headers.get("Retry-After")
        if retry_after is not None:
            try:
                return max(float(retry_after), 0.0)
            except (TypeError, ValueError):
                pass
        reset = headers.get("X-RateLimit-Reset")
        if reset is not None:
            try:
                return max(float(reset) - time.time(), 0.0)
            except (TypeError, ValueError):
                pass
        return 60.0

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

    @_PAGE_RETRY
    def _get_page(self, endpoint: str, params: dict[str, Any], symbol: str) -> dict[str, Any]:
        """Fetch and parse one bars page, retrying transient failures.

        Retry lives at the page level so a failure on a later page never
        refetches earlier pages, and a 429's server-provided delay is honored
        per attempt. Each attempt acquires its own rate-limit token, since each
        attempt is a real request.

        Args:
            endpoint: The bars endpoint URL.
            params: The query parameters for this page.
            symbol: The requested symbol, for error context.

        Returns:
            The parsed JSON payload for this page.

        Raises:
            RateLimitError: On HTTP 429 (after retries are exhausted).
            AuthenticationError: On HTTP 401/403.
            DataNotAvailableError: On HTTP 404.
            NetworkError: On other HTTP errors, transport failures, or a body
                that cannot be decoded as JSON (after retries are exhausted).
        """
        # One acquisition per attempt is the only rate-limit gate, since
        # fetch_ohlcv is fully overridden and the base never acquires.
        self.rate_limiter.acquire(blocking=True)
        try:
            response = self.session.get(endpoint, params=params)
        except Exception as err:
            raise NetworkError(provider="alpaca", message=f"Request failed: {endpoint}") from err
        self._check_response_status(response, symbol)
        return self._parse_bars_response(response)

    @_PAGE_RETRY
    async def _get_page_async(
        self, endpoint: str, params: dict[str, Any], symbol: str
    ) -> dict[str, Any]:
        """Asynchronously fetch and parse one bars page, retrying transient failures.

        Mirrors :meth:`_get_page` over the async transport. The rate-limit
        acquisition is pushed to a worker thread so a throttled fetch never
        blocks the event loop.

        Args:
            endpoint: The bars endpoint URL.
            params: The query parameters for this page.
            symbol: The requested symbol, for error context.

        Returns:
            The parsed JSON payload for this page.

        Raises:
            RateLimitError: On HTTP 429 (after retries are exhausted).
            AuthenticationError: On HTTP 401/403.
            DataNotAvailableError: On HTTP 404.
            NetworkError: On other HTTP errors, transport failures, or a body
                that cannot be decoded as JSON (after retries are exhausted).
        """
        await asyncio.to_thread(self.rate_limiter.acquire, True)
        try:
            response = await self._aget(endpoint, params=params)
        except Exception as err:
            raise NetworkError(provider="alpaca", message=f"Request failed: {endpoint}") from err
        self._check_response_status(response, symbol)
        return self._parse_bars_response(response)

    def _fetch_raw_data(
        self,
        symbol: str,
        start: str,
        end: str,
        frequency: str = "daily",
        asset_class: str | None = None,
    ) -> dict[str, Any]:
        """Fetch a full bars range from Alpaca, following pagination to the end.

        Routes to the stock or crypto bars endpoint based on the resolved asset
        class. A ``BASE/QUOTE`` symbol (e.g. "BTC/USD") routes to crypto unless
        an explicit ``asset_class`` overrides the inference. Requests are repeated
        with each response's ``next_page_token`` until it is null/absent, and the
        per-page bars are merged into the single shape ``_transform_data``
        expects (a list for stocks, a per-symbol dict for crypto).

        Args:
            symbol: The symbol to fetch (case-insensitive; uppercased into the
                request for both asset classes).
            start: Inclusive start date/datetime in ISO-8601 (RFC-3339) form.
            end: Inclusive end date/datetime in ISO-8601 (RFC-3339) form.
            frequency: Canonical frequency key or alias.
            asset_class: Explicit asset class; inferred from the symbol when
                ``None``.

        Returns:
            The parsed JSON payload containing the ``bars`` data.

        Raises:
            RateLimitError: On HTTP 429 (after per-page retries).
            AuthenticationError: On HTTP 401/403.
            DataNotAvailableError: On HTTP 404.
            DataValidationError: On an unsupported frequency or asset class.
            NetworkError: On other HTTP, transport, or JSON-decoding failures.
        """
        resolved = self._resolve_asset_class(symbol, asset_class)
        endpoint, params = self._bars_request(symbol, start, end, frequency, resolved)

        accumulated: Any = None
        token: str | None = None
        while True:
            # A fresh dict per page keeps each request's params independent;
            # mutating one shared dict would otherwise rewrite the token on
            # earlier requests that already went out.
            page_params = {**params, "page_token": token} if token else params
            payload = self._get_page(endpoint, page_params, symbol)
            accumulated = self._merge_bars(accumulated, payload.get("bars"))
            token = payload.get("next_page_token")
            if not token:
                return {"bars": accumulated}

    async def _fetch_raw_data_async(
        self,
        symbol: str,
        start: str,
        end: str,
        frequency: str = "daily",
        asset_class: str | None = None,
    ) -> dict[str, Any]:
        """Asynchronously fetch a full bars range, following pagination to the end.

        Mirrors :meth:`_fetch_raw_data` over the async transport; the async
        session carries the same two-header credentials as the sync session and
        applies the same stock/crypto routing and ``next_page_token`` loop.

        Args:
            symbol: The symbol to fetch (case-insensitive; uppercased into the
                request for both asset classes).
            start: Inclusive start date/datetime in ISO-8601 (RFC-3339) form.
            end: Inclusive end date/datetime in ISO-8601 (RFC-3339) form.
            frequency: Canonical frequency key or alias.
            asset_class: Explicit asset class; inferred from the symbol when
                ``None``.

        Returns:
            The parsed JSON payload containing the ``bars`` data.

        Raises:
            RateLimitError: On HTTP 429 (after per-page retries).
            AuthenticationError: On HTTP 401/403.
            DataNotAvailableError: On HTTP 404.
            DataValidationError: On an unsupported frequency or asset class.
            NetworkError: On other HTTP, transport, or JSON-decoding failures.
        """
        resolved = self._resolve_asset_class(symbol, asset_class)
        endpoint, params = self._bars_request(symbol, start, end, frequency, resolved)

        accumulated: Any = None
        token: str | None = None
        while True:
            # A fresh dict per page keeps each request's params independent;
            # mutating one shared dict would otherwise rewrite the token on
            # earlier requests that already went out.
            page_params = {**params, "page_token": token} if token else params
            payload = await self._get_page_async(endpoint, page_params, symbol)
            accumulated = self._merge_bars(accumulated, payload.get("bars"))
            token = payload.get("next_page_token")
            if not token:
                return {"bars": accumulated}

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
            return bars.get(symbol.upper()) or bars.get(symbol) or []
        return bars or []

    def _transform_data(
        self,
        raw_data: dict[str, Any],
        symbol: str,
        asset_class: str | None = None,  # noqa: ARG002
    ) -> pl.DataFrame:
        """Transform a raw bars payload into the canonical OHLCV schema.

        Args:
            raw_data: The parsed JSON payload from a bars request.
            symbol: The requested symbol; its uppercased form is written into
                the ``symbol`` column (the canonical contract), with a crypto
                slash preserved (e.g. ``"BTC/USD"``).
            asset_class: Unused; kept so both fetch paths can thread the
                resolved asset class through a single transform signature.

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

        # Canonical contract: the symbol column is uppercase for both asset
        # classes; a crypto BASE/QUOTE slash is preserved.
        return self._bars_to_dataframe(bars, symbol.upper(), symbol)

    def _bars_to_dataframe(
        self, bars: list[dict[str, Any]], symbol_literal: str, symbol: str
    ) -> pl.DataFrame:
        """Convert a list of bar records into the canonical OHLCV DataFrame.

        Stock and crypto bars share the same ``o/h/l/c/v/t`` field names, so a
        single conversion serves both branches; only the bars-extraction and the
        symbol literal differ upstream. Crypto bars are 24/7, so no calendar or
        session filtering is applied here.

        Args:
            bars: Non-empty list of bar records with ``o/h/l/c/v/t`` keys.
            symbol_literal: The value to write into the ``symbol`` column.
            symbol: The requested symbol, used only for error context.

        Returns:
            A DataFrame with columns
            ``[timestamp, symbol, open, high, low, close, volume]`` sorted by
            timestamp.

        Raises:
            DataValidationError: If the bars cannot be transformed.
        """
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

    def fetch_ohlcv(
        self,
        symbol: str,
        start: str,
        end: str,
        frequency: str = "daily",
        asset_class: str | None = None,
    ) -> pl.DataFrame:
        """Fetch OHLCV bars for a stock or crypto symbol.

        This fully overrides the base template rather than delegating to it: the
        base signature cannot thread ``asset_class`` through, and it acquires a
        rate-limit token up front whereas this provider rate-limits per page
        inside the fetch. The base contract is reproduced here directly --
        input validation, circuit-breaker-wrapped fetch/transform/validate, and
        the same info-logging. Transient-failure retry happens per page inside
        the fetch (see ``_get_page``) rather than around the whole fetch.

        Args:
            symbol: The symbol to fetch. A ``BASE/QUOTE`` symbol (e.g. "BTC/USD")
                is treated as crypto unless ``asset_class`` overrides it.
            start: Inclusive start, ``YYYY-MM-DD`` or RFC-3339 datetime.
            end: Inclusive end, ``YYYY-MM-DD`` or RFC-3339 datetime.
            frequency: Canonical frequency key or alias.
            asset_class: Explicit asset class ("stock" or "crypto"); inferred
                from the symbol when ``None``.

        Returns:
            A DataFrame in the canonical OHLCV schema
            ``[timestamp, symbol, open, high, low, close, volume]``.

        Raises:
            ValueError: If the symbol is empty, a date bound is malformed, or
                start is after end.
            DataValidationError: If the frequency or asset class is
                unsupported, or the response cannot be transformed/validated.
            AuthenticationError: If the credentials are rejected (HTTP 401/403).
            RateLimitError: If HTTP 429 persists past the per-page retries.
            DataNotAvailableError: If the endpoint reports no data (HTTP 404).
            NetworkError: On other HTTP, transport, or JSON-decoding failures.
            CircuitBreakerOpenError: If repeated failures have opened the
                circuit breaker.
        """
        self.logger.info(
            "Fetching OHLCV data",
            symbol=symbol,
            start=start,
            end=end,
            frequency=frequency,
            provider=self.name,
        )

        self._validate_inputs(symbol, start, end, frequency)
        resolved = self._resolve_asset_class(symbol, asset_class)

        def _fetch_and_process() -> pl.DataFrame:
            raw_data = self._fetch_raw_data(symbol, start, end, frequency, asset_class=resolved)
            df = self._transform_data(raw_data, symbol, asset_class=resolved)
            return self._validate_ohlcv(df, self.name)

        validated_data = self._with_circuit_breaker(_fetch_and_process)

        self.logger.info(
            "Successfully fetched OHLCV data",
            symbol=symbol,
            rows=len(validated_data),
            provider=self.name,
        )

        return validated_data

    async def fetch_ohlcv_async(
        self,
        symbol: str,
        start: str,
        end: str,
        frequency: str = "daily",
        asset_class: str | None = None,
    ) -> pl.DataFrame:
        """Asynchronously fetch OHLCV bars for a stock or crypto symbol.

        Mirrors :meth:`fetch_ohlcv` over the async transport with the same
        resilience semantics: the circuit breaker wraps the full
        fetch/transform/validate pipeline (an open breaker refuses before any
        request goes out, and fetch failures count toward opening it), and
        transient failures retry per page inside the fetch.

        Args:
            symbol: The symbol to fetch. A ``BASE/QUOTE`` symbol (e.g. "BTC/USD")
                is treated as crypto unless ``asset_class`` overrides it.
            start: Inclusive start, ``YYYY-MM-DD`` or RFC-3339 datetime.
            end: Inclusive end, ``YYYY-MM-DD`` or RFC-3339 datetime.
            frequency: Canonical frequency key or alias.
            asset_class: Explicit asset class ("stock" or "crypto"); inferred
                from the symbol when ``None``.

        Returns:
            A DataFrame in the canonical OHLCV schema
            ``[timestamp, symbol, open, high, low, close, volume]``.

        Raises:
            ValueError: If the symbol is empty, a date bound is malformed, or
                start is after end.
            DataValidationError: If the frequency or asset class is
                unsupported, or the response cannot be transformed/validated.
            AuthenticationError: If the credentials are rejected (HTTP 401/403).
            RateLimitError: If HTTP 429 persists past the per-page retries.
            DataNotAvailableError: If the endpoint reports no data (HTTP 404).
            NetworkError: On other HTTP, transport, or JSON-decoding failures.
            CircuitBreakerOpenError: If repeated failures have opened the
                circuit breaker.
        """
        self.logger.info(
            "Fetching OHLCV data (async)",
            symbol=symbol,
            start=start,
            end=end,
            frequency=frequency,
            provider=self.name,
        )

        self._validate_inputs(symbol, start, end, frequency)
        resolved = self._resolve_asset_class(symbol, asset_class)

        async def _fetch_and_process() -> pl.DataFrame:
            raw_data = await self._fetch_raw_data_async(
                symbol, start, end, frequency, asset_class=resolved
            )
            df = self._transform_data(raw_data, symbol, asset_class=resolved)
            return self._validate_ohlcv(df, self.name)

        validated_data = await self._with_circuit_breaker_async(_fetch_and_process)

        self.logger.info(
            "Successfully fetched OHLCV data (async)",
            symbol=symbol,
            rows=len(validated_data),
            provider=self.name,
        )

        return validated_data
