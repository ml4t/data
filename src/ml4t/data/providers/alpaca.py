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
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

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

    def _resolve_asset_class(self, symbol: str, asset_class: str | None) -> str:
        """Resolve the asset class for a request.

        An explicit ``asset_class`` always wins. Otherwise a ``BASE/QUOTE``
        symbol (e.g. ``"BTC/USD"``) is treated as crypto and everything else as
        a stock.

        Args:
            symbol: The requested symbol.
            asset_class: An explicit asset class, or ``None`` to infer it.

        Returns:
            Either ``"crypto"`` or ``"stock"``.
        """
        if asset_class is not None:
            return asset_class
        return "crypto" if "/" in symbol else "stock"

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

    def _crypto_bars_params(
        self, symbol: str, frequency: str, start: str, end: str
    ) -> dict[str, Any]:
        """Build the query parameters for a crypto bars request.

        The crypto bars endpoint is multi-symbol: the symbol travels in the
        ``symbols`` parameter (preserving the ``BASE/QUOTE`` form verbatim) and
        no ``feed`` is sent, since crypto has a single consolidated feed.

        Args:
            symbol: The crypto symbol in ``BASE/QUOTE`` form (e.g. "BTC/USD").
            frequency: Canonical frequency key or alias.
            start: Inclusive start date/datetime in ISO-8601 (RFC-3339) form.
            end: Inclusive end date/datetime in ISO-8601 (RFC-3339) form.

        Returns:
            The query parameter mapping for the crypto bars endpoint.
        """
        return {
            "symbols": symbol,
            "timeframe": self._map_frequency(frequency),
            "start": start,
            "end": end,
            "limit": 10000,
        }

    def _bars_request(
        self, symbol: str, start: str, end: str, frequency: str, asset_class: str
    ) -> tuple[str, dict[str, Any]]:
        """Resolve the endpoint and query params for a bars request.

        Branches on the resolved asset class so the sync and async fetchers share
        one routing decision. The stock branch uppercases the symbol into the
        path; the crypto branch hits the multi-symbol endpoint and preserves the
        ``BASE/QUOTE`` symbol verbatim.

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
            endpoint = f"{self.crypto_base_url}/v1beta3/crypto/us/bars"
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
            symbol: The symbol to fetch (stocks are case-insensitive; crypto
                symbols are preserved verbatim).
            start: Inclusive start date/datetime in ISO-8601 (RFC-3339) form.
            end: Inclusive end date/datetime in ISO-8601 (RFC-3339) form.
            frequency: Canonical frequency key or alias.
            asset_class: Explicit asset class; inferred from the symbol when
                ``None``.

        Returns:
            The parsed JSON payload containing the ``bars`` data.

        Raises:
            RateLimitError, AuthenticationError, DataNotAvailableError,
            NetworkError: Per the HTTP status of the response.
        """
        resolved = self._resolve_asset_class(symbol, asset_class)
        endpoint, params = self._bars_request(symbol, start, end, frequency, resolved)

        accumulated: Any = None
        token: str | None = None
        try:
            while True:
                # A fresh dict per page keeps each request's params independent;
                # mutating one shared dict would otherwise rewrite the token on
                # earlier requests that already went out.
                page_params = {**params, "page_token": token} if token else params
                # One acquisition per page is the only rate-limit gate, since
                # fetch_ohlcv is fully overridden and the base never acquires.
                self.rate_limiter.acquire(blocking=True)
                response = self.session.get(endpoint, params=page_params)
                self._check_response_status(response.status_code, symbol, response.text)
                payload = self._parse_bars_response(response)
                accumulated = self._merge_bars(accumulated, payload.get("bars"))
                token = payload.get("next_page_token")
                if not token:
                    break
            return {"bars": accumulated}
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
        asset_class: str | None = None,
    ) -> dict[str, Any]:
        """Asynchronously fetch a full bars range, following pagination to the end.

        Mirrors :meth:`_fetch_raw_data` over the async transport; the async
        session carries the same two-header credentials as the sync session and
        applies the same stock/crypto routing and ``next_page_token`` loop.

        Args:
            symbol: The symbol to fetch (stocks are case-insensitive; crypto
                symbols are preserved verbatim).
            start: Inclusive start date/datetime in ISO-8601 (RFC-3339) form.
            end: Inclusive end date/datetime in ISO-8601 (RFC-3339) form.
            frequency: Canonical frequency key or alias.
            asset_class: Explicit asset class; inferred from the symbol when
                ``None``.

        Returns:
            The parsed JSON payload containing the ``bars`` data.

        Raises:
            RateLimitError, AuthenticationError, DataNotAvailableError,
            NetworkError: Per the HTTP status of the response.
        """
        resolved = self._resolve_asset_class(symbol, asset_class)
        endpoint, params = self._bars_request(symbol, start, end, frequency, resolved)

        accumulated: Any = None
        token: str | None = None
        try:
            while True:
                # A fresh dict per page keeps each request's params independent;
                # mutating one shared dict would otherwise rewrite the token on
                # earlier requests that already went out.
                page_params = {**params, "page_token": token} if token else params
                # One acquisition per page is the only rate-limit gate, since
                # fetch_ohlcv is fully overridden and the base never acquires.
                self.rate_limiter.acquire(blocking=True)
                response = await self._aget(endpoint, params=page_params)
                self._check_response_status(response.status_code, symbol, response.text)
                payload = self._parse_bars_response(response)
                accumulated = self._merge_bars(accumulated, payload.get("bars"))
                token = payload.get("next_page_token")
                if not token:
                    break
            return {"bars": accumulated}
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

        # Crypto symbols keep their BASE/QUOTE form verbatim; stocks uppercase.
        symbol_literal = symbol if asset_class == "crypto" else symbol.upper()
        return self._bars_to_dataframe(bars, symbol_literal, symbol)

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

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((NetworkError, RateLimitError)),
        reraise=True,
    )
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
        the same info-logging.

        Args:
            symbol: The symbol to fetch. A ``BASE/QUOTE`` symbol (e.g. "BTC/USD")
                is treated as crypto unless ``asset_class`` overrides it.
            start: Start date in YYYY-MM-DD format (inclusive).
            end: End date in YYYY-MM-DD format (inclusive).
            frequency: Canonical frequency key or alias.
            asset_class: Explicit asset class ("stock" or "crypto"); inferred
                from the symbol when ``None``.

        Returns:
            A DataFrame in the canonical OHLCV schema
            ``[timestamp, symbol, open, high, low, close, volume]``.
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

        Mirrors :meth:`fetch_ohlcv` over the async transport. The circuit
        breaker wraps a synchronous callable, so the coroutine is awaited first
        and only the transform/validate step runs inside the breaker, preserving
        failure accounting without blocking the event loop on the fetch.

        Args:
            symbol: The symbol to fetch. A ``BASE/QUOTE`` symbol (e.g. "BTC/USD")
                is treated as crypto unless ``asset_class`` overrides it.
            start: Start date in YYYY-MM-DD format (inclusive).
            end: End date in YYYY-MM-DD format (inclusive).
            frequency: Canonical frequency key or alias.
            asset_class: Explicit asset class ("stock" or "crypto"); inferred
                from the symbol when ``None``.

        Returns:
            A DataFrame in the canonical OHLCV schema
            ``[timestamp, symbol, open, high, low, close, volume]``.
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

        raw_data = await self._fetch_raw_data_async(
            symbol, start, end, frequency, asset_class=resolved
        )

        def _transform_and_validate() -> pl.DataFrame:
            df = self._transform_data(raw_data, symbol, asset_class=resolved)
            return self._validate_ohlcv(df, self.name)

        validated_data = self._with_circuit_breaker(_transform_and_validate)

        self.logger.info(
            "Successfully fetched OHLCV data (async)",
            symbol=symbol,
            rows=len(validated_data),
            provider=self.name,
        )

        return validated_data
