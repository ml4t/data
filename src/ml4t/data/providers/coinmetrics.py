"""Coin Metrics spot-market candles with exact, half-open timestamp requests."""

from __future__ import annotations

import os
import re
from datetime import UTC, datetime, timedelta
from math import isfinite
from typing import Any, ClassVar

import httpx
import polars as pl

from ml4t.data.core.exceptions import (
    AuthenticationError,
    DataValidationError,
    NetworkError,
    RateLimitError,
)
from ml4t.data.providers.base import BaseProvider
from ml4t.data.providers.protocols import ProviderCapabilities


class CoinMetricsProvider(BaseProvider):
    """Fetch real candles for one explicit Coin Metrics spot-market identifier.

    Account entitlements determine available markets, frequencies and history. An empty
    response never authorizes filling a no-trade interval with manufactured observations.
    """

    DEFAULT_RATE_LIMIT: ClassVar[tuple[int, float]] = (60, 60.0)
    FREQUENCIES: ClassVar[dict[str, str]] = {
        "minute": "1m",
        "1min": "1m",
        "1m": "1m",
        "5min": "5m",
        "5minute": "5m",
        "5m": "5m",
        "10min": "10m",
        "10m": "10m",
        "15min": "15m",
        "15minute": "15m",
        "15m": "15m",
        "30min": "30m",
        "30minute": "30m",
        "30m": "30m",
        "hour": "1h",
        "hourly": "1h",
        "1hour": "1h",
        "1h": "1h",
        "4h": "4h",
        "daily": "1d",
        "1day": "1d",
        "1d": "1d",
    }
    PATH: ClassVar[str] = "/v4/timeseries/market-candles"

    def __init__(
        self,
        api_key: str | None = None,
        rate_limit: tuple[int, float] | None = None,
        page_size: int = 1000,
    ) -> None:
        self.api_key = api_key or os.getenv("COINMETRICS_API_KEY")
        if not 1 <= page_size <= 10000:
            raise ValueError("page_size must be between 1 and 10000")
        self.page_size = page_size
        self.base_url = (
            "https://api.coinmetrics.io" if self.api_key else "https://community-api.coinmetrics.io"
        )
        super().__init__(rate_limit=rate_limit or self.DEFAULT_RATE_LIMIT)

    @property
    def name(self) -> str:
        return "coinmetrics"

    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            supports_intraday=True,
            supports_crypto=True,
            requires_api_key=True,
            rate_limit=self.DEFAULT_RATE_LIMIT,
        )

    def _classify_error_response(self, response: httpx.Response) -> Exception | None:
        if response.status_code in (401, 403):
            return AuthenticationError(
                self.name, "Market candles require authorized account access"
            )
        if response.status_code >= 400 and response.status_code not in (404, 429):
            return NetworkError(
                self.name,
                f"HTTP {response.status_code} for market candles",
                retryable=response.status_code >= 500,
            )
        return None

    def _fetch_and_transform_data(
        self, symbol: str, start: str, end: str, frequency: str = "daily"
    ) -> pl.DataFrame:
        return self.fetch_ohlcv_range(
            symbol,
            datetime.fromisoformat(start).replace(tzinfo=UTC),
            datetime.fromisoformat(end).replace(tzinfo=UTC) + timedelta(days=1),
            frequency,
        )

    def fetch_ohlcv_range(
        self, symbol: str, start: datetime, end: datetime, frequency: str = "minute"
    ) -> pl.DataFrame:
        """Return observed candles in [start, end), preserving the exact spot market.

        Bounds must be aware instants. Candle starts use UTC, left labels, a UTC-midnight
        grid and no custom daily offset. Requests never expand a ticker into a universe.
        """
        if not re.fullmatch(r"[a-z0-9_.]+-[a-z0-9_.]+-[a-z0-9_.]+-spot", symbol):
            raise ValueError(
                "symbol must name one exact lower-case exchange-base-quote-spot market"
            )
        if (
            start.tzinfo is None
            or end.tzinfo is None
            or start.utcoffset() is None
            or end.utcoffset() is None
        ):
            raise ValueError("timestamp bounds must be timezone-aware")
        start, end = start.astimezone(UTC), end.astimezone(UTC)
        if start >= end:
            raise ValueError("start must be before end")
        try:
            native_frequency = self.FREQUENCIES[frequency]
        except KeyError as error:
            raise ValueError(f"Unsupported market-candle frequency: {frequency}") from error
        params: dict[str, Any] = {
            "markets": symbol,
            "frequency": native_frequency,
            "start_time": start.isoformat(),
            "end_time": end.isoformat(),
            "start_inclusive": "true",
            "end_inclusive": "false",
            "timezone": "UTC",
            "paging_from": "start",
            "page_size": self.page_size,
        }
        if self.api_key:
            params["api_key"] = self.api_key
        url = f"{self.base_url}{self.PATH}"
        fixed_params = dict(params)
        seen: set[str] = set()
        rows: dict[datetime, tuple[Any, ...]] = {}
        while True:
            try:
                payload = self._request_json(url, resource=symbol, params=params)
            except RateLimitError:
                raise
            except NetworkError as error:
                raise NetworkError(
                    self.name,
                    "Market-candle request failed",
                    retryable=error.retryable,
                    retry_after=error.retry_after,
                ) from None
            data = payload.get("data")
            if not isinstance(data, list):
                raise DataValidationError(self.name, "Market candles response requires a data list")
            for record in data:
                try:
                    if not isinstance(record, dict) or record["market"] != symbol:
                        raise ValueError("response market differs from requested market")
                    timestamp = datetime.fromisoformat(record["time"].replace("Z", "+00:00"))
                    if timestamp.utcoffset() != timedelta(0):
                        raise ValueError("response timestamp must use UTC")
                    numbers = tuple(
                        float(record[key])
                        for key in (
                            "price_open",
                            "price_high",
                            "price_low",
                            "price_close",
                            "volume",
                        )
                    )
                    if not all(isfinite(value) for value in numbers):
                        raise ValueError("non-finite candle value")
                except (KeyError, TypeError, ValueError, AttributeError) as error:
                    raise DataValidationError(
                        self.name, f"Malformed market candle: {error}"
                    ) from error
                if start <= timestamp < end:
                    row = (timestamp, symbol.upper(), *numbers)
                    if timestamp in rows and rows[timestamp] != row:
                        raise DataValidationError(self.name, "Conflicting duplicate market candle")
                    rows[timestamp] = row
            next_url = payload.get("next_page_url")
            if next_url is None:
                if payload.get("next_page_token"):
                    raise DataValidationError(self.name, "Continuation token without next_page_url")
                break
            if not isinstance(next_url, str) or not next_url:
                raise DataValidationError(self.name, "Malformed next_page_url")
            try:
                next_page = httpx.URL(next_url)
            except httpx.InvalidURL:
                raise DataValidationError(self.name, "Malformed next_page_url") from None
            origin = httpx.URL(self.base_url)
            if (
                (next_page.scheme, next_page.host, next_page.port, next_page.path)
                != (origin.scheme, origin.host, origin.port, self.PATH)
                or next_page.userinfo
                or next_page.fragment
            ):
                raise DataValidationError(self.name, "Unsafe market-candle continuation URL")
            page_params = dict(next_page.params)
            if set(page_params) - set(fixed_params) - {"next_page_token"}:
                raise DataValidationError(self.name, "Unexpected continuation parameters")
            for key, value in fixed_params.items():
                values = next_page.params.get_list(key)
                if values and values != [str(value)]:
                    raise DataValidationError(self.name, "Continuation changed a request filter")
            token = page_params.get("next_page_token")
            if not token or len(next_page.params.get_list("next_page_token")) != 1:
                raise DataValidationError(self.name, "Continuation URL has no unique page token")
            if token in seen:
                raise DataValidationError(self.name, "Market-candle page repeated")
            seen.add(token)
            # Reapply original bounds even when the server's continuation link omits them.
            params = {**fixed_params, "next_page_token": token}
        if not rows:
            return self._create_empty_dataframe()
        frame = pl.DataFrame(list(rows.values()), schema=self.CANONICAL_SCHEMA, orient="row").sort(
            "timestamp"
        )
        return self._validate_ohlcv(frame, self.name, symbol)
