"""Finnhub data provider.

Finnhub provides comprehensive financial market data with global coverage.

API Documentation: https://finnhub.io/docs/api
Pricing: https://finnhub.io/pricing

**IMPORTANT - Tier Limitations**:

FREE TIER (60 API calls/minute):
- ✅ Real-time quotes - 60 calls/minute
- ❌ Historical OHLCV - Only 30 candle calls/day (VERY limited)
- Best for: Real-time market data, not historical analysis

PAID TIER ($59.99+/month):
- ✅ Historical OHLCV - 300-600 calls/minute
- ✅ Global coverage (60+ exchanges)
- ✅ Multiple resolutions (minute to monthly)
- Required for: Backtesting, historical analysis

Example:
    >>> from ml4t.data.providers.finnhub import FinnhubProvider
    >>> provider = FinnhubProvider(api_key="your_key")
    >>> data = provider.fetch_ohlcv("AAPL", "2024-01-01", "2024-01-31")
    >>> provider.close()
"""

import os
from datetime import datetime
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
    SymbolNotFoundError,
)
from ml4t.data.providers.base import BaseProvider

logger = structlog.get_logger()


class FinnhubProvider(BaseProvider):
    """Finnhub data provider.

    Supports stocks, ETFs, forex, and crypto with multiple resolutions.

    Rate Limits:
    - Free tier: 60 requests/minute (but only 30 candle calls/day)
    - Paid tier: 300-600 requests/minute ($59.99+/month)
    """

    # Free tier: 60 requests/min = 1 per second
    DEFAULT_RATE_LIMIT: ClassVar[tuple[int, float]] = (1, 1.0)

    # Map frequency to Finnhub resolution codes
    RESOLUTION_MAP: ClassVar[dict[str, str]] = {
        "1min": "1",
        "1m": "1",
        "5min": "5",
        "5m": "5",
        "15min": "15",
        "15m": "15",
        "30min": "30",
        "30m": "30",
        "60min": "60",
        "1h": "60",
        "hour": "60",
        "hourly": "60",
        "daily": "D",
        "1d": "D",
        "day": "D",
        "D": "D",
        "weekly": "W",
        "1w": "W",
        "week": "W",
        "W": "W",
        "monthly": "M",
        "1M": "M",
        "month": "M",
        "M": "M",
    }

    def __init__(self, api_key: str | None = None, rate_limit: tuple[int, float] | None = None):
        """Initialize Finnhub provider.

        Args:
            api_key: Finnhub API key (or set FINNHUB_API_KEY env var)
            rate_limit: Optional custom rate limit (calls, period_seconds)

        Raises:
            AuthenticationError: If API key is not provided
        """
        self.api_key = api_key or os.getenv("FINNHUB_API_KEY")
        if not self.api_key:
            raise AuthenticationError(
                provider="finnhub",
                message="Finnhub API key required. Set FINNHUB_API_KEY "
                "environment variable or pass api_key parameter. "
                "Get free key at: https://finnhub.io/register",
            )

        self.base_url = "https://finnhub.io/api/v1"

        super().__init__(rate_limit=rate_limit or self.DEFAULT_RATE_LIMIT)

        self.logger.info(
            "Initialized Finnhub provider", rate_limit=rate_limit or self.DEFAULT_RATE_LIMIT
        )

    @property
    def name(self) -> str:
        """Return provider name."""
        return "finnhub"

    def _fetch_raw_data(
        self,
        symbol: str,
        start: str,
        end: str,
        frequency: str = "daily",
    ) -> dict[str, Any]:
        """Fetch raw data from Finnhub API."""
        # Map frequency to Finnhub resolution
        finnhub_resolution = self.RESOLUTION_MAP.get(frequency.lower(), frequency)
        if finnhub_resolution not in ["1", "5", "15", "30", "60", "D", "W", "M"]:
            raise DataValidationError(
                provider="finnhub",
                message=f"Unsupported frequency '{frequency}'. "
                f"Supported: {list(self.RESOLUTION_MAP.keys())}",
                field="frequency",
                value=frequency,
            )

        # Convert dates to unix timestamps
        try:
            start_dt = datetime.fromisoformat(start)
            end_dt = datetime.fromisoformat(end)
            start_ts = int(start_dt.timestamp())
            end_ts = int(end_dt.timestamp())
        except ValueError as err:
            raise DataValidationError(
                provider="finnhub",
                message=f"Invalid date format. Use YYYY-MM-DD. Error: {err}",
                field="start/end",
                value=f"{start}/{end}",
            ) from err

        # Build request
        endpoint = f"{self.base_url}/stock/candle"
        params = {
            "symbol": symbol.upper(),
            "resolution": finnhub_resolution,
            "from": start_ts,
            "to": end_ts,
            "token": self.api_key,
        }

        try:
            response = self.session.get(endpoint, params=params)

            # Check for errors
            if response.status_code == 429:
                raise RateLimitError(provider="finnhub", retry_after=60.0)
            if response.status_code in [401, 403]:
                raise AuthenticationError(provider="finnhub", message="Invalid API key")
            if response.status_code == 404:
                raise DataNotAvailableError(provider="finnhub", symbol=symbol)
            if response.status_code != 200:
                raise NetworkError(
                    provider="finnhub", message=f"HTTP {response.status_code}: {response.text}"
                )

            # Parse JSON
            try:
                data = response.json()
            except Exception as err:
                raise NetworkError(provider="finnhub", message="Failed to parse JSON") from err

            # Check status field
            if data.get("s") == "no_data":
                raise SymbolNotFoundError(provider="finnhub", symbol=symbol)
            if data.get("s") == "error":
                raise ProviderError(provider="finnhub", message=f"API error: {data.get('error')}")

            # Verify we have data
            if not data.get("c") or not data.get("t"):
                raise SymbolNotFoundError(provider="finnhub", symbol=symbol)

            return data

        except (
            AuthenticationError,
            RateLimitError,
            NetworkError,
            ProviderError,
            DataNotAvailableError,
            DataValidationError,
            SymbolNotFoundError,
        ):
            raise
        except Exception as err:
            raise NetworkError(provider="finnhub", message=f"Request failed: {endpoint}") from err

    def _transform_data(self, raw_data: dict[str, Any], symbol: str) -> pl.DataFrame:
        """Transform raw API response to Polars DataFrame."""
        try:
            # Finnhub returns arrays: c, h, l, o, t, v, s
            df = pl.DataFrame(
                {
                    "timestamp": pl.Series(raw_data["t"]).cast(pl.Int64),
                    "open": raw_data["o"],
                    "high": raw_data["h"],
                    "low": raw_data["l"],
                    "close": raw_data["c"],
                    "volume": raw_data["v"],
                }
            )

            # Convert unix timestamp to datetime
            df = df.with_columns(pl.from_epoch("timestamp", time_unit="s").alias("timestamp"))

            # Ensure numeric columns are float
            df = df.with_columns(
                [
                    pl.col("open").cast(pl.Float64),
                    pl.col("high").cast(pl.Float64),
                    pl.col("low").cast(pl.Float64),
                    pl.col("close").cast(pl.Float64),
                    pl.col("volume").cast(pl.Float64),
                ]
            )

            # Sort and add symbol
            df = df.sort("timestamp")
            df = df.with_columns(pl.lit(symbol.upper()).alias("symbol"))

            # Reorder columns
            df = df.select(["timestamp", "symbol", "open", "high", "low", "close", "volume"])

            return df

        except Exception as err:
            raise DataValidationError(
                provider="finnhub", message=f"Failed to transform data for {symbol}"
            ) from err

    async def fetch_ohlcv_async(
        self, symbol: str, start: str, end: str, frequency: str = "daily"
    ) -> "pl.DataFrame":
        """Async fetch OHLCV data using thread pool.

        Since the Finnhub SDK is synchronous, this wraps the sync call
        in asyncio.to_thread() to avoid blocking the event loop.

        Args:
            symbol: Symbol to fetch
            start: Start date (YYYY-MM-DD)
            end: End date (YYYY-MM-DD)
            frequency: Data frequency

        Returns:
            Polars DataFrame with OHLCV data
        """
        import asyncio

        return await asyncio.to_thread(self.fetch_ohlcv, symbol, start, end, frequency)
