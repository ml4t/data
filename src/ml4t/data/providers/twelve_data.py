"""Twelve Data provider for multi-asset market data.

Twelve Data provides comprehensive market data for stocks, forex, and cryptocurrencies
across 50+ global exchanges.

API Documentation: https://twelvedata.com/docs

Free Tier Limits:
- 800 API requests per day
- 8 requests per minute (strictest rate limit)
- 15+ years of historical data

Example:
    >>> from ml4t.data.providers.twelve_data import TwelveDataProvider
    >>> provider = TwelveDataProvider(api_key="your_key")
    >>> data = provider.fetch_ohlcv("AAPL", "2024-01-01", "2024-01-31")

Async Example:
    >>> async with TwelveDataProvider(api_key="your_key") as provider:
    ...     data = await provider.fetch_ohlcv_async("AAPL", "2024-01-01", "2024-01-31")
"""

import os
from typing import Any

import httpx
import polars as pl
import structlog

from ml4t.data.core.exceptions import (
    AuthenticationError,
    NetworkError,
    ProviderError,
    RateLimitError,
    SymbolNotFoundError,
)
from ml4t.data.providers.base import BaseProvider
from ml4t.data.providers.mixins import AsyncSessionMixin

logger = structlog.get_logger()


class TwelveDataProvider(AsyncSessionMixin, BaseProvider):
    """Twelve Data provider for multi-asset market data.

    Supports stocks, ETFs, indices, forex, and cryptocurrencies.
    Async support for 10x faster batch fetches.

    Rate Limits (Free Tier):
    - 8 requests per minute (strictest limit)
    - 800 requests per day
    """

    # 8 requests per minute = 0.133/second
    DEFAULT_RATE_LIMIT = (8, 60.0)

    # Map frequency to Twelve Data interval format
    FREQUENCY_MAP: dict[str, str] = {
        "daily": "1day",
        "1min": "1min",
        "5min": "5min",
        "15min": "15min",
        "30min": "30min",
        "1h": "1h",
        "1hour": "1h",
        "4h": "4h",
        "1week": "1week",
        "1month": "1month",
    }

    def __init__(self, api_key: str | None = None, rate_limit: tuple[int, float] | None = None):
        """Initialize Twelve Data provider.

        Args:
            api_key: Twelve Data API key (or set TWELVE_DATA_API_KEY env var)
            rate_limit: Optional (calls, period_seconds) tuple to override default rate limiting

        Raises:
            AuthenticationError: If API key is not provided
        """
        self.api_key = api_key or os.getenv("TWELVE_DATA_API_KEY")
        if not self.api_key:
            raise AuthenticationError(
                provider="twelve_data",
                message="Twelve Data API key required. "
                "Set TWELVE_DATA_API_KEY environment variable or pass api_key parameter. "
                "Get your free API key at: https://twelvedata.com/",
            )

        self.base_url = "https://api.twelvedata.com"

        super().__init__(rate_limit=rate_limit or self.DEFAULT_RATE_LIMIT)

    @property
    def name(self) -> str:
        """Return provider name."""
        return "twelve_data"

    def _fetch_raw_data(
        self,
        symbol: str,
        start: str,
        end: str,
        frequency: str = "daily",
    ) -> dict[str, Any]:
        """Fetch raw OHLCV data from Twelve Data API."""
        # Map frequency to Twelve Data interval format
        interval = self.FREQUENCY_MAP.get(frequency, frequency)

        endpoint = f"{self.base_url}/time_series"
        params = {
            "symbol": symbol,
            "interval": interval,
            "start_date": start,
            "end_date": end,
            "timezone": "America/New_York",
            "apikey": self.api_key,
            "outputsize": 5000,
            "format": "JSON",
        }

        try:
            response = self.session.get(endpoint, params=params)

            # Check for errors
            if response.status_code == 429:
                raise RateLimitError(self.name, retry_after=60.0)
            if response.status_code != 200:
                raise NetworkError(self.name, f"HTTP {response.status_code}: {response.text}")

            data = response.json()

            # Check for API errors
            if "status" in data and data["status"] == "error":
                error_msg = data.get("message", "Unknown error")
                if "rate" in error_msg.lower():
                    raise RateLimitError(provider=self.name, retry_after=60.0)
                elif "not found" in error_msg.lower() or "invalid symbol" in error_msg.lower():
                    raise SymbolNotFoundError(provider=self.name, symbol=symbol)
                else:
                    raise ProviderError(provider=self.name, message=error_msg)

            # Check for empty data
            if "values" not in data or not data["values"]:
                raise SymbolNotFoundError(provider=self.name, symbol=symbol)

            return data

        except httpx.RequestError as e:
            raise NetworkError(self.name, f"Request failed: {e}") from e

    def _transform_data(self, raw_data: dict[str, Any], symbol: str) -> pl.DataFrame:
        """Transform Twelve Data response to standardized schema."""
        if not raw_data.get("values"):
            return self._create_empty_dataframe()

        # Convert to Polars DataFrame
        df = pl.DataFrame(raw_data["values"])

        # Rename columns
        df = df.rename({"datetime": "timestamp"})

        # Convert timestamp to datetime
        df = df.with_columns(pl.col("timestamp").str.to_datetime().alias("timestamp"))

        # Convert OHLCV columns to float
        for col in ["open", "high", "low", "close", "volume"]:
            if col in df.columns:
                df = df.with_columns(pl.col(col).cast(pl.Float64))

        # Sort, add symbol, and select in standard order
        df = df.sort("timestamp")
        df = df.with_columns(pl.lit(symbol.upper()).alias("symbol"))
        df = df.select(["timestamp", "symbol", "open", "high", "low", "close", "volume"])

        return df

    def _create_empty_dataframe(self) -> pl.DataFrame:
        """Create an empty DataFrame with the correct schema."""
        return pl.DataFrame(
            {
                "timestamp": [],
                "symbol": [],
                "open": [],
                "high": [],
                "low": [],
                "close": [],
                "volume": [],
            }
        ).with_columns(
            [
                pl.col("timestamp").cast(pl.Datetime),
                pl.col("symbol").cast(pl.String),
                pl.col("open").cast(pl.Float64),
                pl.col("high").cast(pl.Float64),
                pl.col("low").cast(pl.Float64),
                pl.col("close").cast(pl.Float64),
                pl.col("volume").cast(pl.Float64),
            ]
        )

    async def _fetch_raw_data_async(
        self,
        symbol: str,
        start: str,
        end: str,
        frequency: str = "daily",
    ) -> dict[str, Any]:
        """Async fetch raw OHLCV data from Twelve Data API."""
        interval = self.FREQUENCY_MAP.get(frequency, frequency)

        endpoint = f"{self.base_url}/time_series"
        params = {
            "symbol": symbol,
            "interval": interval,
            "start_date": start,
            "end_date": end,
            "timezone": "America/New_York",
            "apikey": self.api_key,
            "outputsize": 5000,
            "format": "JSON",
        }

        try:
            response = await self._aget(endpoint, params=params)

            if response.status_code == 429:
                raise RateLimitError(self.name, retry_after=60.0)
            if response.status_code != 200:
                raise NetworkError(self.name, f"HTTP {response.status_code}: {response.text}")

            data = response.json()

            if "status" in data and data["status"] == "error":
                error_msg = data.get("message", "Unknown error")
                if "rate" in error_msg.lower():
                    raise RateLimitError(provider=self.name, retry_after=60.0)
                elif "not found" in error_msg.lower() or "invalid symbol" in error_msg.lower():
                    raise SymbolNotFoundError(provider=self.name, symbol=symbol)
                else:
                    raise ProviderError(provider=self.name, message=error_msg)

            if "values" not in data or not data["values"]:
                raise SymbolNotFoundError(provider=self.name, symbol=symbol)

            return data

        except httpx.RequestError as e:
            raise NetworkError(self.name, f"Request failed: {e}") from e

    async def fetch_ohlcv_async(
        self,
        symbol: str,
        start: str,
        end: str,
        frequency: str = "daily",
    ) -> pl.DataFrame:
        """Async fetch OHLCV data for a symbol.

        This is 3-10x faster than sync when fetching multiple symbols
        concurrently using asyncio.gather() or async_batch_load().

        Args:
            symbol: Symbol to fetch (e.g., "AAPL")
            start: Start date (YYYY-MM-DD)
            end: End date (YYYY-MM-DD)
            frequency: Data frequency

        Returns:
            Polars DataFrame with OHLCV data

        Example:
            async with TwelveDataProvider(api_key="key") as provider:
                df = await provider.fetch_ohlcv_async("AAPL", "2024-01-01", "2024-06-30")
        """
        self.logger.info(
            f"Fetching {frequency} OHLCV (async)",
            symbol=symbol,
            start=start,
            end=end,
        )

        raw_data = await self._fetch_raw_data_async(symbol, start, end, frequency)
        df = self._transform_data(raw_data, symbol)

        self.logger.info(f"Fetched {len(df)} records (async)", symbol=symbol)

        return df
