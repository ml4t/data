"""Polygon.io data provider.

Polygon.io offers comprehensive financial data across multiple asset classes:
- Stocks (equities)
- Options
- Cryptocurrencies
- Forex

Rate Limits:
    - Free tier: 5 requests/minute
    - Basic tier: 100 requests/minute

Authentication:
    Requires API key from https://polygon.io/
    Set POLYGON_API_KEY environment variable or pass api_key parameter

Example:
    >>> from ml4t.data.providers.polygon import PolygonProvider
    >>> provider = PolygonProvider(api_key="your_key")
    >>> data = provider.fetch_ohlcv("AAPL", "2024-01-01", "2024-01-31")
    >>> provider.close()
"""

import os
from typing import ClassVar

import polars as pl

from ml4t.data.core.exceptions import (
    AuthenticationError,
    DataValidationError,
    NetworkError,
    ProviderError,
    RateLimitError,
    SymbolNotFoundError,
)
from ml4t.data.providers.base import BaseProvider


class PolygonProvider(BaseProvider):
    """Polygon.io data provider.

    Supports stocks, options, cryptocurrencies, and forex data.
    """

    DEFAULT_RATE_LIMIT: ClassVar[tuple[int, float]] = (5, 1.0)  # 5 requests per second (free tier)

    # Map frequencies to Polygon timespans
    FREQUENCY_MAP: ClassVar[dict[str, str]] = {
        "day": "day",
        "daily": "day",
        "1d": "day",
        "1day": "day",
        "week": "week",
        "weekly": "week",
        "1w": "week",
        "1week": "week",
        "month": "month",
        "monthly": "month",
        "1M": "month",
        "1month": "month",
        "hour": "hour",
        "hourly": "hour",
        "1h": "hour",
        "1hour": "hour",
        "minute": "minute",
        "1m": "minute",
        "1minute": "minute",
    }

    def __init__(self, api_key: str | None = None, rate_limit: tuple[int, float] | None = None):
        """Initialize Polygon provider.

        Args:
            api_key: Polygon API key. If None, reads from POLYGON_API_KEY env var
            rate_limit: Optional custom rate limit (calls, period_seconds)

        Raises:
            AuthenticationError: If API key is not provided
        """
        super().__init__(rate_limit=rate_limit or self.DEFAULT_RATE_LIMIT)

        self.api_key = api_key or os.getenv("POLYGON_API_KEY")
        if not self.api_key:
            raise AuthenticationError(
                provider="polygon",
                message="API key required. Set POLYGON_API_KEY environment variable "
                "or pass api_key parameter. Get your key at: https://polygon.io/",
            )

        self.base_url = "https://api.polygon.io"

        self.logger.info(
            "Initialized Polygon provider", rate_limit=rate_limit or self.DEFAULT_RATE_LIMIT
        )

    @property
    def name(self) -> str:
        """Return provider name."""
        return "polygon"

    def _fetch_raw_data(
        self,
        symbol: str,
        start: str,
        end: str,
        frequency: str = "day",
    ) -> dict:
        """Fetch raw data from Polygon API."""
        timespan = self.FREQUENCY_MAP.get(frequency.lower())
        if not timespan:
            raise DataValidationError(
                provider="polygon",
                message=f"Unsupported frequency '{frequency}'. Supported: {list(self.FREQUENCY_MAP.keys())}",
                field="frequency",
                value=frequency,
            )

        endpoint = (
            f"{self.base_url}/v2/aggs/ticker/{symbol.upper()}/range/1/{timespan}/{start}/{end}"
        )
        params = {
            "adjusted": "true",
            "sort": "asc",
            "limit": 50000,
            "apiKey": self.api_key,
        }

        try:
            response = self.session.get(endpoint, params=params)

            # Check for errors
            if response.status_code == 401:
                raise AuthenticationError(
                    provider="polygon",
                    message="Invalid API key. Get your key at: https://polygon.io/",
                )
            elif response.status_code == 429:
                raise RateLimitError(provider="polygon", retry_after=60.0)
            elif response.status_code != 200:
                raise NetworkError(
                    provider="polygon",
                    message=f"API error (HTTP {response.status_code}): {response.text}",
                )

            # Parse JSON
            try:
                data = response.json()
            except Exception as err:
                raise NetworkError(
                    provider="polygon", message="Failed to parse JSON response"
                ) from err

            # Check for API-level errors
            if data.get("status") == "ERROR":
                error_msg = data.get("error", "Unknown error")
                # Check if it's a symbol not found error
                if "NOT_FOUND" in error_msg or "not found" in error_msg.lower():
                    raise SymbolNotFoundError(
                        provider="polygon", symbol=symbol, details={"error": error_msg}
                    )
                raise ProviderError(provider="polygon", message=f"API error: {error_msg}")

            return data

        except (
            AuthenticationError,
            RateLimitError,
            NetworkError,
            ProviderError,
            SymbolNotFoundError,
        ):
            raise
        except Exception as err:
            raise NetworkError(provider="polygon", message=f"Request failed: {endpoint}") from err

    def _transform_data(self, raw_data: dict, symbol: str) -> pl.DataFrame:
        """Transform raw API response to Polars DataFrame."""
        if not raw_data.get("results"):
            self.logger.warning(f"No data returned for {symbol}")
            raise SymbolNotFoundError(
                provider="polygon",
                symbol=symbol,
                details={"message": "No results returned from API"},
            )

        try:
            df = pl.DataFrame(raw_data["results"])

            # Rename columns
            df = df.rename(
                {
                    "t": "timestamp_ms",
                    "o": "open",
                    "h": "high",
                    "l": "low",
                    "c": "close",
                    "v": "volume",
                }
            )

            # Convert timestamp from milliseconds to datetime
            df = df.with_columns(pl.from_epoch("timestamp_ms", time_unit="ms").alias("timestamp"))
            df = df.drop("timestamp_ms")

            # Convert OHLCV columns to float
            for col in ["open", "high", "low", "close", "volume"]:
                if col in df.columns:
                    df = df.with_columns(pl.col(col).cast(pl.Float64))

            # Sort, add symbol, and select in standard order
            df = df.sort("timestamp")
            df = df.with_columns(pl.lit(symbol.upper()).alias("symbol"))
            df = df.select(["timestamp", "symbol", "open", "high", "low", "close", "volume"])

            return df

        except Exception as err:
            raise DataValidationError(
                provider="polygon", message=f"Failed to transform data for {symbol}"
            ) from err

    async def fetch_ohlcv_async(
        self, symbol: str, start: str, end: str, frequency: str = "daily"
    ) -> "pl.DataFrame":
        """Async fetch OHLCV data using thread pool.

        Since the Polygon SDK is synchronous, this wraps the sync call
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
