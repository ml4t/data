"""Tiingo data provider.

Tiingo provides stock market data with extensive historical coverage and
adjusted prices for splits/dividends.

API Documentation: https://www.tiingo.com/documentation/end-of-day

Free Tier Limits:
- 1000 API calls per day
- Historical daily data (5+ years)
- 500 unique symbols per month

Example:
    >>> from ml4t.data.providers.tiingo import TiingoProvider
    >>> provider = TiingoProvider(api_key="your_key")
    >>> data = provider.fetch_ohlcv("AAPL", "2024-01-01", "2024-01-31")
    >>> provider.close()
"""

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
    SymbolNotFoundError,
)
from ml4t.data.providers.base import BaseProvider

logger = structlog.get_logger()


class TiingoProvider(BaseProvider):
    """Tiingo data provider.

    Supports stocks and ETFs with daily OHLCV data (raw and adjusted).

    Rate Limits (Free Tier):
    - 1000 requests per day
    - 500 unique symbols per month
    """

    # Free tier: 1000 requests/day = ~0.69 per minute = 1 per 86.4 seconds
    # Conservative: 1 request per 90 seconds
    DEFAULT_RATE_LIMIT: ClassVar[tuple[int, float]] = (1, 90.0)

    # Map frequency to Tiingo intervals
    FREQUENCY_MAP: ClassVar[dict[str, str]] = {
        "daily": "daily",
        "1d": "daily",
        "day": "daily",
        "weekly": "weekly",
        "1w": "weekly",
        "week": "weekly",
        "monthly": "monthly",
        "1m": "monthly",
        "month": "monthly",
    }

    def __init__(self, api_key: str | None = None, rate_limit: tuple[int, float] | None = None):
        """Initialize Tiingo provider.

        Args:
            api_key: Tiingo API key (or set TIINGO_API_KEY env var)
            rate_limit: Optional custom rate limit (calls, period_seconds)

        Raises:
            AuthenticationError: If API key is not provided
        """
        self.api_key = api_key or os.getenv("TIINGO_API_KEY")
        if not self.api_key:
            raise AuthenticationError(
                provider="tiingo",
                message="Tiingo API key required. Set TIINGO_API_KEY "
                "environment variable or pass api_key parameter. "
                "Get free key at: https://www.tiingo.com/",
            )

        self.base_url = "https://api.tiingo.com"

        # Add authentication header
        session_config = {"headers": {"Authorization": f"Token {self.api_key}"}}

        super().__init__(
            rate_limit=rate_limit or self.DEFAULT_RATE_LIMIT, session_config=session_config
        )

        self.logger.info(
            "Initialized Tiingo provider", rate_limit=rate_limit or self.DEFAULT_RATE_LIMIT
        )

    @property
    def name(self) -> str:
        """Return provider name."""
        return "tiingo"

    def _fetch_raw_data(
        self,
        symbol: str,
        start: str,
        end: str,
        frequency: str = "daily",
    ) -> list[dict[str, Any]]:
        """Fetch raw data from Tiingo API."""
        # Map frequency to Tiingo interval
        interval = self.FREQUENCY_MAP.get(frequency.lower())
        if not interval:
            raise DataValidationError(
                provider="tiingo",
                message=f"Unsupported frequency '{frequency}'. Supported: {list(self.FREQUENCY_MAP.keys())}",
                field="frequency",
                value=frequency,
            )

        # Build request
        endpoint = f"{self.base_url}/tiingo/daily/{symbol.upper()}/prices"
        params = {
            "startDate": start,
            "endDate": end,
            "resampleFreq": interval,
            "format": "json",
        }

        try:
            response = self.session.get(endpoint, params=params)

            # Check for errors
            if response.status_code == 429:
                raise RateLimitError(provider="tiingo", retry_after=60.0)
            if response.status_code == 401:
                raise AuthenticationError(provider="tiingo", message="Invalid API key")
            if response.status_code == 404:
                raise SymbolNotFoundError(provider="tiingo", symbol=symbol)
            if response.status_code != 200:
                raise NetworkError(
                    provider="tiingo", message=f"HTTP {response.status_code}: {response.text}"
                )

            # Parse JSON
            try:
                data = response.json()
            except Exception as err:
                raise NetworkError(provider="tiingo", message="Failed to parse JSON") from err

            # Check if data is empty
            if not data:
                raise SymbolNotFoundError(provider="tiingo", symbol=symbol)

            # Check for API-level errors
            if isinstance(data, dict) and "detail" in data:
                raise ProviderError(provider="tiingo", message=f"API error: {data['detail']}")

            return data

        except (
            AuthenticationError,
            RateLimitError,
            NetworkError,
            ProviderError,
            DataNotAvailableError,
            SymbolNotFoundError,
        ):
            raise
        except Exception as err:
            raise NetworkError(provider="tiingo", message=f"Request failed: {endpoint}") from err

    def _transform_data(self, raw_data: list[dict[str, Any]], symbol: str) -> pl.DataFrame:
        """Transform raw API response to Polars DataFrame."""
        if not raw_data:
            return self._create_empty_dataframe()

        try:
            df = pl.DataFrame(raw_data)

            # Convert date to datetime (Tiingo returns ISO 8601 format)
            df = df.with_columns(pl.col("date").str.to_datetime(format="%+").alias("timestamp"))
            df = df.drop("date")

            # Rename adjusted columns to adj_* format
            # Tiingo returns both unadjusted (close, high, etc.) and adjusted (adjClose, adjHigh, etc.)
            adj_rename_map = {
                "adjClose": "adj_close",
                "adjHigh": "adj_high",
                "adjLow": "adj_low",
                "adjOpen": "adj_open",
                "adjVolume": "adj_volume",
            }
            df = df.rename({k: v for k, v in adj_rename_map.items() if k in df.columns})

            # Rename corporate action columns
            corp_action_map = {
                "divCash": "dividend",
                "splitFactor": "split_factor",
            }
            df = df.rename({k: v for k, v in corp_action_map.items() if k in df.columns})

            # Convert numeric columns to float
            numeric_cols = [
                "open",
                "high",
                "low",
                "close",
                "volume",
                "adj_open",
                "adj_high",
                "adj_low",
                "adj_close",
                "adj_volume",
                "dividend",
                "split_factor",
            ]
            for col in numeric_cols:
                if col in df.columns:
                    df = df.with_columns(pl.col(col).cast(pl.Float64))

            # Sort, add symbol, and reorder columns (standard OHLCV first, then extras)
            df = df.sort("timestamp")
            df = df.with_columns(pl.lit(symbol.upper()).alias("symbol"))

            # Standard columns in order (only those present), plus any extra columns
            base_cols = ["timestamp", "symbol", "open", "high", "low", "close", "volume"]
            present_base = [c for c in base_cols if c in df.columns]
            extra_cols = [c for c in df.columns if c not in base_cols]
            df = df.select(present_base + extra_cols)

            return df

        except Exception as err:
            raise DataValidationError(
                provider="tiingo", message=f"Failed to transform data for {symbol}"
            ) from err

    async def fetch_ohlcv_async(
        self, symbol: str, start: str, end: str, frequency: str = "daily"
    ) -> "pl.DataFrame":
        """Async fetch OHLCV data using thread pool.

        Since the Tiingo SDK is synchronous, this wraps the sync call
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
