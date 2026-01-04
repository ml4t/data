"""EODHD data provider.

EODHD provides affordable global equities data with 60+ exchanges and
150,000+ tickers worldwide.

API Documentation: https://eodhd.com/financial-apis/

Free Tier Limits:
- 500 API calls per day
- 1 year historical depth
- Daily/weekly/monthly OHLCV data
- Global coverage (60+ exchanges)

Symbol Format:
- Use SYMBOL.EXCHANGE format (e.g., "AAPL.US", "VOD.LSE", "BMW.FRA")

Example:
    >>> from ml4t.data.providers.eodhd import EODHDProvider
    >>> provider = EODHDProvider(api_key="your_key")
    >>> data = provider.fetch_ohlcv("AAPL", "2024-01-01", "2024-01-31", exchange="US")
    >>> provider.close()

Async Example:
    >>> async with EODHDProvider(api_key="your_key") as provider:
    ...     data = await provider.fetch_ohlcv_async("AAPL", "2024-01-01", "2024-01-31")
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
)
from ml4t.data.providers.base import BaseProvider
from ml4t.data.providers.mixins import AsyncSessionMixin

logger = structlog.get_logger()


class EODHDProvider(AsyncSessionMixin, BaseProvider):
    """EODHD data provider.

    Supports global equities with daily/weekly/monthly OHLCV data.

    Rate Limits (Free Tier):
    - 500 requests per day
    - 1 year historical depth

    Supports both sync and async operations:
        # Sync
        provider = EODHDProvider()
        df = provider.fetch_ohlcv("AAPL", start, end)

        # Async (10x faster for batch fetches)
        async with EODHDProvider() as provider:
            df = await provider.fetch_ohlcv_async("AAPL", start, end)
    """

    # Conservative: 500 requests/day = ~1 per 3 minutes
    DEFAULT_RATE_LIMIT: ClassVar[tuple[int, float]] = (1, 180.0)

    # Map frequency to EODHD period codes
    FREQUENCY_MAP: ClassVar[dict[str, str]] = {
        "daily": "d",
        "1d": "d",
        "day": "d",
        "weekly": "w",
        "1w": "w",
        "week": "w",
        "monthly": "m",
        "1M": "m",
        "month": "m",
    }

    def __init__(
        self,
        api_key: str | None = None,
        exchange: str = "US",
        rate_limit: tuple[int, float] | None = None,
    ):
        """Initialize EODHD provider.

        Args:
            api_key: EODHD API key (or set EODHD_API_KEY env var)
            exchange: Default exchange code (e.g., "US", "LSE", "FRA")
            rate_limit: Optional custom rate limit (calls, period_seconds)

        Raises:
            AuthenticationError: If API key is not provided
        """
        self.api_key = api_key or os.getenv("EODHD_API_KEY")
        if not self.api_key:
            raise AuthenticationError(
                provider="eodhd",
                message="EODHD API key required. Set EODHD_API_KEY "
                "environment variable or pass api_key parameter. "
                "Get free key at: https://eodhd.com/register",
            )

        self.base_url = "https://eodhd.com/api"
        self.default_exchange = exchange

        super().__init__(rate_limit=rate_limit or self.DEFAULT_RATE_LIMIT)

        self.logger.info(
            "Initialized EODHD provider",
            default_exchange=exchange,
        )

    @property
    def name(self) -> str:
        """Return provider name."""
        return "eodhd"

    def _fetch_raw_data(
        self,
        symbol: str,
        start: str,
        end: str,
        frequency: str = "daily",
        exchange: str | None = None,
    ) -> list[dict[str, Any]]:
        """Fetch raw data from EODHD API."""
        exchange_code = exchange or self.default_exchange

        # Map frequency to EODHD period
        period = self.FREQUENCY_MAP.get(frequency.lower())
        if not period:
            raise DataValidationError(
                provider="eodhd",
                message=f"Unsupported frequency '{frequency}'. "
                f"Supported: {list(self.FREQUENCY_MAP.keys())}",
                field="frequency",
                value=frequency,
            )

        # Format symbol with exchange: AAPL.US
        if "." in symbol:
            formatted_symbol = symbol.upper()
        else:
            formatted_symbol = f"{symbol.upper()}.{exchange_code.upper()}"

        # Build request
        endpoint = f"{self.base_url}/eod/{formatted_symbol}"
        params = {
            "api_token": self.api_key,
            "fmt": "json",
            "period": period,
            "from": start,
            "to": end,
        }

        try:
            self.rate_limiter.acquire(blocking=True)
            response = self.session.get(endpoint, params=params)

            # Check for errors
            if response.status_code == 429:
                raise RateLimitError(provider="eodhd", retry_after=180.0)
            if response.status_code in [401, 403]:
                raise AuthenticationError(
                    provider="eodhd", message="Invalid API key or unauthorized access"
                )
            if response.status_code == 404:
                raise DataNotAvailableError(
                    provider="eodhd", symbol=formatted_symbol, start=start, end=end
                )
            if response.status_code != 200:
                raise NetworkError(
                    provider="eodhd", message=f"HTTP {response.status_code}: {response.text}"
                )

            # Parse JSON
            try:
                data = response.json()
            except Exception as err:
                raise NetworkError(
                    provider="eodhd", message="Failed to parse JSON response"
                ) from err

            # Check if data is empty
            if not data:
                raise DataNotAvailableError(provider="eodhd", symbol=formatted_symbol)

            # Check for API-level errors
            if isinstance(data, dict) and "errors" in data:
                raise ProviderError(provider="eodhd", message=f"API error: {data['errors']}")

            # Check for tier limitation warnings
            if (
                isinstance(data, list)
                and len(data) == 1
                and isinstance(data[0], dict)
                and "warning" in data[0]
            ):
                raise DataNotAvailableError(
                    provider="eodhd",
                    symbol=formatted_symbol,
                    details={
                        "warning": data[0]["warning"],
                        "tier_limitation": "Free tier is limited to 1 year of historical data",
                    },
                )

            return data

        except (
            AuthenticationError,
            RateLimitError,
            NetworkError,
            ProviderError,
            DataNotAvailableError,
        ):
            raise
        except Exception as err:
            raise NetworkError(provider="eodhd", message=f"Request failed: {endpoint}") from err

    def _transform_data(self, raw_data: list[dict[str, Any]], symbol: str) -> pl.DataFrame:
        """Transform raw API response to Polars DataFrame."""
        if not raw_data:
            return self._create_empty_dataframe()

        try:
            df = pl.DataFrame(raw_data)

            # Convert date to datetime
            df = df.with_columns(pl.col("date").str.to_date().cast(pl.Datetime).alias("timestamp"))
            df = df.drop("date")

            # Use adjusted close, drop unadjusted
            if "close" in df.columns:
                df = df.drop("close")
            if "adjusted_close" in df.columns:
                df = df.rename({"adjusted_close": "close"})

            # Convert numeric columns to float
            for col in ["open", "high", "low", "close", "volume"]:
                if col in df.columns:
                    df = df.with_columns(pl.col(col).cast(pl.Float64))

            # Sort and add symbol
            df = df.sort("timestamp")
            df = df.with_columns(pl.lit(symbol.upper()).alias("symbol"))

            # Select final columns
            df = df.select(["timestamp", "symbol", "open", "high", "low", "close", "volume"])

            return df

        except Exception as err:
            raise DataValidationError(
                provider="eodhd", message=f"Failed to transform data for {symbol}"
            ) from err

    def fetch_ohlcv(
        self,
        symbol: str,
        start: str,
        end: str,
        frequency: str = "daily",
        exchange: str | None = None,
    ) -> pl.DataFrame:
        """Fetch OHLCV data for a symbol.

        Args:
            symbol: Stock symbol (e.g., "AAPL")
            start: Start date (YYYY-MM-DD)
            end: End date (YYYY-MM-DD)
            frequency: Data frequency (daily, weekly, monthly)
            exchange: Exchange code (e.g., "US", "LSE", "FRA")

        Returns:
            Polars DataFrame with OHLCV data

        Example:
            >>> provider = EODHDProvider(api_key="your_key")
            >>> df = provider.fetch_ohlcv("AAPL", "2024-01-01", "2024-01-31", exchange="US")
            >>> df = provider.fetch_ohlcv("VOD", "2024-01-01", "2024-01-31", exchange="LSE")
        """
        exchange_code = exchange or self.default_exchange
        if "." in symbol:
            formatted_symbol = symbol.upper()
        else:
            formatted_symbol = f"{symbol.upper()}.{exchange_code.upper()}"

        self.logger.info(
            f"Fetching {frequency} OHLCV",
            symbol=formatted_symbol,
            start=start,
            end=end,
        )

        raw_data = self._fetch_raw_data(symbol, start, end, frequency, exchange=exchange_code)
        df = self._transform_data(raw_data, symbol)

        self.logger.info(f"Fetched {len(df)} records", symbol=formatted_symbol)

        return df

    async def _fetch_raw_data_async(
        self,
        symbol: str,
        start: str,
        end: str,
        frequency: str = "daily",
        exchange: str | None = None,
    ) -> list[dict[str, Any]]:
        """Async fetch raw data from EODHD API."""
        exchange_code = exchange or self.default_exchange

        # Map frequency to EODHD period
        period = self.FREQUENCY_MAP.get(frequency.lower())
        if not period:
            raise DataValidationError(
                provider="eodhd",
                message=f"Unsupported frequency '{frequency}'. "
                f"Supported: {list(self.FREQUENCY_MAP.keys())}",
                field="frequency",
                value=frequency,
            )

        # Format symbol with exchange: AAPL.US
        if "." in symbol:
            formatted_symbol = symbol.upper()
        else:
            formatted_symbol = f"{symbol.upper()}.{exchange_code.upper()}"

        # Build request
        endpoint = f"{self.base_url}/eod/{formatted_symbol}"
        params = {
            "api_token": self.api_key,
            "fmt": "json",
            "period": period,
            "from": start,
            "to": end,
        }

        try:
            self.rate_limiter.acquire(blocking=True)
            response = await self._aget(endpoint, params=params)

            # Check for errors
            if response.status_code == 429:
                raise RateLimitError(provider="eodhd", retry_after=180.0)
            if response.status_code in [401, 403]:
                raise AuthenticationError(
                    provider="eodhd", message="Invalid API key or unauthorized access"
                )
            if response.status_code == 404:
                raise DataNotAvailableError(
                    provider="eodhd", symbol=formatted_symbol, start=start, end=end
                )
            if response.status_code != 200:
                raise NetworkError(
                    provider="eodhd", message=f"HTTP {response.status_code}: {response.text}"
                )

            # Parse JSON
            try:
                data = response.json()
            except Exception as err:
                raise NetworkError(
                    provider="eodhd", message="Failed to parse JSON response"
                ) from err

            # Check if data is empty
            if not data:
                raise DataNotAvailableError(provider="eodhd", symbol=formatted_symbol)

            # Check for API-level errors
            if isinstance(data, dict) and "errors" in data:
                raise ProviderError(provider="eodhd", message=f"API error: {data['errors']}")

            # Check for tier limitation warnings
            if (
                isinstance(data, list)
                and len(data) == 1
                and isinstance(data[0], dict)
                and "warning" in data[0]
            ):
                raise DataNotAvailableError(
                    provider="eodhd",
                    symbol=formatted_symbol,
                    details={
                        "warning": data[0]["warning"],
                        "tier_limitation": "Free tier is limited to 1 year of historical data",
                    },
                )

            return data

        except (
            AuthenticationError,
            RateLimitError,
            NetworkError,
            ProviderError,
            DataNotAvailableError,
        ):
            raise
        except Exception as err:
            raise NetworkError(provider="eodhd", message=f"Request failed: {endpoint}") from err

    async def fetch_ohlcv_async(
        self,
        symbol: str,
        start: str,
        end: str,
        frequency: str = "daily",
        exchange: str | None = None,
    ) -> pl.DataFrame:
        """Async fetch OHLCV data for a symbol.

        This is 3-10x faster than sync when fetching multiple symbols
        concurrently using asyncio.gather() or async_batch_load().

        Args:
            symbol: Stock symbol (e.g., "AAPL")
            start: Start date (YYYY-MM-DD)
            end: End date (YYYY-MM-DD)
            frequency: Data frequency (daily, weekly, monthly)
            exchange: Exchange code (e.g., "US", "LSE", "FRA")

        Returns:
            Polars DataFrame with OHLCV data

        Example:
            async with EODHDProvider(api_key="key") as provider:
                df = await provider.fetch_ohlcv_async("AAPL", "2024-01-01", "2024-06-30")
        """
        exchange_code = exchange or self.default_exchange
        if "." in symbol:
            formatted_symbol = symbol.upper()
        else:
            formatted_symbol = f"{symbol.upper()}.{exchange_code.upper()}"

        self.logger.info(
            f"Fetching {frequency} OHLCV (async)",
            symbol=formatted_symbol,
            start=start,
            end=end,
        )

        raw_data = await self._fetch_raw_data_async(
            symbol, start, end, frequency, exchange=exchange_code
        )
        df = self._transform_data(raw_data, symbol)

        self.logger.info(f"Fetched {len(df)} records (async)", symbol=formatted_symbol)

        return df

    def close(self) -> None:
        """Close HTTP client."""
        if hasattr(self, "session"):
            self.session.close()
            self.logger.debug("Closed EODHD API client")
