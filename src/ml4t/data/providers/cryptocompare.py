"""CryptoCompare provider for cryptocurrency data."""

from __future__ import annotations

import asyncio
import time
from datetime import UTC, datetime
from typing import Any, ClassVar

import httpx
import polars as pl
import structlog

from ml4t.data.providers.base import BaseProvider

logger = structlog.get_logger()


class CryptoCompareProvider(BaseProvider):
    """
    Provider for cryptocurrency data from CryptoCompare API.

    Features:
    - Spot price data for major cryptocurrencies
    - Historical OHLCV data
    - Multiple exchange support
    - Free tier: 100,000 calls/month

    API Documentation: https://min-api.cryptocompare.com/documentation
    """

    BASE_URL = "https://min-api.cryptocompare.com/data/v2"

    # Map internal frequencies to CryptoCompare API endpoints
    FREQUENCY_MAP: ClassVar[dict[str, str]] = {
        "minute": "histominute",
        "1minute": "histominute",
        "hourly": "histohour",
        "1hour": "histohour",
        "daily": "histoday",
        "1day": "histoday",
    }

    # Limits for each frequency (free tier)
    FREQUENCY_LIMITS: ClassVar[dict[str, int]] = {
        "histominute": 2000,  # Max 2000 minutes per call
        "histohour": 2000,  # Max 2000 hours per call
        "histoday": 2000,  # Max 2000 days per call
    }

    # Override default rate limiting for CryptoCompare (100k calls/month free)
    DEFAULT_RATE_LIMIT: ClassVar[tuple[int, float]] = (
        10,
        60.0,
    )  # 10 requests per minute for free tier

    def __init__(
        self,
        api_key: str | None = None,
        exchange: str = "CCCAGG",  # CryptoCompare aggregate
        timeout: float = 30.0,
        **kwargs,
    ) -> None:
        """
        Initialize CryptoCompare provider.

        Args:
            api_key: Optional API key for higher rate limits
            exchange: Exchange to fetch from (default: aggregate)
            timeout: Request timeout in seconds
            **kwargs: Additional arguments passed to BaseProvider
        """
        # Initialize base provider with rate limiting
        super().__init__(**kwargs)

        self.api_key = api_key
        self.exchange = exchange
        self.timeout = timeout

        # Configure session headers if API key provided
        if self.api_key:
            self.session.headers.update({"authorization": f"Apikey {self.api_key}"})

    @property
    def name(self) -> str:
        """Return provider name."""
        return "cryptocompare"

    @property
    def client(self) -> httpx.Client:
        """Get HTTP client from base provider."""
        return self.session

    def _normalize_symbol(self, symbol: str) -> tuple[str, str]:
        """
        Normalize symbol to base and quote currencies.

        Args:
            symbol: Symbol like BTC, BTC-USD, BTC/USD, BTCUSD

        Returns:
            Tuple of (base, quote) currencies
        """
        # Remove common separators and convert to uppercase
        symbol = symbol.upper()

        # Handle different formats
        if "-" in symbol:
            parts = symbol.split("-")
        elif "/" in symbol:
            parts = symbol.split("/")
        elif len(symbol) == 6:  # Assume format like BTCUSD
            parts = [symbol[:3], symbol[3:]]
        elif len(symbol) == 3:  # Just base currency, default to USD
            parts = [symbol, "USD"]
        else:
            # Try to split on common quote currencies
            for quote in ["USDT", "USD", "EUR", "GBP", "BTC", "ETH"]:
                if symbol.endswith(quote):
                    base = symbol[: -len(quote)]
                    parts = [base, quote]
                    break
            else:
                # Default to treating entire symbol as base, quote as USD
                parts = [symbol, "USD"]

        return parts[0], parts[1] if len(parts) > 1 else "USD"

    def _fetch_raw_data(self, symbol: str, start: str, end: str, frequency: str) -> Any:
        """
        Fetch raw data from CryptoCompare API.

        Args:
            symbol: Cryptocurrency symbol (e.g., BTC, BTC-USD, BTC/USD)
            start: Start date (YYYY-MM-DD)
            end: End date (YYYY-MM-DD)
            frequency: Data frequency (minute, hourly, daily)

        Returns:
            Raw API response data
        """
        # Parse dates
        start_dt = datetime.strptime(start, "%Y-%m-%d").replace(tzinfo=UTC)
        end_dt = datetime.strptime(end, "%Y-%m-%d").replace(tzinfo=UTC)

        # For minute/hourly data, extend end date to end of day
        if frequency.lower() in ["minute", "1minute", "hourly", "1hour"]:
            end_dt = end_dt.replace(hour=23, minute=59, second=59)

        # Normalize symbol
        base, quote = self._normalize_symbol(symbol)

        # Get API endpoint for frequency
        freq_lower = frequency.lower()
        if freq_lower not in self.FREQUENCY_MAP:
            raise ValueError(f"Unsupported frequency: {frequency}")
        endpoint = self.FREQUENCY_MAP[freq_lower]

        logger.info(
            f"Fetching {base}/{quote} data from CryptoCompare",
            frequency=frequency,
            start=start,
            end=end,
            exchange=self.exchange,
        )

        # Calculate time parameters
        current_time = int(end_dt.timestamp())
        total_seconds = (end_dt - start_dt).total_seconds()

        # Determine aggregate parameter based on frequency
        aggregate = 1
        if freq_lower == "5minute":
            aggregate = 5
            endpoint = "histominute"
        elif freq_lower == "15minute":
            aggregate = 15
            endpoint = "histominute"
        elif freq_lower == "30minute":
            aggregate = 30
            endpoint = "histominute"

        # Calculate limit (number of data points)
        if endpoint == "histominute":
            if aggregate > 1:
                limit = min(
                    max(1, int(total_seconds / (60 * aggregate))), self.FREQUENCY_LIMITS[endpoint]
                )
            else:
                limit = min(max(1, int(total_seconds / 60)), self.FREQUENCY_LIMITS[endpoint])
        elif endpoint == "histohour":
            limit = min(max(1, int(total_seconds / 3600)), self.FREQUENCY_LIMITS[endpoint])
        else:  # histoday
            limit = min(max(1, int(total_seconds / 86400)), self.FREQUENCY_LIMITS[endpoint])

        # Fetch data in chunks if needed
        all_data = []

        while current_time >= start_dt.timestamp():
            # Prepare request parameters
            params: dict[str, str | int] = {
                "fsym": base,
                "tsym": quote,
                "limit": min(limit, self.FREQUENCY_LIMITS[endpoint]),
                "toTs": int(current_time),
                "e": self.exchange,
            }

            if aggregate > 1:
                params["aggregate"] = aggregate

            # Make request
            url = f"{self.BASE_URL}/{endpoint}"

            try:
                response = self.client.get(url, params=params)
                response.raise_for_status()
                data = response.json()

                if data.get("Response") != "Success":
                    error_msg = data.get("Message", "Unknown error")
                    raise ValueError(f"API error: {error_msg}")

                # Extract price data
                price_data = data.get("Data", {}).get("Data", [])

                if not price_data:
                    logger.warning(f"No data returned for {base}/{quote}")
                    break

                all_data.extend(price_data)

                # Update current_time for next chunk
                oldest_time = price_data[0]["time"]
                if oldest_time <= start_dt.timestamp():
                    break
                current_time = oldest_time - 1

                # Rate limiting (be conservative)
                if not self.api_key:
                    time.sleep(0.5)  # Slower for free tier
                else:
                    time.sleep(0.1)  # Faster with API key

            except httpx.HTTPStatusError as e:
                if e.response.status_code == 429:
                    logger.warning("Rate limit hit, waiting 60 seconds")
                    time.sleep(60)
                    continue
                raise

        if not all_data:
            logger.warning(f"No data fetched for {symbol}")
            return []

        return {
            "data": all_data,
            "symbol": symbol,
            "start_dt": start_dt,
            "end_dt": end_dt,
            "base": base,
            "quote": quote,
        }

    def _transform_data(self, raw_data: Any, symbol: str) -> pl.DataFrame:
        """
        Transform raw CryptoCompare data to standardized schema.

        Args:
            raw_data: Raw data from _fetch_raw_data
            symbol: Original symbol for logging

        Returns:
            Standardized Polars DataFrame
        """
        if not raw_data or not raw_data.get("data"):
            return pl.DataFrame()

        all_data = raw_data["data"]
        start_dt = raw_data["start_dt"]
        end_dt = raw_data["end_dt"]
        base = raw_data["base"]
        quote = raw_data["quote"]

        # Convert to DataFrame
        df = pl.DataFrame(all_data)

        # Rename columns to match our schema
        df = df.rename(
            {
                "time": "timestamp",
                "high": "high",
                "low": "low",
                "open": "open",
                "close": "close",
                "volumefrom": "volume",  # Volume in base currency
            }
        )

        # Convert timestamp to datetime (nanoseconds for consistency)
        df = df.with_columns(
            pl.col("timestamp").cast(pl.Int64).mul(1_000_000_000).cast(pl.Datetime("ns", "UTC"))
        )

        # Add symbol column
        df = df.with_columns(pl.lit(symbol).alias("symbol"))

        # Select and order columns (standard order: timestamp, symbol, then OHLCV)
        df = df.select(
            [
                "timestamp",
                "symbol",
                "open",
                "high",
                "low",
                "close",
                "volume",
            ]
        )

        # Filter by date range
        df = df.filter((pl.col("timestamp") >= start_dt) & (pl.col("timestamp") <= end_dt))

        # Sort by timestamp
        df = df.sort("timestamp")

        # Remove duplicates (keep last)
        df = df.unique(subset=["timestamp"], keep="last")

        logger.info(
            f"Transformed {len(df)} rows for {base}/{quote}",
            start_date=df["timestamp"].min() if len(df) > 0 else None,
            end_date=df["timestamp"].max() if len(df) > 0 else None,
        )

        return df

    def __del__(self) -> None:
        """Clean up HTTP client."""
        if hasattr(self, "_client") and self._client:
            self._client.close()

    # ===== Async Support =====

    @property
    def _async_session(self) -> httpx.AsyncClient:
        """Lazily create async HTTP client."""
        if not hasattr(self, "_async_client") or self._async_client is None:
            headers = {}
            if self.api_key:
                headers["authorization"] = f"Apikey {self.api_key}"
            self._async_client = httpx.AsyncClient(
                timeout=httpx.Timeout(self.timeout),
                headers=headers,
            )
        return self._async_client

    async def _fetch_raw_data_async(self, symbol: str, start: str, end: str, frequency: str) -> Any:
        """Async version: Fetch raw data from CryptoCompare API.

        Args:
            symbol: Cryptocurrency symbol
            start: Start date (YYYY-MM-DD)
            end: End date (YYYY-MM-DD)
            frequency: Data frequency

        Returns:
            Raw API response data
        """
        start_dt = datetime.strptime(start, "%Y-%m-%d").replace(tzinfo=UTC)
        end_dt = datetime.strptime(end, "%Y-%m-%d").replace(tzinfo=UTC)

        if frequency.lower() in ["minute", "1minute", "hourly", "1hour"]:
            end_dt = end_dt.replace(hour=23, minute=59, second=59)

        base, quote = self._normalize_symbol(symbol)

        freq_lower = frequency.lower()
        if freq_lower not in self.FREQUENCY_MAP:
            raise ValueError(f"Unsupported frequency: {frequency}")
        endpoint = self.FREQUENCY_MAP[freq_lower]

        logger.info(
            f"Fetching {base}/{quote} data async from CryptoCompare",
            frequency=frequency,
            start=start,
            end=end,
            exchange=self.exchange,
        )

        current_time = int(end_dt.timestamp())
        total_seconds = (end_dt - start_dt).total_seconds()

        aggregate = 1
        if freq_lower == "5minute":
            aggregate = 5
            endpoint = "histominute"
        elif freq_lower == "15minute":
            aggregate = 15
            endpoint = "histominute"
        elif freq_lower == "30minute":
            aggregate = 30
            endpoint = "histominute"

        if endpoint == "histominute":
            if aggregate > 1:
                limit = min(
                    max(1, int(total_seconds / (60 * aggregate))), self.FREQUENCY_LIMITS[endpoint]
                )
            else:
                limit = min(max(1, int(total_seconds / 60)), self.FREQUENCY_LIMITS[endpoint])
        elif endpoint == "histohour":
            limit = min(max(1, int(total_seconds / 3600)), self.FREQUENCY_LIMITS[endpoint])
        else:
            limit = min(max(1, int(total_seconds / 86400)), self.FREQUENCY_LIMITS[endpoint])

        all_data = []

        while current_time >= start_dt.timestamp():
            params: dict[str, str | int] = {
                "fsym": base,
                "tsym": quote,
                "limit": min(limit, self.FREQUENCY_LIMITS[endpoint]),
                "toTs": int(current_time),
                "e": self.exchange,
            }

            if aggregate > 1:
                params["aggregate"] = aggregate

            url = f"{self.BASE_URL}/{endpoint}"

            try:
                response = await self._async_session.get(url, params=params)
                response.raise_for_status()
                data = response.json()

                if data.get("Response") != "Success":
                    error_msg = data.get("Message", "Unknown error")
                    raise ValueError(f"API error: {error_msg}")

                price_data = data.get("Data", {}).get("Data", [])

                if not price_data:
                    logger.warning(f"No data returned for {base}/{quote}")
                    break

                all_data.extend(price_data)

                oldest_time = price_data[0]["time"]
                if oldest_time <= start_dt.timestamp():
                    break
                current_time = oldest_time - 1

                # Rate limiting
                delay = 0.5 if not self.api_key else 0.1
                await asyncio.sleep(delay)

            except httpx.HTTPStatusError as e:
                if e.response.status_code == 429:
                    logger.warning("Rate limit hit, waiting 60 seconds")
                    await asyncio.sleep(60)
                    continue
                raise

        if not all_data:
            logger.warning(f"No data fetched for {symbol}")
            return []

        return {
            "data": all_data,
            "symbol": symbol,
            "start_dt": start_dt,
            "end_dt": end_dt,
            "base": base,
            "quote": quote,
        }

    async def fetch_ohlcv_async(
        self,
        symbol: str,
        start: str,
        end: str,
        frequency: str = "daily",
    ) -> pl.DataFrame:
        """Async fetch OHLCV data for a cryptocurrency.

        Args:
            symbol: Cryptocurrency symbol (e.g., "BTC", "BTC-USD")
            start: Start date (YYYY-MM-DD)
            end: End date (YYYY-MM-DD)
            frequency: Data frequency

        Returns:
            DataFrame with OHLCV data
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

        raw_data = await self._fetch_raw_data_async(symbol, start, end, frequency)
        data = self._transform_data(raw_data, symbol)

        if not data.is_empty():
            data = self._validate_ohlcv(data, self.name)

        self.logger.info(
            "Successfully fetched OHLCV data (async)",
            symbol=symbol,
            rows=len(data),
            provider=self.name,
        )

        return data

    async def close_async(self) -> None:
        """Close async HTTP client."""
        if hasattr(self, "_async_client") and self._async_client is not None:
            await self._async_client.aclose()
            self._async_client = None

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close_async()
