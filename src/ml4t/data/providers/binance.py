"""Binance provider for cryptocurrency data.

Supports both sync and async operations for spot and futures markets.

Async Example:
    async with BinanceProvider() as provider:
        df = await provider.fetch_ohlcv_async("BTCUSDT", "2024-01-01", "2024-06-30")
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import ClassVar

import httpx
import polars as pl
import structlog

from ml4t.data.core.exceptions import RateLimitError, SymbolNotFoundError
from ml4t.data.providers.base import BaseProvider
from ml4t.data.providers.mixins import AsyncSessionMixin

logger = structlog.get_logger()


class BinanceProvider(AsyncSessionMixin, BaseProvider):
    """Provider for cryptocurrency data from Binance API.

    Features:
    - No API key required for public market data
    - Spot and futures market support
    - Real-time and historical data
    - High rate limits for public endpoints
    - Async support for 10x faster batch fetches

    API Documentation: https://binance-docs.github.io/apidocs/
    """

    SPOT_BASE_URL = "https://api.binance.com/api/v3"
    FUTURES_BASE_URL = "https://fapi.binance.com/fapi/v1"

    # Map internal frequencies to Binance intervals
    INTERVAL_MAP: ClassVar[dict[str, str]] = {
        "minute": "1m",
        "1minute": "1m",
        "3minute": "3m",
        "5minute": "5m",
        "15minute": "15m",
        "30minute": "30m",
        "hourly": "1h",
        "1hour": "1h",
        "2hour": "2h",
        "4hour": "4h",
        "6hour": "6h",
        "8hour": "8h",
        "12hour": "12h",
        "daily": "1d",
        "1day": "1d",
        "3day": "3d",
        "weekly": "1w",
        "1week": "1w",
        "monthly": "1M",
        "1month": "1M",
    }

    MAX_KLINES = 1000  # Maximum klines per request

    # Default rate limit: 1200 weight/minute
    # klines endpoint has weight 1-20 depending on limit parameter
    # Conservative: 20 calls per 60 seconds (assumes max weight 20 per call)
    DEFAULT_RATE_LIMIT: ClassVar[tuple[int, float]] = (20, 60.0)

    def __init__(
        self,
        market: str = "spot",  # "spot" or "futures"
        timeout: float = 30.0,
        rate_limit: tuple[int, float] | None = None,
    ) -> None:
        """Initialize Binance provider.

        Args:
            market: Market type (spot or futures)
            timeout: Request timeout in seconds
            rate_limit: Optional custom rate limit (calls, period_seconds)
        """
        self.market = market.lower()
        if self.market not in ["spot", "futures"]:
            raise ValueError(f"Invalid market: {market}. Must be 'spot' or 'futures'")

        self.base_url = self.SPOT_BASE_URL if self.market == "spot" else self.FUTURES_BASE_URL
        self.timeout = timeout

        super().__init__(rate_limit=rate_limit or self.DEFAULT_RATE_LIMIT)

    @property
    def name(self) -> str:
        """Return the provider name."""
        return "binance"

    @property
    def client(self) -> httpx.Client:
        """Get HTTP client from base provider."""
        return self.session

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
            },
            schema={
                "timestamp": pl.Datetime("ms", "UTC"),
                "symbol": pl.String,
                "open": pl.Float64,
                "high": pl.Float64,
                "low": pl.Float64,
                "close": pl.Float64,
                "volume": pl.Float64,
            },
        )

    def _normalize_symbol(self, symbol: str) -> str:
        """Normalize symbol to Binance format.

        Args:
            symbol: Symbol like BTC, BTC-USD, BTC/USD, BTCUSD, BTCUSDT

        Returns:
            Binance format symbol (e.g., BTCUSDT)
        """
        # Remove separators and convert to uppercase
        symbol = symbol.upper().replace("-", "").replace("/", "").replace(" ", "")

        # Handle common cases
        if symbol in ["BTC", "BITCOIN"]:
            symbol = "BTCUSDT"
        elif symbol in ["ETH", "ETHEREUM"]:
            symbol = "ETHUSDT"
        elif symbol == "BTCUSD":
            # Binance primarily uses USDT pairs
            symbol = "BTCUSDT"
        elif symbol == "ETHUSD":
            symbol = "ETHUSDT"
        elif len(symbol) == 3:
            # Assume it's a base currency, add USDT
            symbol = symbol + "USDT"
        elif not symbol.endswith(("USDT", "BUSD", "BTC", "ETH", "BNB")):
            # If no quote currency, default to USDT
            symbol = symbol + "USDT"

        return symbol

    def _fetch_and_transform_data(
        self, symbol: str, start: str, end: str, frequency: str
    ) -> pl.DataFrame:
        """Fetch and transform OHLCV data from Binance.

        Note: This method implements Binance-specific pagination to handle
        large date ranges. Rate limiting, circuit breaker, input validation,
        and response validation are handled by the base class.

        Args:
            symbol: Cryptocurrency symbol
            start: Start date (YYYY-MM-DD)
            end: End date (YYYY-MM-DD)
            frequency: Data frequency

        Returns:
            Polars DataFrame with OHLCV data in standardized schema
        """
        # Parse dates
        start_dt = datetime.strptime(start, "%Y-%m-%d").replace(tzinfo=UTC)
        end_dt = datetime.strptime(end, "%Y-%m-%d").replace(tzinfo=UTC)

        # For intraday frequencies, extend end date to end of day
        if frequency.lower() not in ["daily", "1day", "weekly", "1week", "monthly", "1month"]:
            end_dt = end_dt.replace(hour=23, minute=59, second=59)

        # Normalize symbol
        symbol = self._normalize_symbol(symbol)

        # Get interval
        freq_lower = frequency.lower()
        if freq_lower not in self.INTERVAL_MAP:
            raise ValueError(f"Unsupported frequency: {frequency}")
        interval = self.INTERVAL_MAP[freq_lower]

        logger.info(
            f"Fetching {symbol} data from Binance {self.market}",
            frequency=frequency,
            interval=interval,
            start=start,
            end=end,
        )

        # Binance uses millisecond timestamps
        start_ms = int(start_dt.timestamp() * 1000)
        end_ms = int(end_dt.timestamp() * 1000)

        # Fetch data in chunks
        all_klines = []
        current_start = start_ms

        while current_start < end_ms:
            # Prepare request
            endpoint = "/klines"
            url = f"{self.base_url}{endpoint}"

            params = {
                "symbol": symbol,
                "interval": interval,
                "startTime": current_start,
                "endTime": end_ms,
                "limit": self.MAX_KLINES,
            }

            try:
                response = self.client.get(url, params=params)
                response.raise_for_status()
                klines = response.json()

                if not klines:
                    break

                all_klines.extend(klines)

                # Update start time for next batch
                last_time = klines[-1][0]  # Open time of last kline
                if last_time >= end_ms:
                    break
                current_start = last_time + 1

                # Apply rate limiting for pagination (first request handled by base class)
                if current_start < end_ms:
                    self._acquire_rate_limit()

            except httpx.HTTPStatusError as e:
                if e.response.status_code == 429:
                    logger.warning("Rate limited by Binance")
                    raise RateLimitError("binance", retry_after=60.0) from e
                if e.response.status_code == 400:
                    logger.error(f"Invalid symbol or parameters: {symbol}")
                    raise SymbolNotFoundError("binance", symbol) from e
                raise

        # Return empty DataFrame if no data (base class will handle validation)
        if not all_klines:
            logger.info(f"No data returned for {symbol}")
            return self._create_empty_dataframe()

        # Convert to DataFrame
        # Binance kline format:
        # [Open time, Open, High, Low, Close, Volume, Close time, ...]

        data = {
            "timestamp": [k[0] for k in all_klines],
            "open": [float(k[1]) for k in all_klines],
            "high": [float(k[2]) for k in all_klines],
            "low": [float(k[3]) for k in all_klines],
            "close": [float(k[4]) for k in all_klines],
            "volume": [float(k[5]) for k in all_klines],
        }

        df = pl.DataFrame(data)

        # Convert timestamp from milliseconds to datetime
        df = df.with_columns(pl.col("timestamp").cast(pl.Datetime("ms", "UTC")))

        # Add symbol column (using normalized symbol) and select in standard order
        df = df.with_columns(pl.lit(symbol).alias("symbol"))
        df = df.select(["timestamp", "symbol", "open", "high", "low", "close", "volume"])

        # Filter by date range (Binance sometimes returns extra data)
        df = df.filter((pl.col("timestamp") >= start_dt) & (pl.col("timestamp") <= end_dt))

        # Note: Sorting and duplicate removal handled by base class _validate_response

        logger.info(
            f"Fetched {len(df)} rows for {symbol}",
            start_date=df["timestamp"].min() if not df.is_empty() else None,
            end_date=df["timestamp"].max() if not df.is_empty() else None,
        )

        return df

    async def fetch_ohlcv_async(
        self, symbol: str, start: str, end: str, frequency: str = "daily"
    ) -> pl.DataFrame:
        """Async fetch OHLCV data for a cryptocurrency.

        This is 3-10x faster than sync when fetching multiple symbols
        concurrently using asyncio.gather() or async_batch_load().

        Args:
            symbol: Cryptocurrency symbol (e.g., "BTCUSDT")
            start: Start date (YYYY-MM-DD)
            end: End date (YYYY-MM-DD)
            frequency: Data frequency

        Returns:
            Polars DataFrame with OHLCV data

        Example:
            async with BinanceProvider() as provider:
                df = await provider.fetch_ohlcv_async("BTCUSDT", "2024-01-01", "2024-06-30")
        """
        # Parse dates
        start_dt = datetime.strptime(start, "%Y-%m-%d").replace(tzinfo=UTC)
        end_dt = datetime.strptime(end, "%Y-%m-%d").replace(tzinfo=UTC)

        # For intraday frequencies, extend end date to end of day
        if frequency.lower() not in ["daily", "1day", "weekly", "1week", "monthly", "1month"]:
            end_dt = end_dt.replace(hour=23, minute=59, second=59)

        # Normalize symbol
        symbol = self._normalize_symbol(symbol)

        # Get interval
        freq_lower = frequency.lower()
        if freq_lower not in self.INTERVAL_MAP:
            raise ValueError(f"Unsupported frequency: {frequency}")
        interval = self.INTERVAL_MAP[freq_lower]

        logger.info(
            f"Fetching {symbol} data from Binance {self.market} (async)",
            frequency=frequency,
            interval=interval,
            start=start,
            end=end,
        )

        # Binance uses millisecond timestamps
        start_ms = int(start_dt.timestamp() * 1000)
        end_ms = int(end_dt.timestamp() * 1000)

        # Fetch data in chunks
        all_klines = []
        current_start = start_ms

        while current_start < end_ms:
            endpoint = "/klines"
            url = f"{self.base_url}{endpoint}"

            params = {
                "symbol": symbol,
                "interval": interval,
                "startTime": current_start,
                "endTime": end_ms,
                "limit": self.MAX_KLINES,
            }

            try:
                response = await self._aget(url, params=params)
                response.raise_for_status()
                klines = response.json()

                if not klines:
                    break

                all_klines.extend(klines)

                # Update start time for next batch
                last_time = klines[-1][0]
                if last_time >= end_ms:
                    break
                current_start = last_time + 1

                # Apply rate limiting for pagination
                if current_start < end_ms:
                    self._acquire_rate_limit()

            except httpx.HTTPStatusError as e:
                if e.response.status_code == 429:
                    raise RateLimitError("binance", retry_after=60.0) from e
                if e.response.status_code == 400:
                    raise SymbolNotFoundError("binance", symbol) from e
                raise

        if not all_klines:
            return self._create_empty_dataframe()

        # Convert to DataFrame
        data = {
            "timestamp": [k[0] for k in all_klines],
            "open": [float(k[1]) for k in all_klines],
            "high": [float(k[2]) for k in all_klines],
            "low": [float(k[3]) for k in all_klines],
            "close": [float(k[4]) for k in all_klines],
            "volume": [float(k[5]) for k in all_klines],
        }

        df = pl.DataFrame(data)
        df = df.with_columns(pl.col("timestamp").cast(pl.Datetime("ms", "UTC")))
        df = df.with_columns(pl.lit(symbol).alias("symbol"))
        df = df.select(["timestamp", "symbol", "open", "high", "low", "close", "volume"])
        df = df.filter((pl.col("timestamp") >= start_dt) & (pl.col("timestamp") <= end_dt))

        logger.info(
            f"Fetched {len(df)} rows for {symbol} (async)",
            start_date=df["timestamp"].min() if not df.is_empty() else None,
            end_date=df["timestamp"].max() if not df.is_empty() else None,
        )

        return df
