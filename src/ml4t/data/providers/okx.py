"""OKX exchange provider for cryptocurrency perpetuals data.

This provider supports:
- OHLCV data for perpetual swap contracts
- Historical and current funding rates
- No API key required for public market data
- No geo-restrictions (works globally)

API Documentation: https://www.okx.com/docs-v5/en/

Async Example:
    async with OKXProvider() as provider:
        df = await provider.fetch_ohlcv_async("BTC-USDT-SWAP", "2024-01-01", "2024-06-30")
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, ClassVar

import httpx
import polars as pl
import structlog

from ml4t.data.core.exceptions import RateLimitError, SymbolNotFoundError
from ml4t.data.providers.base import BaseProvider
from ml4t.data.providers.mixins import AsyncSessionMixin

logger = structlog.get_logger()


class OKXProvider(AsyncSessionMixin, BaseProvider):
    """Provider for cryptocurrency perpetuals data from OKX API.

    Features:
    - No API key required for public market data
    - No geo-restrictions (unlike Binance/Bybit)
    - Perpetual swap contracts with funding rates
    - High-quality historical data
    - Async support for 10x faster batch fetches

    Symbols use OKX format: BTC-USDT-SWAP, ETH-USDT-SWAP, etc.
    """

    BASE_URL = "https://www.okx.com/api/v5"

    # Map internal frequencies to OKX bar sizes
    # OKX uses: 1m, 3m, 5m, 15m, 30m, 1H, 2H, 4H, 6H, 12H, 1D, 1W, 1M
    INTERVAL_MAP: ClassVar[dict[str, str]] = {
        "minute": "1m",
        "1minute": "1m",
        "3minute": "3m",
        "5minute": "5m",
        "15minute": "15m",
        "30minute": "30m",
        "hourly": "1H",
        "1hour": "1H",
        "2hour": "2H",
        "4hour": "4H",
        "6hour": "6H",
        "12hour": "12H",
        "daily": "1D",
        "1day": "1D",
        "weekly": "1W",
        "1week": "1W",
        "monthly": "1M",
        "1month": "1M",
    }

    MAX_CANDLES = 100  # OKX returns max 100 candles per request

    # Rate limit: 20 requests per 2 seconds for market data
    DEFAULT_RATE_LIMIT: ClassVar[tuple[int, float]] = (20, 2.0)

    def __init__(
        self,
        market: str = "swap",  # "swap" for perpetuals, "spot" for spot
        timeout: float = 30.0,
        rate_limit: tuple[int, float] | None = None,
    ) -> None:
        """Initialize OKX provider.

        Args:
            market: Market type (swap for perpetuals, spot for spot)
            timeout: Request timeout in seconds
            rate_limit: Optional custom rate limit (calls, period_seconds)
        """
        self.market = market.lower()
        if self.market not in ["swap", "spot"]:
            raise ValueError(f"Invalid market: {market}. Must be 'swap' or 'spot'")

        self.timeout = timeout
        super().__init__(rate_limit=rate_limit or self.DEFAULT_RATE_LIMIT)

    @property
    def name(self) -> str:
        """Return the provider name."""
        return "okx"

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

    def _create_empty_funding_dataframe(self) -> pl.DataFrame:
        """Create an empty DataFrame for funding rate data."""
        return pl.DataFrame(
            {
                "timestamp": [],
                "symbol": [],
                "funding_rate": [],
                "realized_rate": [],
            },
            schema={
                "timestamp": pl.Datetime("ms", "UTC"),
                "symbol": pl.String,
                "funding_rate": pl.Float64,
                "realized_rate": pl.Float64,
            },
        )

    def _normalize_symbol(self, symbol: str) -> str:
        """Normalize symbol to OKX perpetual swap format.

        Args:
            symbol: Symbol like BTC, BTC-USD, BTC/USDT, BTCUSDT, BTC-USDT-SWAP

        Returns:
            OKX format symbol (e.g., BTC-USDT-SWAP)
        """
        symbol = symbol.upper().strip()

        # Already in OKX format
        if symbol.endswith("-SWAP"):
            return symbol

        # Remove common separators
        symbol = symbol.replace("/", "-").replace(" ", "")

        # Handle various input formats
        if symbol in ["BTC", "BITCOIN"]:
            return "BTC-USDT-SWAP"
        elif symbol in ["ETH", "ETHEREUM"]:
            return "ETH-USDT-SWAP"
        elif symbol == "BTCUSDT":
            return "BTC-USDT-SWAP"
        elif symbol == "ETHUSDT":
            return "ETH-USDT-SWAP"
        elif symbol.endswith("USDT"):
            # SOLUSDT -> SOL-USDT-SWAP
            base = symbol[:-4]
            return f"{base}-USDT-SWAP"
        elif "-USDT" in symbol:
            # Already has separator
            return f"{symbol}-SWAP" if not symbol.endswith("-SWAP") else symbol
        elif len(symbol) <= 5:
            # Assume it's a base currency
            return f"{symbol}-USDT-SWAP"

        # Default: add -USDT-SWAP
        return f"{symbol}-USDT-SWAP"

    def _fetch_and_transform_data(
        self, symbol: str, start: str, end: str, frequency: str
    ) -> pl.DataFrame:
        """Fetch and transform OHLCV data from OKX.

        Note: OKX returns newest data first and uses "before" parameter
        for pagination (timestamp before which to fetch).

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
        inst_id = self._normalize_symbol(symbol)

        # Get interval
        freq_lower = frequency.lower()
        if freq_lower not in self.INTERVAL_MAP:
            raise ValueError(f"Unsupported frequency: {frequency}")
        bar = self.INTERVAL_MAP[freq_lower]

        logger.info(
            f"Fetching {inst_id} data from OKX",
            frequency=frequency,
            bar=bar,
            start=start,
            end=end,
        )

        # OKX uses millisecond timestamps
        start_ms = int(start_dt.timestamp() * 1000)
        end_ms = int(end_dt.timestamp() * 1000)

        # Fetch data in chunks (OKX returns newest first, paginate backwards)
        all_candles: list[list[Any]] = []
        current_before = end_ms + 1  # Start from end, go backwards

        while True:
            url = f"{self.BASE_URL}/market/candles"
            params = {
                "instId": inst_id,
                "bar": bar,
                "before": str(current_before),
                "limit": str(self.MAX_CANDLES),
            }

            try:
                response = self.client.get(url, params=params)
                response.raise_for_status()
                result = response.json()

                if result.get("code") != "0":
                    msg = result.get("msg", "Unknown error")
                    if "instrument" in msg.lower() or "instid" in msg.lower():
                        raise SymbolNotFoundError("okx", inst_id)
                    raise ValueError(f"OKX API error: {msg}")

                candles = result.get("data", [])

                if not candles:
                    break

                # Filter candles within our date range
                for candle in candles:
                    ts = int(candle[0])
                    if ts >= start_ms:
                        all_candles.append(candle)

                # Check if we've gone past start date
                oldest_ts = int(candles[-1][0])
                if oldest_ts <= start_ms:
                    break

                # Update pagination cursor
                current_before = oldest_ts

                # Rate limit for pagination
                self._acquire_rate_limit()

            except httpx.HTTPStatusError as e:
                if e.response.status_code == 429:
                    logger.warning("Rate limited by OKX")
                    raise RateLimitError("okx", retry_after=2.0) from e
                raise

        # Return empty DataFrame if no data
        if not all_candles:
            logger.info(f"No data returned for {inst_id}")
            return self._create_empty_dataframe()

        # Convert to DataFrame
        # OKX candle format: [ts, o, h, l, c, vol, volCcy, volCcyQuote, confirm]
        data = {
            "timestamp": [int(c[0]) for c in all_candles],
            "open": [float(c[1]) for c in all_candles],
            "high": [float(c[2]) for c in all_candles],
            "low": [float(c[3]) for c in all_candles],
            "close": [float(c[4]) for c in all_candles],
            "volume": [float(c[5]) for c in all_candles],
        }

        df = pl.DataFrame(data)

        # Convert timestamp from milliseconds to datetime
        df = df.with_columns(pl.col("timestamp").cast(pl.Datetime("ms", "UTC")))

        # Add symbol column and select in standard order
        df = df.with_columns(pl.lit(inst_id).alias("symbol"))
        df = df.select(["timestamp", "symbol", "open", "high", "low", "close", "volume"])

        # Sort by timestamp (OKX returns newest first)
        df = df.sort("timestamp")

        logger.info(
            f"Fetched {len(df)} rows for {inst_id}",
            start_date=df["timestamp"].min() if not df.is_empty() else None,
            end_date=df["timestamp"].max() if not df.is_empty() else None,
        )

        return df

    def fetch_funding_rates(
        self,
        symbol: str,
        start: str,
        end: str,
    ) -> pl.DataFrame:
        """Fetch historical funding rates from OKX.

        Funding rates are settled every 8 hours (00:00, 08:00, 16:00 UTC).

        Args:
            symbol: Perpetual swap symbol (e.g., BTC-USDT-SWAP)
            start: Start date (YYYY-MM-DD)
            end: End date (YYYY-MM-DD)

        Returns:
            DataFrame with columns: timestamp, symbol, funding_rate, realized_rate
        """
        # Validate and normalize
        start_dt = datetime.strptime(start, "%Y-%m-%d").replace(tzinfo=UTC)
        end_dt = datetime.strptime(end, "%Y-%m-%d").replace(
            hour=23, minute=59, second=59, tzinfo=UTC
        )
        inst_id = self._normalize_symbol(symbol)

        logger.info(
            f"Fetching funding rates for {inst_id}",
            start=start,
            end=end,
        )

        start_ms = int(start_dt.timestamp() * 1000)
        end_ms = int(end_dt.timestamp() * 1000)

        all_rates: list[dict[str, Any]] = []
        current_before = end_ms + 1

        while True:
            self._acquire_rate_limit()

            url = f"{self.BASE_URL}/public/funding-rate-history"
            params = {
                "instId": inst_id,
                "before": str(current_before),
                "limit": "100",
            }

            try:
                response = self.client.get(url, params=params)
                response.raise_for_status()
                result = response.json()

                if result.get("code") != "0":
                    msg = result.get("msg", "Unknown error")
                    if "instrument" in msg.lower():
                        raise SymbolNotFoundError("okx", inst_id)
                    raise ValueError(f"OKX API error: {msg}")

                rates = result.get("data", [])

                if not rates:
                    break

                # Filter rates within our date range
                for rate in rates:
                    ts = int(rate["fundingTime"])
                    if ts >= start_ms:
                        all_rates.append(rate)

                # Check if we've gone past start date
                oldest_ts = int(rates[-1]["fundingTime"])
                if oldest_ts <= start_ms:
                    break

                current_before = oldest_ts

            except httpx.HTTPStatusError as e:
                if e.response.status_code == 429:
                    raise RateLimitError("okx", retry_after=2.0) from e
                raise

        if not all_rates:
            return self._create_empty_funding_dataframe()

        # Convert to DataFrame
        data = {
            "timestamp": [int(r["fundingTime"]) for r in all_rates],
            "symbol": [inst_id] * len(all_rates),
            "funding_rate": [float(r["fundingRate"]) for r in all_rates],
            "realized_rate": [float(r["realizedRate"]) for r in all_rates],
        }

        df = pl.DataFrame(data)
        df = df.with_columns(pl.col("timestamp").cast(pl.Datetime("ms", "UTC")))
        df = df.sort("timestamp")

        logger.info(
            f"Fetched {len(df)} funding rate records for {inst_id}",
            start_date=df["timestamp"].min() if not df.is_empty() else None,
            end_date=df["timestamp"].max() if not df.is_empty() else None,
        )

        return df

    def fetch_current_funding_rate(self, symbol: str) -> dict[str, Any]:
        """Fetch current funding rate for a symbol.

        Args:
            symbol: Perpetual swap symbol

        Returns:
            Dict with current and next funding rate info
        """
        inst_id = self._normalize_symbol(symbol)
        self._acquire_rate_limit()

        url = f"{self.BASE_URL}/public/funding-rate"
        params = {"instId": inst_id}

        response = self.client.get(url, params=params)
        response.raise_for_status()
        result = response.json()

        if result.get("code") != "0":
            msg = result.get("msg", "Unknown error")
            raise ValueError(f"OKX API error: {msg}")

        data = result.get("data", [{}])[0]
        return {
            "symbol": inst_id,
            "funding_rate": float(data.get("fundingRate", 0)),
            "next_funding_rate": float(data.get("nextFundingRate", 0))
            if data.get("nextFundingRate")
            else None,
            "funding_time": datetime.fromtimestamp(int(data.get("fundingTime", 0)) / 1000, tz=UTC),
            "next_funding_time": datetime.fromtimestamp(
                int(data.get("nextFundingTime", 0)) / 1000, tz=UTC
            )
            if data.get("nextFundingTime")
            else None,
        }

    async def fetch_ohlcv_async(
        self, symbol: str, start: str, end: str, frequency: str = "daily"
    ) -> pl.DataFrame:
        """Async fetch OHLCV data from OKX.

        This is 3-10x faster than sync when fetching multiple symbols
        concurrently using asyncio.gather() or async_batch_load().

        Args:
            symbol: Cryptocurrency symbol (e.g., "BTC-USDT-SWAP")
            start: Start date (YYYY-MM-DD)
            end: End date (YYYY-MM-DD)
            frequency: Data frequency

        Returns:
            Polars DataFrame with OHLCV data

        Example:
            async with OKXProvider() as provider:
                df = await provider.fetch_ohlcv_async("BTC-USDT-SWAP", "2024-01-01", "2024-06-30")
        """
        # Parse dates
        start_dt = datetime.strptime(start, "%Y-%m-%d").replace(tzinfo=UTC)
        end_dt = datetime.strptime(end, "%Y-%m-%d").replace(tzinfo=UTC)

        # For intraday frequencies, extend end date to end of day
        if frequency.lower() not in ["daily", "1day", "weekly", "1week", "monthly", "1month"]:
            end_dt = end_dt.replace(hour=23, minute=59, second=59)

        inst_id = self._normalize_symbol(symbol)
        freq_lower = frequency.lower()
        if freq_lower not in self.INTERVAL_MAP:
            raise ValueError(f"Unsupported frequency: {frequency}")
        bar = self.INTERVAL_MAP[freq_lower]

        logger.info(
            f"Fetching {inst_id} data from OKX (async)",
            frequency=frequency,
            bar=bar,
            start=start,
            end=end,
        )

        start_ms = int(start_dt.timestamp() * 1000)
        end_ms = int(end_dt.timestamp() * 1000)

        all_candles: list[list[Any]] = []
        current_before = end_ms + 1

        while True:
            url = f"{self.BASE_URL}/market/candles"
            params = {
                "instId": inst_id,
                "bar": bar,
                "before": str(current_before),
                "limit": str(self.MAX_CANDLES),
            }

            try:
                response = await self._aget(url, params=params)
                response.raise_for_status()
                result = response.json()

                if result.get("code") != "0":
                    msg = result.get("msg", "Unknown error")
                    if "instrument" in msg.lower() or "instid" in msg.lower():
                        raise SymbolNotFoundError("okx", inst_id)
                    raise ValueError(f"OKX API error: {msg}")

                candles = result.get("data", [])

                if not candles:
                    break

                for candle in candles:
                    ts = int(candle[0])
                    if ts >= start_ms:
                        all_candles.append(candle)

                oldest_ts = int(candles[-1][0])
                if oldest_ts <= start_ms:
                    break

                current_before = oldest_ts
                self._acquire_rate_limit()

            except httpx.HTTPStatusError as e:
                if e.response.status_code == 429:
                    raise RateLimitError("okx", retry_after=2.0) from e
                raise

        if not all_candles:
            return self._create_empty_dataframe()

        data = {
            "timestamp": [int(c[0]) for c in all_candles],
            "open": [float(c[1]) for c in all_candles],
            "high": [float(c[2]) for c in all_candles],
            "low": [float(c[3]) for c in all_candles],
            "close": [float(c[4]) for c in all_candles],
            "volume": [float(c[5]) for c in all_candles],
        }

        df = pl.DataFrame(data)
        df = df.with_columns(pl.col("timestamp").cast(pl.Datetime("ms", "UTC")))
        df = df.with_columns(pl.lit(inst_id).alias("symbol"))
        df = df.select(["timestamp", "symbol", "open", "high", "low", "close", "volume"])
        df = df.sort("timestamp")

        logger.info(
            f"Fetched {len(df)} rows for {inst_id} (async)",
            start_date=df["timestamp"].min() if not df.is_empty() else None,
            end_date=df["timestamp"].max() if not df.is_empty() else None,
        )

        return df
