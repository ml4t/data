"""OANDA provider implementation for FX market data.

This provider supports:
- Major, minor, and exotic FX pairs
- Multiple timeframes (M1, M5, H1, D, etc.)
- High-frequency data with generous rate limits
"""

from __future__ import annotations

import os
from datetime import UTC, datetime
from typing import Any, ClassVar

import oandapyV20  # type: ignore
import oandapyV20.endpoints.instruments as instruments  # type: ignore
import polars as pl
import structlog
from oandapyV20.exceptions import V20Error  # type: ignore

from ml4t.data.core.exceptions import (
    AuthenticationError,
    DataNotAvailableError,
    DataValidationError,
    NetworkError,
    RateLimitError,
    SymbolNotFoundError,
)
from ml4t.data.providers.base import BaseProvider

logger = structlog.get_logger()


class OandaProvider(BaseProvider):
    """OANDA provider for institutional-grade FX market data.

    OANDA provides high-quality foreign exchange data with excellent coverage
    of major, minor, and exotic currency pairs.
    """

    # OANDA allows up to 100 requests per second
    DEFAULT_RATE_LIMIT: ClassVar[tuple[int, float]] = (100, 1.0)

    # Supported timeframes
    TIMEFRAMES = {
        "S5": "5s",
        "S10": "10s",
        "S15": "15s",
        "S30": "30s",
        "M1": "1m",
        "M2": "2m",
        "M4": "4m",
        "M5": "5m",
        "M10": "10m",
        "M15": "15m",
        "M30": "30m",
        "H1": "1h",
        "H2": "2h",
        "H3": "3h",
        "H4": "4h",
        "H6": "6h",
        "H8": "8h",
        "H12": "12h",
        "D": "1d",
        "W": "1w",
        "M": "1mo",
    }

    def __init__(
        self,
        api_key: str | None = None,
        practice: bool = True,
        rate_limit: tuple[int, float] | None = None,
    ):
        """Initialize OANDA provider.

        Args:
            api_key: OANDA API key (or set OANDA_API_KEY env var)
            practice: Use practice account (True) or live account (False)
            rate_limit: Optional custom rate limit (calls, period_seconds)

        Raises:
            AuthenticationError: If API key is not provided
        """
        self.api_key = api_key or os.getenv("OANDA_API_KEY")
        if not self.api_key:
            raise AuthenticationError(
                provider="oanda",
                message="OANDA API key required. Set OANDA_API_KEY "
                "environment variable or pass api_key parameter. "
                "Get API key at: https://www.oanda.com/",
            )

        self.practice = practice

        # Set base URL based on account type
        self.base_url = (
            "https://api-fxpractice.oanda.com" if practice else "https://api-fxtrade.oanda.com"
        )

        # Initialize OANDA client
        try:
            environment = "practice" if practice else "live"
            self.client = oandapyV20.API(
                access_token=self.api_key,
                environment=environment,
            )
        except Exception as e:
            raise AuthenticationError(
                provider="oanda", message=f"Failed to initialize OANDA client: {e}"
            )

        super().__init__(rate_limit=rate_limit or self.DEFAULT_RATE_LIMIT)

        self.logger.info(
            "Initialized OANDA provider",
            practice=practice,
            rate_limit=rate_limit or self.DEFAULT_RATE_LIMIT,
        )

    @property
    def name(self) -> str:
        """Return provider name."""
        return "oanda"

    def _validate_pair(self, symbol: str) -> str:
        """Validate and format currency pair.

        Normalizes various forex pair formats to OANDA's EUR_USD format.

        Args:
            symbol: Currency pair (e.g., 'EURUSD', 'EUR_USD', 'EUR/USD')

        Returns:
            Formatted pair for OANDA API (with underscore)
        """
        # Handle EUR/USD format (forward slash)
        if "/" in symbol:
            parts = symbol.split("/")
            if len(parts) == 2 and len(parts[0]) == 3 and len(parts[1]) == 3:
                return f"{parts[0].upper()}_{parts[1].upper()}"
            raise ValueError(
                f"Invalid currency pair format: {symbol}. Use format 'EURUSD', 'EUR_USD', or 'EUR/USD'"
            )

        # Handle EURUSD format (6 characters, no separator)
        if len(symbol) == 6 and "_" not in symbol and "/" not in symbol:
            return f"{symbol[:3].upper()}_{symbol[3:].upper()}"

        # Handle EUR_USD format (already has underscore)
        if "_" in symbol:
            return symbol.upper()

        raise ValueError(
            f"Invalid currency pair format: {symbol}. Use format 'EURUSD', 'EUR_USD', or 'EUR/USD'"
        )

    def _map_frequency_to_granularity(self, frequency: str) -> str:
        """Map frequency parameter to OANDA granularity."""
        freq_lower = frequency.lower()

        # Common aliases
        mappings = {
            "daily": "D",
            "day": "D",
            "1d": "D",
            "hourly": "H1",
            "hour": "H1",
            "1h": "H1",
            "minute": "M1",
            "min": "M1",
            "1m": "M1",
            "5min": "M5",
            "5m": "M5",
            "15min": "M15",
            "15m": "M15",
            "30min": "M30",
            "30m": "M30",
            "4hour": "H4",
            "4h": "H4",
            "weekly": "W",
            "week": "W",
            "1w": "W",
        }

        # Check direct mapping first
        if freq_lower in mappings:
            return mappings[freq_lower]

        # Check if it's a valid OANDA timeframe
        freq_upper = frequency.upper()
        if freq_upper in self.TIMEFRAMES:
            return freq_upper

        # Default to daily
        return "D"

    def _fetch_raw_data(
        self,
        symbol: str,
        start: str,
        end: str,
        frequency: str = "daily",
    ) -> Any:
        """Fetch raw data from OANDA API."""
        pair = self._validate_pair(symbol)
        granularity = self._map_frequency_to_granularity(frequency)

        # Parse dates
        start_dt = datetime.strptime(start, "%Y-%m-%d").replace(
            hour=0, minute=0, second=0, tzinfo=UTC
        )
        end_dt = datetime.strptime(end, "%Y-%m-%d").replace(
            hour=23, minute=59, second=59, tzinfo=UTC
        )

        # Fetch in chunks (OANDA max 5000 candles per request)
        all_candles = []
        chunk_start = int(start_dt.timestamp())
        end_unix = int(end_dt.timestamp())

        try:
            while chunk_start < end_unix:
                params = {
                    "from": chunk_start,
                    "count": 5000,
                    "granularity": granularity,
                    "price": "M",  # Mid prices
                }

                request = instruments.InstrumentsCandles(instrument=pair, params=params)
                self.client.request(request)
                response = request.response

                if "candles" not in response:
                    break

                candles = response["candles"]
                if not candles:
                    break

                # Filter complete candles only
                complete_candles = [c for c in candles if c.get("complete", True)]
                all_candles.extend(complete_candles)

                # Update for next chunk
                if complete_candles:
                    last_time = complete_candles[-1]["time"]
                    last_dt = datetime.fromisoformat(last_time.replace("Z", "+00:00"))
                    new_chunk_start = int(last_dt.timestamp()) + 1

                    if new_chunk_start <= chunk_start:
                        break
                    chunk_start = new_chunk_start
                else:
                    break

                if chunk_start >= end_unix:
                    break

        except V20Error as e:
            error_msg = str(e)
            if "401" in error_msg or "unauthorized" in error_msg.lower():
                raise AuthenticationError(provider="oanda", message=f"Authentication failed: {e}")
            if "429" in error_msg or "rate" in error_msg.lower():
                raise RateLimitError(provider="oanda", retry_after=60.0)
            if "404" in error_msg:
                raise SymbolNotFoundError(provider="oanda", symbol=pair)
            if "invalid value specified for 'instrument'" in error_msg.lower():
                raise SymbolNotFoundError(provider="oanda", symbol=pair)
            raise NetworkError(provider="oanda", message=f"API error: {e}")

        except (
            AuthenticationError,
            RateLimitError,
            NetworkError,
            DataNotAvailableError,
            SymbolNotFoundError,
        ):
            raise
        except Exception as e:
            self.logger.error("Error fetching from OANDA", error=str(e), symbol=symbol)
            raise NetworkError(provider="oanda", message=f"Failed to fetch data: {e}")

        return all_candles

    def _transform_data(self, raw_data: Any, symbol: str) -> pl.DataFrame:
        """Transform OANDA data to standard schema."""
        if not raw_data:
            return self._create_empty_dataframe()

        try:
            processed_data = []
            for candle in raw_data:
                mid = candle.get("mid", {})
                processed_data.append(
                    {
                        "timestamp": candle["time"],
                        "open": float(mid.get("o", 0)),
                        "high": float(mid.get("h", 0)),
                        "low": float(mid.get("l", 0)),
                        "close": float(mid.get("c", 0)),
                        "volume": int(candle.get("volume", 0)),
                        "symbol": symbol,
                    }
                )

            df = pl.DataFrame(processed_data)

            # Convert timestamp
            df = df.with_columns(
                pl.col("timestamp")
                .str.replace("Z", "+00:00")
                .str.to_datetime("%Y-%m-%dT%H:%M:%S%.f%z")
                .alias("timestamp")
            )

            # Sort, remove duplicates, and select in standard order
            df = df.sort("timestamp").unique(subset=["timestamp"], keep="first")
            df = df.select(["timestamp", "symbol", "open", "high", "low", "close", "volume"])

            return df

        except Exception as e:
            self.logger.error("Failed to transform OANDA data", error=str(e), symbol=symbol)
            raise DataValidationError(
                provider="oanda", message=f"Failed to transform data for {symbol}: {e}"
            )

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

    async def fetch_ohlcv_async(
        self, symbol: str, start: str, end: str, frequency: str = "daily"
    ) -> pl.DataFrame:
        """Async fetch OHLCV data using thread pool.

        Since the OANDA SDK is synchronous, this wraps the sync call
        in asyncio.to_thread() to avoid blocking the event loop.

        Args:
            symbol: Forex pair (e.g., "EUR_USD")
            start: Start date (YYYY-MM-DD)
            end: End date (YYYY-MM-DD)
            frequency: Data frequency

        Returns:
            Polars DataFrame with OHLCV data
        """
        import asyncio

        return await asyncio.to_thread(self.fetch_ohlcv, symbol, start, end, frequency)
