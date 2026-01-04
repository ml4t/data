"""CoinGecko cryptocurrency data provider.

CoinGecko provides free cryptocurrency data with API key (Demo plan).
Demo plan (free): 30 calls/minute, 10,000 calls/month.
Public API (no key): 5-15 calls/minute (varies).

Performance Note:
- This is a simple wrapper for consistent API across providers
- For advanced features (coin list, price API, market data), use the CoinGecko API directly
- Rate limiting is handled by BaseProvider

Async Example:
    async with CoinGeckoProvider() as provider:
        df = await provider.fetch_ohlcv_async("BTC", "2024-01-01", "2024-06-30")
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import ClassVar

import httpx
import polars as pl
import structlog

from ml4t.data.core.exceptions import (
    DataNotAvailableError,
    DataValidationError,
    NetworkError,
    RateLimitError,
    SymbolNotFoundError,
)
from ml4t.data.providers.base import BaseProvider
from ml4t.data.providers.mixins import AsyncSessionMixin

logger = structlog.get_logger()


class CoinGeckoProvider(AsyncSessionMixin, BaseProvider):
    """Simple CoinGecko wrapper for OHLCV data with consistent API.

    This provider exists primarily to provide:
    - Consistent API across all providers (fetch_ohlcv interface)
    - Polars DataFrame output (standardized schema)
    - Symbol column in results
    - Structured logging and typed exceptions
    - Async support for 10x faster batch fetches

    For advanced CoinGecko features (coin lists, price API, market metrics),
    use the CoinGecko API directly or dedicated crypto data libraries.

    Rate Limits:
    - Demo plan (with API key): 30 calls/minute
    - Public API (no key): 10 calls/minute
    - Pro tier: 500 calls/minute (requires paid key)
    """

    # Common symbol to CoinGecko ID mappings
    SYMBOL_TO_ID_MAP: ClassVar[dict[str, str]] = {
        "BTC": "bitcoin",
        "ETH": "ethereum",
        "USDT": "tether",
        "BNB": "binancecoin",
        "USDC": "usd-coin",
        "XRP": "ripple",
        "ADA": "cardano",
        "DOGE": "dogecoin",
        "SOL": "solana",
        "TRX": "tron",
        "DOT": "polkadot",
        "MATIC": "matic-network",
        "LTC": "litecoin",
        "SHIB": "shiba-inu",
        "AVAX": "avalanche-2",
        "LINK": "chainlink",
        "UNI": "uniswap",
        "ATOM": "cosmos",
        "XLM": "stellar",
        "ETC": "ethereum-classic",
    }

    def __init__(
        self,
        api_key: str | None = None,
        use_pro: bool = False,
        rate_limit: tuple[int, float] | None = None,
    ) -> None:
        """Initialize CoinGecko provider.

        Args:
            api_key: Optional API key for Demo/Pro plan (default: from COINGECKO_API_KEY env var)
            use_pro: Use Pro API endpoint (requires Pro subscription)
            rate_limit: Optional custom rate limit (calls, period_seconds)
        """
        # Get API key from parameter or environment
        self.api_key = api_key or os.getenv("COINGECKO_API_KEY")
        self.use_pro = use_pro

        # Set base URL based on tier
        if self.use_pro:
            self.base_url = "https://pro-api.coingecko.com/api/v3"
        else:
            self.base_url = "https://api.coingecko.com/api/v3"

        # Set rate limit based on API key if not provided
        # Demo plan: 30 calls/minute, Public: 10 calls/minute (conservative)
        # Pro tier: 500 calls/minute
        if rate_limit is None:
            if self.use_pro:
                rate_limit = (500, 60.0)
            elif self.api_key:
                rate_limit = (30, 60.0)
            else:
                rate_limit = (10, 60.0)

        # Add API key to headers if provided
        session_config = {}
        if self.api_key:
            session_config["headers"] = {"x-cg-demo-api-key": self.api_key}

        # Initialize base provider with rate limiting
        super().__init__(rate_limit=rate_limit, session_config=session_config)

    @property
    def name(self) -> str:
        """Return the provider name."""
        return "coingecko"

    def _fetch_and_transform_data(
        self,
        symbol: str,
        start: str,
        end: str,
        frequency: str,
    ) -> pl.DataFrame:
        """Fetch and transform OHLCV data from CoinGecko.

        Args:
            symbol: Crypto symbol (e.g., "BTC") or CoinGecko ID (e.g., "bitcoin")
            start: Start date in YYYY-MM-DD format
            end: End date in YYYY-MM-DD format
            frequency: Data frequency (only "daily" supported by CoinGecko)

        Returns:
            DataFrame with OHLCV data in standardized schema

        Raises:
            DataValidationError: If frequency is not "daily"
        """
        # Validate frequency - CoinGecko OHLC endpoint only supports daily data
        if frequency.lower() != "daily":
            raise DataValidationError(
                provider=self.name,
                message=f"CoinGecko only supports 'daily' frequency, got '{frequency}'. "
                "For intraday data, consider using Binance or CryptoCompare providers.",
            )

        # Convert symbol to CoinGecko ID if needed
        coin_id = self.symbol_to_id(symbol)

        # Parse dates
        start_dt = datetime.strptime(start, "%Y-%m-%d")
        end_dt = datetime.strptime(end, "%Y-%m-%d")

        # CoinGecko works with "days from now" parameter
        # Calculate how many days back from now to start date
        days_from_now = (datetime.now() - start_dt).days + 1

        # Round to valid days parameter (1, 7, 14, 30, 90, 180, 365, max)
        valid_days = self._round_to_valid_days(days_from_now)

        logger.info(
            "Fetching data from CoinGecko",
            coin_id=coin_id,
            symbol=symbol,
            start=start,
            end=end,
            days_requested=days_from_now,
            days_rounded=valid_days,
        )

        # Fetch OHLC data (CoinGecko provides this in one endpoint)
        df = self._fetch_ohlc(coin_id, days=valid_days)

        # Filter to requested date range
        df = df.filter(
            (pl.col("timestamp").dt.truncate("1d") >= start_dt.date())
            & (pl.col("timestamp").dt.truncate("1d") <= end_dt.date())
        )

        if df.is_empty():
            logger.warning("No data found for date range", coin_id=coin_id, start=start, end=end)
            return self._create_empty_dataframe()

        # Add symbol column and select in standard order
        df = df.with_columns(pl.lit(symbol.upper()).alias("symbol"))
        df = df.select(["timestamp", "symbol", "open", "high", "low", "close", "volume"])

        logger.info("Successfully fetched data from CoinGecko", coin_id=coin_id, rows=len(df))

        return df

    def _fetch_ohlc(self, coin_id: str, days: int | str, vs_currency: str = "usd") -> pl.DataFrame:
        """Fetch OHLC data from CoinGecko.

        Args:
            coin_id: CoinGecko coin ID (e.g., "bitcoin")
            days: Number of days back from now
            vs_currency: Target currency (default: usd)

        Returns:
            DataFrame with OHLCV data
        """
        endpoint = f"{self.base_url}/coins/{coin_id}/ohlc"
        params = {"vs_currency": vs_currency, "days": days}

        try:
            response = self.session.get(endpoint, params=params)
            response.raise_for_status()
            ohlc_data = response.json()
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429:
                raise RateLimitError(self.name, retry_after=60.0) from e
            elif e.response.status_code == 404:
                raise SymbolNotFoundError(self.name, coin_id) from e
            else:
                raise DataNotAvailableError(self.name, coin_id, details={"error": str(e)}) from e
        except httpx.RequestError as e:
            raise NetworkError(self.name, str(e)) from e

        if not ohlc_data:
            return self._create_empty_dataframe()

        # Convert OHLC data to DataFrame
        # Format: [[timestamp_ms, open, high, low, close], ...]
        df = pl.DataFrame(
            ohlc_data,
            schema=["timestamp_ms", "open", "high", "low", "close"],
            orient="row",
        )

        # Convert timestamp from milliseconds to datetime
        df = df.with_columns(pl.col("timestamp_ms").cast(pl.Datetime("ms")).alias("timestamp"))

        # Add volume column (CoinGecko OHLC endpoint doesn't provide volume)
        # Set to 0 as placeholder - use market_chart endpoint if volume needed
        df = df.with_columns(pl.lit(0.0).alias("volume"))

        # Select and order columns
        df = df.select(["timestamp", "open", "high", "low", "close", "volume"])

        return df

    def _round_to_valid_days(self, days: int) -> str | int:
        """Round days to valid CoinGecko API parameter.

        Valid values: 1, 7, 14, 30, 90, 180, 365, "max"

        Args:
            days: Number of days requested

        Returns:
            Valid days parameter (int or "max")
        """
        valid_values = [1, 7, 14, 30, 90, 180, 365]

        # If days exceeds 365, use "max"
        if days > 365:
            return "max"

        # Round up to next valid value
        for valid_days in valid_values:
            if days <= valid_days:
                return valid_days

        # Fallback to max
        return "max"

    def symbol_to_id(self, symbol: str) -> str:
        """Convert symbol to CoinGecko coin ID.

        Args:
            symbol: Crypto symbol (e.g., "BTC") or CoinGecko ID (e.g., "bitcoin")

        Returns:
            CoinGecko coin ID in lowercase (e.g., "bitcoin")
        """
        # Check if symbol is in our mapping
        symbol_upper = symbol.upper()
        if symbol_upper in self.SYMBOL_TO_ID_MAP:
            return self.SYMBOL_TO_ID_MAP[symbol_upper]

        # If already lowercase, assume it's a valid coin ID
        if symbol.islower():
            return symbol

        # Otherwise, convert to lowercase (unknown symbol → coin ID format)
        return symbol.lower()

    def get_coin_list(self) -> pl.DataFrame:
        """Fetch list of all available coins from CoinGecko.

        Returns:
            DataFrame with columns: id, symbol, name
        """
        endpoint = f"{self.base_url}/coins/list"

        try:
            response = self.session.get(endpoint)
            response.raise_for_status()
            coins = response.json()
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429:
                raise RateLimitError(self.name, retry_after=60.0) from e
            else:
                raise DataNotAvailableError(
                    self.name, "coin_list", details={"error": str(e)}
                ) from e
        except httpx.RequestError as e:
            raise NetworkError(self.name, str(e)) from e

        if not coins:
            return pl.DataFrame(schema={"id": pl.String, "symbol": pl.String, "name": pl.String})

        # Convert to DataFrame
        df = pl.DataFrame(coins)

        # Select relevant columns
        df = df.select(["id", "symbol", "name"])

        logger.info("Fetched coin list from CoinGecko", coin_count=len(df))

        return df

    def get_price(
        self, coin_ids: list[str], vs_currencies: list[str] | None = None
    ) -> pl.DataFrame:
        """Fetch current prices for coins.

        Args:
            coin_ids: List of CoinGecko coin IDs (e.g., ["bitcoin", "ethereum"])
            vs_currencies: List of target currencies (default: ["usd"])

        Returns:
            DataFrame with columns: coin_id, currency, price
        """
        if vs_currencies is None:
            vs_currencies = ["usd"]

        endpoint = f"{self.base_url}/simple/price"
        params = {
            "ids": ",".join(coin_ids),
            "vs_currencies": ",".join(vs_currencies),
        }

        try:
            response = self.session.get(endpoint, params=params)
            response.raise_for_status()
            prices = response.json()
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429:
                raise RateLimitError(self.name, retry_after=60.0) from e
            else:
                raise DataNotAvailableError(self.name, "prices", details={"error": str(e)}) from e
        except httpx.RequestError as e:
            raise NetworkError(self.name, str(e)) from e

        if not prices:
            return pl.DataFrame(
                schema={"coin_id": pl.String, "currency": pl.String, "price": pl.Float64}
            )

        # Flatten the nested dictionary to rows
        rows = []
        for coin_id, currencies in prices.items():
            for currency, price in currencies.items():
                rows.append({"coin_id": coin_id, "currency": currency, "price": price})

        df = pl.DataFrame(rows)

        logger.info(
            "Fetched prices from CoinGecko", coins=len(coin_ids), currencies=len(vs_currencies)
        )

        return df

    def _create_empty_dataframe(self) -> pl.DataFrame:
        """Create an empty DataFrame with the correct schema."""
        return pl.DataFrame(
            schema={
                "timestamp": pl.Datetime,
                "open": pl.Float64,
                "high": pl.Float64,
                "low": pl.Float64,
                "close": pl.Float64,
                "volume": pl.Float64,
                "symbol": pl.String,
            }
        )

    async def _fetch_ohlc_async(
        self, coin_id: str, days: int | str, vs_currency: str = "usd"
    ) -> pl.DataFrame:
        """Async fetch OHLC data from CoinGecko."""
        endpoint = f"{self.base_url}/coins/{coin_id}/ohlc"
        params = {"vs_currency": vs_currency, "days": days}

        try:
            response = await self._aget(endpoint, params=params)
            response.raise_for_status()
            ohlc_data = response.json()
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429:
                raise RateLimitError(self.name, retry_after=60.0) from e
            elif e.response.status_code == 404:
                raise SymbolNotFoundError(self.name, coin_id) from e
            else:
                raise DataNotAvailableError(self.name, coin_id, details={"error": str(e)}) from e
        except httpx.RequestError as e:
            raise NetworkError(self.name, str(e)) from e

        if not ohlc_data:
            return self._create_empty_dataframe()

        df = pl.DataFrame(
            ohlc_data,
            schema=["timestamp_ms", "open", "high", "low", "close"],
            orient="row",
        )
        df = df.with_columns(pl.col("timestamp_ms").cast(pl.Datetime("ms")).alias("timestamp"))
        df = df.with_columns(pl.lit(0.0).alias("volume"))
        df = df.select(["timestamp", "open", "high", "low", "close", "volume"])

        return df

    async def fetch_ohlcv_async(
        self,
        symbol: str,
        start: str,
        end: str,
        frequency: str = "daily",
    ) -> pl.DataFrame:
        """Async fetch OHLCV data for a cryptocurrency.

        This is 3-10x faster than sync when fetching multiple symbols
        concurrently using asyncio.gather() or async_batch_load().

        Args:
            symbol: Crypto symbol (e.g., "BTC") or CoinGecko ID (e.g., "bitcoin")
            start: Start date in YYYY-MM-DD format
            end: End date in YYYY-MM-DD format
            frequency: Data frequency (only "daily" supported by CoinGecko)

        Returns:
            DataFrame with OHLCV data

        Example:
            async with CoinGeckoProvider() as provider:
                df = await provider.fetch_ohlcv_async("BTC", "2024-01-01", "2024-06-30")
        """
        if frequency.lower() != "daily":
            raise DataValidationError(
                provider=self.name,
                message=f"CoinGecko only supports 'daily' frequency, got '{frequency}'.",
            )

        coin_id = self.symbol_to_id(symbol)
        start_dt = datetime.strptime(start, "%Y-%m-%d")
        end_dt = datetime.strptime(end, "%Y-%m-%d")
        days_from_now = (datetime.now() - start_dt).days + 1
        valid_days = self._round_to_valid_days(days_from_now)

        logger.info(
            "Fetching data from CoinGecko (async)",
            coin_id=coin_id,
            symbol=symbol,
            start=start,
            end=end,
        )

        df = await self._fetch_ohlc_async(coin_id, days=valid_days)

        df = df.filter(
            (pl.col("timestamp").dt.truncate("1d") >= start_dt.date())
            & (pl.col("timestamp").dt.truncate("1d") <= end_dt.date())
        )

        if df.is_empty():
            return self._create_empty_dataframe()

        df = df.with_columns(pl.lit(symbol.upper()).alias("symbol"))
        df = df.select(["timestamp", "symbol", "open", "high", "low", "close", "volume"])

        logger.info("Fetched data from CoinGecko (async)", coin_id=coin_id, rows=len(df))

        return df
