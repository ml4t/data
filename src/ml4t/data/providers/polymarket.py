"""Polymarket prediction market data provider.

Polymarket is the largest prediction market by volume, operating on Polygon blockchain.
Their CLOB (Central Limit Order Book) API provides historical price data for all markets.

API Documentation:
- CLOB Timeseries: https://docs.polymarket.com/developers/CLOB/timeseries
- Gamma Markets: https://docs.polymarket.com/developers/gamma-markets-api/get-markets

Rate Limits:
- CLOB API: ~60 requests per minute (estimated)
- Gamma API: ~30 requests per minute (estimated)

Market Structure:
- Each market has YES and NO outcome tokens
- Condition ID: Unique market identifier (0x...)
- Slug: Human-readable URL slug
- Token IDs: Separate tokens for YES and NO outcomes

Price Interpretation:
- Prices are probabilities (0.00 to 1.00)
- YES + NO prices should sum to ~1.00 (minus spread)

Example:
    >>> from ml4t.data.providers.polymarket import PolymarketProvider
    >>> provider = PolymarketProvider()  # No auth required
    >>> data = provider.fetch_ohlcv("will-bitcoin-exceed-100k-2025", "2024-01-01", "2024-12-31")
    >>> markets = provider.list_markets(active=True)
    >>> provider.close()
"""

import hashlib
import json
from collections.abc import Iterator
from datetime import UTC, date, datetime, timedelta
from math import isfinite
from typing import Any, ClassVar

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
from ml4t.data.providers.prediction_markets import (
    RESOLUTION_SCHEMA,
    RESULT_NO,
    RESULT_OTHER,
    RESULT_VOID,
    RESULT_YES,
    TRADE_SCHEMA,
    UTC_DATETIME,
    empty_frame,
    parse_utc_timestamp,
    to_unix_seconds,
)

logger = structlog.get_logger()

TimestampArg = int | float | str | date | datetime | None

# Normalized market columns returned by PolymarketProvider.fetch_markets.
MARKET_SCHEMA: dict[str, pl.DataType] = {
    "ticker": pl.Utf8(),
    "market_id": pl.Utf8(),
    "slug": pl.Utf8(),
    "question": pl.Utf8(),
    "event_slug": pl.Utf8(),
    "event_title": pl.Utf8(),
    "category": pl.Utf8(),
    "tags": pl.List(pl.Utf8()),
    "yes_outcome": pl.Utf8(),
    "no_outcome": pl.Utf8(),
    "yes_token_id": pl.Utf8(),
    "no_token_id": pl.Utf8(),
    "closed": pl.Boolean(),
    "open_time": UTC_DATETIME,
    "close_time": UTC_DATETIME,
    **RESOLUTION_SCHEMA,
    "uma_resolution_status": pl.Utf8(),
    "outcome_prices": pl.List(pl.Float64()),
    "volume": pl.Float64(),
    "neg_risk": pl.Boolean(),
    "neg_risk_market_id": pl.Utf8(),
    "fees_enabled": pl.Boolean(),
    "fee_type": pl.Utf8(),
    "maker_base_fee": pl.Float64(),
    "taker_base_fee": pl.Float64(),
    "fee_schedule": pl.Utf8(),
}

# Price points returned by PolymarketProvider.fetch_candles.
CANDLE_SCHEMA: dict[str, pl.DataType] = {
    "timestamp": UTC_DATETIME,
    "token_id": pl.Utf8(),
    "price": pl.Float64(),
}

# Normalized trade columns returned by PolymarketProvider.fetch_trades.
POLYMARKET_TRADE_SCHEMA: dict[str, pl.DataType] = {
    **TRADE_SCHEMA,
    "outcome": pl.Utf8(),
    "outcome_index": pl.Int64(),
    "side": pl.Utf8(),
    "outcome_price": pl.Float64(),
    "asset": pl.Utf8(),
    "proxy_wallet": pl.Utf8(),
    "transaction_hash": pl.Utf8(),
}


def _json_list(value: Any) -> list[Any] | None:
    """Parse a Gamma list field, which arrives as a JSON-encoded string or a list."""
    if value is None or value == "":
        return None
    if isinstance(value, list):
        return value
    parsed = json.loads(value)
    if not isinstance(parsed, list):
        raise ValueError(f"expected a JSON list, got {value!r}")
    return parsed


def _optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        raise ValueError(f"boolean where a number was expected: {value!r}")
    return float(value)


def _iso_utc(seconds: int) -> str:
    return datetime.fromtimestamp(seconds, UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


class PolymarketProvider(BaseProvider):
    """Polymarket prediction market data provider.

    Provides access to Polymarket prediction market data with support for:
    - Price history with OHLC aggregation
    - Market listing and search
    - Symbol resolution (slug/condition_id → token_id)
    - Both YES and NO outcome tokens

    Prices represent probabilities (0.00 to 1.00):
    - 0.65 means 65% implied probability of event occurring
    - YES + NO prices should sum to ~1.00

    Rate Limits:
    - ~30 requests per minute (conservative)
    """

    # Conservative rate limit: 30 req/min
    DEFAULT_RATE_LIMIT: ClassVar[tuple[int, float]] = (30, 60.0)
    HISTORY_CHUNK_DAYS: ClassVar[int] = 14
    MARKET_PAGE_SIZE: ClassVar[int] = 500
    MAX_MARKET_SCAN: ClassVar[int] = 5000

    # API base URLs
    CLOB_URL: ClassVar[str] = "https://clob.polymarket.com"
    GAMMA_URL: ClassVar[str] = "https://gamma-api.polymarket.com"
    DATA_API_URL: ClassVar[str] = "https://data-api.polymarket.com"

    # GET /markets/keyset returns at most 100 markets per page whatever ``limit`` asks for.
    KEYSET_PAGE_SIZE: ClassVar[int] = 100

    # GET /prices-history rejects a startTs..endTs span longer than 15 days (HTTP 400).
    PRICE_HISTORY_MAX_WINDOW_SECONDS: ClassVar[int] = 15 * 86_400

    # GET /trades serves up to 10000 trades per request and rejects offset > 10000 (HTTP 400).
    TRADES_MAX_LIMIT: ClassVar[int] = 10_000
    TRADES_MAX_OFFSET: ClassVar[int] = 10_000
    TRADES_PAGE_SIZE: ClassVar[int] = 1_000

    # Map common frequency names to Polymarket interval values
    INTERVAL_MAP: ClassVar[dict[str, str]] = {
        "1m": "1m",
        "minute": "1m",
        "1h": "1h",
        "hourly": "1h",
        "hour": "1h",
        "6h": "6h",
        "1d": "1d",
        "daily": "1d",
        "day": "1d",
        "1w": "1w",
        "weekly": "1w",
        "week": "1w",
        "max": "max",
    }

    # Fidelity in minutes for each interval (for OHLC aggregation)
    FIDELITY_MAP: ClassVar[dict[str, int]] = {
        "1m": 1,
        "minute": 1,
        "1h": 60,
        "hourly": 60,
        "hour": 60,
        "6h": 360,
        "1d": 1440,
        "daily": 1440,
        "day": 1440,
        "1w": 10080,
        "weekly": 10080,
        "week": 10080,
    }

    def __init__(
        self,
        rate_limit: tuple[int, float] | None = None,
    ):
        """Initialize Polymarket provider.

        Args:
            rate_limit: Optional custom rate limit (calls, period_seconds)
        """
        super().__init__(rate_limit=rate_limit or self.DEFAULT_RATE_LIMIT)

        # Cache for symbol resolution (slug/condition_id → market data)
        self._market_cache: dict[str, dict[str, Any]] = {}

        self.logger.info("Initialized Polymarket provider")

    @property
    def name(self) -> str:
        """Return provider name."""
        return "polymarket"

    def _is_token_id(self, symbol: str) -> bool:
        """Check if symbol is a token ID (long numeric string).

        Args:
            symbol: Symbol to check

        Returns:
            True if symbol appears to be a token ID
        """
        return symbol.isdigit() and len(symbol) > 15

    def _is_condition_id(self, symbol: str) -> bool:
        """Check if symbol is a condition ID (starts with 0x).

        Args:
            symbol: Symbol to check

        Returns:
            True if symbol appears to be a condition ID
        """
        return symbol.startswith("0x")

    def _get_market_by_slug(self, slug: str) -> dict[str, Any]:
        """Get market data by slug from Gamma API.

        Args:
            slug: Human-readable market slug (e.g., "will-bitcoin-exceed-100k-2025")

        Returns:
            Market dictionary with tokens, condition_id, etc.

        Raises:
            SymbolNotFoundError: If market not found
        """
        # Check cache first
        cache_key = f"slug:{slug}"
        if cache_key in self._market_cache:
            return self._market_cache[cache_key]

        try:
            endpoint = f"{self.GAMMA_URL}/markets/slug/{slug}"
            self._acquire_rate_limit()
            response = self.session.get(endpoint)

            if response.status_code == 429:
                raise RateLimitError(provider="polymarket", retry_after=60.0)
            if response.status_code == 200:
                market = response.json()
                if isinstance(market, dict) and market.get("slug"):
                    self._market_cache[cache_key] = market
                    condition_id = market.get("conditionId") or market.get("condition_id")
                    if condition_id:
                        self._market_cache[f"condition:{condition_id}"] = market
                    return market
            elif response.status_code != 404:
                raise NetworkError(
                    provider="polymarket",
                    message=f"HTTP {response.status_code}: {response.text[:200]}",
                )

            normalized_slug = slug.strip().lower()
            for active, closed in ((True, False), (None, None)):
                offset = 0
                while offset < self.MAX_MARKET_SCAN:
                    markets = self.list_markets(
                        active=active,
                        closed=closed,
                        limit=self.MARKET_PAGE_SIZE,
                        offset=offset,
                    )
                    if not markets:
                        break

                    for market in markets:
                        market_slug = str(market.get("slug", "")).strip().lower()
                        if market_slug == normalized_slug:
                            self._market_cache[cache_key] = market
                            condition_id = market.get("conditionId") or market.get("condition_id")
                            if condition_id:
                                self._market_cache[f"condition:{condition_id}"] = market
                            return market

                    if len(markets) < self.MARKET_PAGE_SIZE:
                        break

                    offset += len(markets)

            raise SymbolNotFoundError(
                provider="polymarket",
                symbol=slug,
                details={"type": "slug"},
            )

        except (RateLimitError, NetworkError, SymbolNotFoundError):
            raise
        except Exception as err:
            raise NetworkError(
                provider="polymarket",
                message=f"Failed to get market by slug: {slug}",
            ) from err

    def _get_market_by_condition(self, condition_id: str) -> dict[str, Any]:
        """Get market data by condition ID from Gamma API.

        Args:
            condition_id: Market condition ID (0x...)

        Returns:
            Market dictionary with tokens, slug, etc.

        Raises:
            SymbolNotFoundError: If market not found
        """
        # Check cache first
        cache_key = f"condition:{condition_id}"
        if cache_key in self._market_cache:
            return self._market_cache[cache_key]

        try:
            for active, closed in ((True, False), (None, None)):
                offset = 0
                while offset < self.MAX_MARKET_SCAN:
                    markets = self.list_markets(
                        active=active,
                        closed=closed,
                        limit=self.MARKET_PAGE_SIZE,
                        offset=offset,
                    )
                    if not markets:
                        break

                    for market in markets:
                        market_condition = market.get("conditionId") or market.get("condition_id")
                        if market_condition == condition_id:
                            self._market_cache[cache_key] = market
                            market_slug = market.get("slug")
                            if market_slug:
                                self._market_cache[f"slug:{market_slug}"] = market
                            return market

                    if len(markets) < self.MARKET_PAGE_SIZE:
                        break

                    offset += len(markets)

            raise SymbolNotFoundError(
                provider="polymarket",
                symbol=condition_id,
                details={"type": "condition_id"},
            )

        except (RateLimitError, NetworkError, SymbolNotFoundError):
            raise
        except Exception as err:
            raise NetworkError(
                provider="polymarket",
                message=f"Failed to get market by condition ID: {condition_id}",
            ) from err

    def resolve_symbol(self, symbol: str, outcome: str = "yes") -> str:
        """Resolve various symbol formats to token_id.

        Accepts:
        - Token ID directly: "12345678901234567890"
        - Condition ID: "0xabcd1234..."
        - Slug: "will-bitcoin-exceed-100k-2025"

        Args:
            symbol: Symbol in any supported format
            outcome: Outcome to get token for ("yes" or "no")

        Returns:
            Token ID for the specified outcome

        Raises:
            SymbolNotFoundError: If symbol not found or outcome not available
            DataValidationError: If outcome is invalid
        """
        outcome = outcome.lower()
        if outcome not in ("yes", "no"):
            raise DataValidationError(
                provider="polymarket",
                message=f"Invalid outcome '{outcome}'. Must be 'yes' or 'no'",
                field="outcome",
                value=outcome,
            )

        # Already a token ID
        if self._is_token_id(symbol):
            return symbol

        # Get market data
        if self._is_condition_id(symbol):
            market = self._get_market_by_condition(symbol)
        else:
            market = self._get_market_by_slug(symbol)

        # Extract token_id for requested outcome
        # Try clobTokenIds first (new API format) - string array [YES_id, NO_id]
        # Note: API returns this as a JSON string, not a list
        clob_ids_raw = market.get("clobTokenIds", [])
        clob_ids = []
        if isinstance(clob_ids_raw, str) and clob_ids_raw:
            try:
                clob_ids = json.loads(clob_ids_raw)
            except json.JSONDecodeError:
                pass
        elif isinstance(clob_ids_raw, list):
            clob_ids = clob_ids_raw

        if clob_ids and len(clob_ids) >= 2:
            if outcome == "yes":
                return clob_ids[0]
            elif outcome == "no":
                return clob_ids[1]

        # Fall back to tokens array (legacy format) - array of {outcome, token_id}
        tokens = market.get("tokens", [])
        for token in tokens:
            token_outcome = token.get("outcome", "").lower()
            if token_outcome == outcome:
                return token.get("token_id", "")

        raise SymbolNotFoundError(
            provider="polymarket",
            symbol=symbol,
            details={"outcome": outcome, "clobTokenIds": clob_ids, "tokens": tokens},
        )

    def _fetch_price_history(
        self,
        token_id: str,
        start: str,
        end: str,
        interval: str = "1d",
    ) -> list[dict[str, Any]]:
        """Fetch raw price history from CLOB API.

        Args:
            token_id: Token ID to fetch
            start: Start date (YYYY-MM-DD)
            end: End date (YYYY-MM-DD)
            interval: Data interval (1m, 1h, 6h, 1d, 1w)

        Returns:
            List of price points [{t: timestamp, p: price}, ...]
        """
        try:
            start_dt = datetime.strptime(start, "%Y-%m-%d")
            end_dt = datetime.strptime(end, "%Y-%m-%d")

            history: list[dict[str, Any]] = []
            chunk_days = self.HISTORY_CHUNK_DAYS if interval != "max" else self.MAX_MARKET_SCAN
            chunk_start = start_dt

            while chunk_start <= end_dt:
                chunk_end = min(chunk_start + timedelta(days=chunk_days - 1), end_dt)
                chunk_end = chunk_end.replace(hour=23, minute=59, second=59)
                history.extend(
                    self._fetch_price_history_chunk(
                        token_id,
                        int(chunk_start.timestamp()),
                        int(chunk_end.timestamp()),
                        interval,
                    )
                )
                chunk_start = (chunk_end + timedelta(seconds=1)).replace(
                    hour=0,
                    minute=0,
                    second=0,
                )

            deduped: dict[int, dict[str, Any]] = {}
            dropped = 0
            for entry in history:
                timestamp = entry.get("t")
                if isinstance(timestamp, int) and not isinstance(timestamp, bool):
                    normalized_timestamp = timestamp
                    deduped[normalized_timestamp] = {**entry, "t": normalized_timestamp}
                elif isinstance(timestamp, float) and isfinite(timestamp):
                    normalized_timestamp = int(timestamp)
                    deduped[normalized_timestamp] = {**entry, "t": normalized_timestamp}
                else:
                    dropped += 1
            if dropped:
                self.logger.warning(
                    "Dropped Polymarket price history entries with invalid timestamps",
                    dropped=dropped,
                    total=len(history),
                )
            return [deduped[timestamp] for timestamp in sorted(deduped)]
        except (RateLimitError, NetworkError, SymbolNotFoundError):
            raise
        except Exception as err:
            raise NetworkError(
                provider="polymarket",
                message=f"Failed to fetch price history for token {token_id}",
            ) from err

    def _fetch_price_history_chunk(
        self,
        token_id: str,
        start_ts: int,
        end_ts: int,
        interval: str,
    ) -> list[dict[str, Any]]:
        """Fetch a single chunk of raw price history from the CLOB API."""
        endpoint = f"{self.CLOB_URL}/prices-history"
        params = {
            "market": token_id,
            "startTs": start_ts,
            "endTs": end_ts,
            "interval": interval,
        }

        self._acquire_rate_limit()
        response = self.session.get(endpoint, params=params)

        if response.status_code == 429:
            raise RateLimitError(provider="polymarket", retry_after=60.0)
        if response.status_code == 404:
            raise SymbolNotFoundError(
                provider="polymarket",
                symbol=token_id,
                details={"type": "token_id"},
            )
        if response.status_code != 200:
            raise NetworkError(
                provider="polymarket",
                message=f"HTTP {response.status_code}: {response.text[:200]}",
            )

        data = response.json()
        return data.get("history", [])

    def _aggregate_to_ohlc(
        self,
        price_data: list[dict[str, Any]],
        symbol: str,
        target_interval: str,
    ) -> pl.DataFrame:
        """Aggregate price-only data to OHLC format.

        For high-frequency data, aggregates to target interval.
        For sparse data, uses price as all OHLC values.

        Args:
            price_data: List of {t: timestamp, p: price} dictionaries
            symbol: Symbol name for labeling
            target_interval: Target interval (daily, hourly, etc.)

        Returns:
            Polars DataFrame with OHLCV schema
        """
        if not price_data:
            return self._create_empty_dataframe()

        try:
            # Convert to DataFrame
            df = pl.DataFrame(price_data)

            # Rename columns: t → timestamp, p → price
            if "t" in df.columns:
                df = df.rename({"t": "timestamp_raw", "p": "price"})
            elif "timestamp" in df.columns and "p" in df.columns:
                df = df.rename({"timestamp": "timestamp_raw", "p": "price"})

            # Convert timestamp from unix seconds
            df = df.with_columns(
                pl.from_epoch("timestamp_raw", time_unit="s")
                .dt.replace_time_zone("UTC")
                .alias("timestamp"),
                pl.col("price").cast(pl.Float64),
            )

            # Determine aggregation period
            period_map = {
                "1m": "1m",
                "minute": "1m",
                "1h": "1h",
                "hourly": "1h",
                "hour": "1h",
                "6h": "6h",
                "1d": "1d",
                "daily": "1d",
                "day": "1d",
                "1w": "1w",
                "weekly": "1w",
                "week": "1w",
            }
            period = period_map.get(target_interval.lower(), "1d")

            # Check if we have enough data points to aggregate
            if len(df) > 1:
                # Aggregate using group_by_dynamic
                df_ohlc = (
                    df.sort("timestamp")
                    .group_by_dynamic("timestamp", every=period)
                    .agg(
                        [
                            pl.col("price").first().alias("open"),
                            pl.col("price").max().alias("high"),
                            pl.col("price").min().alias("low"),
                            pl.col("price").last().alias("close"),
                            pl.len().alias("volume"),  # Count of price points as proxy for volume
                        ]
                    )
                )
            else:
                # Single data point - use price for all OHLC
                df_ohlc = df.select(
                    [
                        "timestamp",
                        pl.col("price").alias("open"),
                        pl.col("price").alias("high"),
                        pl.col("price").alias("low"),
                        pl.col("price").alias("close"),
                        pl.lit(1.0).alias("volume"),
                    ]
                )

            # Add symbol
            df_ohlc = df_ohlc.with_columns(
                [
                    pl.lit(symbol.upper()).alias("symbol"),
                    pl.col("volume").cast(pl.Float64),
                ]
            )

            # Select final schema
            df_ohlc = df_ohlc.select(
                ["timestamp", "symbol", "open", "high", "low", "close", "volume"]
            )

            # Sort and deduplicate
            df_ohlc = df_ohlc.sort("timestamp").unique(subset=["timestamp"], maintain_order=True)

            return df_ohlc

        except Exception as err:
            raise DataValidationError(
                provider="polymarket",
                message=f"Failed to aggregate price data for {symbol}",
            ) from err

    def _validate_response(self, df: pl.DataFrame) -> pl.DataFrame:
        """Override base validation for prediction market data.

        Prediction market data may have:
        - Identical OHLC values (when only price is available)
        - Low volume (count-based proxy)

        Args:
            df: DataFrame to validate

        Returns:
            Validated DataFrame
        """
        # Handle empty responses
        if df.is_empty():
            self.logger.info(
                "Provider returned empty DataFrame - no data available for requested range"
            )
            return df

        # Check required columns exist
        required_columns = ["timestamp", "open", "high", "low", "close", "volume"]
        for col in required_columns:
            if col not in df.columns:
                raise DataValidationError(self.name, f"Missing required column: {col}")

        # For prediction markets, we accept:
        # 1. Standard OHLC invariants (high >= low, etc.)
        # 2. OR identical values (all price = same)
        # Check if all values are identical (price-only scenario)
        identical_ohlc = (
            (df["open"] == df["close"]) & (df["high"] == df["close"]) & (df["low"] == df["close"])
        )

        if not identical_ohlc.all():
            # Standard OHLC data - validate normally
            invalid_ohlc = (
                (df["high"] < df["low"])
                | (df["high"] < df["open"])
                | (df["high"] < df["close"])
                | (df["low"] > df["open"])
                | (df["low"] > df["close"])
            )

            if invalid_ohlc.any():
                n_invalid = invalid_ohlc.sum()
                raise DataValidationError(
                    self.name, f"Found {n_invalid} rows with invalid OHLC relationships"
                )

        # Sort and deduplicate
        df = df.sort("timestamp").unique(subset=["timestamp"], maintain_order=True)

        return df

    def fetch_ohlcv(
        self,
        symbol: str,
        start: str,
        end: str,
        frequency: str = "daily",
        outcome: str = "yes",
    ) -> pl.DataFrame:
        """Fetch OHLCV data for a Polymarket market.

        Prices represent probabilities (0.00 to 1.00):
        - 0.65 means 65% implied probability of event occurring

        Note: Volume is a proxy based on number of price updates.

        Args:
            symbol: Market identifier (slug, condition_id, or token_id)
                    Examples:
                    - "will-bitcoin-exceed-100k-2025" (slug)
                    - "0xabcd..." (condition_id)
                    - "12345678901234567890" (token_id)
            start: Start date (YYYY-MM-DD)
            end: End date (YYYY-MM-DD)
            frequency: Data frequency (minute, hourly, daily, weekly)
            outcome: Outcome to fetch ("yes" or "no"), ignored if symbol is token_id

        Returns:
            Polars DataFrame with OHLCV data

        Example:
            >>> provider = PolymarketProvider()
            >>> # Daily data by slug
            >>> data = provider.fetch_ohlcv(
            ...     "will-bitcoin-exceed-100k-2025",
            ...     "2024-01-01", "2024-12-31"
            ... )
            >>>
            >>> # Hourly data for NO outcome
            >>> data = provider.fetch_ohlcv(
            ...     "will-bitcoin-exceed-100k-2025",
            ...     "2024-01-01", "2024-12-31",
            ...     frequency="hourly",
            ...     outcome="no"
            ... )
        """
        self.logger.info(
            f"Fetching {frequency} OHLCV",
            symbol=symbol,
            start=start,
            end=end,
            outcome=outcome,
        )

        # Validate inputs
        self._validate_inputs(symbol, start, end, frequency)

        # Map frequency to interval
        interval = self.INTERVAL_MAP.get(frequency.lower())
        if interval is None:
            raise DataValidationError(
                provider="polymarket",
                message=f"Unsupported frequency '{frequency}'. "
                f"Supported: {list(self.INTERVAL_MAP.keys())}",
                field="frequency",
                value=frequency,
            )

        # Resolve symbol to token_id
        if self._is_token_id(symbol):
            token_id = symbol
            display_symbol = symbol[:8] + "..."
        else:
            token_id = self.resolve_symbol(symbol, outcome)
            display_symbol = f"{symbol}:{outcome.upper()}"

        # Fetch price history
        # For better OHLC, fetch at higher frequency and aggregate
        fetch_interval = interval
        if interval in ("1d", "1w") and frequency.lower() in ("daily", "day", "1d"):
            # Fetch hourly for daily aggregation (if data is dense)
            fetch_interval = "1h"
        elif interval == "1w":
            # Fetch daily for weekly aggregation
            fetch_interval = "1d"

        price_data = self._fetch_price_history(token_id, start, end, fetch_interval)

        # Aggregate to OHLC
        df = self._aggregate_to_ohlc(price_data, display_symbol, frequency)

        df = self._validate_ohlcv(df, self.name, display_symbol)

        self.logger.info(f"Fetched {len(df)} records", symbol=display_symbol)

        return df

    def fetch_both_outcomes(
        self,
        symbol: str,
        start: str,
        end: str,
        frequency: str = "daily",
    ) -> pl.DataFrame:
        """Fetch OHLCV data for both YES and NO outcomes.

        Returns a DataFrame with data for both outcomes, useful for
        analyzing the spread and arbitrage opportunities.

        Args:
            symbol: Market identifier (slug or condition_id, not token_id)
            start: Start date (YYYY-MM-DD)
            end: End date (YYYY-MM-DD)
            frequency: Data frequency (minute, hourly, daily, weekly)

        Returns:
            Long-format DataFrame with symbol column containing outcome suffix
            (e.g., "WILL-BITCOIN-EXCEED-100K-2025:YES")

        Example:
            >>> provider = PolymarketProvider()
            >>> df = provider.fetch_both_outcomes(
            ...     "will-bitcoin-exceed-100k-2025",
            ...     "2024-01-01", "2024-12-31"
            ... )
        """
        if self._is_token_id(symbol):
            raise DataValidationError(
                provider="polymarket",
                message="Cannot fetch both outcomes from token_id. Use slug or condition_id.",
                field="symbol",
                value=symbol,
            )

        dataframes = []
        for outcome in ["yes", "no"]:
            try:
                df = self.fetch_ohlcv(symbol, start, end, frequency, outcome)
                if not df.is_empty():
                    dataframes.append(df)
            except (SymbolNotFoundError, DataNotAvailableError) as err:
                self.logger.warning(f"No data for {outcome} outcome: {err}")

        if not dataframes:
            raise DataNotAvailableError(
                provider="polymarket",
                symbol=symbol,
                start=start,
                end=end,
                details={"error": "No data available for either outcome"},
            )

        return pl.concat(dataframes).sort(["timestamp", "symbol"])

    def list_markets(
        self,
        active: bool | None = True,
        closed: bool | None = None,
        category: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """List available markets from Gamma API.

        Args:
            active: Filter for active markets (True/False/None for all)
            closed: Filter for closed markets (True/False/None for all)
            category: Filter by category (e.g., "Politics", "Crypto", "Sports")
            limit: Maximum number of markets to return
            offset: Pagination offset

        Returns:
            List of market dictionaries containing:
            - id: Condition ID
            - question: The prediction question
            - slug: URL slug
            - tokens: List of {token_id, outcome, price}
            - volume: Total trading volume
            - liquidity: Current liquidity
            - startDate, endDate: Market dates
            - category: Market category

        Example:
            >>> provider = PolymarketProvider()
            >>> # Get active crypto markets
            >>> markets = provider.list_markets(active=True, category="Crypto")
            >>> for m in markets:
            ...     print(f"{m['slug']}: {m['question'][:50]}")
        """
        endpoint = f"{self.GAMMA_URL}/markets"
        if active is True and closed is None:
            closed = False

        params: dict[str, Any] = {
            "limit": min(limit, 1000),
            "offset": offset,
            "order": "volume24hr",
            "ascending": "false",
        }

        if active is not None:
            params["active"] = str(active).lower()
        if closed is not None:
            params["closed"] = str(closed).lower()

        self._acquire_rate_limit()

        try:
            response = self.session.get(endpoint, params=params)

            if response.status_code == 429:
                raise RateLimitError(provider="polymarket", retry_after=60.0)
            if response.status_code != 200:
                raise NetworkError(
                    provider="polymarket",
                    message=f"HTTP {response.status_code}: {response.text[:200]}",
                )

            markets = response.json()

            # Filter by category if specified (API may not support this directly)
            if category and isinstance(markets, list):
                markets = [m for m in markets if m.get("category", "").lower() == category.lower()]

            return markets if isinstance(markets, list) else []

        except (RateLimitError, NetworkError):
            raise
        except Exception as err:
            raise NetworkError(
                provider="polymarket",
                message="Failed to list markets",
            ) from err

    def search_markets(self, query: str, limit: int = 20) -> list[dict[str, Any]]:
        """Search markets by question text.

        Args:
            query: Search query string
            limit: Maximum number of results

        Returns:
            List of markets matching the query

        Example:
            >>> provider = PolymarketProvider()
            >>> markets = provider.search_markets("bitcoin")
            >>> for m in markets:
            ...     print(m['question'][:60])
        """
        # Gamma API doesn't have a search endpoint, so we fetch and filter
        # This is less efficient but works for basic searches
        all_markets = self.list_markets(active=True, limit=1000)

        query_lower = query.lower()
        results = []
        for market in all_markets:
            question = market.get("question", "").lower()
            slug = market.get("slug", "").lower()
            if query_lower in question or query_lower in slug:
                results.append(market)
                if len(results) >= limit:
                    break

        return results

    def get_market_metadata(self, symbol: str) -> dict[str, Any]:
        """Get detailed metadata for a market.

        Args:
            symbol: Market identifier (slug or condition_id)

        Returns:
            Market dictionary with all metadata

        Example:
            >>> provider = PolymarketProvider()
            >>> meta = provider.get_market_metadata("will-bitcoin-exceed-100k-2025")
            >>> print(f"Question: {meta['question']}")
            >>> print(f"Volume: ${meta.get('volume', 0):,.2f}")
        """
        if self._is_token_id(symbol):
            raise DataValidationError(
                provider="polymarket",
                message="Cannot get metadata from token_id. Use slug or condition_id.",
                field="symbol",
                value=symbol,
            )

        if self._is_condition_id(symbol):
            return self._get_market_by_condition(symbol)
        else:
            return self._get_market_by_slug(symbol)

    def get_token_prices(self, symbol: str) -> dict[str, float]:
        """Get current prices for both YES and NO tokens.

        Args:
            symbol: Market identifier (slug or condition_id)

        Returns:
            Dictionary with "yes" and "no" prices

        Example:
            >>> provider = PolymarketProvider()
            >>> prices = provider.get_token_prices("will-bitcoin-exceed-100k-2025")
            >>> print(f"YES: {prices['yes']:.2%}, NO: {prices['no']:.2%}")
        """
        market = self.get_market_metadata(symbol)
        prices: dict[str, float] = {}

        # Try new API format: outcomes + outcomePrices arrays
        outcomes_raw = market.get("outcomes", [])
        prices_raw = market.get("outcomePrices", [])

        # Parse JSON strings if needed
        outcomes = json.loads(outcomes_raw) if isinstance(outcomes_raw, str) else outcomes_raw or []
        outcome_prices = json.loads(prices_raw) if isinstance(prices_raw, str) else prices_raw or []

        if outcomes and outcome_prices and len(outcomes) == len(outcome_prices):
            for outcome, price in zip(outcomes, outcome_prices):
                outcome_lower = outcome.lower() if isinstance(outcome, str) else ""
                if outcome_lower in ("yes", "no"):
                    try:
                        prices[outcome_lower] = float(price)
                    except (ValueError, TypeError):
                        pass

        # Fall back to tokens array (legacy format)
        if not prices:
            tokens = market.get("tokens", []) or []
            for token in tokens:
                outcome = token.get("outcome", "").lower()
                price = token.get("price", 0.0)
                if outcome in ("yes", "no"):
                    prices[outcome] = float(price)

        return prices

    # ------------------------------------------------------------------
    # Resolved-market history: market listing, price history, trades
    # ------------------------------------------------------------------

    def iter_markets(
        self,
        closed: bool = True,
        min_volume: float | None = None,
        end_date_min: TimestampArg = None,
        end_date_max: TimestampArg = None,
        page_size: int = KEYSET_PAGE_SIZE,
        max_pages: int | None = None,
    ) -> Iterator[dict[str, Any]]:
        """Iterate over every Gamma market matching the filters.

        Pages through ``GET /markets/keyset`` (``GET /markets`` refuses offsets beyond 2000),
        passing ``next_cursor`` back as ``after_cursor``; markets come in ascending Gamma id
        order. The server applies ``closed``, ``volume_num_min``, ``end_date_min`` and
        ``end_date_max`` (verified 2026-10-08); the same filters are applied again on the
        client, so a filter the server stopped honouring cannot leak rows. A cursor the
        server repeats raises instead of looping forever.

        Args:
            closed: True for closed (resolved or awaiting resolution) markets, False for
                markets still trading.
            min_volume: Keep markets whose lifetime ``volumeNum`` (USDC) is at least this.
            end_date_min: Keep markets whose scheduled ``endDate`` is at or after this time
                (Unix seconds, ISO string, date or datetime; naive values are UTC).
            end_date_max: Keep markets whose ``endDate`` is at or before this time.
            page_size: Markets per request (1-100).
            max_pages: Optional cap on the number of pages, for sampling.

        Yields:
            Raw Gamma market dictionaries, including the market's ``tags``.
        """
        limit = max(1, min(int(page_size), self.KEYSET_PAGE_SIZE))
        min_end = to_unix_seconds(end_date_min)
        max_end = to_unix_seconds(end_date_max)
        params: dict[str, Any] = {
            "limit": limit,
            "closed": str(bool(closed)).lower(),
            "include_tag": "true",
        }
        if min_volume is not None:
            params["volume_num_min"] = float(min_volume)
        if min_end is not None:
            params["end_date_min"] = _iso_utc(min_end)
        if max_end is not None:
            params["end_date_max"] = _iso_utc(max_end)

        def selected(market: dict[str, Any]) -> bool:
            if market.get("closed") is not None and bool(market["closed"]) != bool(closed):
                return False
            if min_volume is not None:
                volume = self._market_volume(market)
                if volume is None or volume < float(min_volume):
                    return False
            if min_end is None and max_end is None:
                return True
            end = self._lenient_timestamp(market, "endDate")
            if end is None:
                return False
            end_seconds = end.timestamp()
            if min_end is not None and end_seconds < min_end:
                return False
            return not (max_end is not None and end_seconds > max_end)

        cursor: str | None = None
        seen_cursors: set[str] = set()
        pages = 0
        while True:
            page_params = dict(params)
            if cursor:
                page_params["after_cursor"] = cursor
            payload = self._request_json(
                f"{self.GAMMA_URL}/markets/keyset", resource="markets", params=page_params
            )
            markets = payload.get("markets")
            if not isinstance(markets, list):
                raise DataValidationError(
                    provider="polymarket",
                    message="Keyset response has no 'markets' list",
                    field="markets",
                )
            for market in markets:
                if not isinstance(market, dict):
                    raise DataValidationError(
                        provider="polymarket",
                        message="Non-object entry in keyset 'markets'",
                        field="markets",
                    )
                if selected(market):
                    yield market
            pages += 1
            cursor = payload.get("next_cursor")
            if not cursor or not markets or (max_pages is not None and pages >= max_pages):
                return
            if not isinstance(cursor, str):
                raise DataValidationError(
                    provider="polymarket", message="Non-string next_cursor", field="next_cursor"
                )
            if cursor in seen_cursors:
                raise DataValidationError(
                    provider="polymarket",
                    message="Keyset cursor repeated; the server ignored after_cursor",
                    field="next_cursor",
                    value=cursor,
                )
            seen_cursors.add(cursor)

    def fetch_markets(
        self,
        closed: bool = True,
        min_volume: float | None = None,
        end_date_min: TimestampArg = None,
        end_date_max: TimestampArg = None,
        page_size: int = KEYSET_PAGE_SIZE,
        max_pages: int | None = None,
    ) -> pl.DataFrame:
        """Return every matching market with its normalized resolution outcome.

        Accepts the same filters as ``iter_markets``; see ``iter_market_frames`` for the
        columns.

        Example:
            >>> provider = PolymarketProvider()
            >>> big = provider.fetch_markets(min_volume=100_000, end_date_min="2026-01-01")
            >>> big.select("ticker", "question", "result", "settlement_ts")
        """
        frames = list(
            self.iter_market_frames(
                closed=closed,
                min_volume=min_volume,
                end_date_min=end_date_min,
                end_date_max=end_date_max,
                page_size=page_size,
                max_pages=max_pages,
            )
        )
        if not frames:
            return empty_frame(MARKET_SCHEMA)
        return pl.concat(frames, how="vertical")

    def iter_market_frames(
        self,
        chunk_size: int = 100_000,
        closed: bool = True,
        min_volume: float | None = None,
        end_date_min: TimestampArg = None,
        end_date_max: TimestampArg = None,
        page_size: int = KEYSET_PAGE_SIZE,
        max_pages: int | None = None,
    ) -> Iterator[pl.DataFrame]:
        """Stream normalized markets in frames of at most ``chunk_size`` rows.

        Takes the filters of ``iter_markets``. Each frame can be written out before the next
        page is requested; the closed listing runs to hundreds of thousands of markets.

        Gamma markets are binary: two outcomes, two CLOB tokens and two final prices, all
        in the same order. The first outcome is treated as the YES side and the second as
        NO, whatever their labels: in ``["Trump", "Boden"]`` "Trump" is ``yes_outcome``,
        and ``result == "yes"`` means the first outcome won.

        ``result`` comes from the final ``outcomePrices`` of a closed market: ``[1, 0]`` is
        ``"yes"``, ``[0, 1]`` ``"no"`` and ``[0.5, 0.5]`` (a 50/50 resolution) ``"void"``. A
        market whose ``umaResolutionStatus`` is ``"resolved"`` with any other prices is
        ``"other"``; everything else (open markets, closed markets without final prices) is
        null. Closed markets snap to final prices even while the UMA status is still
        ``"proposed"``; filter on ``uma_resolution_status == "resolved"`` to exclude them.

        Yields:
            Non-empty DataFrames with ``MARKET_SCHEMA`` columns: ``ticker`` (condition id),
            ``market_id`` (Gamma id), ``slug``, ``question``, ``event_slug``,
            ``event_title``, ``category`` (the market's or event's category, mostly null
            since 2024), ``tags`` (tag labels, the reliable topic field), ``yes_outcome``,
            ``no_outcome``, ``yes_token_id``, ``no_token_id``, ``closed``, ``open_time``
            (``startDate``), ``close_time`` (scheduled ``endDate``), ``result``,
            ``settlement_value`` (payout of the first outcome token), ``settlement_ts``
            (``closedTime``, else ``umaEndDate``; null while unresolved),
            ``uma_resolution_status``, ``outcome_prices``, ``volume`` (lifetime USDC),
            ``neg_risk``, ``neg_risk_market_id``, ``fees_enabled``, ``fee_type``,
            ``maker_base_fee``, ``taker_base_fee`` (Gamma's raw base-fee fields) and
            ``fee_schedule`` (Gamma's fee schedule as JSON text). Timestamps Gamma cannot
            parse (``umaEndDate`` holds ``"NOW*()"`` on some 2024 markets) are null. Markets
            without a condition id are skipped with a warning.
        """
        if chunk_size < 1:
            raise DataValidationError(
                provider="polymarket",
                message=f"chunk_size must be positive, got {chunk_size}",
                field="chunk_size",
                value=chunk_size,
            )
        rows: list[tuple[Any, ...]] = []
        for market in self.iter_markets(
            closed=closed,
            min_volume=min_volume,
            end_date_min=end_date_min,
            end_date_max=end_date_max,
            page_size=page_size,
            max_pages=max_pages,
        ):
            row = self._normalize_market(market)
            if row is None:
                continue
            rows.append(row)
            if len(rows) >= chunk_size:
                yield pl.DataFrame(rows, schema=MARKET_SCHEMA, orient="row")
                rows = []
        if rows:
            yield pl.DataFrame(rows, schema=MARKET_SCHEMA, orient="row")

    @staticmethod
    def _market_volume(market: dict[str, Any]) -> float | None:
        volume = market.get("volumeNum")
        if volume is None or volume == "":
            volume = market.get("volume")
        return _optional_float(volume)

    def _lenient_timestamp(self, market: dict[str, Any], key: str) -> datetime | None:
        """Parse a Gamma timestamp field; an unparseable value is logged and read as null."""
        try:
            return parse_utc_timestamp(market.get(key))
        except (TypeError, ValueError):
            self.logger.warning(
                "Unparseable Polymarket timestamp",
                field=key,
                value=market.get(key),
                market_id=market.get("id"),
            )
            return None

    @staticmethod
    def _market_result(
        closed: bool, uma_status: str | None, prices: list[float] | None
    ) -> str | None:
        if closed and prices is not None and len(prices) == 2:
            first, second = prices
            if first == 1.0 and second == 0.0:
                return RESULT_YES
            if first == 0.0 and second == 1.0:
                return RESULT_NO
            if first == 0.5 and second == 0.5:
                return RESULT_VOID
        if uma_status == "resolved":
            return RESULT_OTHER
        return None

    def _normalize_market(self, market: dict[str, Any]) -> tuple[Any, ...] | None:
        condition_id = market.get("conditionId")
        market_id = market.get("id")
        if not condition_id:
            self.logger.warning("Skipping Polymarket market without conditionId", id=market_id)
            return None
        try:
            outcomes = _json_list(market.get("outcomes")) or []
            raw_prices = _json_list(market.get("outcomePrices"))
            prices = [float(price) for price in raw_prices] if raw_prices else None
            token_ids = _json_list(market.get("clobTokenIds")) or []
            closed = bool(market.get("closed"))
            uma_status = market.get("umaResolutionStatus") or None
            result = self._market_result(closed, uma_status, prices)
            binary = len(outcomes) == 2
            events = market.get("events") or []
            event = events[0] if events and isinstance(events[0], dict) else {}
            tags = [
                str(tag["label"])
                for tag in market.get("tags") or event.get("tags") or []
                if isinstance(tag, dict) and tag.get("label")
            ]
            settlement_ts = None
            if result is not None:
                settlement_ts = self._lenient_timestamp(
                    market, "closedTime"
                ) or self._lenient_timestamp(market, "umaEndDate")
            fee_schedule = market.get("feeSchedule")
            fees_enabled = market.get("feesEnabled")
            neg_risk = market.get("negRisk")
            row = {
                "ticker": str(condition_id),
                "market_id": None if market_id is None else str(market_id),
                "slug": market.get("slug"),
                "question": market.get("question"),
                "event_slug": event.get("slug"),
                "event_title": event.get("title"),
                "category": market.get("category") or event.get("category"),
                "tags": tags,
                "yes_outcome": str(outcomes[0]) if binary else None,
                "no_outcome": str(outcomes[1]) if binary else None,
                "yes_token_id": str(token_ids[0]) if len(token_ids) == 2 else None,
                "no_token_id": str(token_ids[1]) if len(token_ids) == 2 else None,
                "closed": closed,
                "open_time": self._lenient_timestamp(market, "startDate")
                or self._lenient_timestamp(market, "createdAt"),
                "close_time": self._lenient_timestamp(market, "endDate"),
                "result": result,
                "settlement_value": prices[0] if result is not None and prices else None,
                "settlement_ts": settlement_ts,
                "uma_resolution_status": uma_status,
                "outcome_prices": prices,
                "volume": self._market_volume(market),
                "neg_risk": None if neg_risk is None else bool(neg_risk),
                "neg_risk_market_id": market.get("negRiskMarketID") or None,
                "fees_enabled": None if fees_enabled is None else bool(fees_enabled),
                "fee_type": market.get("feeType"),
                "maker_base_fee": _optional_float(market.get("makerBaseFee")),
                "taker_base_fee": _optional_float(market.get("takerBaseFee")),
                "fee_schedule": None
                if fee_schedule is None
                else json.dumps(fee_schedule, sort_keys=True),
            }
        except (TypeError, ValueError, KeyError) as err:
            raise DataValidationError(
                provider="polymarket",
                message=f"Malformed market record {market_id} ({condition_id}): {err}",
                field="markets",
                value=market_id,
            ) from err
        return tuple(row[column] for column in MARKET_SCHEMA)

    def fetch_candles(
        self,
        token_id: str,
        start: TimestampArg,
        end: TimestampArg,
        fidelity_minutes: int = 60,
    ) -> pl.DataFrame:
        """Return the CLOB price history of one outcome token between ``start`` and ``end``.

        ``GET /prices-history`` with ``interval=max`` returns nothing for a resolved market
        at fidelities finer than one day, and rejects a ``startTs``..``endTs`` span longer
        than 15 days. This method therefore always sends explicit bounds and splits the
        range into windows of at most 15 days, which serves resolved markets at any
        fidelity down to one minute (verified 2026-10-08).

        Args:
            token_id: CLOB token id (``yes_token_id`` or ``no_token_id`` of
                ``fetch_markets``).
            start: Range start (Unix seconds, ISO string, date or datetime; naive is UTC).
            end: Range end, inclusive.
            fidelity_minutes: Sampling step in minutes (1 or more).

        Returns:
            DataFrame with ``CANDLE_SCHEMA`` columns: ``timestamp`` (UTC), ``token_id`` and
            ``price`` (the token's price, 0-1), one row per sample, sorted. These are price
            samples, not OHLC bars, and carry no volume. An unknown token or a range without
            trading yields an empty frame (the CLOB answers an empty history, not 404).
        """
        start_seconds = to_unix_seconds(start)
        end_seconds = to_unix_seconds(end)
        if start_seconds is None or end_seconds is None or end_seconds < start_seconds:
            raise DataValidationError(
                provider="polymarket",
                message="fetch_candles needs start and end, with end >= start",
                field="start",
            )
        fidelity = int(fidelity_minutes)
        if fidelity < 1:
            raise DataValidationError(
                provider="polymarket",
                message=f"fidelity_minutes must be at least 1, got {fidelity_minutes}",
                field="fidelity_minutes",
                value=fidelity_minutes,
            )
        resource = f"price history for token {token_id}"
        points: dict[int, float] = {}
        chunk_start = start_seconds
        while chunk_start <= end_seconds:
            chunk_end = min(chunk_start + self.PRICE_HISTORY_MAX_WINDOW_SECONDS, end_seconds)
            payload = self._request_json(
                f"{self.CLOB_URL}/prices-history",
                resource=resource,
                params={
                    "market": token_id,
                    "startTs": chunk_start,
                    "endTs": chunk_end,
                    "fidelity": fidelity,
                },
            )
            history = payload.get("history")
            if not isinstance(history, list):
                raise DataValidationError(
                    provider="polymarket",
                    message=f"Response for {resource} has no 'history' list",
                    field="history",
                )
            for point in history:
                try:
                    timestamp = point["t"]
                    price = point["p"]
                    if isinstance(timestamp, bool) or isinstance(price, bool):
                        raise ValueError("boolean field")
                    seconds = int(timestamp)
                    value = float(price)
                    if seconds != timestamp or not isfinite(value):
                        raise ValueError("non-integer timestamp or non-finite price")
                except (KeyError, TypeError, ValueError) as err:
                    raise DataValidationError(
                        provider="polymarket",
                        message=f"Malformed price point for {resource}: {point!r}",
                        field="history",
                    ) from err
                if start_seconds <= seconds <= end_seconds:
                    points[seconds] = value
            chunk_start = chunk_end + 1
        if not points:
            return empty_frame(CANDLE_SCHEMA)
        ordered = sorted(points)
        return pl.DataFrame(
            {
                "timestamp": [datetime.fromtimestamp(t, UTC) for t in ordered],
                "token_id": [str(token_id)] * len(ordered),
                "price": [points[t] for t in ordered],
            },
            schema=CANDLE_SCHEMA,
        )

    @staticmethod
    def _trade_key(trade: dict[str, Any]) -> str:
        fields = ("transactionHash", "asset", "proxyWallet", "side", "size", "price", "timestamp")
        return hashlib.sha1(
            "|".join(str(trade.get(field)) for field in fields).encode(), usedforsecurity=False
        ).hexdigest()

    def iter_trades(
        self,
        condition_id: str,
        start: TimestampArg = None,
        end: TimestampArg = None,
        page_size: int = TRADES_PAGE_SIZE,
        max_pages: int | None = None,
    ) -> Iterator[dict[str, Any]]:
        """Iterate over the taker trades of one market, newest first.

        ``GET https://data-api.polymarket.com/trades`` returns trades newest first, takes
        inclusive ``start``/``end`` bounds in Unix seconds, serves up to 10000 trades per
        request and rejects ``offset`` above 10000. To reach older trades this method pages
        by offset until the cap, then moves ``end`` to the oldest second seen and starts
        again at offset 0, dropping the trades of that boundary second it already yielded.
        Records carry no trade id; a trade is identified by its transaction hash, token,
        wallet, side, size, price and time, so two otherwise identical fills in one
        transaction are yielded once.

        Args:
            condition_id: Market condition id (``ticker`` of ``fetch_markets``).
            start: Keep trades at or after this time (Unix seconds, ISO string, date or
                datetime; naive is UTC).
            end: Keep trades at or before this time.
            page_size: Trades per request (1-10000).
            max_pages: Optional cap on the number of requests, for sampling.

        Yields:
            Raw data-api trade dictionaries.

        Raises:
            DataValidationError: The server repeated a page (offset ignored) or one second
                holds more trades than one offset window can reach.
        """
        limit = max(1, min(int(page_size), self.TRADES_MAX_LIMIT))
        start_seconds = to_unix_seconds(start)
        window_end = to_unix_seconds(end)
        resource = f"trades for {condition_id}"
        boundary_keys: set[str] = set()
        pages = 0
        while True:
            offset = 0
            window_keys = set(boundary_keys)
            oldest: int | None = None
            oldest_keys: set[str] = set()
            previous_page: list[str] | None = None
            while True:
                params: dict[str, Any] = {"market": condition_id, "limit": limit, "offset": offset}
                if start_seconds is not None:
                    params["start"] = start_seconds
                if window_end is not None:
                    params["end"] = window_end
                response = self._request(
                    f"{self.DATA_API_URL}/trades", resource=resource, params=params
                )
                try:
                    trades = response.json()
                except ValueError as err:
                    raise DataValidationError(
                        provider="polymarket", message=f"Malformed JSON response for {resource}"
                    ) from err
                if not isinstance(trades, list):
                    raise DataValidationError(
                        provider="polymarket",
                        message=f"Expected a JSON list for {resource}, got {type(trades).__name__}",
                    )
                pages += 1
                page_keys: list[str] = []
                for trade in trades:
                    if not isinstance(trade, dict) or isinstance(trade.get("timestamp"), bool):
                        raise DataValidationError(
                            provider="polymarket",
                            message=f"Malformed trade entry for {resource}: {trade!r}",
                            field="trades",
                        )
                    try:
                        seconds = int(trade["timestamp"])
                    except (KeyError, TypeError, ValueError) as err:
                        raise DataValidationError(
                            provider="polymarket",
                            message=f"Trade without a timestamp for {resource}: {trade!r}",
                            field="timestamp",
                        ) from err
                    key = self._trade_key(trade)
                    page_keys.append(key)
                    if oldest is None or seconds < oldest:
                        oldest, oldest_keys = seconds, {key}
                    elif seconds == oldest:
                        oldest_keys.add(key)
                    if key in window_keys:
                        continue
                    window_keys.add(key)
                    if start_seconds is not None and seconds < start_seconds:
                        continue
                    if window_end is not None and seconds > window_end:
                        continue
                    yield trade
                if len(trades) < limit or (max_pages is not None and pages >= max_pages):
                    return
                if offset > 0 and page_keys == previous_page:
                    raise DataValidationError(
                        provider="polymarket",
                        message=f"Trade page repeated for {resource}; the server ignored offset",
                        field="offset",
                        value=offset,
                    )
                previous_page = page_keys
                offset += len(trades)
                if offset > self.TRADES_MAX_OFFSET:
                    break
            if oldest is None or (window_end is not None and oldest >= window_end):
                raise DataValidationError(
                    provider="polymarket",
                    message=(
                        f"More than {self.TRADES_MAX_OFFSET + limit} trades at second "
                        f"{window_end} for {resource}; cannot page past it"
                    ),
                    field="end",
                    value=window_end,
                )
            window_end = oldest
            boundary_keys = oldest_keys

    def fetch_trades(
        self,
        condition_id: str,
        start: TimestampArg = None,
        end: TimestampArg = None,
        page_size: int = TRADES_PAGE_SIZE,
        max_pages: int | None = None,
    ) -> pl.DataFrame:
        """Return the taker trades of one market, sorted by time.

        Accepts the arguments of ``iter_trades``. The data API's default ``takerOnly=true``
        view is used: one record per taker fill, without the matching maker records.

        Returns:
            DataFrame with ``POLYMARKET_TRADE_SCHEMA`` columns: ``trade_id`` (hash of the
            identifying fields), ``ticker`` (condition id), ``timestamp`` (UTC, whole
            seconds), ``price`` (YES price: the traded token's price, or one minus it for
            the second outcome), ``count`` (shares), ``taker_side`` (``"yes"`` when the
            taker bought the first outcome or sold the second, else ``"no"``),
            ``is_block_trade`` (always False), ``outcome`` (traded outcome label),
            ``outcome_index`` (0 or 1), ``side`` (taker ``BUY``/``SELL`` of that outcome),
            ``outcome_price`` (price of the traded token), ``asset`` (token id),
            ``proxy_wallet`` (taker wallet) and ``transaction_hash``.

        Example:
            >>> provider = PolymarketProvider()
            >>> trades = provider.fetch_trades(
            ...     "0x8ee2f1640386310eb5e7ffa596ba9335f2d324e303d21b0dfea6998874445791",
            ...     start="2025-12-31",
            ... )
        """
        rows = [
            self._normalize_trade(trade, condition_id)
            for trade in self.iter_trades(
                condition_id, start=start, end=end, page_size=page_size, max_pages=max_pages
            )
        ]
        if not rows:
            return empty_frame(POLYMARKET_TRADE_SCHEMA)
        return pl.DataFrame(rows, schema=POLYMARKET_TRADE_SCHEMA, orient="row").sort(
            ["timestamp", "trade_id"]
        )

    def _normalize_trade(self, trade: dict[str, Any], condition_id: str) -> tuple[Any, ...]:
        try:
            outcome_index = trade["outcomeIndex"]
            outcome_price = float(trade["price"])
            size = float(trade["size"])
            side = str(trade["side"]).upper()
            timestamp = datetime.fromtimestamp(int(trade["timestamp"]), UTC)
            if isinstance(outcome_index, bool) or outcome_index not in (0, 1):
                raise ValueError(f"outcomeIndex {outcome_index!r} is not 0 or 1")
            if side not in ("BUY", "SELL"):
                raise ValueError(f"side {side!r} is not BUY or SELL")
        except (KeyError, TypeError, ValueError) as err:
            raise DataValidationError(
                provider="polymarket",
                message=f"Malformed trade record for {condition_id}: {err}",
                field="trades",
            ) from err
        first_outcome = outcome_index == 0
        yes_price = outcome_price if first_outcome else round(1.0 - outcome_price, 10)
        taker_yes = first_outcome == (side == "BUY")
        return (
            self._trade_key(trade),
            trade.get("conditionId") or condition_id,
            timestamp,
            yes_price,
            size,
            RESULT_YES if taker_yes else RESULT_NO,
            False,
            trade.get("outcome"),
            int(outcome_index),
            side,
            outcome_price,
            None if trade.get("asset") is None else str(trade["asset"]),
            trade.get("proxyWallet"),
            trade.get("transactionHash"),
        )

    def close(self) -> None:
        """Close HTTP client and clear caches."""
        self._market_cache.clear()
        if hasattr(self, "session"):
            self.session.close()
            self._log_close_event("Closed Polymarket API client")
