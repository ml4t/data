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

import json
from datetime import datetime
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

logger = structlog.get_logger()


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

    # API base URLs
    CLOB_URL: ClassVar[str] = "https://clob.polymarket.com"
    GAMMA_URL: ClassVar[str] = "https://gamma-api.polymarket.com"

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

        endpoint = f"{self.GAMMA_URL}/markets"
        params = {"slug": slug}

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

            data = response.json()

            # Gamma API returns a list
            if not data:
                raise SymbolNotFoundError(
                    provider="polymarket",
                    symbol=slug,
                    details={"type": "slug"},
                )

            market = data[0] if isinstance(data, list) else data

            # Cache the result
            self._market_cache[cache_key] = market

            return market

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

        endpoint = f"{self.GAMMA_URL}/markets/{condition_id}"

        self._acquire_rate_limit()

        try:
            response = self.session.get(endpoint)

            if response.status_code == 429:
                raise RateLimitError(provider="polymarket", retry_after=60.0)
            if response.status_code == 404:
                raise SymbolNotFoundError(
                    provider="polymarket",
                    symbol=condition_id,
                    details={"type": "condition_id"},
                )
            if response.status_code != 200:
                raise NetworkError(
                    provider="polymarket",
                    message=f"HTTP {response.status_code}: {response.text[:200]}",
                )

            market = response.json()

            # Cache the result
            self._market_cache[cache_key] = market

            return market

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
        # Convert dates to unix timestamps
        start_dt = datetime.strptime(start, "%Y-%m-%d")
        end_dt = datetime.strptime(end, "%Y-%m-%d")
        # Set end to end of day
        end_dt = end_dt.replace(hour=23, minute=59, second=59)

        start_ts = int(start_dt.timestamp())
        end_ts = int(end_dt.timestamp())

        endpoint = f"{self.CLOB_URL}/prices-history"
        params = {
            "market": token_id,
            "startTs": start_ts,
            "endTs": end_ts,
            "interval": interval,
        }

        self._acquire_rate_limit()

        try:
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

        except (RateLimitError, NetworkError, SymbolNotFoundError):
            raise
        except Exception as err:
            raise NetworkError(
                provider="polymarket",
                message=f"Failed to fetch price history for token {token_id}",
            ) from err

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
                pl.from_epoch("timestamp_raw", time_unit="s").alias("timestamp"),
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

    def _create_empty_dataframe(self) -> pl.DataFrame:
        """Create an empty DataFrame with the correct schema."""
        return pl.DataFrame(
            schema={
                "timestamp": pl.Datetime,
                "symbol": pl.Utf8,
                "open": pl.Float64,
                "high": pl.Float64,
                "low": pl.Float64,
                "close": pl.Float64,
                "volume": pl.Float64,
            }
        )

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

        # Validate response
        df = self._validate_response(df)

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
        params: dict[str, Any] = {
            "limit": min(limit, 1000),
            "offset": offset,
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

    def close(self) -> None:
        """Close HTTP client and clear caches."""
        self._market_cache.clear()
        if hasattr(self, "session"):
            self.session.close()
            self.logger.debug("Closed Polymarket API client")
