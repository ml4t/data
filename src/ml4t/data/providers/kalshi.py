"""Kalshi prediction market data provider.

Kalshi is a CFTC-regulated prediction market exchange offering binary outcome
contracts on economics, politics, climate, tech, and more.

API Documentation: https://docs.kalshi.com/getting_started/quick_start_market_data

Rate Limits:
- ~10 requests per second (conservative estimate)
- No daily limit documented

Market Taxonomy:
- Series: Recurring event templates (e.g., KXINFL for monthly CPI)
- Event: Specific instances (e.g., KXINFL-25JAN for January 2025 CPI)
- Market: Tradeable contracts with specific strikes

Key Series Examples:
- KXINFL: CPI Inflation (monthly)
- KXFED: Fed Funds Rate (per FOMC)
- KXGDP: GDP Growth (quarterly)
- KXUNEMPLOY: Unemployment (monthly)
- KXSPX: S&P 500 Range (daily/weekly)
- KXBTC: Bitcoin Range (daily/weekly)

Price Interpretation:
- Prices are probabilities (0.00 to 1.00)
- 0.45 = 45% implied probability of event occurring

Example:
    >>> from ml4t.data.providers.kalshi import KalshiProvider
    >>> provider = KalshiProvider()  # No auth required for public data
    >>> data = provider.fetch_ohlcv("KXINFL-25JAN", "2024-01-01", "2024-12-31")
    >>> markets = provider.list_markets(status="open")
    >>> provider.close()
"""

import time
from collections.abc import Iterator
from datetime import date, datetime
from typing import Any, ClassVar

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
from ml4t.data.providers.prediction_markets import (
    RESOLUTION_SCHEMA,
    TRADE_SCHEMA,
    UTC_DATETIME,
    empty_frame,
    parse_utc_timestamp,
    result_from_label,
    to_unix_seconds,
)

logger = structlog.get_logger()

TimestampArg = int | float | str | date | datetime | None

# Normalized market columns returned by KalshiProvider.fetch_markets.
MARKET_SCHEMA: dict[str, pl.DataType] = {
    "ticker": pl.Utf8(),
    "event_ticker": pl.Utf8(),
    "series": pl.Utf8(),
    "title": pl.Utf8(),
    "status": pl.Utf8(),
    "market_type": pl.Utf8(),
    "open_time": UTC_DATETIME,
    "close_time": UTC_DATETIME,
    **RESOLUTION_SCHEMA,
    "result_raw": pl.Utf8(),
    "expiration_value": pl.Utf8(),
    "last_price": pl.Float64(),
    "previous_price": pl.Float64(),
    "volume": pl.Float64(),
    "open_interest": pl.Float64(),
    "source": pl.Utf8(),
}

# Normalized trade columns returned by KalshiProvider.fetch_trades.
KALSHI_TRADE_SCHEMA: dict[str, pl.DataType] = {
    **TRADE_SCHEMA,
    "taker_outcome_side": pl.Utf8(),
    "taker_book_side": pl.Utf8(),
    "source": pl.Utf8(),
}

_LIVE = "live"
_HISTORICAL = "historical"


class KalshiProvider(BaseProvider):
    """Kalshi prediction market data provider.

    Provides access to Kalshi prediction market data with support for:
    - Native OHLC candlestick data
    - Multiple timeframes (minute, hourly, daily)
    - Market listing and filtering
    - Series browsing

    Prices represent probabilities:
    - 0.45 = 45% probability of event occurring
    - Volume is in contracts traded

    Rate Limits:
    - ~10 requests per second (conservative)
    - No daily limit documented
    """

    # Conservative rate limit: 10 req/sec
    DEFAULT_RATE_LIMIT: ClassVar[tuple[int, float]] = (10, 1.0)

    # Kalshi API base URL (elections domain provides all markets)
    BASE_URL: ClassVar[str] = "https://api.elections.kalshi.com/trade-api/v2"

    # Public requests can be rejected transiently by Kalshi's edge layer.
    EDGE_403_RETRY_DELAY: ClassVar[float] = 1.0

    # Map common frequency names to Kalshi period_interval values
    FREQUENCY_MAP: ClassVar[dict[str, int]] = {
        "1m": 1,  # 1 minute
        "minute": 1,
        "1h": 60,  # 1 hour
        "hourly": 60,
        "hour": 60,
        "1d": 1440,  # 1 day
        "daily": 1440,
        "day": 1440,
    }

    def __init__(
        self,
        api_key: str | None = None,
        rate_limit: tuple[int, float] | None = None,
    ):
        """Initialize Kalshi provider.

        Args:
            api_key: Optional API key for authenticated endpoints (not needed for
                     public data like prices and markets). Get key at:
                     https://kalshi.com/account/api
            rate_limit: Optional custom rate limit (calls, period_seconds)
        """
        # API key is optional for public data
        self.api_key = api_key

        super().__init__(rate_limit=rate_limit or self.DEFAULT_RATE_LIMIT)

        self.logger.info("Initialized Kalshi provider")

    @property
    def name(self) -> str:
        """Return provider name."""
        return "kalshi"

    def _get_headers(self) -> dict[str, str]:
        """Get request headers with optional auth."""
        headers = {"Accept": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _is_unauthenticated_edge_forbidden(self, response: Any) -> bool:
        """Identify an HTML 403 from the public API edge layer."""
        if self.api_key or response.status_code != 403:
            return False

        headers = getattr(response, "headers", {})
        content_type = str(headers.get("content-type", "")).lower()
        body = str(getattr(response, "text", "")).lstrip().lower()
        return (
            "text/html" in content_type
            or body.startswith("<!doctype html")
            or body.startswith("<html")
        )

    def _extract_series_ticker(self, market_ticker: str) -> str:
        """Extract series ticker from market ticker.

        Market tickers follow format: {series}-{date}[-{strike}]
        Examples:
        - "KXINFL-25JAN" -> "KXINFL"
        - "KXSPX-25JAN03-T5950" -> "KXSPX"

        Args:
            market_ticker: Full market ticker

        Returns:
            Series ticker (first part before dash)
        """
        if not market_ticker or "-" not in market_ticker:
            raise DataValidationError(
                provider="kalshi",
                message=f"Invalid market ticker format: {market_ticker}. "
                "Expected format: SERIES-DATE (e.g., KXINFL-25JAN)",
                field="symbol",
                value=market_ticker,
            )
        parts = market_ticker.split("-")
        return parts[0]

    def _fetch_raw_data(
        self,
        symbol: str,
        start: str,
        end: str,
        frequency: str = "daily",
        series_ticker: str | None = None,
    ) -> list[dict[str, Any]]:
        """Fetch raw candlestick data from Kalshi API.

        Args:
            symbol: Market ticker (e.g., "KXINFL-25JAN")
            start: Start date (YYYY-MM-DD)
            end: End date (YYYY-MM-DD)
            frequency: Data frequency (minute, hourly, daily)
            series_ticker: Optional series ticker (auto-detected from symbol)

        Returns:
            List of candlestick dictionaries from Kalshi API
        """
        market_ticker = symbol.upper()

        # Auto-detect series ticker if not provided
        if series_ticker is None:
            series_ticker = self._extract_series_ticker(market_ticker)
        series_ticker = series_ticker.upper()

        # Map frequency to period_interval
        period_interval = self.FREQUENCY_MAP.get(frequency.lower())
        if period_interval is None:
            raise DataValidationError(
                provider="kalshi",
                message=f"Unsupported frequency '{frequency}'. "
                f"Supported: {list(self.FREQUENCY_MAP.keys())}",
                field="frequency",
                value=frequency,
            )

        # Convert dates to unix timestamps
        start_dt = datetime.strptime(start, "%Y-%m-%d")
        end_dt = datetime.strptime(end, "%Y-%m-%d")
        # Set end to end of day
        end_dt = end_dt.replace(hour=23, minute=59, second=59)

        start_ts = int(start_dt.timestamp())
        end_ts = int(end_dt.timestamp())

        # Build request
        endpoint = f"{self.BASE_URL}/series/{series_ticker}/markets/{market_ticker}/candlesticks"
        params = {
            "start_ts": start_ts,
            "end_ts": end_ts,
            "period_interval": period_interval,
        }

        try:
            response = self.session.get(
                endpoint,
                params=params,
                headers=self._get_headers(),
            )

            # Check for errors
            if response.status_code == 429:
                raise RateLimitError(provider="kalshi", retry_after=60.0)
            if response.status_code == 404:
                # Could be invalid series or market ticker
                raise SymbolNotFoundError(
                    provider="kalshi",
                    symbol=market_ticker,
                    details={"series_ticker": series_ticker},
                )
            if response.status_code != 200:
                # Try to parse error message
                try:
                    error_data = response.json()
                    error_msg = error_data.get("message", f"HTTP {response.status_code}")
                except Exception:
                    error_msg = f"HTTP {response.status_code}: {response.text[:200]}"

                raise NetworkError(provider="kalshi", message=error_msg)

            # Parse JSON response
            try:
                data = response.json()
            except Exception as err:
                raise NetworkError(
                    provider="kalshi",
                    message="Failed to parse JSON response",
                ) from err

            # Extract candlesticks
            candlesticks = data.get("candlesticks", [])

            return candlesticks

        except (RateLimitError, NetworkError, SymbolNotFoundError, DataValidationError):
            raise
        except Exception as err:
            raise NetworkError(
                provider="kalshi",
                message=f"Request failed for market {market_ticker}",
            ) from err

    @staticmethod
    def _struct_field_names(dtype: pl.DataType | None) -> set[str]:
        """Return the available field names for a Polars struct dtype."""
        if not isinstance(dtype, pl.Struct):
            return set()
        return {field.name for field in dtype.fields}

    @staticmethod
    def _struct_price_expr(column: str, field: str, fields: set[str]) -> pl.Expr:
        """Extract a probability price from a Kalshi nested struct."""
        if field in fields:
            return pl.col(column).struct.field(field).cast(pl.Float64) / 100.0

        dollar_field = f"{field}_dollars"
        if dollar_field in fields:
            return pl.col(column).struct.field(dollar_field).cast(pl.Float64)

        return pl.lit(None, dtype=pl.Float64)

    def _transform_data(self, raw_data: list[dict[str, Any]], symbol: str) -> pl.DataFrame:
        """Transform raw Kalshi API response to Polars DataFrame.

        Kalshi API returns nested structs for price data:
        - price.open/high/low/close or price.*_dollars for actual trades
        - yes_bid.*_dollars and yes_ask.*_dollars for quote-side OHLC
        - volume or volume_fp for contracts traded

        When trade prices are missing, we fall back to the YES bid-side implied
        probability, which matches how the book downloader consumes Kalshi OHLC.
        Prices are converted to probability (0-1) format.

        Args:
            raw_data: List of candlestick dictionaries from Kalshi API
            symbol: Market ticker for labeling

        Returns:
            Polars DataFrame with OHLCV schema
        """
        if not raw_data:
            return self._create_empty_dataframe()

        try:
            # Convert to DataFrame
            df = pl.DataFrame(raw_data)

            # Kalshi returns end_period_ts as unix timestamp
            df = df.with_columns(
                pl.from_epoch("end_period_ts", time_unit="s")
                .dt.replace_time_zone("UTC")
                .alias("timestamp")
            )

            volume_expr = pl.lit(0.0).alias("volume")
            if "volume" in df.columns:
                volume_expr = pl.col("volume").cast(pl.Float64).alias("volume")
            elif "volume_fp" in df.columns:
                volume_expr = pl.col("volume_fp").cast(pl.Float64).alias("volume")

            price_fields = self._struct_field_names(df.schema.get("price"))
            bid_fields = self._struct_field_names(df.schema.get("yes_bid"))
            ask_fields = self._struct_field_names(df.schema.get("yes_ask"))

            # Check data schema - Kalshi returns nested structs
            if price_fields or bid_fields or ask_fields:
                df = df.with_columns(
                    [
                        self._struct_price_expr("price", "open", price_fields).alias("trade_open"),
                        self._struct_price_expr("price", "high", price_fields).alias("trade_high"),
                        self._struct_price_expr("price", "low", price_fields).alias("trade_low"),
                        self._struct_price_expr("price", "close", price_fields).alias(
                            "trade_close"
                        ),
                        self._struct_price_expr("yes_bid", "open", bid_fields).alias("bid_open"),
                        self._struct_price_expr("yes_bid", "high", bid_fields).alias("bid_high"),
                        self._struct_price_expr("yes_bid", "low", bid_fields).alias("bid_low"),
                        self._struct_price_expr("yes_bid", "close", bid_fields).alias("bid_close"),
                        self._struct_price_expr("yes_ask", "open", ask_fields).alias("ask_open"),
                        self._struct_price_expr("yes_ask", "high", ask_fields).alias("ask_high"),
                        self._struct_price_expr("yes_ask", "low", ask_fields).alias("ask_low"),
                        self._struct_price_expr("yes_ask", "close", ask_fields).alias("ask_close"),
                    ]
                )
                df = df.with_columns(
                    [
                        pl.coalesce(
                            pl.col("trade_open"),
                            pl.col("bid_open"),
                            pl.col("ask_open"),
                        ).alias("open"),
                        pl.coalesce(
                            pl.col("trade_high"),
                            pl.col("bid_high"),
                            pl.col("ask_high"),
                        ).alias("high"),
                        pl.coalesce(
                            pl.col("trade_low"),
                            pl.col("bid_low"),
                            pl.col("ask_low"),
                        ).alias("low"),
                        pl.coalesce(
                            pl.col("trade_close"),
                            pl.col("bid_close"),
                            pl.col("ask_close"),
                        ).alias("close"),
                        volume_expr,
                        pl.lit(symbol.upper()).alias("symbol"),
                    ]
                )
            elif "price" in df.columns:
                # price is a scalar value (simple format)
                df = df.with_columns(
                    [
                        (pl.col("price").cast(pl.Float64) / 100.0).alias("open"),
                        (pl.col("price").cast(pl.Float64) / 100.0).alias("high"),
                        (pl.col("price").cast(pl.Float64) / 100.0).alias("low"),
                        (pl.col("price").cast(pl.Float64) / 100.0).alias("close"),
                        volume_expr,
                        pl.lit(symbol.upper()).alias("symbol"),
                    ]
                )
            elif all(col in df.columns for col in ["open", "high", "low", "close"]):
                # Standard flat OHLC format (from mocked tests)
                df = df.with_columns(
                    [
                        pl.col("open").cast(pl.Float64),
                        pl.col("high").cast(pl.Float64),
                        pl.col("low").cast(pl.Float64),
                        pl.col("close").cast(pl.Float64),
                        volume_expr,
                        pl.lit(symbol.upper()).alias("symbol"),
                    ]
                )
            else:
                raise DataValidationError(
                    provider="kalshi",
                    message=f"Unknown data schema. Expected 'price' struct or flat OHLC columns. "
                    f"Got columns: {df.columns}",
                    field="columns",
                    value=str(df.columns),
                )

            # Select final schema
            df = df.select(["timestamp", "symbol", "open", "high", "low", "close", "volume"])

            # Sort by timestamp and remove duplicates
            df = df.sort("timestamp").unique(subset=["timestamp"], maintain_order=True)

            return df

        except DataValidationError:
            raise
        except Exception as err:
            raise DataValidationError(
                provider="kalshi",
                message=f"Failed to transform data for {symbol}",
            ) from err

    def _validate_response(self, df: pl.DataFrame) -> pl.DataFrame:
        """Override base validation for prediction market data.

        Prediction market data may have:
        - Identical OHLC values (when only price is available)
        - Bid/ask spreads where high=ask, low=bid

        We validate required columns and handle these cases appropriately.

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
        series_ticker: str | None = None,
    ) -> pl.DataFrame:
        """Fetch OHLCV candlestick data for a Kalshi market.

        Prices represent probabilities (0.00 to 1.00):
        - 0.45 means 45% implied probability of event occurring
        - Volume is in contracts traded

        Args:
            symbol: Market ticker (e.g., "KXINFL-25JAN", "KXSPX-25JAN03-T5950")
            start: Start date (YYYY-MM-DD)
            end: End date (YYYY-MM-DD)
            frequency: Data frequency (minute, hourly, daily)
            series_ticker: Optional series ticker (auto-detected from symbol)

        Returns:
            Polars DataFrame with OHLCV data

        Example:
            >>> provider = KalshiProvider()
            >>> # Daily inflation market data
            >>> data = provider.fetch_ohlcv("KXINFL-25JAN", "2024-01-01", "2024-12-31")
            >>>
            >>> # Hourly SPX market data
            >>> spx = provider.fetch_ohlcv(
            ...     "KXSPX-25JAN03-T5950",
            ...     "2025-01-01", "2025-01-03",
            ...     frequency="hourly"
            ... )
        """
        market_ticker = symbol.upper()

        self.logger.info(
            f"Fetching {frequency} OHLCV",
            symbol=market_ticker,
            start=start,
            end=end,
            series_ticker=series_ticker,
        )

        # Validate inputs
        self._validate_inputs(symbol, start, end, frequency)

        # Acquire rate limit
        self._acquire_rate_limit()

        # Fetch and transform
        raw_data = self._fetch_raw_data(symbol, start, end, frequency, series_ticker=series_ticker)
        df = self._transform_data(raw_data, symbol)

        df = df.drop_nulls(["open", "high", "low", "close"])
        df = self._validate_ohlcv(df, self.name, market_ticker)

        self.logger.info(f"Fetched {len(df)} records", symbol=market_ticker)

        return df

    def list_markets(
        self,
        status: str | None = "open",
        series_ticker: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """List available markets with optional filters.

        Args:
            status: Filter by status ("unopened", "open", "closed", "settled")
                    or None for all statuses
            series_ticker: Optional series ticker to filter by
            limit: Maximum number of markets to return (1-1000)

        Returns:
            List of market dictionaries containing:
            - ticker: Market ticker
            - title: Human-readable title
            - status: Market status
            - yes_bid/yes_ask: Current bid/ask prices
            - last_price: Last traded price
            - volume: Total volume
            - open_interest: Current open interest
            - close_time: When market closes for trading
            - expiration_time: When market settles

        Example:
            >>> provider = KalshiProvider()
            >>> # Get open inflation markets
            >>> markets = provider.list_markets(status="open", series_ticker="KXINFL")
            >>> for m in markets:
            ...     print(f"{m['ticker']}: {m['title']}")
        """
        endpoint = f"{self.BASE_URL}/markets"
        params: dict[str, Any] = {"limit": min(limit, 1000)}

        if status:
            params["status"] = status
        if series_ticker:
            params["series_ticker"] = series_ticker.upper()

        self._acquire_rate_limit()

        try:
            response = self.session.get(
                endpoint,
                params=params,
                headers=self._get_headers(),
            )

            if self._is_unauthenticated_edge_forbidden(response):
                self.logger.warning(
                    "Kalshi public endpoint returned an HTML 403; retrying once",
                    endpoint=endpoint,
                )
                time.sleep(self.EDGE_403_RETRY_DELAY)
                self._acquire_rate_limit()
                response = self.session.get(
                    endpoint,
                    params=params,
                    headers=self._get_headers(),
                )

            if response.status_code != 200:
                raise NetworkError(
                    provider="kalshi",
                    message=f"HTTP {response.status_code}: {response.text[:200]}",
                )

            data = response.json()
            return data.get("markets", [])

        except NetworkError:
            raise
        except Exception as err:
            raise NetworkError(
                provider="kalshi",
                message="Failed to list markets",
            ) from err

    def list_series(self) -> list[dict[str, Any]]:
        """List all available series (event templates).

        Returns:
            List of series dictionaries containing:
            - ticker: Series ticker (e.g., "KXINFL")
            - title: Human-readable title
            - category: Market category
            - frequency: How often events occur

        Example:
            >>> provider = KalshiProvider()
            >>> series = provider.list_series()
            >>> for s in series:
            ...     print(f"{s['ticker']}: {s.get('title', 'N/A')}")
        """
        endpoint = f"{self.BASE_URL}/series"

        self._acquire_rate_limit()

        try:
            response = self.session.get(
                endpoint,
                headers=self._get_headers(),
            )

            if response.status_code != 200:
                raise NetworkError(
                    provider="kalshi",
                    message=f"HTTP {response.status_code}: {response.text[:200]}",
                )

            data = response.json()
            return data.get("series", [])

        except NetworkError:
            raise
        except Exception as err:
            raise NetworkError(
                provider="kalshi",
                message="Failed to list series",
            ) from err

    def get_market_metadata(self, ticker: str) -> dict[str, Any]:
        """Get detailed metadata for a specific market.

        Args:
            ticker: Market ticker (e.g., "KXINFL-25JAN")

        Returns:
            Dictionary with market details including:
            - ticker: Market ticker
            - title: Human-readable title
            - subtitle: Additional description
            - status: Market status
            - yes_bid/yes_ask: Current bid/ask
            - last_price: Last traded price
            - volume: Total volume
            - open_interest: Current open interest
            - close_time: When trading closes
            - expiration_time: When market settles
            - result: Settlement result (if settled)

        Example:
            >>> provider = KalshiProvider()
            >>> meta = provider.get_market_metadata("KXINFL-25JAN")
            >>> print(f"Title: {meta['title']}")
            >>> print(f"Status: {meta['status']}")
        """
        ticker = ticker.upper()

        # First try to find in markets list (more efficient)
        # The /markets endpoint filters can find specific tickers
        endpoint = f"{self.BASE_URL}/markets/{ticker}"

        self._acquire_rate_limit()

        try:
            response = self.session.get(
                endpoint,
                headers=self._get_headers(),
            )

            if response.status_code == 404:
                raise SymbolNotFoundError(
                    provider="kalshi",
                    symbol=ticker,
                )
            if response.status_code != 200:
                raise NetworkError(
                    provider="kalshi",
                    message=f"HTTP {response.status_code}: {response.text[:200]}",
                )

            data = response.json()
            # API returns {"market": {...}}
            return data.get("market", data)

        except (SymbolNotFoundError, NetworkError):
            raise
        except Exception as err:
            raise NetworkError(
                provider="kalshi",
                message=f"Failed to get metadata for {ticker}",
            ) from err

    def fetch_multiple_markets(
        self,
        tickers: list[str],
        start: str,
        end: str,
        frequency: str = "daily",
        align: bool = True,
    ) -> pl.DataFrame:
        """Fetch OHLCV data for multiple markets and optionally align.

        Args:
            tickers: List of market tickers
            start: Start date (YYYY-MM-DD)
            end: End date (YYYY-MM-DD)
            frequency: Data frequency (minute, hourly, daily)
            align: Whether to align timestamps across markets

        Returns:
            Long-format DataFrame with columns:
            - timestamp
            - symbol
            - open, high, low, close, volume

            Or if align=True, wide-format with:
            - timestamp
            - {ticker}_close for each ticker

        Example:
            >>> provider = KalshiProvider()
            >>> df = provider.fetch_multiple_markets(
            ...     ["KXINFL-25JAN", "KXFED-25JAN"],
            ...     "2024-01-01", "2024-12-31",
            ...     frequency="daily"
            ... )
        """
        if not tickers:
            raise DataValidationError(
                provider="kalshi",
                message="tickers cannot be empty",
                field="tickers",
            )

        self.logger.info(
            "Fetching multiple markets",
            tickers=tickers,
            start=start,
            end=end,
            frequency=frequency,
        )

        # Fetch each market
        dataframes: list[pl.DataFrame] = []
        for ticker in tickers:
            try:
                df = self.fetch_ohlcv(ticker, start, end, frequency=frequency)
                if not df.is_empty():
                    dataframes.append(df)
            except (SymbolNotFoundError, DataNotAvailableError):
                self.logger.warning(f"No data available for market {ticker}")
                continue

        if not dataframes:
            raise DataNotAvailableError(
                provider="kalshi",
                symbol=",".join(tickers),
                start=start,
                end=end,
                details={"error": "No data available for any requested markets"},
            )

        if not align:
            # Return long-format (stacked)
            result = pl.concat(dataframes)
            return result.sort(["timestamp", "symbol"])

        # Wide format: join on timestamp
        result = None
        for df in dataframes:
            symbol = df["symbol"][0]
            df_wide = df.select(
                [
                    "timestamp",
                    pl.col("close").alias(f"{symbol}_close"),
                ]
            )
            if result is None:
                result = df_wide
            else:
                result = result.join(df_wide, on="timestamp", how="full", coalesce=True)

        if result is None:
            return self._create_empty_dataframe()

        return result.sort("timestamp")

    # ------------------------------------------------------------------
    # Full market enumeration, resolution outcomes and trade history
    # ------------------------------------------------------------------

    MAX_PAGE_SIZE: ClassVar[int] = 1000

    def _classify_error_response(self, response: httpx.Response) -> Exception | None:
        """Treat Kalshi's HTML 403 edge rejection of public requests as transient."""
        if self._is_unauthenticated_edge_forbidden(response):
            return NetworkError(
                provider="kalshi",
                message="Kalshi public endpoint returned an HTML 403 (edge layer)",
                retry_after=self.EDGE_403_RETRY_DELAY,
            )
        return None

    def get_historical_cutoff(self) -> dict[str, datetime] | None:
        """Return the live/archive boundary timestamps from ``GET /historical/cutoff``.

        Markets settled before ``market_settled_ts`` and trades created before
        ``trades_created_ts`` are served only by the ``/historical/*`` endpoints.

        Returns:
            Mapping of cutoff name (e.g. ``"market_settled_ts"``, ``"trades_created_ts"``) to
            an aware UTC datetime, or None when the endpoint does not exist (HTTP 404). The
            value is fetched once per provider instance.
        """
        if not hasattr(self, "_historical_cutoff"):
            try:
                payload = self._request_json(
                    f"{self.BASE_URL}/historical/cutoff",
                    resource="historical cutoff",
                    headers=self._get_headers(),
                )
            except SymbolNotFoundError:
                self.logger.warning("Kalshi /historical/cutoff not found; querying both tiers")
                self._historical_cutoff: dict[str, datetime] | None = None
            else:
                cutoff: dict[str, datetime] = {}
                for key, value in payload.items():
                    if not key.endswith("_ts"):
                        continue
                    try:
                        parsed = parse_utc_timestamp(value)
                    except ValueError as err:
                        raise DataValidationError(
                            provider="kalshi",
                            message=f"Malformed historical cutoff {key}={value!r}",
                            field=key,
                            value=value,
                        ) from err
                    if parsed is not None:
                        cutoff[key] = parsed
                self._historical_cutoff = cutoff
        return self._historical_cutoff

    def _cutoff_seconds(self, key: str) -> int | None:
        cutoff = self.get_historical_cutoff()
        if not cutoff or key not in cutoff:
            return None
        return int(cutoff[key].timestamp())

    def _paginate(
        self,
        path: str,
        params: dict[str, Any],
        *,
        items_key: str,
        resource: str,
        max_pages: int | None = None,
    ) -> Iterator[dict[str, Any]]:
        """Yield items from a cursor-paginated Kalshi endpoint until the cursor is empty."""
        cursor = ""
        seen_cursors: set[str] = set()
        pages = 0
        while True:
            page_params = dict(params)
            if cursor:
                page_params["cursor"] = cursor
            payload = self._request_json(
                f"{self.BASE_URL}{path}",
                resource=resource,
                params=page_params,
                headers=self._get_headers(),
            )
            items = payload.get(items_key)
            if not isinstance(items, list):
                raise DataValidationError(
                    provider="kalshi",
                    message=f"Response for {resource} has no '{items_key}' list",
                    field=items_key,
                )
            for item in items:
                if not isinstance(item, dict):
                    raise DataValidationError(
                        provider="kalshi",
                        message=f"Non-object entry in '{items_key}' for {resource}",
                        field=items_key,
                    )
                yield item
            pages += 1
            cursor = payload.get("cursor") or ""
            if not isinstance(cursor, str):
                raise DataValidationError(
                    provider="kalshi", message=f"Non-string cursor for {resource}", field="cursor"
                )
            if not cursor or (max_pages is not None and pages >= max_pages):
                return
            if cursor in seen_cursors:
                raise DataValidationError(
                    provider="kalshi",
                    message=f"Pagination cursor repeated for {resource}; aborting",
                    field="cursor",
                    value=cursor,
                )
            seen_cursors.add(cursor)

    def iter_markets(
        self,
        status: str | None = None,
        series_ticker: str | None = None,
        event_ticker: str | None = None,
        min_close_ts: TimestampArg = None,
        max_close_ts: TimestampArg = None,
        include_historical: bool = True,
        mve_filter: str | None = None,
        page_size: int = MAX_PAGE_SIZE,
        max_pages: int | None = None,
    ) -> Iterator[dict[str, Any]]:
        """Iterate over every market matching the filters, across the live and archive tiers.

        Kalshi serves markets settled before the historical cutoff (``get_historical_cutoff``)
        only from ``GET /historical/markets``, which accepts a single ``event_ticker`` or
        ``series_ticker`` filter and no status or time filters. This method pages through
        ``GET /markets`` and, when the filters can match archived markets, through the
        archive; it applies the filters the archive cannot take on the client and drops
        archive duplicates of live tickers.

        Args:
            status: ``"unopened"``, ``"open"``, ``"paused"``, ``"closed"``, ``"settled"`` or
                None for all. Archived markets are all settled, so the archive is only
                queried for ``"settled"`` or None.
            series_ticker: Series filter (e.g. ``"KXFED"``).
            event_ticker: Event filter (e.g. ``"FED-23DEC"``).
            min_close_ts: Keep markets closing at or after this time (Unix seconds, ISO
                string, date or datetime; naive values are UTC).
            max_close_ts: Keep markets closing at or before this time.
            include_historical: Also page through the archive tier.
            mve_filter: Multivariate (combo/parlay, ``KXMVE...``) selection: None includes
                them (Kalshi's default), ``"exclude"`` drops them, ``"only"`` keeps only them.
                Sent to ``/markets`` as ``mve_filter``. The archive accepts only
                ``"exclude"`` and only without a ticker filter, so the rest is applied on the
                client there.
            page_size: Markets per request (1-1000).
            max_pages: Optional cap on pages per tier, for sampling.

        Yields:
            Raw Kalshi market dictionaries with an added ``"_source"`` key
            (``"live"`` or ``"historical"``).
        """
        min_close = to_unix_seconds(min_close_ts)
        max_close = to_unix_seconds(max_close_ts)
        limit = max(1, min(int(page_size), self.MAX_PAGE_SIZE))

        def close_in_range(market: dict[str, Any]) -> bool:
            if min_close is None and max_close is None:
                return True
            close_time = parse_utc_timestamp(market.get("close_time"))
            if close_time is None:
                return False
            close_seconds = close_time.timestamp()
            if min_close is not None and close_seconds < min_close:
                return False
            return not (max_close is not None and close_seconds > max_close)

        if mve_filter not in (None, "exclude", "only"):
            raise DataValidationError(
                provider="kalshi",
                message=f"mve_filter must be None, 'exclude' or 'only', got {mve_filter!r}",
                field="mve_filter",
                value=mve_filter,
            )

        def mve_selected(market: dict[str, Any]) -> bool:
            if mve_filter is None:
                return True
            multivariate = bool(market.get("mve_collection_ticker")) or str(
                market.get("ticker", "")
            ).startswith("KXMVE")
            return multivariate == (mve_filter == "only")

        series_filter = series_ticker.upper() if series_ticker else None
        live_params: dict[str, Any] = {"limit": limit}
        if status:
            live_params["status"] = status
        if series_ticker:
            live_params["series_ticker"] = series_ticker.upper()
        if event_ticker:
            live_params["event_ticker"] = event_ticker.upper()
        if mve_filter:
            live_params["mve_filter"] = mve_filter
        # Kalshi accepts close-time filters only without a status or with status=closed.
        if status in (None, "closed"):
            if min_close is not None:
                live_params["min_close_ts"] = min_close
            if max_close is not None:
                live_params["max_close_ts"] = max_close

        seen: set[str] = set()
        for market in self._paginate(
            "/markets",
            live_params,
            items_key="markets",
            resource="markets",
            max_pages=max_pages,
        ):
            if not close_in_range(market) or not mve_selected(market):
                continue
            seen.add(str(market.get("ticker")))
            yield {**market, "_source": _LIVE, "_series": series_filter}

        if not include_historical or status not in (None, "settled"):
            return
        settled_cutoff = self._cutoff_seconds("market_settled_ts")
        if settled_cutoff is not None and min_close is not None and min_close >= settled_cutoff:
            return  # archived markets settled, hence closed, before the cutoff

        archive_params: dict[str, Any] = {"limit": limit}
        # The archive takes exactly one of event_ticker, series_ticker, mve_filter=exclude.
        if event_ticker:
            archive_params["event_ticker"] = event_ticker.upper()
        elif series_ticker:
            archive_params["series_ticker"] = series_ticker.upper()
        elif mve_filter == "exclude":
            archive_params["mve_filter"] = "exclude"
        event_prefix = event_ticker.upper() if event_ticker else None

        for market in self._paginate(
            "/historical/markets",
            archive_params,
            items_key="markets",
            resource="historical markets",
            max_pages=max_pages,
        ):
            ticker = str(market.get("ticker"))
            if ticker in seen:
                continue
            if event_prefix and str(market.get("event_ticker", "")).upper() != event_prefix:
                continue
            if not close_in_range(market) or not mve_selected(market):
                continue
            seen.add(ticker)
            yield {**market, "_source": _HISTORICAL, "_series": series_filter}

    def fetch_markets(
        self,
        status: str | None = None,
        series_ticker: str | None = None,
        event_ticker: str | None = None,
        min_close_ts: TimestampArg = None,
        max_close_ts: TimestampArg = None,
        include_historical: bool = True,
        mve_filter: str | None = None,
        page_size: int = MAX_PAGE_SIZE,
        max_pages: int | None = None,
    ) -> pl.DataFrame:
        """Return every matching market with its normalized resolution outcome.

        Accepts the same filters as ``iter_markets``, which covers both the live and the
        archive tier.

        Returns:
            DataFrame with ``MARKET_SCHEMA`` columns, one row per market. ``result`` is
            ``"yes"``/``"no"`` for a binary market resolved to that side, ``"other"`` for a
            scalar settlement, and null while unresolved; ``result_raw`` keeps Kalshi's
            label. ``settlement_value`` is the YES payout in dollars and ``settlement_ts``
            the UTC settlement time. ``series`` follows ``_series_of``; ``source`` names the
            tier that served the row.

        Example:
            >>> provider = KalshiProvider()
            >>> fed = provider.fetch_markets(status="settled", series_ticker="KXFEDDECISION")
            >>> fed.select("ticker", "result", "settlement_value", "settlement_ts")
        """
        rows = [
            self._normalize_market(market)
            for market in self.iter_markets(
                status=status,
                series_ticker=series_ticker,
                event_ticker=event_ticker,
                min_close_ts=min_close_ts,
                max_close_ts=max_close_ts,
                include_historical=include_historical,
                mve_filter=mve_filter,
                page_size=page_size,
                max_pages=max_pages,
            )
        ]
        if not rows:
            return empty_frame(MARKET_SCHEMA)
        return pl.DataFrame(rows, schema=MARKET_SCHEMA, orient="row")

    @staticmethod
    def _dollars(record: dict[str, Any], key: str) -> float | None:
        """Read a price in dollars from ``{key}_dollars`` or a legacy unsuffixed field.

        Unsuffixed JSON integers follow Kalshi's legacy cent encoding; unsuffixed strings and
        floats (archive candles) are already dollars.
        """
        value = record.get(f"{key}_dollars")
        if value is not None and value != "":
            return float(value)
        value = record.get(key)
        if value is None or value == "":
            return None
        if isinstance(value, bool):
            raise ValueError(f"Boolean value for price field {key}")
        if isinstance(value, int):
            return value / 100.0
        return float(value)

    @staticmethod
    def _fixed_point(record: dict[str, Any], key: str) -> float | None:
        """Read a contract count from ``{key}_fp`` or the unsuffixed field."""
        for name in (f"{key}_fp", key):
            value = record.get(name)
            if value is not None and value != "":
                if isinstance(value, bool):
                    raise ValueError(f"Boolean value for count field {name}")
                return float(value)
        return None

    @staticmethod
    def _series_of(market: dict[str, Any]) -> str | None:
        """Series of a market: the record's own field, the series filter, or the event prefix.

        Market records carry no series field. The event-ticker prefix equals the series for
        current tickers (``KXFEDDECISION-26SEP``) but not for some legacy events
        (``FED-23DEC`` belongs to series ``KXFED``); filtering by ``series_ticker`` gives the
        authoritative value.
        """
        explicit = market.get("series_ticker") or market.get("_series")
        if explicit:
            return str(explicit)
        event = market.get("event_ticker")
        return str(event).split("-", 1)[0] if event else None

    def _normalize_market(self, market: dict[str, Any]) -> tuple[Any, ...]:
        ticker = market.get("ticker")
        if not ticker:
            raise DataValidationError(
                provider="kalshi", message="Market entry without a ticker", field="ticker"
            )
        try:
            row = {
                "ticker": str(ticker),
                "event_ticker": market.get("event_ticker"),
                "series": self._series_of(market),
                "title": market.get("title"),
                "status": market.get("status"),
                "market_type": market.get("market_type"),
                "open_time": parse_utc_timestamp(market.get("open_time")),
                "close_time": parse_utc_timestamp(market.get("close_time")),
                "result": result_from_label(market.get("result")),
                "settlement_value": self._dollars(market, "settlement_value"),
                "settlement_ts": parse_utc_timestamp(market.get("settlement_ts")),
                "result_raw": market.get("result"),
                "expiration_value": market.get("expiration_value"),
                "last_price": self._dollars(market, "last_price"),
                "previous_price": self._dollars(market, "previous_price"),
                "volume": self._fixed_point(market, "volume"),
                "open_interest": self._fixed_point(market, "open_interest"),
                "source": market.get("_source"),
            }
        except (TypeError, ValueError) as err:
            raise DataValidationError(
                provider="kalshi",
                message=f"Malformed market record for {ticker}: {err}",
                field="markets",
                value=ticker,
            ) from err
        return tuple(row[column] for column in MARKET_SCHEMA)

    def iter_trades(
        self,
        ticker: str,
        min_ts: TimestampArg = None,
        max_ts: TimestampArg = None,
        include_historical: bool = True,
        page_size: int = MAX_PAGE_SIZE,
        max_pages: int | None = None,
    ) -> Iterator[dict[str, Any]]:
        """Iterate over the trades of one market across the live and archive tiers.

        Trades created before the ``trades_created_ts`` cutoff are served only by
        ``GET /historical/trades``. The archive is skipped when ``min_ts`` is at or after
        the cutoff; trades returned by both tiers are yielded once.

        Args:
            ticker: Market ticker.
            min_ts: Keep trades after this time (Unix seconds, ISO string, date or datetime).
            max_ts: Keep trades before this time.
            include_historical: Also page through the archive tier.
            page_size: Trades per request (1-1000).
            max_pages: Optional cap on pages per tier.

        Yields:
            Raw Kalshi trade dictionaries with an added ``"_source"`` key.
        """
        market_ticker = ticker.upper()
        params: dict[str, Any] = {
            "ticker": market_ticker,
            "limit": max(1, min(int(page_size), self.MAX_PAGE_SIZE)),
        }
        min_seconds = to_unix_seconds(min_ts)
        max_seconds = to_unix_seconds(max_ts)
        if min_seconds is not None:
            params["min_ts"] = min_seconds
        if max_seconds is not None:
            params["max_ts"] = max_seconds

        trades_cutoff = self._cutoff_seconds("trades_created_ts") if include_historical else None
        seen: set[str] = set()
        if trades_cutoff is None or max_seconds is None or max_seconds >= trades_cutoff:
            for trade in self._paginate(
                "/markets/trades",
                params,
                items_key="trades",
                resource=f"trades for {market_ticker}",
                max_pages=max_pages,
            ):
                seen.add(str(trade.get("trade_id")))
                yield {**trade, "_source": _LIVE}

        if not include_historical:
            return
        if trades_cutoff is not None and min_seconds is not None and min_seconds >= trades_cutoff:
            return
        for trade in self._paginate(
            "/historical/trades",
            params,
            items_key="trades",
            resource=f"historical trades for {market_ticker}",
            max_pages=max_pages,
        ):
            trade_id = str(trade.get("trade_id"))
            if trade_id in seen:
                continue
            seen.add(trade_id)
            yield {**trade, "_source": _HISTORICAL}

    def fetch_trades(
        self,
        ticker: str,
        min_ts: TimestampArg = None,
        max_ts: TimestampArg = None,
        include_historical: bool = True,
        page_size: int = MAX_PAGE_SIZE,
        max_pages: int | None = None,
    ) -> pl.DataFrame:
        """Return the trade history of one market, sorted by time.

        Accepts the same arguments as ``iter_trades``.

        Returns:
            DataFrame with ``KALSHI_TRADE_SCHEMA`` columns: ``trade_id``, ``ticker``,
            ``timestamp`` (UTC), ``price`` (YES price in dollars, 0-1), ``count``
            (contracts, fractional allowed), ``taker_side`` (``"yes"``/``"no"``; falls back
            to ``taker_outcome_side``), ``is_block_trade``, ``taker_outcome_side``,
            ``taker_book_side`` (``"bid"``/``"ask"``, null when absent) and ``source``.

        Example:
            >>> provider = KalshiProvider()
            >>> trades = provider.fetch_trades("KXFEDDECISION-26SEP-H25")
        """
        rows = [
            self._normalize_trade(trade)
            for trade in self.iter_trades(
                ticker,
                min_ts=min_ts,
                max_ts=max_ts,
                include_historical=include_historical,
                page_size=page_size,
                max_pages=max_pages,
            )
        ]
        if not rows:
            return empty_frame(KALSHI_TRADE_SCHEMA)
        return pl.DataFrame(rows, schema=KALSHI_TRADE_SCHEMA, orient="row").sort(
            ["timestamp", "trade_id"]
        )

    def _normalize_trade(self, trade: dict[str, Any]) -> tuple[Any, ...]:
        trade_id = trade.get("trade_id")
        try:
            timestamp = parse_utc_timestamp(trade.get("created_time"))
            price = self._dollars(trade, "yes_price")
            count = self._fixed_point(trade, "count")
        except (TypeError, ValueError) as err:
            raise DataValidationError(
                provider="kalshi",
                message=f"Malformed trade record {trade_id}: {err}",
                field="trades",
                value=trade_id,
            ) from err
        if not trade_id or timestamp is None or price is None or count is None:
            raise DataValidationError(
                provider="kalshi",
                message=f"Trade record missing trade_id, created_time, price or count: {trade}",
                field="trades",
            )
        taker_side = trade.get("taker_side") or trade.get("taker_outcome_side")
        return (
            str(trade_id),
            trade.get("ticker"),
            timestamp,
            price,
            count,
            taker_side,
            bool(trade.get("is_block_trade", False)),
            trade.get("taker_outcome_side"),
            trade.get("taker_book_side"),
            trade.get("_source"),
        )

    def close(self) -> None:
        """Close HTTP client."""
        if hasattr(self, "session"):
            self.session.close()
            self._log_close_event("Closed Kalshi API client")
