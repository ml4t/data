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

    def _transform_data(self, raw_data: list[dict[str, Any]], symbol: str) -> pl.DataFrame:
        """Transform raw Kalshi API response to Polars DataFrame.

        Kalshi API returns nested structs for price data:
        - price.open, price.high, price.low, price.close (in cents, 0-100) - actual trades
        - yes_bid.open/high/low/close - bid side quotes (in cents)
        - yes_ask.open/high/low/close - ask side quotes (in cents)
        - volume (contracts traded)

        When price values are null (no trades), we use mid-price from bid/ask.
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
            df = df.with_columns(pl.from_epoch("end_period_ts", time_unit="s").alias("timestamp"))

            # Check data schema - Kalshi returns nested structs
            if "price" in df.columns:
                price_schema = df.schema.get("price")

                if price_schema is not None and hasattr(price_schema, "fields"):
                    # Extract trade prices first
                    df = df.with_columns(
                        [
                            (pl.col("price").struct.field("open").cast(pl.Float64) / 100.0).alias(
                                "trade_open"
                            ),
                            (pl.col("price").struct.field("high").cast(pl.Float64) / 100.0).alias(
                                "trade_high"
                            ),
                            (pl.col("price").struct.field("low").cast(pl.Float64) / 100.0).alias(
                                "trade_low"
                            ),
                            (pl.col("price").struct.field("close").cast(pl.Float64) / 100.0).alias(
                                "trade_close"
                            ),
                        ]
                    )

                    # Check if we have yes_bid/yes_ask for fallback (when no trades)
                    if "yes_bid" in df.columns and "yes_ask" in df.columns:
                        # Extract bid/ask OHLC and compute mid-prices
                        df = df.with_columns(
                            [
                                (
                                    pl.col("yes_bid").struct.field("open").cast(pl.Float64) / 100.0
                                ).alias("bid_open"),
                                (
                                    pl.col("yes_bid").struct.field("high").cast(pl.Float64) / 100.0
                                ).alias("bid_high"),
                                (
                                    pl.col("yes_bid").struct.field("low").cast(pl.Float64) / 100.0
                                ).alias("bid_low"),
                                (
                                    pl.col("yes_bid").struct.field("close").cast(pl.Float64) / 100.0
                                ).alias("bid_close"),
                                (
                                    pl.col("yes_ask").struct.field("open").cast(pl.Float64) / 100.0
                                ).alias("ask_open"),
                                (
                                    pl.col("yes_ask").struct.field("high").cast(pl.Float64) / 100.0
                                ).alias("ask_high"),
                                (
                                    pl.col("yes_ask").struct.field("low").cast(pl.Float64) / 100.0
                                ).alias("ask_low"),
                                (
                                    pl.col("yes_ask").struct.field("close").cast(pl.Float64) / 100.0
                                ).alias("ask_close"),
                            ]
                        )

                        # Use trade price if available, otherwise mid-price from bid/ask
                        df = df.with_columns(
                            [
                                pl.coalesce(
                                    pl.col("trade_open"),
                                    (pl.col("bid_open") + pl.col("ask_open")) / 2.0,
                                ).alias("open"),
                                pl.coalesce(
                                    pl.col("trade_high"),
                                    (pl.col("bid_high") + pl.col("ask_high")) / 2.0,
                                ).alias("high"),
                                pl.coalesce(
                                    pl.col("trade_low"),
                                    (pl.col("bid_low") + pl.col("ask_low")) / 2.0,
                                ).alias("low"),
                                pl.coalesce(
                                    pl.col("trade_close"),
                                    (pl.col("bid_close") + pl.col("ask_close")) / 2.0,
                                ).alias("close"),
                                pl.col("volume").cast(pl.Float64),
                                pl.lit(symbol.upper()).alias("symbol"),
                            ]
                        )
                    else:
                        # No bid/ask data, just use trade prices (may be null)
                        df = df.with_columns(
                            [
                                pl.col("trade_open").alias("open"),
                                pl.col("trade_high").alias("high"),
                                pl.col("trade_low").alias("low"),
                                pl.col("trade_close").alias("close"),
                                pl.col("volume").cast(pl.Float64),
                                pl.lit(symbol.upper()).alias("symbol"),
                            ]
                        )
                else:
                    # price is a scalar value (simple format)
                    df = df.with_columns(
                        [
                            (pl.col("price").cast(pl.Float64) / 100.0).alias("open"),
                            (pl.col("price").cast(pl.Float64) / 100.0).alias("high"),
                            (pl.col("price").cast(pl.Float64) / 100.0).alias("low"),
                            (pl.col("price").cast(pl.Float64) / 100.0).alias("close"),
                            pl.col("volume").cast(pl.Float64)
                            if "volume" in df.columns
                            else pl.lit(1.0).alias("volume"),
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
                        pl.col("volume").cast(pl.Float64),
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

        # Validate response (standard OHLC validation)
        df = self._validate_response(df)

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
                result = result.join(df_wide, on="timestamp", how="outer_coalesce")

        if result is None:
            return self._create_empty_dataframe()

        return result.sort("timestamp")

    def close(self) -> None:
        """Close HTTP client."""
        if hasattr(self, "session"):
            self.session.close()
            self.logger.debug("Closed Kalshi API client")
