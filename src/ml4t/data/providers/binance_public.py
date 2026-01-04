"""Binance Public Data provider for bulk historical downloads.

This provider downloads from data.binance.vision, Binance's public data repository.
Unlike the live API (BinanceProvider), this:
- Works globally without geographic restrictions
- Downloads bulk ZIP files (not rate-limited)
- Provides historical data going back years
- Uses MIT-licensed data distribution

Repository: https://data.binance.vision
GitHub: https://github.com/binance/binance-public-data
"""

from __future__ import annotations

import asyncio
import io
import zipfile
from datetime import UTC, datetime, timedelta
from typing import ClassVar

import httpx
import polars as pl
import structlog

from ml4t.data.core.exceptions import DataValidationError, SymbolNotFoundError
from ml4t.data.providers.base import BaseProvider

logger = structlog.get_logger()


class BinancePublicProvider(BaseProvider):
    """Provider for bulk historical data from Binance Public Data repository.

    Downloads from data.binance.vision - Binance's public S3 bucket with
    historical market data. This is separate from the live API and has
    no geographic restrictions.

    Features:
    - No API key required
    - No rate limits (public S3 bucket)
    - Works globally (no geo-restrictions)
    - Historical data going back years
    - Spot and USD-M futures markets
    - Daily and monthly aggregated files

    Data Structure:
    - Spot: /data/spot/daily/klines/{symbol}/{interval}/
    - Futures: /data/futures/um/daily/klines/{symbol}/{interval}/

    Example:
        >>> provider = BinancePublicProvider(market="spot")
        >>> df = provider.fetch_ohlcv("BTCUSDT", "2024-01-01", "2024-01-31", "daily")
    """

    BASE_URL = "https://data.binance.vision/data"

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

    # Very permissive rate limit since it's public S3
    # S3 has no real rate limits - 1000/min is plenty fast without overwhelming
    DEFAULT_RATE_LIMIT: ClassVar[tuple[int, float]] = (1000, 60.0)

    def __init__(
        self,
        market: str = "spot",
        timeout: float = 60.0,
        rate_limit: tuple[int, float] | None = None,
    ) -> None:
        """Initialize Binance Public Data provider.

        Args:
            market: Market type - "spot" or "futures" (USD-M perpetuals)
            timeout: Request timeout in seconds (higher for large downloads)
            rate_limit: Optional custom rate limit (calls, period_seconds)
        """
        self.market = market.lower()
        if self.market not in ["spot", "futures"]:
            raise ValueError(f"Invalid market: {market}. Must be 'spot' or 'futures'")

        self.timeout = timeout

        super().__init__(rate_limit=rate_limit or self.DEFAULT_RATE_LIMIT)

        # Override session with longer timeout for bulk downloads
        self.session = httpx.Client(
            timeout=httpx.Timeout(timeout),
            limits=httpx.Limits(max_connections=10, max_keepalive_connections=5),
            follow_redirects=True,
        )

    @property
    def name(self) -> str:
        """Return the provider name."""
        return "binance_public"

    def _create_empty_dataframe(self) -> pl.DataFrame:
        """Create an empty DataFrame with the correct schema."""
        return pl.DataFrame(
            {
                "timestamp": [],
                "open": [],
                "high": [],
                "low": [],
                "close": [],
                "volume": [],
            },
            schema={
                "timestamp": pl.Datetime("ms", "UTC"),
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
            symbol = "BTCUSDT"
        elif symbol == "ETHUSD":
            symbol = "ETHUSDT"
        elif len(symbol) == 3 or not symbol.endswith(("USDT", "BUSD", "BTC", "ETH", "BNB")):
            symbol = symbol + "USDT"

        return symbol

    def _build_url(self, symbol: str, interval: str, date: datetime) -> str:
        """Build download URL for a specific date.

        Args:
            symbol: Normalized symbol (e.g., BTCUSDT)
            interval: Binance interval (e.g., 1d, 1h, 1m)
            date: Date to download

        Returns:
            Full URL to the ZIP file
        """
        date_str = date.strftime("%Y-%m-%d")

        if self.market == "spot":
            path = f"spot/daily/klines/{symbol}/{interval}/{symbol}-{interval}-{date_str}.zip"
        else:  # futures
            path = f"futures/um/daily/klines/{symbol}/{interval}/{symbol}-{interval}-{date_str}.zip"

        return f"{self.BASE_URL}/{path}"

    def _build_monthly_url(self, symbol: str, interval: str, year: int, month: int) -> str:
        """Build download URL for monthly aggregated data.

        Args:
            symbol: Normalized symbol
            interval: Binance interval
            year: Year (e.g., 2024)
            month: Month (1-12)

        Returns:
            Full URL to the monthly ZIP file
        """
        month_str = f"{year}-{month:02d}"

        if self.market == "spot":
            path = f"spot/monthly/klines/{symbol}/{interval}/{symbol}-{interval}-{month_str}.zip"
        else:
            path = (
                f"futures/um/monthly/klines/{symbol}/{interval}/{symbol}-{interval}-{month_str}.zip"
            )

        return f"{self.BASE_URL}/{path}"

    def _download_and_parse_zip(self, url: str) -> pl.DataFrame | None:
        """Download ZIP file and parse CSV contents.

        Args:
            url: URL to the ZIP file

        Returns:
            DataFrame with parsed data, or None if file not found
        """
        try:
            response = self.session.get(url)

            if response.status_code == 404:
                # File doesn't exist (date not available)
                return None

            response.raise_for_status()

            # Extract CSV from ZIP
            with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
                # ZIP contains single CSV file
                csv_name = zf.namelist()[0]
                with zf.open(csv_name) as csv_file:
                    csv_content = csv_file.read().decode("utf-8")

            # Check if first line looks like a header (contains text like "open", "high")
            first_line = csv_content.split("\n")[0].lower()
            has_header = "open" in first_line or "close" in first_line

            # Column names for Binance kline data
            # Columns: Open time, Open, High, Low, Close, Volume, Close time,
            #          Quote asset volume, Number of trades, Taker buy base,
            #          Taker buy quote, Ignore
            column_names = [
                "open_time",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "close_time",
                "quote_volume",
                "num_trades",
                "taker_buy_base",
                "taker_buy_quote",
                "ignore",
            ]

            if has_header:
                # Parse with header, then rename columns
                df = pl.read_csv(
                    io.StringIO(csv_content),
                    has_header=True,
                    infer_schema_length=1000,
                )
                # Map common header names to our standard names
                rename_map = {
                    "open_time": "open_time",
                    "Open time": "open_time",
                    "open": "open",
                    "Open": "open",
                    "high": "high",
                    "High": "high",
                    "low": "low",
                    "Low": "low",
                    "close": "close",
                    "Close": "close",
                    "volume": "volume",
                    "Volume": "volume",
                }
                # Apply renaming for columns that exist
                for old_name, new_name in rename_map.items():
                    if old_name in df.columns and old_name != new_name:
                        df = df.rename({old_name: new_name})
            else:
                # Parse without header
                df = pl.read_csv(
                    io.StringIO(csv_content),
                    has_header=False,
                    new_columns=column_names,
                )

            # Select and transform columns
            df = df.select(
                [
                    pl.col("open_time").alias("timestamp"),
                    pl.col("open").cast(pl.Float64),
                    pl.col("high").cast(pl.Float64),
                    pl.col("low").cast(pl.Float64),
                    pl.col("close").cast(pl.Float64),
                    pl.col("volume").cast(pl.Float64),
                ]
            )

            # Convert timestamp from milliseconds
            df = df.with_columns(pl.col("timestamp").cast(pl.Datetime("ms", "UTC")))

            return df

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                return None
            raise

    def _fetch_and_transform_data(
        self, symbol: str, start: str, end: str, frequency: str
    ) -> pl.DataFrame:
        """Fetch and transform OHLCV data from Binance Public Data.

        Downloads daily ZIP files for the date range and concatenates them.

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

        # For intraday frequencies, extend end_dt to end of day
        freq_lower = frequency.lower()
        if freq_lower not in ["daily", "1day", "weekly", "1week", "monthly", "1month"]:
            end_dt = end_dt.replace(hour=23, minute=59, second=59)

        # Normalize symbol
        symbol = self._normalize_symbol(symbol)

        # Get interval
        freq_lower = frequency.lower()
        if freq_lower not in self.INTERVAL_MAP:
            raise ValueError(f"Unsupported frequency: {frequency}")
        interval = self.INTERVAL_MAP[freq_lower]

        logger.info(
            f"Fetching {symbol} data from Binance Public Data ({self.market})",
            frequency=frequency,
            interval=interval,
            start=start,
            end=end,
        )

        # Determine download strategy
        days_requested = (end_dt - start_dt).days + 1

        all_data: list[pl.DataFrame] = []

        if days_requested > 60:
            # Use monthly files for efficiency
            all_data = self._fetch_monthly_data(symbol, interval, start_dt, end_dt)
        else:
            # Use daily files for shorter ranges
            all_data = self._fetch_daily_data(symbol, interval, start_dt, end_dt)

        if not all_data:
            logger.info(f"No data found for {symbol} in requested range")
            return self._create_empty_dataframe()

        # Concatenate all data
        df = pl.concat(all_data)

        # Filter to exact date range
        df = df.filter((pl.col("timestamp") >= start_dt) & (pl.col("timestamp") <= end_dt))

        # Sort and deduplicate
        df = df.sort("timestamp").unique(subset=["timestamp"], maintain_order=True)

        logger.info(
            f"Fetched {len(df)} rows for {symbol}",
            start_date=df["timestamp"].min() if not df.is_empty() else None,
            end_date=df["timestamp"].max() if not df.is_empty() else None,
        )

        return df

    def _fetch_daily_data(
        self, symbol: str, interval: str, start_dt: datetime, end_dt: datetime
    ) -> list[pl.DataFrame]:
        """Fetch data using daily files.

        Args:
            symbol: Normalized symbol
            interval: Binance interval
            start_dt: Start datetime
            end_dt: End datetime

        Returns:
            List of DataFrames
        """
        all_data: list[pl.DataFrame] = []
        current_date = start_dt
        not_found_count = 0

        while current_date <= end_dt:
            url = self._build_url(symbol, interval, current_date)

            try:
                df = self._download_and_parse_zip(url)
                if df is not None and not df.is_empty():
                    all_data.append(df)
                    not_found_count = 0
                else:
                    not_found_count += 1
                    # If too many consecutive not found, symbol may not exist
                    if not_found_count > 7:
                        if not all_data:
                            raise SymbolNotFoundError(self.name, symbol)
                        break

            except Exception as e:
                logger.warning(f"Failed to download {current_date}: {e}")

            current_date += timedelta(days=1)

            # Rate limit between requests
            if current_date <= end_dt:
                self._acquire_rate_limit()

        return all_data

    def _fetch_monthly_data(
        self, symbol: str, interval: str, start_dt: datetime, end_dt: datetime
    ) -> list[pl.DataFrame]:
        """Fetch data using monthly files for efficiency.

        Args:
            symbol: Normalized symbol
            interval: Binance interval
            start_dt: Start datetime
            end_dt: End datetime

        Returns:
            List of DataFrames
        """
        all_data: list[pl.DataFrame] = []

        # Generate list of months to download
        current = start_dt.replace(day=1)
        months: list[tuple[int, int]] = []

        while current <= end_dt:
            months.append((current.year, current.month))
            # Move to next month
            if current.month == 12:
                current = current.replace(year=current.year + 1, month=1)
            else:
                current = current.replace(month=current.month + 1)

        not_found_count = 0

        for year, month in months:
            url = self._build_monthly_url(symbol, interval, year, month)

            try:
                df = self._download_and_parse_zip(url)
                if df is not None and not df.is_empty():
                    all_data.append(df)
                    not_found_count = 0
                else:
                    not_found_count += 1
                    # Fall back to daily files for this month
                    month_start = datetime(year, month, 1, tzinfo=UTC)
                    if month == 12:
                        month_end = datetime(year + 1, 1, 1, tzinfo=UTC) - timedelta(days=1)
                    else:
                        month_end = datetime(year, month + 1, 1, tzinfo=UTC) - timedelta(days=1)

                    # Clip to requested range
                    fetch_start = max(month_start, start_dt)
                    fetch_end = min(month_end, end_dt)

                    daily_data = self._fetch_daily_data(symbol, interval, fetch_start, fetch_end)
                    all_data.extend(daily_data)

            except Exception as e:
                logger.warning(f"Failed to download {year}-{month:02d}: {e}")

            # Rate limit between requests
            self._acquire_rate_limit()

        return all_data

    def fetch_metrics(
        self,
        symbol: str,
        start: str,
        end: str,
    ) -> pl.DataFrame:
        """Fetch futures metrics data (Open Interest, Long/Short Ratios).

        Downloads metrics data from Binance Public Data. This data is ONLY
        available for USD-M futures markets, at 5-minute intervals.

        Available since: 2021-12-01

        Columns returned:
        - timestamp: 5-minute intervals
        - symbol: Trading pair
        - open_interest: Sum of open interest (contracts)
        - open_interest_value: Sum of open interest (USD)
        - toptrader_long_short_ratio: Top trader long/short ratio
        - account_long_short_ratio: Account long/short ratio
        - taker_volume_ratio: Taker buy/sell volume ratio

        Args:
            symbol: Futures symbol (e.g., "BTCUSDT")
            start: Start date (YYYY-MM-DD)
            end: End date (YYYY-MM-DD)

        Returns:
            Polars DataFrame with metrics data

        Example:
            >>> provider = BinancePublicProvider(market="futures")
            >>> df = provider.fetch_metrics("BTCUSDT", "2024-01-01", "2024-01-31")
            >>> df.columns
            ['timestamp', 'symbol', 'open_interest', 'open_interest_value', ...]
        """
        # Parse dates
        start_dt = datetime.strptime(start, "%Y-%m-%d").replace(tzinfo=UTC)
        end_dt = datetime.strptime(end, "%Y-%m-%d").replace(
            hour=23, minute=59, second=59, tzinfo=UTC
        )

        # Normalize symbol
        symbol = self._normalize_symbol(symbol)

        logger.info(
            f"Fetching metrics for {symbol} from Binance Public Data",
            start=start,
            end=end,
        )

        all_data: list[pl.DataFrame] = []
        current_date = start_dt
        not_found_count = 0

        while current_date <= end_dt:
            url = self._build_metrics_url(symbol, current_date)

            try:
                df = self._download_and_parse_metrics_zip(url)
                if df is not None and not df.is_empty():
                    all_data.append(df)
                    not_found_count = 0
                else:
                    not_found_count += 1
                    if not_found_count > 7 and not all_data:
                        raise SymbolNotFoundError(self.name, symbol)

            except Exception as e:
                logger.warning(f"Failed to download metrics for {current_date}: {e}")

            current_date += timedelta(days=1)

            if current_date <= end_dt:
                self._acquire_rate_limit()

        if not all_data:
            logger.info(f"No metrics data found for {symbol}")
            return self._create_empty_metrics_dataframe()

        # Concatenate all data
        df = pl.concat(all_data)

        # Filter to exact date range and deduplicate
        df = df.filter((pl.col("timestamp") >= start_dt) & (pl.col("timestamp") <= end_dt))
        df = df.sort("timestamp").unique(subset=["timestamp"], maintain_order=True)

        logger.info(
            f"Fetched {len(df)} metrics rows for {symbol}",
            start_date=df["timestamp"].min() if not df.is_empty() else None,
            end_date=df["timestamp"].max() if not df.is_empty() else None,
        )

        return df

    def _build_metrics_url(self, symbol: str, date: datetime) -> str:
        """Build URL for metrics data.

        Args:
            symbol: Normalized symbol
            date: Date to download

        Returns:
            Full URL to the metrics ZIP file
        """
        date_str = date.strftime("%Y-%m-%d")
        path = f"futures/um/daily/metrics/{symbol}/{symbol}-metrics-{date_str}.zip"
        return f"{self.BASE_URL}/{path}"

    def _download_and_parse_metrics_zip(self, url: str) -> pl.DataFrame | None:
        """Download and parse metrics ZIP file.

        Args:
            url: URL to the ZIP file

        Returns:
            DataFrame with parsed metrics data, or None if not found
        """
        try:
            response = self.session.get(url)

            if response.status_code == 404:
                return None

            response.raise_for_status()

            # Extract CSV from ZIP
            with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
                csv_name = zf.namelist()[0]
                with zf.open(csv_name) as csv_file:
                    csv_content = csv_file.read().decode("utf-8")

            # Parse CSV - metrics files have headers
            df = pl.read_csv(
                io.StringIO(csv_content),
                has_header=True,
                infer_schema_length=1000,
            )

            # Rename columns to standardized names
            rename_map = {
                "create_time": "timestamp",
                "sum_open_interest": "open_interest",
                "sum_open_interest_value": "open_interest_value",
                "count_toptrader_long_short_ratio": "toptrader_long_short_ratio",
                "sum_toptrader_long_short_ratio": "toptrader_long_short_ratio_sum",
                "count_long_short_ratio": "account_long_short_ratio",
                "sum_taker_long_short_vol_ratio": "taker_volume_ratio",
            }

            for old_name, new_name in rename_map.items():
                if old_name in df.columns:
                    df = df.rename({old_name: new_name})

            # Convert timestamp to datetime
            if "timestamp" in df.columns:
                df = df.with_columns(
                    pl.col("timestamp")
                    .str.to_datetime("%Y-%m-%d %H:%M:%S")
                    .dt.replace_time_zone("UTC")
                )

            # Cast numeric columns
            numeric_cols = [
                "open_interest",
                "open_interest_value",
                "toptrader_long_short_ratio",
                "toptrader_long_short_ratio_sum",
                "account_long_short_ratio",
                "taker_volume_ratio",
            ]
            for col in numeric_cols:
                if col in df.columns:
                    df = df.with_columns(pl.col(col).cast(pl.Float64))

            return df

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                return None
            raise

    def _create_empty_metrics_dataframe(self) -> pl.DataFrame:
        """Create an empty DataFrame with metrics schema."""
        return pl.DataFrame(
            {
                "timestamp": [],
                "symbol": [],
                "open_interest": [],
                "open_interest_value": [],
                "toptrader_long_short_ratio": [],
                "account_long_short_ratio": [],
                "taker_volume_ratio": [],
            },
            schema={
                "timestamp": pl.Datetime("us", "UTC"),
                "symbol": pl.Utf8,
                "open_interest": pl.Float64,
                "open_interest_value": pl.Float64,
                "toptrader_long_short_ratio": pl.Float64,
                "account_long_short_ratio": pl.Float64,
                "taker_volume_ratio": pl.Float64,
            },
        )

    def fetch_premium_index(
        self,
        symbol: str,
        start: str,
        end: str,
        interval: str = "8h",
    ) -> pl.DataFrame:
        """Fetch premium index data for perpetual futures.

        The premium index measures the basis between perpetual and spot prices,
        and is the primary driver of funding rates. Use this for funding rate
        arbitrage and mean-reversion strategies.

        Premium Index = (Perpetual Price - Spot Price) / Spot Price
        - High premium → Crowded longs → Expected underperformance
        - Low/negative premium → Crowded shorts → Expected outperformance

        Args:
            symbol: Futures symbol (e.g., "BTCUSDT", "ETHUSDT")
            start: Start date (YYYY-MM-DD)
            end: End date (YYYY-MM-DD)
            interval: Data interval (default "8h" for funding rate alignment)
                      Options: 1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d

        Returns:
            Polars DataFrame with columns:
            - timestamp: Datetime (UTC)
            - symbol: Trading pair
            - premium_index_open: Opening premium index
            - premium_index_high: High premium index
            - premium_index_low: Low premium index
            - premium_index_close: Closing premium index

        Example:
            >>> provider = BinancePublicProvider(market="futures")
            >>> df = provider.fetch_premium_index("BTCUSDT", "2024-01-01", "2024-01-31")
            >>> df.head()
            shape: (5, 6)
            ┌─────────────────────┬──────────┬────────────────────┬───────────────────┬...
            │ timestamp           ┆ symbol   ┆ premium_index_open ┆ premium_index_high┆...
        """
        # Parse dates
        start_dt = datetime.strptime(start, "%Y-%m-%d").replace(tzinfo=UTC)
        end_dt = datetime.strptime(end, "%Y-%m-%d").replace(
            hour=23, minute=59, second=59, tzinfo=UTC
        )

        # Normalize symbol
        symbol = self._normalize_symbol(symbol)

        # Validate interval - premium index supports various intervals
        valid_intervals = [
            "1m",
            "3m",
            "5m",
            "15m",
            "30m",
            "1h",
            "2h",
            "4h",
            "6h",
            "8h",
            "12h",
            "1d",
            "3d",
            "1w",
            "1mo",
        ]
        if interval not in valid_intervals:
            raise DataValidationError(
                provider=self.name,
                message=f"Invalid interval '{interval}'. Valid options: {valid_intervals}",
                field="interval",
                value=interval,
            )

        logger.info(
            f"Fetching premium index for {symbol} from Binance Public Data",
            interval=interval,
            start=start,
            end=end,
        )

        all_data: list[pl.DataFrame] = []

        # Use monthly files for efficiency (~30x fewer downloads than daily)
        current = start_dt.replace(day=1)
        months: list[tuple[int, int]] = []

        while current <= end_dt:
            months.append((current.year, current.month))
            if current.month == 12:
                current = current.replace(year=current.year + 1, month=1)
            else:
                current = current.replace(month=current.month + 1)

        not_found_count = 0

        for year, month in months:
            url = self._build_premium_index_monthly_url(symbol, interval, year, month)

            try:
                df = self._download_and_parse_premium_index_zip(url, symbol)
                if df is not None and not df.is_empty():
                    all_data.append(df)
                    not_found_count = 0
                else:
                    not_found_count += 1
                    # Fall back to daily files for this month (e.g., current incomplete month)
                    logger.info(
                        f"Monthly file not found for {year}-{month:02d}, trying daily files"
                    )
                    month_start = datetime(year, month, 1, tzinfo=UTC)
                    if month == 12:
                        month_end = datetime(year + 1, 1, 1, tzinfo=UTC) - timedelta(days=1)
                    else:
                        month_end = datetime(year, month + 1, 1, tzinfo=UTC) - timedelta(days=1)

                    # Clip to requested range
                    fetch_start = max(month_start, start_dt)
                    fetch_end = min(month_end, end_dt)

                    daily_data = self._fetch_premium_index_daily(
                        symbol, interval, fetch_start, fetch_end
                    )
                    if daily_data:
                        all_data.extend(daily_data)
                        not_found_count = 0

                    if not_found_count > 3 and not all_data:
                        raise SymbolNotFoundError(self.name, symbol)

            except Exception as e:
                logger.warning(f"Failed to download premium index for {year}-{month:02d}: {e}")

            self._acquire_rate_limit()

        if not all_data:
            logger.info(f"No premium index data found for {symbol}")
            return self._create_empty_premium_index_dataframe()

        # Concatenate all data
        df = pl.concat(all_data)

        # Filter to exact date range and deduplicate
        df = df.filter((pl.col("timestamp") >= start_dt) & (pl.col("timestamp") <= end_dt))
        df = df.sort("timestamp").unique(subset=["timestamp"], maintain_order=True)

        logger.info(
            f"Fetched {len(df)} premium index rows for {symbol}",
            start_date=df["timestamp"].min() if not df.is_empty() else None,
            end_date=df["timestamp"].max() if not df.is_empty() else None,
        )

        return df

    def fetch_premium_index_multi(
        self,
        symbols: list[str],
        start: str,
        end: str,
        interval: str = "8h",
    ) -> pl.DataFrame:
        """Fetch premium index for multiple symbols.

        Returns a single DataFrame with all symbols, suitable for
        cross-sectional analysis and ranking.

        Args:
            symbols: List of futures symbols (e.g., ["BTCUSDT", "ETHUSDT", "SOLUSDT"])
            start: Start date (YYYY-MM-DD)
            end: End date (YYYY-MM-DD)
            interval: Data interval (default "8h" for funding rate alignment)

        Returns:
            Combined DataFrame with symbol column for grouping

        Example:
            >>> provider = BinancePublicProvider(market="futures")
            >>> df = provider.fetch_premium_index_multi(
            ...     ["BTCUSDT", "ETHUSDT", "SOLUSDT"],
            ...     "2024-01-01", "2024-01-31"
            ... )
            >>> df["symbol"].n_unique()
            3
        """
        if not symbols:
            raise DataValidationError(
                provider=self.name,
                message="symbols list cannot be empty",
                field="symbols",
            )

        logger.info(
            f"Fetching premium index for {len(symbols)} symbols",
            symbols=symbols,
            start=start,
            end=end,
            interval=interval,
        )

        all_data: list[pl.DataFrame] = []

        for symbol in symbols:
            try:
                df = self.fetch_premium_index(symbol, start, end, interval)
                if not df.is_empty():
                    all_data.append(df)
            except SymbolNotFoundError:
                logger.warning(f"No premium index data for {symbol}")
            except Exception as e:
                logger.warning(f"Failed to fetch premium index for {symbol}: {e}")

        if not all_data:
            logger.info("No premium index data found for any symbol")
            return self._create_empty_premium_index_dataframe()

        # Concatenate all data
        df = pl.concat(all_data)

        # Sort by timestamp and symbol
        df = df.sort(["timestamp", "symbol"])

        logger.info(
            f"Fetched premium index for {df['symbol'].n_unique()} symbols, {len(df)} total rows"
        )

        return df

    def _build_premium_index_url(self, symbol: str, interval: str, date: datetime) -> str:
        """Build URL for daily premium index data.

        Args:
            symbol: Normalized symbol (e.g., BTCUSDT)
            interval: Data interval (e.g., 8h)
            date: Date to download

        Returns:
            Full URL to the premium index ZIP file
        """
        date_str = date.strftime("%Y-%m-%d")
        path = f"futures/um/daily/premiumIndexKlines/{symbol}/{interval}/{symbol}-{interval}-{date_str}.zip"
        return f"{self.BASE_URL}/{path}"

    def _fetch_premium_index_daily(
        self, symbol: str, interval: str, start_dt: datetime, end_dt: datetime
    ) -> list[pl.DataFrame]:
        """Fetch premium index using daily files.

        Used as fallback when monthly files aren't available (e.g., current month).

        Args:
            symbol: Normalized symbol
            interval: Data interval
            start_dt: Start datetime
            end_dt: End datetime

        Returns:
            List of DataFrames
        """
        all_data: list[pl.DataFrame] = []
        current_date = start_dt
        not_found_count = 0

        while current_date <= end_dt:
            url = self._build_premium_index_url(symbol, interval, current_date)

            try:
                df = self._download_and_parse_premium_index_zip(url, symbol)
                if df is not None and not df.is_empty():
                    all_data.append(df)
                    not_found_count = 0
                else:
                    not_found_count += 1
                    # If too many consecutive not found, stop
                    if not_found_count > 7:
                        break

            except Exception as e:
                logger.warning(f"Failed to download premium index for {current_date}: {e}")

            current_date += timedelta(days=1)

            if current_date <= end_dt:
                self._acquire_rate_limit()

        return all_data

    def _build_premium_index_monthly_url(
        self, symbol: str, interval: str, year: int, month: int
    ) -> str:
        """Build URL for monthly premium index data.

        Args:
            symbol: Normalized symbol (e.g., BTCUSDT)
            interval: Data interval (e.g., 8h)
            year: Year (e.g., 2024)
            month: Month (1-12)

        Returns:
            Full URL to the monthly premium index ZIP file
        """
        month_str = f"{year}-{month:02d}"
        path = f"futures/um/monthly/premiumIndexKlines/{symbol}/{interval}/{symbol}-{interval}-{month_str}.zip"
        return f"{self.BASE_URL}/{path}"

    def _download_and_parse_premium_index_zip(self, url: str, symbol: str) -> pl.DataFrame | None:
        """Download and parse premium index ZIP file.

        Args:
            url: URL to the ZIP file
            symbol: Symbol for labeling

        Returns:
            DataFrame with parsed premium index data, or None if not found
        """
        try:
            response = self.session.get(url)

            if response.status_code == 404:
                return None

            response.raise_for_status()

            # Extract CSV from ZIP
            with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
                csv_name = zf.namelist()[0]
                with zf.open(csv_name) as csv_file:
                    csv_content = csv_file.read().decode("utf-8")

            # Premium index klines have same format as regular klines
            # Columns: open_time, open, high, low, close, volume, close_time,
            #          quote_volume, count, taker_buy_base, taker_buy_quote, ignore
            column_names = [
                "open_time",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "close_time",
                "quote_volume",
                "count",
                "taker_buy_base",
                "taker_buy_quote",
                "ignore",
            ]

            # Check if file has header
            first_line = csv_content.split("\n")[0].lower()
            has_header = "open" in first_line or "close" in first_line

            if has_header:
                df = pl.read_csv(
                    io.StringIO(csv_content),
                    has_header=True,
                    infer_schema_length=1000,
                )
                # Standardize column names
                rename_map = {
                    "open_time": "open_time",
                    "Open time": "open_time",
                    "open": "open",
                    "Open": "open",
                    "high": "high",
                    "High": "high",
                    "low": "low",
                    "Low": "low",
                    "close": "close",
                    "Close": "close",
                }
                for old_name, new_name in rename_map.items():
                    if old_name in df.columns and old_name != new_name:
                        df = df.rename({old_name: new_name})
            else:
                df = pl.read_csv(
                    io.StringIO(csv_content),
                    has_header=False,
                    new_columns=column_names,
                )

            # Select and transform to premium index schema
            df = df.select(
                [
                    pl.from_epoch(pl.col("open_time"), time_unit="ms")
                    .dt.replace_time_zone("UTC")
                    .alias("timestamp"),
                    pl.lit(symbol).alias("symbol"),
                    pl.col("open").cast(pl.Float64).alias("premium_index_open"),
                    pl.col("high").cast(pl.Float64).alias("premium_index_high"),
                    pl.col("low").cast(pl.Float64).alias("premium_index_low"),
                    pl.col("close").cast(pl.Float64).alias("premium_index_close"),
                ]
            )

            return df

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                return None
            raise

    def _create_empty_premium_index_dataframe(self) -> pl.DataFrame:
        """Create an empty DataFrame with premium index schema."""
        return pl.DataFrame(
            {
                "timestamp": [],
                "symbol": [],
                "premium_index_open": [],
                "premium_index_high": [],
                "premium_index_low": [],
                "premium_index_close": [],
            },
            schema={
                "timestamp": pl.Datetime("ms", "UTC"),
                "symbol": pl.Utf8,
                "premium_index_open": pl.Float64,
                "premium_index_high": pl.Float64,
                "premium_index_low": pl.Float64,
                "premium_index_close": pl.Float64,
            },
        )

    def get_available_symbols(self, search: str | None = None) -> list[str]:
        """Get list of available symbols.

        Note: This is a convenience method. For the full list, see:
        https://data.binance.vision/data/spot/daily/klines/

        Args:
            search: Optional search term to filter symbols

        Returns:
            List of common trading pairs
        """
        # Common trading pairs - not exhaustive
        common_pairs = [
            "BTCUSDT",
            "ETHUSDT",
            "BNBUSDT",
            "SOLUSDT",
            "XRPUSDT",
            "DOGEUSDT",
            "ADAUSDT",
            "AVAXUSDT",
            "DOTUSDT",
            "MATICUSDT",
            "LINKUSDT",
            "LTCUSDT",
            "ATOMUSDT",
            "UNIUSDT",
            "ETCUSDT",
            "XLMUSDT",
            "FILUSDT",
            "AAVEUSDT",
            "SANDUSDT",
            "MANAUSDT",
        ]

        if search:
            search = search.upper()
            return [s for s in common_pairs if search in s]

        return common_pairs

    # ===== Async Support =====

    @property
    def _async_session(self) -> httpx.AsyncClient:
        """Lazily create async HTTP client."""
        if not hasattr(self, "_async_client") or self._async_client is None:
            self._async_client = httpx.AsyncClient(
                timeout=httpx.Timeout(self.timeout),
                limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
                follow_redirects=True,
            )
        return self._async_client

    async def _download_and_parse_zip_async(self, url: str) -> pl.DataFrame | None:
        """Async version: Download ZIP file and parse CSV contents.

        Args:
            url: URL to the ZIP file

        Returns:
            DataFrame with parsed data, or None if file not found
        """
        try:
            response = await self._async_session.get(url)

            if response.status_code == 404:
                return None

            response.raise_for_status()

            # Extract CSV from ZIP (CPU-bound, run in thread)
            def parse_zip_content(content: bytes) -> pl.DataFrame | None:
                with zipfile.ZipFile(io.BytesIO(content)) as zf:
                    csv_name = zf.namelist()[0]
                    with zf.open(csv_name) as csv_file:
                        csv_content = csv_file.read().decode("utf-8")

                first_line = csv_content.split("\n")[0].lower()
                has_header = "open" in first_line or "close" in first_line

                column_names = [
                    "open_time",
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                    "close_time",
                    "quote_volume",
                    "num_trades",
                    "taker_buy_base",
                    "taker_buy_quote",
                    "ignore",
                ]

                if has_header:
                    df = pl.read_csv(
                        io.StringIO(csv_content),
                        has_header=True,
                        infer_schema_length=1000,
                    )
                    rename_map = {
                        "open_time": "open_time",
                        "Open time": "open_time",
                        "open": "open",
                        "Open": "open",
                        "high": "high",
                        "High": "high",
                        "low": "low",
                        "Low": "low",
                        "close": "close",
                        "Close": "close",
                        "volume": "volume",
                        "Volume": "volume",
                    }
                    for old_name, new_name in rename_map.items():
                        if old_name in df.columns and old_name != new_name:
                            df = df.rename({old_name: new_name})
                else:
                    df = pl.read_csv(
                        io.StringIO(csv_content),
                        has_header=False,
                        new_columns=column_names,
                    )

                df = df.select(
                    [
                        pl.col("open_time").alias("timestamp"),
                        pl.col("open").cast(pl.Float64),
                        pl.col("high").cast(pl.Float64),
                        pl.col("low").cast(pl.Float64),
                        pl.col("close").cast(pl.Float64),
                        pl.col("volume").cast(pl.Float64),
                    ]
                )

                df = df.with_columns(pl.col("timestamp").cast(pl.Datetime("ms", "UTC")))
                return df

            return await asyncio.to_thread(parse_zip_content, response.content)

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                return None
            raise

    async def _fetch_daily_data_async(
        self, symbol: str, interval: str, start_dt: datetime, end_dt: datetime
    ) -> list[pl.DataFrame]:
        """Async version: Fetch data using daily files concurrently.

        Args:
            symbol: Normalized symbol
            interval: Binance interval
            start_dt: Start datetime
            end_dt: End datetime

        Returns:
            List of DataFrames
        """
        # Generate all URLs
        urls: list[tuple[datetime, str]] = []
        current_date = start_dt
        while current_date <= end_dt:
            url = self._build_url(symbol, interval, current_date)
            urls.append((current_date, url))
            current_date += timedelta(days=1)

        # Fetch all concurrently with semaphore for rate limiting
        semaphore = asyncio.Semaphore(20)  # Max concurrent requests

        async def fetch_one(date: datetime, url: str) -> tuple[datetime, pl.DataFrame | None]:
            async with semaphore:
                df = await self._download_and_parse_zip_async(url)
                return (date, df)

        tasks = [fetch_one(date, url) for date, url in urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Collect successful results in order
        all_data: list[pl.DataFrame] = []
        for result in results:
            if isinstance(result, Exception):
                logger.warning(f"Failed to download: {result}")
                continue
            date, df = result
            if df is not None and not df.is_empty():
                all_data.append(df)

        return all_data

    async def _fetch_and_transform_data_async(
        self, symbol: str, start: str, end: str, frequency: str
    ) -> pl.DataFrame:
        """Async version: Fetch and transform OHLCV data.

        Args:
            symbol: Cryptocurrency symbol
            start: Start date (YYYY-MM-DD)
            end: End date (YYYY-MM-DD)
            frequency: Data frequency

        Returns:
            Polars DataFrame with OHLCV data
        """
        start_dt = datetime.strptime(start, "%Y-%m-%d").replace(tzinfo=UTC)
        end_dt = datetime.strptime(end, "%Y-%m-%d").replace(tzinfo=UTC)

        freq_lower = frequency.lower()
        if freq_lower not in ["daily", "1day", "weekly", "1week", "monthly", "1month"]:
            end_dt = end_dt.replace(hour=23, minute=59, second=59)

        symbol = self._normalize_symbol(symbol)

        if freq_lower not in self.INTERVAL_MAP:
            raise ValueError(f"Unsupported frequency: {frequency}")
        interval = self.INTERVAL_MAP[freq_lower]

        logger.info(
            f"Fetching {symbol} data async from Binance Public Data ({self.market})",
            frequency=frequency,
            interval=interval,
            start=start,
            end=end,
        )

        # Use daily async fetching (more efficient for async)
        all_data = await self._fetch_daily_data_async(symbol, interval, start_dt, end_dt)

        if not all_data:
            logger.info(f"No data found for {symbol} in requested range")
            return self._create_empty_dataframe()

        df = pl.concat(all_data)
        df = df.filter((pl.col("timestamp") >= start_dt) & (pl.col("timestamp") <= end_dt))
        df = df.sort("timestamp").unique(subset=["timestamp"], maintain_order=True)

        logger.info(
            f"Fetched {len(df)} rows for {symbol} (async)",
            start_date=df["timestamp"].min() if not df.is_empty() else None,
            end_date=df["timestamp"].max() if not df.is_empty() else None,
        )

        return df

    async def fetch_ohlcv_async(
        self,
        symbol: str,
        start: str,
        end: str,
        frequency: str = "daily",
    ) -> pl.DataFrame:
        """Async fetch OHLCV data for a symbol.

        This is 3-5x faster than sync for multi-day fetches due to
        concurrent HTTP requests.

        Args:
            symbol: Symbol to fetch (e.g., "BTCUSDT")
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

        # Validate inputs
        self._validate_inputs(symbol, start, end, frequency)

        # Fetch and transform
        data = await self._fetch_and_transform_data_async(symbol, start, end, frequency)

        # Validate OHLCV data
        validated = self._validate_ohlcv(data, self.name)

        self.logger.info(
            "Successfully fetched OHLCV data (async)",
            symbol=symbol,
            rows=len(validated),
            provider=self.name,
        )

        return validated

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
