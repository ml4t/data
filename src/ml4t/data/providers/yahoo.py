"""Yahoo Finance data provider - simplified wrapper around yfinance."""

from __future__ import annotations

import asyncio
import time
from datetime import datetime, timedelta
from typing import ClassVar

import pandas as pd
import polars as pl
import structlog

# yfinance is an optional dependency
try:
    import yfinance as yf

    _YFINANCE_AVAILABLE = True
except ImportError:
    yf = None  # type: ignore[assignment]
    _YFINANCE_AVAILABLE = False

from ml4t.data.core.exceptions import (
    DataNotAvailableError,
    DataValidationError,
    SymbolNotFoundError,
)
from ml4t.data.providers.base import BaseProvider

logger = structlog.get_logger()


def _chunks(lst: list, n: int):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


class YahooFinanceProvider(BaseProvider):
    """
    Thin wrapper around yfinance for API consistency and incremental updates.

    This provider exists primarily to provide:
    - Consistent API across all providers (fetch_ohlcv interface)
    - Polars DataFrame output (instead of pandas)
    - Standardized column schema (with symbol column)
    - OHLC data validation (catches bad data before backtests)
    - Integration with automated update infrastructure

    Performance Note (IMPORTANT):
    - With provider REUSE: overhead is negligible (<10%, often faster due to variance)
    - With NEW instance per call: ~2.5x slower due to httpx.Client initialization (30ms)
    - ALWAYS reuse provider instances in production:

      # Good - reuse provider
      provider = YahooFinanceProvider()
      for symbol in symbols:
          data = provider.fetch_ohlcv(symbol, start, end)

      # Bad - creates overhead
      for symbol in symbols:
          provider = YahooFinanceProvider()  # ← 30ms initialization per call!
          data = provider.fetch_ohlcv(symbol, start, end)

    When to use this wrapper vs native yfinance:
    - Use wrapper: automated incremental updates, cross-provider consistency, data validation
    - Use native: one-off analysis in notebooks, need yfinance-specific features
    """

    # Map our frequency names to yfinance intervals
    FREQUENCY_MAP: ClassVar[dict[str, str]] = {
        "minute": "1m",
        "1minute": "1m",
        "5minute": "5m",
        "15minute": "15m",
        "30minute": "30m",
        "hourly": "1h",
        "1hour": "1h",
        "daily": "1d",
        "1day": "1d",
        "weekly": "1wk",
        "1week": "1wk",
        "monthly": "1mo",
        "1month": "1mo",
    }

    def __init__(self, enable_progress: bool = False) -> None:
        """
        Initialize Yahoo Finance provider.

        Args:
            enable_progress: Show progress bars for downloads (default False)

        Raises:
            ImportError: If yfinance is not installed
        """
        if not _YFINANCE_AVAILABLE:
            raise ImportError(
                "YahooFinanceProvider requires yfinance. "
                "Install with: pip install 'ml4t-data[yahoo]'"
            )
        # Don't use BaseProvider's rate limiting - yfinance handles this
        super().__init__(rate_limit=None)
        self.enable_progress = enable_progress

    @property
    def name(self) -> str:
        """Return the provider name."""
        return "yahoo"

    def _fetch_and_transform_data(
        self, symbol: str, start: str, end: str, frequency: str
    ) -> pl.DataFrame:
        """
        Fetch and transform OHLCV data from Yahoo Finance.

        Args:
            symbol: Stock symbol (e.g., "AAPL")
            start: Start date in YYYY-MM-DD format
            end: End date in YYYY-MM-DD format
            frequency: Data frequency (minute, hourly, daily, etc.)

        Returns:
            Polars DataFrame with standardized schema

        Raises:
            DataNotAvailableError: If no data available for the period
            DataValidationError: If data format is invalid
        """
        interval = self.FREQUENCY_MAP.get(frequency.lower(), "1d")

        # yfinance end date is exclusive, add one day
        end_date = datetime.strptime(end, "%Y-%m-%d") + timedelta(days=1)
        end_str = end_date.strftime("%Y-%m-%d")

        logger.info(
            "Fetching data from Yahoo Finance",
            symbol=symbol,
            start=start,
            end=end_str,
            interval=interval,
        )

        try:
            df_pandas = yf.download(
                symbol,
                start=start,
                end=end_str,
                interval=interval,
                progress=self.enable_progress,
                auto_adjust=True,
                actions=False,
            )

            if df_pandas.empty:
                raise SymbolNotFoundError(
                    "yahoo",
                    symbol,
                    details={"start": start, "end": end_str, "frequency": frequency},
                )

            # Convert to Polars with symbol column
            df = self._convert_to_polars(df_pandas, symbol)

            logger.info("Successfully fetched data", symbol=symbol, rows=len(df))
            return df

        except (DataNotAvailableError, SymbolNotFoundError):
            raise

        except Exception as e:
            logger.error("Error fetching data", symbol=symbol, error=str(e))
            raise DataValidationError(
                "yahoo",
                f"Failed to fetch {symbol}: {e}",
                details={"symbol": symbol, "error": str(e)},
            ) from e

    def _convert_to_polars(self, df_pandas: pd.DataFrame, symbol: str) -> pl.DataFrame:
        """
        Convert pandas DataFrame from yfinance to Polars DataFrame.

        Args:
            df_pandas: Pandas DataFrame from yfinance
            symbol: Stock symbol (added as column)

        Returns:
            Polars DataFrame with standardized schema
        """
        # Handle multi-level columns (yfinance always returns MultiIndex now)
        if df_pandas.columns.nlevels > 1:
            # Extract columns for the specific symbol to avoid duplicates
            # yfinance format: [('Close', 'AAPL'), ('High', 'AAPL'), ...]
            symbol_upper = symbol.upper()

            # Check if we have data for this specific symbol
            if (df_pandas.columns.get_level_values(1) == symbol_upper).any():
                # Select columns for this symbol only
                df_pandas = df_pandas.loc[:, (slice(None), symbol_upper)]
                # Flatten to single level
                df_pandas.columns = df_pandas.columns.get_level_values(0)
            else:
                # Single symbol download - just flatten
                df_pandas.columns = df_pandas.columns.get_level_values(0)

        df_pandas = df_pandas.reset_index()

        # Find timestamp column
        timestamp_col = None
        for col in df_pandas.columns:
            if isinstance(col, str) and col in ["Date", "Datetime", "index"]:
                timestamp_col = col
                break
        if timestamp_col is None:
            timestamp_col = df_pandas.columns[0]

        # Create Polars DataFrame
        df = pl.DataFrame(
            {
                "timestamp": df_pandas[timestamp_col],
                "open": df_pandas.get("Open", []),
                "high": df_pandas.get("High", []),
                "low": df_pandas.get("Low", []),
                "close": df_pandas.get("Close", []),
                "volume": df_pandas.get("Volume", []),
            }
        )

        # Cast to proper types, add symbol, and select in standard order
        return (
            df.with_columns(
                [
                    pl.col("timestamp").cast(pl.Datetime),
                    pl.col("open").cast(pl.Float64),
                    pl.col("high").cast(pl.Float64),
                    pl.col("low").cast(pl.Float64),
                    pl.col("close").cast(pl.Float64),
                    pl.col("volume").cast(pl.Float64),
                ]
            )
            .with_columns(pl.lit(symbol.upper()).alias("symbol"))
            .select(["timestamp", "symbol", "open", "high", "low", "close", "volume"])
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
                pl.col("symbol").cast(pl.Utf8),
                pl.col("open").cast(pl.Float64),
                pl.col("high").cast(pl.Float64),
                pl.col("low").cast(pl.Float64),
                pl.col("close").cast(pl.Float64),
                pl.col("volume").cast(pl.Float64),
            ]
        )

    def fetch_batch_ohlcv(
        self,
        symbols: list[str],
        start: str,
        end: str,
        frequency: str = "daily",
        chunk_size: int = 50,
        delay_seconds: float = 1.0,
    ) -> pl.DataFrame:
        """
        Fetch OHLCV data for multiple symbols efficiently using batch downloads.

        This method is ~20x faster than calling fetch_ohlcv() for each symbol
        individually because it uses yfinance's multi-ticker download capability.

        Use this for:
        - Initial data acquisition (hundreds of symbols)
        - Bulk historical downloads
        - One-time data setup

        Use fetch_ohlcv() instead for:
        - Incremental daily updates
        - Small symbol counts (<10)
        - Integration with UpdateManager

        Args:
            symbols: List of stock symbols (e.g., ["AAPL", "MSFT", "GOOGL"])
            start: Start date in YYYY-MM-DD format
            end: End date in YYYY-MM-DD format
            frequency: Data frequency (daily, weekly, monthly, etc.)
            chunk_size: Number of symbols per batch (default 50, max ~100)
            delay_seconds: Delay between chunks to avoid rate limiting (default 1.0)

        Returns:
            Polars DataFrame with columns: timestamp, open, high, low, close, volume, symbol
            Data is in long format (one row per symbol per timestamp)

        Raises:
            DataValidationError: If batch download fails

        Example:
            >>> provider = YahooFinanceProvider()
            >>> df = provider.fetch_batch_ohlcv(
            ...     symbols=["AAPL", "MSFT", "GOOGL"],
            ...     start="2020-01-01",
            ...     end="2024-01-01",
            ... )
            >>> print(df.shape)  # (3000+, 7)
        """
        interval = self.FREQUENCY_MAP.get(frequency.lower(), "1d")

        # yfinance end date is exclusive, add one day
        end_date = datetime.strptime(end, "%Y-%m-%d") + timedelta(days=1)
        end_str = end_date.strftime("%Y-%m-%d")

        logger.info(
            "Starting batch download",
            total_symbols=len(symbols),
            chunk_size=chunk_size,
            start=start,
            end=end_str,
        )

        all_data: list[pl.DataFrame] = []
        successful_symbols: set[str] = set()
        failed_symbols: list[str] = []

        n_chunks = (len(symbols) + chunk_size - 1) // chunk_size

        for i, chunk in enumerate(_chunks(symbols, chunk_size), 1):
            logger.info(
                "Downloading chunk",
                chunk=i,
                total_chunks=n_chunks,
                symbols_in_chunk=len(chunk),
            )

            try:
                df_pandas = yf.download(
                    chunk,
                    start=start,
                    end=end_str,
                    interval=interval,
                    progress=self.enable_progress,
                    auto_adjust=True,
                    actions=False,
                    threads=True,
                )

                if df_pandas.empty:
                    logger.warning("Empty response for chunk", chunk=i)
                    failed_symbols.extend(chunk)
                    continue

                # Convert batch result to long-format Polars
                df_polars = self._convert_batch_to_polars(df_pandas, chunk)
                all_data.append(df_polars)

                # Track successful symbols
                chunk_symbols = df_polars["symbol"].unique().to_list()
                successful_symbols.update(chunk_symbols)

                # Track failed symbols (requested but not returned)
                for sym in chunk:
                    if sym.upper() not in [s.upper() for s in chunk_symbols]:
                        failed_symbols.append(sym)

            except Exception as e:
                logger.error("Chunk download failed", chunk=i, error=str(e))
                failed_symbols.extend(chunk)

            # Delay between chunks (except after last chunk)
            if i < n_chunks and delay_seconds > 0:
                time.sleep(delay_seconds)

        # Combine all chunks
        if not all_data:
            logger.error("No data downloaded", failed_symbols=failed_symbols)
            return self._create_empty_dataframe()

        result = pl.concat(all_data).sort(["symbol", "timestamp"])

        logger.info(
            "Batch download complete",
            successful_symbols=len(successful_symbols),
            failed_symbols=len(failed_symbols),
            total_rows=len(result),
        )

        if failed_symbols:
            logger.warning(
                "Some symbols failed to download",
                count=len(failed_symbols),
                symbols=failed_symbols[:10],  # Log first 10
            )

        return result

    def _convert_batch_to_polars(
        self, df_pandas: pd.DataFrame, requested_symbols: list[str]
    ) -> pl.DataFrame:
        """
        Convert multi-symbol pandas DataFrame to long-format Polars DataFrame.

        Args:
            df_pandas: Pandas DataFrame from yfinance batch download
                       (MultiIndex columns: [('Close', 'AAPL'), ('Close', 'MSFT'), ...])
            requested_symbols: List of symbols that were requested

        Returns:
            Polars DataFrame in long format with symbol column
        """
        if df_pandas.empty:
            return self._create_empty_dataframe()

        # Reset index to get timestamp as column
        df_pandas = df_pandas.reset_index()

        # Find timestamp column
        timestamp_col = None
        for col in df_pandas.columns:
            col_name = col[0] if isinstance(col, tuple) else col
            if col_name in ["Date", "Datetime", "index"]:
                timestamp_col = col
                break
        if timestamp_col is None:
            timestamp_col = df_pandas.columns[0]

        timestamps = df_pandas[timestamp_col]

        # Handle multi-level columns from batch download
        if df_pandas.columns.nlevels > 1:
            # Get unique symbols from column level 1
            symbols_in_data = [
                s
                for s in df_pandas.columns.get_level_values(1).unique()
                if s and s != ""  # Filter empty strings
            ]
        else:
            # Single symbol case (shouldn't happen in batch, but handle it)
            symbols_in_data = requested_symbols[:1]

        # Build long-format data
        records = []

        for symbol in symbols_in_data:
            try:
                if df_pandas.columns.nlevels > 1:
                    # Extract OHLCV for this symbol
                    symbol_data = {
                        "timestamp": timestamps,
                        "open": df_pandas.get(
                            ("Open", symbol), pd.Series([None] * len(timestamps))
                        ),
                        "high": df_pandas.get(
                            ("High", symbol), pd.Series([None] * len(timestamps))
                        ),
                        "low": df_pandas.get(("Low", symbol), pd.Series([None] * len(timestamps))),
                        "close": df_pandas.get(
                            ("Close", symbol), pd.Series([None] * len(timestamps))
                        ),
                        "volume": df_pandas.get(
                            ("Volume", symbol), pd.Series([None] * len(timestamps))
                        ),
                        "symbol": symbol,
                    }
                else:
                    # Single symbol (flat columns)
                    symbol_data = {
                        "timestamp": timestamps,
                        "open": df_pandas.get("Open", pd.Series([None] * len(timestamps))),
                        "high": df_pandas.get("High", pd.Series([None] * len(timestamps))),
                        "low": df_pandas.get("Low", pd.Series([None] * len(timestamps))),
                        "close": df_pandas.get("Close", pd.Series([None] * len(timestamps))),
                        "volume": df_pandas.get("Volume", pd.Series([None] * len(timestamps))),
                        "symbol": symbol,
                    }

                # Create DataFrame for this symbol
                df_symbol = pl.DataFrame(symbol_data)

                # Cast to proper types BEFORE concat (ensures schema compatibility)
                df_symbol = df_symbol.with_columns(
                    [
                        pl.col("timestamp").cast(pl.Datetime),
                        pl.col("open").cast(pl.Float64),
                        pl.col("high").cast(pl.Float64),
                        pl.col("low").cast(pl.Float64),
                        pl.col("close").cast(pl.Float64),
                        pl.col("volume").cast(pl.Float64),
                        pl.col("symbol").cast(pl.Utf8),
                    ]
                )

                # Drop rows where all OHLCV are null (symbol had no data for that date)
                df_symbol = df_symbol.filter(
                    pl.col("close").is_not_null() | pl.col("open").is_not_null()
                )

                if len(df_symbol) > 0:
                    records.append(df_symbol)

            except Exception as e:
                logger.warning(f"Failed to extract data for {symbol}: {e}")
                continue

        if not records:
            return self._create_empty_dataframe()

        # Combine all symbols (schemas now guaranteed compatible)
        return pl.concat(records)

    # ===== Async Support =====
    # Note: yfinance is sync-only, so we use asyncio.to_thread()
    # for non-blocking async operations.

    async def fetch_ohlcv_async(
        self,
        symbol: str,
        start: str,
        end: str,
        frequency: str = "daily",
    ) -> pl.DataFrame:
        """Async fetch OHLCV data for a symbol.

        Since yfinance is synchronous, this wraps the sync call in
        asyncio.to_thread() to avoid blocking the event loop.

        This is useful when fetching multiple symbols concurrently:

            async def fetch_many(symbols):
                provider = YahooFinanceProvider()
                tasks = [
                    provider.fetch_ohlcv_async(s, start, end)
                    for s in symbols
                ]
                return await asyncio.gather(*tasks)

        Args:
            symbol: Symbol to fetch (e.g., "AAPL")
            start: Start date (YYYY-MM-DD)
            end: End date (YYYY-MM-DD)
            frequency: Data frequency

        Returns:
            DataFrame with OHLCV data
        """
        return await asyncio.to_thread(self.fetch_ohlcv, symbol, start, end, frequency)

    async def fetch_batch_ohlcv_async(
        self,
        symbols: list[str],
        start: str,
        end: str,
        frequency: str = "daily",
        chunk_size: int = 50,
        delay_seconds: float = 1.0,
    ) -> pl.DataFrame:
        """Async batch fetch OHLCV data for multiple symbols.

        Wraps the sync batch download in asyncio.to_thread() for
        non-blocking operation.

        Args:
            symbols: List of stock symbols
            start: Start date (YYYY-MM-DD)
            end: End date (YYYY-MM-DD)
            frequency: Data frequency
            chunk_size: Symbols per batch
            delay_seconds: Delay between chunks

        Returns:
            Multi-asset DataFrame in long format
        """
        return await asyncio.to_thread(
            self.fetch_batch_ohlcv,
            symbols,
            start,
            end,
            frequency,
            chunk_size,
            delay_seconds,
        )

    async def close_async(self) -> None:
        """Close provider resources (no-op for Yahoo)."""
        pass

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close_async()
