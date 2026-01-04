"""Databento provider implementation for futures and equities market data.

This provider supports:
- Multiple schemas (ohlcv-1m, ohlcv-1h, ohlcv-1d)
- Continuous futures contracts (symbol.v.0)
- CME session date logic for futures
- Native Polars output
"""

from __future__ import annotations

import os
from datetime import UTC, datetime
from typing import Any, ClassVar

import polars as pl
import structlog
from databento import Historical
from databento.common.error import BentoClientError, BentoServerError

from ml4t.data.core.exceptions import (
    AuthenticationError,
    DataNotAvailableError,
    NetworkError,
    RateLimitError,
)
from ml4t.data.providers.base import BaseProvider

logger = structlog.get_logger()


class DataBentoProvider(BaseProvider):
    """Thin wrapper around databento.Historical for API consistency and incremental updates.

    **When to use this wrapper:**
    - Automated data pipelines with incremental updates
    - Cross-provider comparisons (Yahoo vs Databento vs EODHD)
    - OHLCV bars only (daily/hourly/minute)
    - Consistent Polars DataFrame output

    **When to use databento.Historical directly:**
    - Advanced schemas: trades, MBO, MBP-10, quotes, imbalance, statistics
    - Symbology API: symbol resolution, contract specifications
    - Cost estimation: metadata.get_cost() before fetching
    - Live streaming: WebSocket real-time data
    - Batch operations: multi-symbol, multi-schema requests

    **Quick start with native SDK:**
        >>> import databento as db
        >>> client = db.Historical(api_key)
        >>> # Get continuous front month futures
        >>> data = client.timeseries.get_range(
        ...     dataset='GLBX.MDP3',
        ...     symbols='ES.c.0',  # Continuous front month
        ...     schema='ohlcv-1d',
        ...     stype_in='continuous',
        ...     start='2024-01-01',
        ...     end='2024-12-31'
        ... )
        >>> import polars as pl
        >>> df = pl.from_pandas(data.to_df())

    **This wrapper exposes the native client:**
        >>> provider = DataBentoProvider(api_key)
        >>> provider.client  # Access databento.Historical directly
        >>> # Use for advanced features while keeping incremental update infrastructure

    See: https://docs.databento.com/ for full native SDK capabilities.
    """

    # Databento has generous rate limits
    DEFAULT_RATE_LIMIT: ClassVar[tuple[int, float]] = (100, 1.0)

    # Schema mappings
    SCHEMA_MAPPING = {
        "ohlcv-1m": "ohlcv-1m",
        "ohlcv-1h": "ohlcv-1h",
        "ohlcv-1d": "ohlcv-1d",
        "trades": "trades",
        "quotes": "tbbo",
        "mbo": "mbo",
    }

    def __init__(
        self,
        api_key: str | None = None,
        dataset: str = "GLBX.MDP3",
        rate_limit: tuple[int, float] | None = None,
        adjust_session_dates: bool = False,
        session_start_hour_utc: int = 0,
    ):
        """Initialize Databento provider.

        Args:
            api_key: Databento API key (or set DATABENTO_API_KEY env var)
            dataset: Default dataset to use (e.g., GLBX.MDP3, XNAS.ITCH)
            rate_limit: Optional custom rate limit (calls, period_seconds)
            adjust_session_dates: Whether to adjust dates for CME session logic
            session_start_hour_utc: Hour in UTC when trading session starts (for futures)
        """
        super().__init__(rate_limit=rate_limit or self.DEFAULT_RATE_LIMIT)

        self.api_key = api_key or os.getenv("DATABENTO_API_KEY")
        if not self.api_key:
            raise AuthenticationError(
                provider="databento",
                message="Databento API key not provided. "
                "Set DATABENTO_API_KEY environment variable or pass api_key parameter.",
            )

        try:
            self.client = Historical(self.api_key)
        except Exception as e:
            raise AuthenticationError(
                provider="databento",
                message=f"Failed to initialize Databento client: {e}",
            )

        self.dataset = dataset
        self.default_schema = "ohlcv-1m"
        self.adjust_session_dates = adjust_session_dates
        self.session_start_hour_utc = session_start_hour_utc

        self.logger.info(
            "Initialized Databento provider",
            dataset=dataset,
            rate_limit=rate_limit or self.DEFAULT_RATE_LIMIT,
        )

    @property
    def name(self) -> str:
        """Return provider name."""
        return "databento"

    def _create_empty_dataframe(self) -> pl.DataFrame:
        """Return empty DataFrame with correct OHLCV schema."""
        return pl.DataFrame(
            schema={
                "timestamp": pl.Datetime("ns", "UTC"),
                "symbol": pl.String,
                "open": pl.Float64,
                "high": pl.Float64,
                "low": pl.Float64,
                "close": pl.Float64,
                "volume": pl.Float64,
            }
        )

    def _map_frequency_to_schema(self, frequency: str) -> str:
        """Map frequency parameter to Databento schema."""
        freq_lower = frequency.lower()

        if freq_lower in ["daily", "day", "1d", "d"]:
            return "ohlcv-1d"
        if freq_lower in ["hourly", "hour", "1h", "h"]:
            return "ohlcv-1h"
        if freq_lower in ["minute", "min", "1m", "m"]:
            return "ohlcv-1m"
        if freq_lower in ["tick", "trades"]:
            return "trades"
        if freq_lower in ["quote", "quotes", "tbbo"]:
            return "tbbo"
        if freq_lower in ["mbo"]:
            return "mbo"

        # Default to daily
        return "ohlcv-1d"

    def _fetch_raw_data(
        self,
        symbol: str,
        start: str,
        end: str,
        frequency: str = "daily",
    ) -> Any:
        """Fetch raw data from Databento API."""
        schema = self._map_frequency_to_schema(frequency)

        # Parse dates
        start_dt = datetime.strptime(start, "%Y-%m-%d").replace(
            hour=0, minute=0, second=0, tzinfo=UTC
        )
        end_dt = datetime.strptime(end, "%Y-%m-%d").replace(
            hour=23, minute=59, second=59, tzinfo=UTC
        )

        # Adjust for session dates if enabled (for futures with CME session logic)
        if self.adjust_session_dates:
            from datetime import timedelta

            # Move start back by one day and set to session start hour
            start_dt = (start_dt - timedelta(days=1)).replace(hour=self.session_start_hour_utc)
            # End stays at end of requested day
            end_dt = end_dt.replace(hour=23, minute=59, second=59)

        try:
            self.logger.debug(
                "Fetching from Databento",
                symbol=symbol,
                dataset=self.dataset,
                schema=schema,
            )

            response = self.client.timeseries.get_range(
                dataset=self.dataset,
                start=start_dt,
                end=end_dt,
                symbols=[symbol],
                schema=schema,
                stype_in="raw_symbol",
            )

            return response

        except BentoClientError as e:
            if "unauthorized" in str(e).lower():
                raise AuthenticationError(
                    provider=self.name,
                    message=f"Databento authentication failed: {e}",
                )
            if "rate limit" in str(e).lower():
                raise RateLimitError(provider=self.name)
            raise DataNotAvailableError(self.name, f"Client error: {e}")

        except BentoServerError as e:
            raise NetworkError(
                provider=self.name,
                message=f"Databento server error: {e}",
            )

        except Exception as e:
            self.logger.error("Error fetching from Databento", error=str(e), symbol=symbol)
            raise NetworkError(
                provider=self.name,
                message=f"Failed to fetch data from Databento: {e}",
            )

    def _transform_data(self, raw_data: Any, symbol: str) -> pl.DataFrame:
        """Transform Databento data to standard schema."""
        try:
            # Convert Databento response to DataFrame
            if hasattr(raw_data, "to_df"):
                df_pandas = raw_data.to_df()

                # Databento uses timestamp as DataFrame index
                df_pandas = df_pandas.reset_index()

                # Rename index column to timestamp
                if "index" in df_pandas.columns:
                    df_pandas = df_pandas.rename(columns={"index": "timestamp"})
                elif "ts_event" in df_pandas.columns:
                    df_pandas = df_pandas.rename(columns={"ts_event": "timestamp"})

                df = pl.from_pandas(df_pandas)
            else:
                df = pl.DataFrame(raw_data)

            # Ensure timestamp column exists and is datetime
            if "timestamp" in df.columns:
                if df["timestamp"].dtype == pl.Int64:
                    # Convert nanoseconds to datetime
                    df = df.with_columns(pl.col("timestamp").cast(pl.Datetime("ns")))

            # Add symbol column
            if "symbol" not in df.columns:
                df = df.with_columns(pl.lit(symbol).alias("symbol"))

            # For OHLCV data, ensure proper column types
            ohlcv_columns = ["open", "high", "low", "close", "volume"]
            for col in ohlcv_columns:
                if col in df.columns:
                    df = df.with_columns(pl.col(col).cast(pl.Float64))

            # Sort by timestamp
            if "timestamp" in df.columns:
                df = df.sort("timestamp")

            # For OHLCV data, select columns in standard order
            required_ohlcv = ["timestamp", "symbol", "open", "high", "low", "close", "volume"]
            if all(col in df.columns for col in required_ohlcv):
                # Keep any extra columns after the standard ones
                extra_cols = [c for c in df.columns if c not in required_ohlcv]
                df = df.select(required_ohlcv + extra_cols)

            return df

        except Exception as e:
            self.logger.error("Failed to transform Databento data", error=str(e), symbol=symbol)
            raise DataNotAvailableError(self.name, f"Failed to transform data for {symbol}: {e}")

    def fetch_continuous_futures(
        self,
        root_symbol: str,
        start: str,
        end: str,
        frequency: str = "daily",
        version: int = 0,
    ) -> pl.DataFrame:
        """Fetch continuous futures contract data.

        Databento supports continuous futures with the .v.N notation where
        N is the version/roll number (0 = front month).

        Args:
            root_symbol: Root futures symbol (e.g., "ES", "CL")
            start: Start date
            end: End date
            frequency: Data frequency
            version: Contract version (0 = front month, 1 = second month, etc.)

        Returns:
            DataFrame with continuous contract data
        """
        continuous_symbol = f"{root_symbol}.v.{version}"

        self.logger.info(
            "Fetching continuous futures",
            root=root_symbol,
            version=version,
            symbol=continuous_symbol,
        )

        return self.fetch_ohlcv(continuous_symbol, start, end, frequency)

    def fetch_multiple_schemas(
        self,
        symbol: str,
        start: str,
        end: str,
        schemas: list[str],
    ) -> dict[str, pl.DataFrame]:
        """Fetch data for multiple schemas at once.

        Args:
            symbol: Symbol to fetch
            start: Start date (YYYY-MM-DD)
            end: End date (YYYY-MM-DD)
            schemas: List of schemas to fetch (e.g., ["ohlcv-1m", "trades"])

        Returns:
            Dictionary mapping schema names to DataFrames
        """
        results = {}
        for schema in schemas:
            # Map schema back to frequency
            if schema == "ohlcv-1d":
                frequency = "daily"
            elif schema == "ohlcv-1h":
                frequency = "hourly"
            elif schema == "ohlcv-1m":
                frequency = "minute"
            elif schema == "trades":
                frequency = "trades"
            elif schema == "tbbo":
                frequency = "quotes"
            else:
                frequency = schema

            try:
                # Use low-level methods to avoid OHLCV validation for non-OHLCV schemas
                raw_data = self._fetch_raw_data(symbol, start, end, frequency)
                df = self._transform_data(raw_data, symbol)
                results[schema] = df
            except Exception as e:
                self.logger.warning(
                    "Failed to fetch schema",
                    schema=schema,
                    symbol=symbol,
                    error=str(e),
                )
                results[schema] = None

        return results

    def get_available_datasets(self) -> list[str]:
        """Get list of available datasets.

        Returns:
            List of dataset names (e.g., ["GLBX.MDP3", "XNAS.ITCH"])
        """
        try:
            return self.client.metadata.list_datasets()
        except Exception as e:
            self.logger.error("Failed to list datasets", error=str(e))
            return []

    def get_available_schemas(self, dataset: str | None = None) -> list[str]:
        """Get list of available schemas for a dataset.

        Args:
            dataset: Dataset name (uses self.dataset if not provided)

        Returns:
            List of schema names (e.g., ["ohlcv-1m", "trades", "tbbo"])
        """
        dataset = dataset or self.dataset
        try:
            return self.client.metadata.list_schemas(dataset=dataset)
        except Exception as e:
            self.logger.error("Failed to list schemas", dataset=dataset, error=str(e))
            return []

    async def fetch_ohlcv_async(
        self, symbol: str, start: str, end: str, frequency: str = "daily"
    ) -> pl.DataFrame:
        """Async fetch OHLCV data using thread pool.

        Since the Databento SDK is synchronous, this wraps the sync call
        in asyncio.to_thread() to avoid blocking the event loop.

        This is useful when fetching multiple symbols concurrently.

        Args:
            symbol: Symbol to fetch
            start: Start date (YYYY-MM-DD)
            end: End date (YYYY-MM-DD)
            frequency: Data frequency

        Returns:
            Polars DataFrame with OHLCV data

        Example:
            async def fetch_many():
                provider = DataBentoProvider()
                tasks = [provider.fetch_ohlcv_async(s, start, end) for s in symbols]
                return await asyncio.gather(*tasks)
        """
        import asyncio

        return await asyncio.to_thread(self.fetch_ohlcv, symbol, start, end, frequency)
