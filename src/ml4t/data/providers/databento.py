"""Databento provider implementation for futures, equities, and OPRA options data.

This provider supports:
- Multiple schemas (ohlcv-1m, ohlcv-1h, ohlcv-1d)
- Continuous futures contracts (symbol.v.0)
- CME session date logic for futures
- OPRA option chains, bars, quotes, and cost estimates
- Native Polars output
"""

from __future__ import annotations

import os
from datetime import UTC, date, datetime, timedelta
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
    OPRA_DATASET: ClassVar[str] = "OPRA.PILLAR"
    OPRA_CONSOLIDATED_PUBLISHER_ID: ClassVar[int] = 30

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

    @staticmethod
    def _date_string(value: str | date | datetime) -> str:
        """Return a Databento-compatible date or timestamp string."""
        if isinstance(value, datetime):
            return value.isoformat()
        if isinstance(value, date):
            return value.isoformat()
        return value

    @classmethod
    def _next_date_string(cls, value: str | date | datetime) -> str:
        """Return the exclusive next-day endpoint for date-scoped requests."""
        if isinstance(value, datetime):
            parsed = value.date()
        elif isinstance(value, date):
            parsed = value
        else:
            parsed = datetime.strptime(value, "%Y-%m-%d").date()
        return (parsed + timedelta(days=1)).isoformat()

    @classmethod
    def _inclusive_end_string(cls, value: str | date | datetime) -> str:
        """Convert an inclusive date endpoint to Databento's exclusive endpoint."""
        if isinstance(value, datetime):
            if value.time() == datetime.min.time():
                return (value.date() + timedelta(days=1)).isoformat()
            return value.isoformat()
        if isinstance(value, date):
            return (value + timedelta(days=1)).isoformat()
        try:
            parsed = datetime.strptime(value, "%Y-%m-%d").date()
        except ValueError:
            return value
        return (parsed + timedelta(days=1)).isoformat()

    def _raise_databento_error(self, error: Exception, context: str) -> None:
        """Map Databento SDK exceptions to ml4t-data provider exceptions."""
        if isinstance(error, BentoClientError):
            message = str(error)
            message_lower = message.lower()
            if "unauthorized" in message_lower:
                raise AuthenticationError(
                    provider=self.name,
                    message=f"Databento authentication failed: {message}",
                )
            if "rate limit" in message_lower:
                raise RateLimitError(provider=self.name)
            raise DataNotAvailableError(self.name, f"{context}: {message}")

        if isinstance(error, BentoServerError):
            raise NetworkError(
                provider=self.name,
                message=f"Databento server error during {context}: {error}",
            )

        raise NetworkError(
            provider=self.name,
            message=f"Databento request failed during {context}: {error}",
        )

    def _fetch_timeseries(
        self,
        *,
        dataset: str,
        symbols: str | list[str],
        schema: str,
        stype_in: str,
        start: Any,
        end: Any,
    ) -> Any:
        """Fetch a Databento timeseries range with shared error handling."""
        symbols_list = [symbols] if isinstance(symbols, str) else symbols
        try:
            self.logger.debug(
                "Fetching from Databento",
                dataset=dataset,
                schema=schema,
                symbols=symbols_list,
                stype_in=stype_in,
            )
            return self.client.timeseries.get_range(
                dataset=dataset,
                start=start,
                end=end,
                symbols=symbols_list,
                schema=schema,
                stype_in=stype_in,
            )
        except Exception as e:
            self._raise_databento_error(e, f"fetching {dataset} {schema}")
            raise

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
            response = self._fetch_timeseries(
                dataset=self.dataset,
                start=start_dt,
                end=end_dt,
                symbols=symbol,
                schema=schema,
                stype_in="raw_symbol",
            )

            return response

        except Exception as e:
            if isinstance(
                e,
                AuthenticationError | DataNotAvailableError | NetworkError | RateLimitError,
            ):
                raise
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

                # Prefer Databento's event timestamp over a default pandas index.
                if "ts_event" in df_pandas.columns:
                    if "index" in df_pandas.columns:
                        df_pandas = df_pandas.drop(columns=["index"])
                    df_pandas = df_pandas.rename(columns={"ts_event": "timestamp"})
                elif "index" in df_pandas.columns:
                    df_pandas = df_pandas.rename(columns={"index": "timestamp"})

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
                if "raw_symbol" in df.columns:
                    df = df.with_columns(pl.col("raw_symbol").cast(pl.String).alias("symbol"))
                else:
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

    def estimate_opra_cost(
        self,
        symbols: str | list[str],
        start: str | date | datetime,
        end: str | date | datetime,
        schema: str = "ohlcv-1d",
        stype_in: str = "raw_symbol",
        include_records: bool = False,
    ) -> dict[str, Any]:
        """Estimate OPRA request size and cost before fetching data.

        Args:
            symbols: OPRA raw symbols or symbols accepted by ``stype_in``
            start: Start date
            end: End date
            schema: Databento schema, such as ``ohlcv-1d`` or ``cbbo-1m``
            stype_in: Databento input symbol type
            include_records: Include record count when supported by the account

        Returns:
            Dictionary with dataset, schema, symbols, billable size, and cost estimate
        """
        symbols_list = [symbols] if isinstance(symbols, str) else list(symbols)
        try:
            billable_size = self.client.metadata.get_billable_size(
                dataset=self.OPRA_DATASET,
                start=self._date_string(start),
                end=self._inclusive_end_string(end),
                symbols=symbols_list,
                schema=schema,
                stype_in=stype_in,
            )
            estimated_cost = self.client.metadata.get_cost(
                dataset=self.OPRA_DATASET,
                start=self._date_string(start),
                end=self._inclusive_end_string(end),
                symbols=symbols_list,
                schema=schema,
                stype_in=stype_in,
            )
            result: dict[str, Any] = {
                "dataset": self.OPRA_DATASET,
                "schema": schema,
                "symbols": symbols_list,
                "stype_in": stype_in,
                "billable_size": billable_size,
                "estimated_cost": estimated_cost,
            }
            if include_records:
                result["record_count"] = self.client.metadata.get_record_count(
                    dataset=self.OPRA_DATASET,
                    start=self._date_string(start),
                    end=self._inclusive_end_string(end),
                    symbols=symbols_list,
                    schema=schema,
                    stype_in=stype_in,
                )
            return result
        except Exception as e:
            self._raise_databento_error(e, f"estimating {self.OPRA_DATASET} {schema} cost")
            raise

    def fetch_option_chain(
        self,
        underlying: str,
        session_date: str | date | datetime,
        *,
        expiry: str | date | datetime | None = None,
        right: str = "both",
        min_strike: float | None = None,
        max_strike: float | None = None,
    ) -> pl.DataFrame:
        """Fetch OPRA option definitions for an underlying and apply common filters.

        Databento's OPRA definitions are requested with parent symbology so callers can
        discover contracts before fetching bars or quotes.
        """
        raw_data = self._fetch_timeseries(
            dataset=self.OPRA_DATASET,
            symbols=underlying,
            schema="definition",
            stype_in="parent",
            start=self._date_string(session_date),
            end=self._next_date_string(session_date),
        )
        df = self._transform_data(raw_data, underlying)

        if df.is_empty():
            return df

        df = df.with_columns(pl.lit(underlying).alias("underlying"))
        expiry_column = self._first_existing_column(
            df, ["expiration", "expiration_date", "expire_date", "maturity_date"]
        )
        if expiry is not None and expiry_column is not None:
            expiry_value = self._date_string(expiry)
            df = df.filter(pl.col(expiry_column).cast(pl.Utf8).str.starts_with(expiry_value))

        right_column = self._first_existing_column(
            df, ["right", "put_call", "option_type", "instrument_class"]
        )
        right_lower = right.lower()
        if right_lower not in {"both", "all"} and right_column is not None:
            if right_lower in {"call", "c"}:
                right_prefix = "C"
            elif right_lower in {"put", "p"}:
                right_prefix = "P"
            else:
                raise ValueError("right must be 'call', 'put', or 'both'")
            df = df.filter(
                pl.col(right_column).cast(pl.Utf8).str.to_uppercase().str.starts_with(right_prefix)
            )

        strike_column = self._first_existing_column(df, ["strike_price", "strike"])
        if strike_column is not None and (min_strike is not None or max_strike is not None):
            df = df.with_columns(pl.col(strike_column).cast(pl.Float64, strict=False))
            if min_strike is not None:
                df = df.filter(pl.col(strike_column) >= min_strike)
            if max_strike is not None:
                df = df.filter(pl.col(strike_column) <= max_strike)

        return df

    def fetch_option_ohlcv(
        self,
        contract: str,
        start: str | date | datetime,
        end: str | date | datetime,
        *,
        frequency: str = "daily",
        stype_in: str = "raw_symbol",
        consolidate_publishers: bool = True,
    ) -> pl.DataFrame:
        """Fetch OPRA option OHLCV bars for a contract."""
        schema = self._map_frequency_to_schema(frequency)
        if not schema.startswith("ohlcv-"):
            raise ValueError("fetch_option_ohlcv requires an OHLCV frequency")

        raw_data = self._fetch_timeseries(
            dataset=self.OPRA_DATASET,
            symbols=contract,
            schema=schema,
            stype_in=stype_in,
            start=self._date_string(start),
            end=self._inclusive_end_string(end),
        )
        df = self._transform_data(raw_data, contract)
        if consolidate_publishers:
            df = self._consolidate_ohlcv_publishers(df)
        return df

    def fetch_option_quotes(
        self,
        contract: str,
        start: str | date | datetime,
        end: str | date | datetime,
        *,
        schema: str = "cbbo-1m",
        stype_in: str = "raw_symbol",
        consolidated_only: bool = True,
    ) -> pl.DataFrame:
        """Fetch OPRA quotes for a contract.

        By default this keeps Databento's consolidated OPRA publisher when
        ``publisher_id`` is present in the returned data.
        """
        raw_data = self._fetch_timeseries(
            dataset=self.OPRA_DATASET,
            symbols=contract,
            schema=schema,
            stype_in=stype_in,
            start=self._date_string(start),
            end=self._inclusive_end_string(end),
        )
        df = self._transform_data(raw_data, contract)
        if consolidated_only and "publisher_id" in df.columns:
            df = df.filter(pl.col("publisher_id") == self.OPRA_CONSOLIDATED_PUBLISHER_ID)
        return df

    @staticmethod
    def _first_existing_column(df: pl.DataFrame, candidates: list[str]) -> str | None:
        """Return the first candidate column present in a DataFrame."""
        for column in candidates:
            if column in df.columns:
                return column
        return None

    @staticmethod
    def _consolidate_ohlcv_publishers(df: pl.DataFrame) -> pl.DataFrame:
        """Aggregate per-publisher OPRA OHLCV bars into one row per timestamp/symbol."""
        if df.is_empty() or "publisher_id" not in df.columns:
            return df

        group_columns = [column for column in ["timestamp", "symbol"] if column in df.columns]
        if not group_columns:
            return df

        aggregations = []
        if "open" in df.columns:
            aggregations.append(pl.col("open").drop_nulls().first().alias("open"))
        if "high" in df.columns:
            aggregations.append(pl.col("high").max().alias("high"))
        if "low" in df.columns:
            aggregations.append(pl.col("low").min().alias("low"))
        if "close" in df.columns:
            aggregations.append(pl.col("close").drop_nulls().last().alias("close"))
        if "volume" in df.columns:
            aggregations.append(pl.col("volume").sum().alias("volume"))

        if not aggregations:
            return df

        consolidated = df.group_by(group_columns, maintain_order=True).agg(aggregations)
        if "timestamp" in consolidated.columns:
            consolidated = consolidated.sort("timestamp")

        required_ohlcv = ["timestamp", "symbol", "open", "high", "low", "close", "volume"]
        if all(column in consolidated.columns for column in required_ohlcv):
            extra_columns = [
                column for column in consolidated.columns if column not in required_ohlcv
            ]
            consolidated = consolidated.select(required_ohlcv + extra_columns)

        return consolidated

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
