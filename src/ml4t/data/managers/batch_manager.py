"""Batch operations for DataManager.

This module handles parallel batch loading operations including:
- Multi-symbol parallel fetch
- Universe loading
- Storage-first loading with fetch fallback
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import polars as pl
import structlog

from ml4t.data.utils.conversion import pandas_to_polars

if TYPE_CHECKING:
    from ml4t.data.managers.fetch_manager import FetchManager

logger = structlog.get_logger()


class BatchManager:
    """Manages parallel batch loading operations.

    This class provides high-performance parallel data fetching for
    multiple symbols, with support for universes and storage integration.

    Attributes:
        fetch_manager: FetchManager for individual fetches
        storage: Optional storage backend for cache-first loading

    Example:
        >>> batch_mgr = BatchManager(fetch_manager, storage)
        >>> df = batch_mgr.batch_load(["AAPL", "MSFT", "GOOG"], "2024-01-01", "2024-12-31")
    """

    def __init__(
        self,
        fetch_manager: FetchManager,
        storage: Any | None = None,
    ) -> None:
        """Initialize BatchManager.

        Args:
            fetch_manager: FetchManager instance
            storage: Optional storage backend
        """
        self.fetch_manager = fetch_manager
        self.storage = storage

    def batch_load(
        self,
        symbols: list[str],
        start: str,
        end: str,
        frequency: str = "daily",
        provider: str | None = None,
        max_workers: int = 4,
        fail_on_partial: bool = False,
        **kwargs: Any,
    ) -> pl.DataFrame:
        """Fetch data for multiple symbols in parallel.

        Returns a multi-asset stacked DataFrame with a 'symbol' column.

        Args:
            symbols: List of symbols to fetch
            start: Start date (YYYY-MM-DD)
            end: End date (YYYY-MM-DD)
            frequency: Data frequency
            provider: Optional provider override
            max_workers: Maximum parallel workers
            fail_on_partial: If True, raise error if ANY symbol fails
            **kwargs: Additional provider-specific parameters

        Returns:
            Single DataFrame with all symbols in stacked format

        Raises:
            ValueError: If all symbols fail or fail_on_partial and any fail
        """
        if not symbols:
            raise ValueError("symbols list cannot be empty")

        # Validate inputs
        self.fetch_manager.validate_dates(start, end)

        logger.info(
            f"Starting batch load for {len(symbols)} symbols",
            start=start,
            end=end,
            frequency=frequency,
            max_workers=max_workers,
        )

        from ml4t.data.core.schemas import MultiAssetSchema

        results: dict[str, pl.DataFrame] = {}
        failed_symbols: list[tuple[str, str]] = []

        def fetch_symbol(symbol: str) -> tuple[str, pl.DataFrame | None, str | None]:
            """Fetch a single symbol and return (symbol, dataframe, error)."""
            try:
                df = self.fetch_manager.fetch(
                    symbol, start, end, frequency, provider=provider, **kwargs
                )

                # Convert to Polars DataFrame if needed
                if not isinstance(df, pl.DataFrame):
                    if hasattr(df, "collect"):
                        df = df.collect()
                    elif hasattr(df, "columns"):
                        df = pandas_to_polars(df)

                # Add symbol column
                df_with_symbol = MultiAssetSchema.add_symbol_column(df, symbol)
                return (symbol, df_with_symbol, None)

            except Exception as e:
                error_msg = str(e)
                logger.warning(f"Failed to fetch {symbol}: {error_msg}")
                return (symbol, None, error_msg)

        # Execute parallel fetches
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(fetch_symbol, symbol): symbol for symbol in symbols}

            for future in as_completed(futures):
                symbol, df, error = future.result()

                if df is not None:
                    results[symbol] = df
                    logger.debug(f"Successfully fetched {symbol}: {len(df)} rows")
                else:
                    failed_symbols.append((symbol, error or "Unknown error"))

        # Report results
        success_count = len(results)
        failure_count = len(failed_symbols)

        logger.info(
            f"Batch load completed: {success_count} succeeded, {failure_count} failed",
            succeeded=success_count,
            failed=failure_count,
        )

        # Handle failures
        if failure_count > 0:
            logger.warning(
                f"Failed symbols: {[s for s, _ in failed_symbols]}",
                failures=failed_symbols,
            )

            if fail_on_partial:
                raise ValueError(
                    f"Batch load failed for {failure_count} symbols: "
                    f"{[s for s, _ in failed_symbols]}"
                )

        # Check if we got ANY data
        if not results:
            raise ValueError(
                f"All {len(symbols)} symbols failed to fetch. Failures: {failed_symbols}"
            )

        # Concatenate all DataFrames

        combined_dfs = list(results.values())
        multi_asset_df = pl.concat(combined_dfs, how="vertical_relaxed")
        multi_asset_df = MultiAssetSchema.standardize_order(multi_asset_df)

        logger.info(
            f"Multi-asset DataFrame created: {len(multi_asset_df)} rows, {len(results)} symbols",
            total_rows=len(multi_asset_df),
            symbol_count=len(results),
        )

        return multi_asset_df

    def batch_load_universe(
        self,
        universe: str,
        start: str,
        end: str,
        frequency: str = "daily",
        provider: str | None = None,
        max_workers: int = 4,
        fail_on_partial: bool = False,
        **kwargs: Any,
    ) -> pl.DataFrame:
        """Fetch data for all symbols in a pre-defined universe.

        Args:
            universe: Universe name (e.g., "sp500", "NASDAQ100")
            start: Start date (YYYY-MM-DD)
            end: End date (YYYY-MM-DD)
            frequency: Data frequency
            provider: Optional provider override
            max_workers: Maximum parallel workers
            fail_on_partial: If True, raise error if ANY symbol fails
            **kwargs: Additional provider-specific parameters

        Returns:
            Single DataFrame with all symbols in stacked format

        Raises:
            ValueError: If universe name is invalid or all symbols fail
        """
        from ml4t.data.universe import Universe

        try:
            symbols = Universe.get(universe)
        except ValueError as e:
            raise ValueError(f"Invalid universe '{universe}'. {str(e)}") from e

        logger.info(
            f"Loading universe '{universe}' with {len(symbols)} symbols",
            universe=universe,
            symbol_count=len(symbols),
        )

        return self.batch_load(
            symbols=symbols,
            start=start,
            end=end,
            frequency=frequency,
            provider=provider,
            max_workers=max_workers,
            fail_on_partial=fail_on_partial,
            **kwargs,
        )

    def batch_load_from_storage(
        self,
        symbols: list[str],
        start: str,
        end: str,
        frequency: str = "daily",
        asset_class: str = "equities",
        provider: str | None = None,
        fetch_missing: bool = True,
        max_workers: int = 4,
        **kwargs: Any,
    ) -> pl.DataFrame:
        """Load multiple symbols from storage with optional fetch fallback.

        Prioritizes reading from local storage (fast) and only fetches
        from network for symbols not in storage.

        Args:
            symbols: List of symbols to load
            start: Start date (YYYY-MM-DD)
            end: End date (YYYY-MM-DD)
            frequency: Data frequency
            asset_class: Asset class for storage path
            provider: Provider for missing symbols
            fetch_missing: If True, fetch symbols not in storage
            max_workers: Maximum parallel workers
            **kwargs: Additional parameters for fetch

        Returns:
            Multi-asset DataFrame with all symbols

        Raises:
            ValueError: If storage not configured or no data loaded
        """
        if not self.storage:
            raise ValueError("Storage not configured")
        storage = self.storage

        if not symbols:
            raise ValueError("symbols list cannot be empty")

        # Validate inputs
        self.fetch_manager.validate_dates(start, end)

        start_dt = datetime.strptime(start, "%Y-%m-%d").replace(tzinfo=UTC)
        end_dt = datetime.strptime(end, "%Y-%m-%d").replace(tzinfo=UTC)

        logger.info(
            f"Starting batch load from storage for {len(symbols)} symbols",
            start=start,
            end=end,
            frequency=frequency,
            asset_class=asset_class,
            fetch_missing=fetch_missing,
        )

        from ml4t.data.core.schemas import MultiAssetSchema

        storage_data: dict[str, pl.DataFrame] = {}
        missing_symbols: list[str] = []
        failed_symbols: list[tuple[str, str]] = []

        def read_symbol_from_storage(symbol: str) -> tuple[str, pl.DataFrame | None, str | None]:
            """Read a single symbol from storage."""
            try:
                key = f"{asset_class}/{frequency}/{symbol}"

                if not storage.exists(key):
                    return (symbol, None, "not_in_storage")

                lazy_df = storage.read(key, start_date=start_dt, end_date=end_dt)
                df = lazy_df.collect()

                if df.is_empty():
                    logger.warning(f"Storage returned empty data for {symbol}")
                    return (symbol, None, "empty_data")

                if "symbol" not in df.columns:
                    df = MultiAssetSchema.add_symbol_column(df, symbol)

                logger.debug(f"Loaded {symbol} from storage: {len(df)} rows")
                return (symbol, df, None)

            except KeyError:
                return (symbol, None, "not_in_storage")
            except Exception as e:
                error_msg = str(e)
                logger.warning(f"Failed to read {symbol} from storage: {error_msg}")
                return (symbol, None, f"storage_error: {error_msg}")

        # Execute parallel storage reads
        logger.debug("Reading symbols from storage in parallel")
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(read_symbol_from_storage, symbol): symbol for symbol in symbols
            }

            for future in as_completed(futures):
                symbol, df, error = future.result()

                if df is not None:
                    storage_data[symbol] = df
                elif error == "not_in_storage":
                    missing_symbols.append(symbol)
                else:
                    failed_symbols.append((symbol, error or "Unknown error"))

        logger.info(
            f"Storage read completed: {len(storage_data)} found, {len(missing_symbols)} missing, "
            f"{len(failed_symbols)} failed",
            found=len(storage_data),
            missing=len(missing_symbols),
            failed=len(failed_symbols),
        )

        # Handle missing symbols
        fetched_df: pl.DataFrame | None = None

        if missing_symbols:
            if fetch_missing:
                logger.info(f"Fetching {len(missing_symbols)} missing symbols from provider")

                fetched_df = self.batch_load(
                    symbols=missing_symbols,
                    start=start,
                    end=end,
                    frequency=frequency,
                    provider=provider,
                    max_workers=max_workers,
                    fail_on_partial=False,
                    **kwargs,
                )

                logger.info(
                    f"Fetched {len(missing_symbols)} symbols from provider",
                    fetched_rows=len(fetched_df) if fetched_df is not None else 0,
                )
            else:
                raise ValueError(
                    f"Storage fetch_missing=False but {len(missing_symbols)} symbols not in "
                    f"storage: {missing_symbols}"
                )

        # Check if we got ANY data
        if not storage_data and fetched_df is None:
            raise ValueError(
                f"No data loaded for any of {len(symbols)} symbols. "
                f"Missing: {missing_symbols}, Failed: {failed_symbols}"
            )

        # Combine storage data with fetched data
        all_dfs: list[pl.DataFrame] = []

        if storage_data:
            all_dfs.extend(storage_data.values())

        if fetched_df is not None and not fetched_df.is_empty():
            all_dfs.append(fetched_df)

        # Concatenate all DataFrames
        logger.debug(f"Concatenating {len(all_dfs)} DataFrames")
        combined_df = pl.concat(all_dfs, how="vertical_relaxed")
        result = MultiAssetSchema.standardize_order(combined_df)

        logger.info(
            f"Multi-asset DataFrame created: {len(result)} rows, "
            f"{len(storage_data) + (1 if fetched_df is not None else 0)} sources",
            total_rows=len(result),
            storage_symbols=len(storage_data),
            fetched_symbols=len(missing_symbols) if fetch_missing else 0,
        )

        return result
