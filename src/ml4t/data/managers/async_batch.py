"""Async batch operations for high-performance data loading.

This module provides async batch loading using asyncio.gather() for
3-5x speedup over thread-based parallel loading.

Usage:
    from ml4t.data.managers.async_batch import async_batch_load
    from ml4t.data.providers.async_base import AsyncBaseProvider

    async with MyAsyncProvider() as provider:
        results = await async_batch_load(
            provider=provider,
            symbols=["AAPL", "MSFT", "GOOGL"],
            start="2024-01-01",
            end="2024-12-31",
        )

Performance Comparison:
    - Thread-based (BatchManager): ~1x baseline
    - Async (async_batch_load): ~3-5x faster for I/O-bound operations
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import polars as pl
import structlog

if TYPE_CHECKING:
    from ml4t.data.providers.protocols import AsyncOHLCVProvider

logger = structlog.get_logger()


async def async_batch_load(
    provider: AsyncOHLCVProvider,
    symbols: list[str],
    start: str,
    end: str,
    frequency: str = "daily",
    max_concurrent: int = 10,
    fail_on_partial: bool = False,
) -> pl.DataFrame:
    """Load OHLCV data for multiple symbols concurrently.

    Uses asyncio.gather() for concurrent fetching, providing 3-5x
    speedup over thread-based approaches for I/O-bound operations.

    Args:
        provider: Async provider implementing fetch_ohlcv_async
        symbols: List of symbols to fetch
        start: Start date (YYYY-MM-DD)
        end: End date (YYYY-MM-DD)
        frequency: Data frequency
        max_concurrent: Maximum concurrent requests
        fail_on_partial: If True, raise error if any symbol fails

    Returns:
        Multi-asset DataFrame with 'symbol' column

    Raises:
        ValueError: If all symbols fail or fail_on_partial and any fail
    """
    if not symbols:
        raise ValueError("symbols list cannot be empty")

    logger.info(
        f"Starting async batch load for {len(symbols)} symbols",
        start=start,
        end=end,
        frequency=frequency,
        max_concurrent=max_concurrent,
    )

    # Semaphore for concurrency control
    semaphore = asyncio.Semaphore(max_concurrent)

    async def fetch_with_semaphore(symbol: str) -> tuple[str, pl.DataFrame | None, str | None]:
        """Fetch with semaphore for concurrency control."""
        async with semaphore:
            try:
                df = await provider.fetch_ohlcv_async(symbol, start, end, frequency)

                # Add symbol column if not present
                if "symbol" not in df.columns:
                    df = df.with_columns(pl.lit(symbol).alias("symbol"))

                return (symbol, df, None)
            except Exception as e:
                error_msg = str(e)
                logger.warning(f"Failed to fetch {symbol}: {error_msg}")
                return (symbol, None, error_msg)

    # Create tasks for all symbols
    tasks = [fetch_with_semaphore(symbol) for symbol in symbols]

    # Execute concurrently
    results = await asyncio.gather(*tasks)

    # Process results
    successful: dict[str, pl.DataFrame] = {}
    failed: list[tuple[str, str]] = []

    for symbol, df, error in results:
        if df is not None:
            successful[symbol] = df
        else:
            failed.append((symbol, error or "Unknown error"))

    # Log results
    logger.info(
        f"Async batch load completed: {len(successful)} succeeded, {len(failed)} failed",
        succeeded=len(successful),
        failed=len(failed),
    )

    # Handle failures
    if failed:
        logger.warning(
            f"Failed symbols: {[s for s, _ in failed]}",
            failures=failed,
        )
        if fail_on_partial:
            raise ValueError(
                f"Batch load failed for {len(failed)} symbols: {[s for s, _ in failed]}"
            )

    # Check if we got any data
    if not successful:
        raise ValueError(f"All {len(symbols)} symbols failed to fetch. Failures: {failed}")

    # Concatenate all DataFrames
    combined_dfs = list(successful.values())
    multi_asset_df = pl.concat(combined_dfs, how="vertical_relaxed")

    # Sort by symbol and timestamp
    if "timestamp" in multi_asset_df.columns and "symbol" in multi_asset_df.columns:
        multi_asset_df = multi_asset_df.sort(["symbol", "timestamp"])

    logger.info(
        f"Multi-asset DataFrame created: {len(multi_asset_df)} rows, {len(successful)} symbols",
        total_rows=len(multi_asset_df),
        symbol_count=len(successful),
    )

    return multi_asset_df


async def async_batch_load_dict(
    provider: AsyncOHLCVProvider,
    symbols: list[str],
    start: str,
    end: str,
    frequency: str = "daily",
    max_concurrent: int = 10,
    return_exceptions: bool = False,
) -> dict[str, pl.DataFrame | Exception]:
    """Load OHLCV data for multiple symbols, returning a dictionary.

    Unlike async_batch_load which returns a stacked DataFrame, this
    returns a dictionary mapping symbols to their individual DataFrames.

    Args:
        provider: Async provider implementing fetch_ohlcv_async
        symbols: List of symbols to fetch
        start: Start date (YYYY-MM-DD)
        end: End date (YYYY-MM-DD)
        frequency: Data frequency
        max_concurrent: Maximum concurrent requests
        return_exceptions: If True, include exceptions in dict instead of raising

    Returns:
        Dictionary mapping symbols to DataFrames (or exceptions)
    """
    if not symbols:
        return {}

    semaphore = asyncio.Semaphore(max_concurrent)

    async def fetch_with_semaphore(symbol: str) -> tuple[str, pl.DataFrame | Exception]:
        async with semaphore:
            try:
                df = await provider.fetch_ohlcv_async(symbol, start, end, frequency)
                return (symbol, df)
            except Exception as e:
                if return_exceptions:
                    return (symbol, e)
                raise

    # Create and execute tasks
    tasks = [fetch_with_semaphore(symbol) for symbol in symbols]
    results = await asyncio.gather(*tasks)
    return dict(results)


class AsyncBatchManager:
    """Async batch manager for high-performance data loading.

    Provides async versions of BatchManager methods using
    asyncio.gather() for concurrent operations.

    Example:
        async_batch = AsyncBatchManager(provider)
        df = await async_batch.load(["AAPL", "MSFT"], "2024-01-01", "2024-12-31")
    """

    def __init__(
        self,
        provider: AsyncOHLCVProvider,
        max_concurrent: int = 10,
    ):
        """Initialize async batch manager.

        Args:
            provider: Async provider
            max_concurrent: Maximum concurrent requests
        """
        self.provider = provider
        self.max_concurrent = max_concurrent

    async def load(
        self,
        symbols: list[str],
        start: str,
        end: str,
        frequency: str = "daily",
        fail_on_partial: bool = False,
    ) -> pl.DataFrame:
        """Load multiple symbols concurrently.

        Args:
            symbols: Symbols to load
            start: Start date
            end: End date
            frequency: Data frequency
            fail_on_partial: Fail if any symbol fails

        Returns:
            Multi-asset stacked DataFrame
        """
        return await async_batch_load(
            provider=self.provider,
            symbols=symbols,
            start=start,
            end=end,
            frequency=frequency,
            max_concurrent=self.max_concurrent,
            fail_on_partial=fail_on_partial,
        )

    async def load_dict(
        self,
        symbols: list[str],
        start: str,
        end: str,
        frequency: str = "daily",
        return_exceptions: bool = False,
    ) -> dict[str, pl.DataFrame | Exception]:
        """Load multiple symbols, returning dictionary.

        Args:
            symbols: Symbols to load
            start: Start date
            end: End date
            frequency: Data frequency
            return_exceptions: Include exceptions in result

        Returns:
            Dictionary of symbol -> DataFrame
        """
        return await async_batch_load_dict(
            provider=self.provider,
            symbols=symbols,
            start=start,
            end=end,
            frequency=frequency,
            max_concurrent=self.max_concurrent,
            return_exceptions=return_exceptions,
        )
