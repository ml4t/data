"""Async base provider for high-performance concurrent fetching.

This module provides AsyncBaseProvider for providers that support
async/await operations, enabling 3-5x speedup for batch operations.

Usage:
    class MyAsyncProvider(AsyncBaseProvider):
        @property
        def name(self) -> str:
            return "my_async_provider"

        async def _fetch_and_transform_data_async(
            self, symbol, start, end, frequency
        ) -> pl.DataFrame:
            # Async implementation
            data = await self._aget_json(f"{self.base_url}?symbol={symbol}")
            return self._transform(data)

    # Usage
    async with MyAsyncProvider() as provider:
        # Single fetch
        df = await provider.fetch_ohlcv_async("AAPL", "2024-01-01", "2024-12-31")

        # Batch fetch (3-5x faster than sync)
        results = await provider.batch_fetch_async(
            ["AAPL", "MSFT", "GOOGL"],
            "2024-01-01",
            "2024-12-31",
        )
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from typing import ClassVar

import polars as pl
import structlog

from ml4t.data.providers.mixins.async_session import AsyncSessionMixin
from ml4t.data.providers.mixins.rate_limit import RateLimitMixin
from ml4t.data.providers.mixins.validation import ValidationMixin
from ml4t.data.providers.protocols import ProviderCapabilities

logger = structlog.get_logger()


class AsyncBaseProvider(
    RateLimitMixin,
    ValidationMixin,
    AsyncSessionMixin,
    ABC,
):
    """Async base provider for concurrent data fetching.

    Provides async/await interface for OHLCV data fetching with:
    - Concurrent batch operations via asyncio.gather()
    - Rate limiting (async-aware)
    - Data validation
    - Connection pooling

    Subclasses must implement:
    - name property
    - _fetch_and_transform_data_async() method
    """

    DEFAULT_RATE_LIMIT: ClassVar[tuple[int, float]] = (60, 60.0)
    MAX_CONCURRENT_REQUESTS: ClassVar[int] = 10

    def __init__(
        self,
        rate_limit: tuple[int, float] | None = None,
        max_concurrent: int | None = None,
    ):
        """Initialize async provider.

        Args:
            rate_limit: Tuple of (calls, period_seconds) for rate limiting
            max_concurrent: Maximum concurrent requests
        """
        self.logger = structlog.get_logger(name=self.__class__.__name__)
        self.max_concurrent = max_concurrent or self.MAX_CONCURRENT_REQUESTS
        self._semaphore: asyncio.Semaphore | None = None

        # Initialize rate limiting
        self.init_rate_limit(self.name, rate_limit)

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the provider name."""

    def capabilities(self) -> ProviderCapabilities:
        """Return provider capabilities."""
        return ProviderCapabilities(
            rate_limit=self.DEFAULT_RATE_LIMIT,
        )

    async def fetch_ohlcv_async(
        self,
        symbol: str,
        start: str,
        end: str,
        frequency: str = "daily",
    ) -> pl.DataFrame:
        """Async fetch OHLCV data for a symbol.

        Args:
            symbol: Symbol to fetch
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

        # Rate limiting
        self._acquire_rate_limit()

        # Initialize session if needed
        if self.async_session is None:
            await self.init_async_session()

        # Use semaphore for concurrency control
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(self.max_concurrent)

        async with self._semaphore:
            # Fetch and transform
            data = await self._fetch_and_transform_data_async(symbol, start, end, frequency)

            # Validate
            validated = self._validate_ohlcv(data, self.name, symbol)

        self.logger.info(
            "Successfully fetched OHLCV data (async)",
            symbol=symbol,
            rows=len(validated),
            provider=self.name,
        )

        return validated

    async def batch_fetch_async(
        self,
        symbols: list[str],
        start: str,
        end: str,
        frequency: str = "daily",
        return_exceptions: bool = False,
    ) -> dict[str, pl.DataFrame | Exception]:
        """Fetch OHLCV data for multiple symbols concurrently.

        Args:
            symbols: List of symbols to fetch
            start: Start date (YYYY-MM-DD)
            end: End date (YYYY-MM-DD)
            frequency: Data frequency
            return_exceptions: If True, return exceptions instead of raising

        Returns:
            Dictionary mapping symbols to DataFrames (or exceptions)
        """
        self.logger.info(
            "Batch fetching OHLCV data (async)",
            symbols=symbols,
            start=start,
            end=end,
            frequency=frequency,
            provider=self.name,
        )

        # Create tasks for all symbols
        tasks = [self.fetch_ohlcv_async(symbol, start, end, frequency) for symbol in symbols]

        # Execute concurrently
        results = await asyncio.gather(*tasks, return_exceptions=return_exceptions)

        # Build result dictionary
        result_dict: dict[str, pl.DataFrame | Exception] = {}
        for symbol, result in zip(symbols, results):
            result_dict[symbol] = result

        successful = sum(1 for r in results if isinstance(r, pl.DataFrame))
        self.logger.info(
            "Batch fetch complete",
            total=len(symbols),
            successful=successful,
            failed=len(symbols) - successful,
        )

        return result_dict

    # Sync wrapper for compatibility
    def fetch_ohlcv(
        self,
        symbol: str,
        start: str,
        end: str,
        frequency: str = "daily",
    ) -> pl.DataFrame:
        """Sync wrapper for fetch_ohlcv_async.

        Args:
            symbol: Symbol to fetch
            start: Start date (YYYY-MM-DD)
            end: End date (YYYY-MM-DD)
            frequency: Data frequency

        Returns:
            DataFrame with OHLCV data
        """
        return asyncio.run(self.fetch_ohlcv_async(symbol, start, end, frequency))

    @abstractmethod
    async def _fetch_and_transform_data_async(
        self,
        symbol: str,
        start: str,
        end: str,
        frequency: str,
    ) -> pl.DataFrame:
        """Async fetch and transform data.

        Subclasses must implement this method.

        Args:
            symbol: Symbol to fetch
            start: Start date
            end: End date
            frequency: Data frequency

        Returns:
            DataFrame with standardized OHLCV schema
        """

    async def close(self) -> None:
        """Close async resources."""
        await self.close_async_session()


# Type alias for static type checking
AsyncProvider = AsyncBaseProvider
