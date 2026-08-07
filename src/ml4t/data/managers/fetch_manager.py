"""Fetch operations for DataManager.

This module handles core data fetching operations including:
- Date validation
- Output format conversion
- Single symbol fetch
- Simple batch fetch (sequential)
"""

from __future__ import annotations

from datetime import datetime
from functools import cache
from importlib.util import find_spec
from typing import TYPE_CHECKING, Any

import pandas as pd
import polars as pl
import structlog

if TYPE_CHECKING:
    from ml4t.data.managers.provider_manager import ProviderManager, ProviderRouter

logger = structlog.get_logger()


class ProviderRoutingError(ValueError):
    """Raised when a request has no explicit or unambiguous provider route."""


@cache
def _has_pyarrow() -> bool:
    """Return whether the optional Arrow conversion dependency is installed."""
    return find_spec("pyarrow") is not None


class FetchManager:
    """Manages core data fetching operations.

    This class provides the fundamental fetch operations used by DataManager,
    handling validation, provider selection, and output format conversion.

    Attributes:
        provider_manager: ProviderManager for provider access
        router: ProviderRouter for symbol-to-provider mapping
        output_format: Output format ('polars', 'pandas', 'lazy')

    Example:
        >>> fetch_mgr = FetchManager(provider_manager, router, "polars")
        >>> df = fetch_mgr.fetch("AAPL", "2024-01-01", "2024-12-31", provider="yahoo")
    """

    def __init__(
        self,
        provider_manager: ProviderManager,
        router: ProviderRouter,
        output_format: str = "polars",
    ) -> None:
        """Initialize FetchManager.

        Args:
            provider_manager: ProviderManager instance
            router: ProviderRouter instance
            output_format: Output format ('polars', 'pandas', 'lazy')
        """
        self.provider_manager = provider_manager
        self.router = router
        self.output_format = output_format

    def validate_dates(self, start: str, end: str) -> tuple[datetime, datetime]:
        """Validate date inputs.

        Args:
            start: Start date string (YYYY-MM-DD)
            end: End date string (YYYY-MM-DD)

        Returns:
            Tuple of (start_datetime, end_datetime)

        Raises:
            ValueError: If dates are invalid or end < start
        """
        try:
            start_dt = datetime.strptime(start, "%Y-%m-%d")
            end_dt = datetime.strptime(end, "%Y-%m-%d")
        except ValueError:
            raise ValueError("Invalid date format. Use YYYY-MM-DD format (e.g., 2024-01-01)")

        if end_dt < start_dt:
            raise ValueError("End date must be after start date")

        return start_dt, end_dt

    def convert_output(self, df: pl.DataFrame) -> pl.DataFrame | pl.LazyFrame | Any:
        """Convert DataFrame to requested output format.

        Args:
            df: Polars DataFrame

        Returns:
            Data in requested format (polars, pandas, or lazy)
        """
        if self.output_format == "polars":
            return df
        if self.output_format == "lazy":
            return df.lazy()
        if self.output_format == "pandas":
            if not _has_pyarrow():
                return pd.DataFrame(df.to_dict(as_series=False))
            return df.to_pandas()
        return df

    def fetch(
        self,
        symbol: str,
        start: str,
        end: str,
        frequency: str = "daily",
        provider: str | None = None,
        **kwargs: Any,
    ) -> pl.DataFrame | pl.LazyFrame | Any:
        """Fetch data for a symbol.

        Args:
            symbol: Symbol to fetch
            start: Start date (YYYY-MM-DD)
            end: End date (YYYY-MM-DD)
            frequency: Data frequency (daily, hourly, etc.)
            provider: Optional provider override
            **kwargs: Additional provider-specific parameters

        Returns:
            Data in configured output format

        Raises:
            ValueError: If no provider found or data fetch fails
        """
        # Validate inputs
        self.validate_dates(start, end)

        # Determine provider
        provider_name = self.router.get_provider(symbol, override=provider)
        if not provider_name:
            raise ProviderRoutingError(
                f"No provider found for symbol: {symbol}. "
                f"Configure routing patterns or specify provider explicitly."
            )

        logger.info(
            f"Fetching {symbol} from {provider_name}",
            start=start,
            end=end,
            frequency=frequency,
        )

        try:
            # Get provider instance
            provider_instance = self.provider_manager.get_provider(provider_name)

            # Fetch data
            df = provider_instance.fetch_ohlcv(symbol, start, end, frequency, **kwargs)

            # Convert to requested format
            return self.convert_output(df)

        except ValueError:
            raise  # Re-raise ValueError as-is
        except Exception as e:
            logger.error(f"Failed to fetch {symbol}: {e}")
            raise Exception(f"Failed to fetch data for {symbol}: {e}") from e

    def fetch_batch(
        self,
        symbols: list[str],
        start: str,
        end: str,
        frequency: str = "daily",
        **kwargs: Any,
    ) -> dict[str, pl.DataFrame | pl.LazyFrame | Any | None]:
        """Fetch data for multiple symbols sequentially.

        This is a simple sequential batch fetch. For parallel fetching,
        use BatchManager.batch_load().

        Args:
            symbols: List of symbols to fetch
            start: Start date (YYYY-MM-DD)
            end: End date (YYYY-MM-DD)
            frequency: Data frequency
            **kwargs: Additional parameters

        Returns:
            Dictionary mapping symbols to data (or None if fetch failed)
        """
        results: dict[str, pl.DataFrame | pl.LazyFrame | Any | None] = {}

        for symbol in symbols:
            try:
                results[symbol] = self.fetch(symbol, start, end, frequency, **kwargs)
            except ProviderRoutingError:
                raise
            except Exception as e:
                logger.warning(f"Failed to fetch {symbol}: {e}")
                results[symbol] = None

        return results

    def fetch_raw(
        self,
        symbol: str,
        start: str,
        end: str,
        frequency: str = "daily",
        provider: str | None = None,
        **kwargs: Any,
    ) -> pl.DataFrame:
        """Fetch data without output format conversion.

        Useful when you need the raw Polars DataFrame for further processing.

        Args:
            symbol: Symbol to fetch
            start: Start date (YYYY-MM-DD)
            end: End date (YYYY-MM-DD)
            frequency: Data frequency
            provider: Optional provider override
            **kwargs: Additional provider-specific parameters

        Returns:
            Polars DataFrame (always)

        Raises:
            ValueError: If no provider found or data fetch fails
        """
        # Validate inputs
        self.validate_dates(start, end)

        # Determine provider
        provider_name = self.router.get_provider(symbol, override=provider)
        if not provider_name:
            raise ValueError(
                f"No provider found for symbol: {symbol}. "
                f"Configure routing patterns or specify provider explicitly."
            )

        # Get provider instance and fetch
        provider_instance = self.provider_manager.get_provider(provider_name)
        return provider_instance.fetch_ohlcv(symbol, start, end, frequency, **kwargs)

    def get_max_history_days(self, symbol: str, provider: str | None = None) -> int | None:
        """Return a provider's declared history bound for managed initial loads."""
        provider_name = self.router.get_provider(symbol, override=provider)
        if provider_name is None:
            return None
        capabilities = self.provider_manager.get_provider(provider_name).capabilities()
        return capabilities.max_history_days
