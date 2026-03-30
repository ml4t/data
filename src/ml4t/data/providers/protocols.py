"""Protocol definitions for provider interfaces.

This module defines Protocol-based interfaces for different provider types,
enabling structural subtyping and better type checking without inheritance.

Provider Types:
    - OHLCVProvider: Price/volume data (Yahoo, Binance API, Binance bulk, etc.)
    - FactorProvider: Academic factor data (Fama-French, AQR)
    - EventProvider: Event/prediction data (Kalshi, Polymarket)

Usage:
    # Any class implementing the protocol works
    def fetch_data(provider: OHLCVProvider, symbol: str) -> pl.DataFrame:
        return provider.fetch_ohlcv(symbol, "2024-01-01", "2024-12-31")

    # Works with any provider implementing fetch_ohlcv
    yahoo = YahooFinanceProvider()
    binance = BinanceAPIProvider()
    fetch_data(yahoo, "AAPL")
    fetch_data(binance, "BTCUSDT")
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    import polars as pl


@dataclass(frozen=True)
class ProviderCapabilities:
    """Describes what a provider can do.

    Attributes:
        supports_intraday: Can fetch minute/hourly data
        supports_crypto: Handles cryptocurrency symbols
        supports_forex: Handles forex pairs
        supports_futures: Handles futures contracts
        requires_api_key: Needs authentication
        max_history_days: Maximum historical data available
        rate_limit: (calls, period_seconds) tuple
    """

    supports_intraday: bool = False
    supports_crypto: bool = False
    supports_forex: bool = False
    supports_futures: bool = False
    requires_api_key: bool = False
    max_history_days: int | None = None
    rate_limit: tuple[int, float] = (60, 60.0)


@runtime_checkable
class OHLCVProvider(Protocol):
    """Protocol for OHLCV data providers.

    Any class implementing these methods is considered an OHLCVProvider,
    regardless of inheritance. This enables duck typing with type safety.

    Example:
        >>> class MyCustomProvider:
        ...     @property
        ...     def name(self) -> str:
        ...         return "custom"
        ...
        ...     def fetch_ohlcv(self, symbol, start, end, frequency="daily"):
        ...         # Custom implementation
        ...         pass
        ...
        ...     def capabilities(self) -> ProviderCapabilities:
        ...         return ProviderCapabilities()
        ...
        >>> isinstance(MyCustomProvider(), OHLCVProvider)  # True
    """

    @property
    def name(self) -> str:
        """Return the provider name (e.g., 'yahoo', 'binance_api')."""
        ...

    def fetch_ohlcv(
        self,
        symbol: str,
        start: str,
        end: str,
        frequency: str = "daily",
    ) -> pl.DataFrame:
        """Fetch OHLCV data for a symbol.

        Args:
            symbol: Symbol to fetch (e.g., 'AAPL', 'BTCUSDT')
            start: Start date in YYYY-MM-DD format
            end: End date in YYYY-MM-DD format
            frequency: Data frequency ('daily', 'hourly', 'minute', etc.)

        Returns:
            DataFrame with columns: [timestamp, symbol, open, high, low, close, volume]
        """
        ...

    def capabilities(self) -> ProviderCapabilities:
        """Return provider capabilities."""
        ...


@runtime_checkable
class FactorProvider(Protocol):
    """Protocol for academic factor data providers.

    Factor providers have a different interface than OHLCV providers:
    - They fetch named datasets rather than symbols
    - They may return multiple factors per request
    - They have discoverable dataset catalogs

    Example:
        >>> class MyFactorProvider:
        ...     @property
        ...     def name(self) -> str:
        ...         return "custom_factors"
        ...
        ...     def fetch(self, dataset, frequency="monthly"):
        ...         pass
        ...
        ...     def list_datasets(self) -> list[str]:
        ...         return ["ff3", "ff5", "momentum"]
    """

    @property
    def name(self) -> str:
        """Return the provider name."""
        ...

    def fetch(
        self,
        dataset: str,
        frequency: str = "monthly",
    ) -> pl.DataFrame:
        """Fetch factor data for a dataset.

        Args:
            dataset: Dataset name (e.g., 'ff3', 'qmj_factors')
            frequency: Data frequency ('daily', 'monthly')

        Returns:
            DataFrame with factor returns
        """
        ...

    def list_datasets(self) -> list[str]:
        """List available datasets.

        Returns:
            List of dataset names
        """
        ...


@runtime_checkable
class EventProvider(Protocol):
    """Protocol for event/prediction market providers.

    Event providers handle prediction market data:
    - Markets with probability outcomes
    - Events with resolution dates
    - Price history as implied probabilities

    Example:
        >>> class MyEventProvider:
        ...     @property
        ...     def name(self) -> str:
        ...         return "custom_events"
        ...
        ...     def fetch_events(self, market, start, end):
        ...         pass
        ...
        ...     def list_markets(self) -> list[str]:
        ...         return ["politics", "sports", "crypto"]
    """

    @property
    def name(self) -> str:
        """Return the provider name."""
        ...

    def fetch_events(
        self,
        market: str,
        start: str,
        end: str,
    ) -> pl.DataFrame:
        """Fetch event/prediction data.

        Args:
            market: Market identifier
            start: Start date in YYYY-MM-DD format
            end: End date in YYYY-MM-DD format

        Returns:
            DataFrame with event data
        """
        ...

    def list_markets(self) -> list[str]:
        """List available markets.

        Returns:
            List of market identifiers
        """
        ...


@runtime_checkable
class AsyncOHLCVProvider(Protocol):
    """Async version of OHLCVProvider for concurrent operations.

    Providers implementing this protocol support async/await for
    better performance in batch operations.
    """

    @property
    def name(self) -> str:
        """Return the provider name."""
        ...

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
            start: Start date in YYYY-MM-DD format
            end: End date in YYYY-MM-DD format
            frequency: Data frequency

        Returns:
            DataFrame with OHLCV data
        """
        ...

    def capabilities(self) -> ProviderCapabilities:
        """Return provider capabilities."""
        ...
