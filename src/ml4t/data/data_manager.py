"""Unified data management interface for ML4T Data.

This module provides the DataManager class which abstracts away provider complexity
and offers a single, consistent interface for fetching financial data from multiple sources.

Architecture:
    DataManager is a thin facade that delegates to focused manager classes:
    - ConfigManager: Configuration loading and merging
    - ProviderManager: Provider lifecycle and caching
    - FetchManager: Core fetch operations
    - BatchManager: Parallel batch loading
    - StorageManager: Storage operations (load, import, update)
    - MetadataManager: Discovery and session operations
    - BulkManager: Bulk update operations
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import polars as pl
import structlog

from ml4t.data.managers.batch_manager import BatchManager
from ml4t.data.managers.bulk_manager import BulkManager
from ml4t.data.managers.config_manager import ConfigManager
from ml4t.data.managers.fetch_manager import FetchManager
from ml4t.data.managers.metadata_manager import MetadataManager
from ml4t.data.managers.provider_manager import ProviderManager, ProviderRouter
from ml4t.data.managers.storage_manager import StorageManager

logger = structlog.get_logger()


class DataManager:
    """Unified interface for financial data access and storage.

    The DataManager provides a single, consistent API for fetching and managing
    data from multiple providers. It handles:

    **Data Fetching:**
    - Provider selection based on symbol patterns
    - Configuration management (YAML, environment, parameters)
    - Connection pooling and session management
    - Output format conversion (Polars, pandas, lazy)
    - Batch fetching with error handling

    **Storage Operations (when storage configured):**
    - Initial data loading with validation
    - Incremental updates with gap detection and filling
    - Transaction support for ACID guarantees
    - Progress callbacks for UI integration
    - Data validation (OHLCV, cross-validation)

    **Usage:**

    Fetch only (no storage):
        >>> manager = DataManager()
        >>> df = manager.fetch("AAPL", "2024-01-01", "2024-12-31", provider="yahoo")

    With storage for load/update:
        >>> from ml4t.data.storage.hive import HiveStorage
        >>> from ml4t.data.storage.backend import StorageConfig
        >>> storage = HiveStorage(StorageConfig(base_path="./data"))
        >>> manager = DataManager(storage=storage, use_transactions=True)
        >>> key = manager.load("AAPL", "2024-01-01", "2024-12-31")
        >>> key = manager.update("AAPL")  # Incremental update
    """

    def __init__(
        self,
        config_path: str | None = None,
        output_format: str = "polars",
        providers: dict[str, dict[str, Any]] | None = None,
        storage: Any | None = None,
        use_transactions: bool = False,
        enable_validation: bool = True,
        progress_callback: Callable[[str, float], None] | None = None,
        **kwargs,
    ) -> None:
        """Initialize DataManager.

        Args:
            config_path: Path to YAML configuration file
            output_format: Output format ('polars', 'pandas', 'lazy')
            providers: Provider-specific configuration overrides
            storage: Optional storage backend for load/update operations
            use_transactions: Enable transactional storage for ACID guarantees
            enable_validation: Enable data validation during load/update
            progress_callback: Optional callback for progress updates (message, progress)
            **kwargs: Additional configuration parameters
        """
        # Initialize configuration manager
        self._config_manager = ConfigManager(
            config_path=config_path,
            output_format=output_format,
            providers=providers,
            **kwargs,
        )

        # Initialize provider router with patterns from config
        self._router = ProviderRouter()
        for pattern_config in self._config_manager.get_routing_patterns():
            pattern = pattern_config.get("pattern")
            provider = pattern_config.get("provider")
            if pattern and provider:
                self._router.add_pattern(pattern, provider)
        self._router.setup_default_patterns()

        # Initialize provider manager
        self._provider_manager = ProviderManager(self._config_manager.config)

        # Initialize fetch manager
        self._fetch_manager = FetchManager(
            provider_manager=self._provider_manager,
            router=self._router,
            output_format=self._config_manager.output_format,
        )

        # Setup storage with optional transactions
        self._storage = storage
        if storage and use_transactions:
            from ml4t.data.storage.transaction import TransactionalStorage

            self._storage = TransactionalStorage(storage)
            logger.info("TransactionalStorage enabled for ACID guarantees")

        # Initialize storage manager (if storage configured)
        self._storage_manager: StorageManager | None = None
        if self._storage:
            self._storage_manager = StorageManager(
                storage=self._storage,
                fetch_manager=self._fetch_manager,
                enable_validation=enable_validation,
                progress_callback=progress_callback,
            )

        # Initialize batch manager
        self._batch_manager = BatchManager(
            fetch_manager=self._fetch_manager,
            storage=self._storage,
        )

        # Initialize metadata manager (if storage configured)
        self._metadata_manager: MetadataManager | None = None
        if self._storage:
            self._metadata_manager = MetadataManager(storage=self._storage)

        # Initialize bulk manager (if storage configured)
        self._bulk_manager: BulkManager | None = None
        if self._storage_manager and self._metadata_manager:
            self._bulk_manager = BulkManager(
                storage_manager=self._storage_manager,
                metadata_manager=self._metadata_manager,
            )

        # Store settings for backward compatibility
        self.enable_validation = enable_validation
        self.progress_callback = progress_callback

        logger.info(
            "DataManager initialized",
            output_format=self._config_manager.output_format,
            available_providers=self._provider_manager.available_providers,
            storage_enabled=storage is not None,
            validation_enabled=enable_validation,
        )

    # ========================================================================
    # Properties for backward compatibility
    # ========================================================================

    @property
    def config(self) -> dict[str, Any]:
        """Get configuration dictionary."""
        return self._config_manager.config

    @property
    def output_format(self) -> str:
        """Get output format."""
        return self._config_manager.output_format

    @property
    def router(self) -> ProviderRouter:
        """Get provider router."""
        return self._router

    @property
    def providers(self) -> dict[str, Any]:
        """Get cached provider instances."""
        return self._provider_manager.providers

    @property
    def storage(self) -> Any:
        """Get storage backend."""
        return self._storage

    @property
    def _available_providers(self) -> list[str]:
        """Get available providers (backward compatibility)."""
        return self._provider_manager.available_providers

    # ========================================================================
    # Core fetch operations (delegate to FetchManager)
    # ========================================================================

    def fetch(
        self,
        symbol: str,
        start: str,
        end: str,
        frequency: str = "daily",
        provider: str | None = None,
        **kwargs,
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
        return self._fetch_manager.fetch(symbol, start, end, frequency, provider, **kwargs)

    def fetch_batch(
        self,
        symbols: list[str],
        start: str,
        end: str,
        frequency: str = "daily",
        **kwargs,
    ) -> dict[str, pl.DataFrame | pl.LazyFrame | Any | None]:
        """Fetch data for multiple symbols.

        Args:
            symbols: List of symbols to fetch
            start: Start date (YYYY-MM-DD)
            end: End date (YYYY-MM-DD)
            frequency: Data frequency
            **kwargs: Additional parameters

        Returns:
            Dictionary mapping symbols to data (or None if fetch failed)
        """
        return self._fetch_manager.fetch_batch(symbols, start, end, frequency, **kwargs)

    # ========================================================================
    # Batch operations (delegate to BatchManager)
    # ========================================================================

    def batch_load(
        self,
        symbols: list[str],
        start: str,
        end: str,
        frequency: str = "daily",
        provider: str | None = None,
        max_workers: int = 4,
        fail_on_partial: bool = False,
        **kwargs,
    ) -> pl.DataFrame:
        """Fetch data for multiple symbols and return in multi-asset stacked format."""
        return self._batch_manager.batch_load(
            symbols, start, end, frequency, provider, max_workers, fail_on_partial, **kwargs
        )

    def batch_load_universe(
        self,
        universe: str,
        start: str,
        end: str,
        frequency: str = "daily",
        provider: str | None = None,
        max_workers: int = 4,
        fail_on_partial: bool = False,
        **kwargs,
    ) -> pl.DataFrame:
        """Fetch data for all symbols in a pre-defined universe."""
        return self._batch_manager.batch_load_universe(
            universe, start, end, frequency, provider, max_workers, fail_on_partial, **kwargs
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
        **kwargs,
    ) -> pl.DataFrame:
        """Load multiple symbols from storage with optional fetch fallback."""
        return self._batch_manager.batch_load_from_storage(
            symbols,
            start,
            end,
            frequency,
            asset_class,
            provider,
            fetch_missing,
            max_workers,
            **kwargs,
        )

    # ========================================================================
    # Storage operations (delegate to StorageManager)
    # ========================================================================

    def load(
        self,
        symbol: str,
        start: str,
        end: str,
        frequency: str = "daily",
        asset_class: str = "equities",
        provider: str | None = None,
        bar_type: str = "time",
        bar_threshold: int | None = None,
        exchange: str = "UNKNOWN",
        calendar: str | None = None,
    ) -> str:
        """Load data from provider and store it."""
        if not self._storage_manager:
            raise ValueError(
                "Storage not configured. Pass storage= parameter to DataManager.__init__()"
            )
        return self._storage_manager.load(
            symbol,
            start,
            end,
            frequency,
            asset_class,
            provider,
            bar_type,
            bar_threshold,
            exchange,
            calendar,
        )

    def import_data(
        self,
        data: pl.DataFrame | pl.LazyFrame,
        symbol: str,
        provider: str,
        frequency: str = "daily",
        asset_class: str = "equities",
        bar_type: str = "time",
        bar_threshold: int | None = None,
        exchange: str = "UNKNOWN",
        calendar: str | None = None,
    ) -> str:
        """Import external data into storage with metadata."""
        if not self._storage_manager:
            raise ValueError(
                "Storage not configured. Pass storage= parameter to DataManager.__init__()"
            )
        return self._storage_manager.import_data(
            data,
            symbol,
            provider,
            frequency,
            asset_class,
            bar_type,
            bar_threshold,
            exchange,
            calendar,
        )

    def update(
        self,
        symbol: str,
        frequency: str = "daily",
        asset_class: str = "equities",
        lookback_days: int = 7,
        fill_gaps: bool = True,
        provider: str | None = None,
    ) -> str:
        """Update existing data with incremental fetch."""
        if not self._storage_manager:
            raise ValueError(
                "Storage not configured. Pass storage= parameter to DataManager.__init__()"
            )
        return self._storage_manager.update(
            symbol, frequency, asset_class, lookback_days, fill_gaps, provider
        )

    # ========================================================================
    # Metadata and discovery (delegate to MetadataManager)
    # ========================================================================

    def list_symbols(
        self,
        provider: str | None = None,
        asset_class: str | None = None,
        exchange: str | None = None,
        bar_type: str | None = None,
    ) -> list[str]:
        """List all symbols in storage, optionally filtered by metadata."""
        if not self._metadata_manager:
            raise ValueError("Storage not configured")
        return self._metadata_manager.list_symbols(provider, asset_class, exchange, bar_type)

    def get_metadata(
        self,
        symbol: str,
        asset_class: str = "equities",
        frequency: str = "daily",
    ) -> dict | None:
        """Get metadata for a specific symbol."""
        if not self._metadata_manager:
            raise ValueError("Storage not configured")
        return self._metadata_manager.get_metadata(symbol, asset_class, frequency)

    def get_metadata_for_key(self, key: str) -> dict | None:
        """Get metadata for a storage key."""
        if not self._metadata_manager:
            return None
        return self._metadata_manager.get_metadata_for_key(key)

    def assign_sessions(
        self,
        df: pl.DataFrame,
        exchange: str | None = None,
        calendar: str | None = None,
    ) -> pl.DataFrame:
        """Assign session_date column to DataFrame based on exchange calendar."""
        if not self._metadata_manager:
            # Fall back to direct implementation
            from ml4t.data.sessions import SessionAssigner

            if calendar is None:
                if exchange is None:
                    raise ValueError("Must provide either exchange or calendar parameter")
                assigner = SessionAssigner.from_exchange(exchange)
            else:
                assigner = SessionAssigner(calendar)
            return assigner.assign_sessions(df)

        return self._metadata_manager.assign_sessions(df, exchange, calendar)

    def complete_sessions(
        self,
        df: pl.DataFrame,
        exchange: str | None = None,
        calendar: str | None = None,
        fill_gaps: bool = True,
        fill_method: str = "forward",
        zero_volume: bool = True,
    ) -> pl.DataFrame:
        """Complete sessions by filling gaps."""
        if not self._metadata_manager:
            # Fall back to direct implementation
            from ml4t.data.sessions import SessionCompleter
            from ml4t.data.sessions.assigner import EXCHANGE_CALENDARS

            if calendar is None:
                if exchange is None:
                    raise ValueError("Must provide either exchange or calendar parameter")
                calendar = EXCHANGE_CALENDARS.get(exchange.upper())
                if not calendar:
                    raise ValueError(f"Unknown exchange '{exchange}'")

            completer = SessionCompleter(calendar)
            return completer.complete_sessions(df, fill_method=fill_method, zero_volume=zero_volume)

        return self._metadata_manager.complete_sessions(
            df, exchange, calendar, fill_gaps, fill_method, zero_volume
        )

    # ========================================================================
    # Bulk operations (delegate to BulkManager)
    # ========================================================================

    def update_all(
        self,
        provider: str | None = None,
        asset_class: str | None = None,
        exchange: str | None = None,
    ) -> dict[str, str]:
        """Update all stored data matching the filters."""
        if not self._bulk_manager:
            raise ValueError("Storage not configured")
        return self._bulk_manager.update_all(provider, asset_class, exchange)

    # ========================================================================
    # Provider management (delegate to ProviderManager)
    # ========================================================================

    def list_providers(self) -> list[str]:
        """List available providers."""
        return self._provider_manager.available_providers

    def get_provider_info(self, provider_name: str) -> dict[str, Any]:
        """Get information about a provider."""
        return self._provider_manager.get_provider_info(provider_name)

    def clear_cache(self) -> None:
        """Clear routing cache and close provider connections."""
        self._router.clear_cache()
        self._provider_manager.close_all()
        logger.info("Cleared provider cache and connections")

    # ========================================================================
    # Internal methods (for backward compatibility)
    # ========================================================================

    def _get_provider(self, provider_name: str) -> Any:
        """Get or create provider instance (backward compatibility)."""
        return self._provider_manager.get_provider(provider_name)

    def _validate_dates(self, start: str, end: str) -> None:
        """Validate date inputs (backward compatibility)."""
        self._fetch_manager.validate_dates(start, end)

    def _convert_output(self, df: pl.DataFrame) -> pl.DataFrame | pl.LazyFrame | Any:
        """Convert DataFrame to requested output format (backward compatibility)."""
        return self._fetch_manager.convert_output(df)

    def _report_progress(self, message: str, progress: float) -> None:
        """Report progress if callback is configured (backward compatibility)."""
        if self._storage_manager:
            self._storage_manager._report_progress(message, progress)

    def _validate_data(self, df: pl.DataFrame, symbol: str, asset_class: str = "equity") -> Any:
        """Validate data (backward compatibility)."""
        if self._storage_manager:
            return self._storage_manager._validate_data(df, symbol, asset_class)
        return None

    def _merge_data(self, existing: pl.DataFrame, new: pl.DataFrame) -> pl.DataFrame:
        """Merge existing and new data (backward compatibility)."""
        if self._storage_manager:
            return self._storage_manager._merge_data(existing, new)
        raise ValueError("Storage not configured")

    # ========================================================================
    # Context manager
    # ========================================================================

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - cleanup connections."""
        self.clear_cache()


# Set class-level PROVIDER_CLASSES attribute for backwards compatibility
# This must be done after the class is defined
DataManager.PROVIDER_CLASSES = ProviderManager._get_provider_classes()
