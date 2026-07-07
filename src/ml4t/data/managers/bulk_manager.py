"""Bulk operations for DataManager.

This module handles bulk update operations across multiple symbols.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import structlog

if TYPE_CHECKING:
    from ml4t.data.managers.metadata_manager import MetadataManager
    from ml4t.data.managers.storage_manager import StorageManager

logger = structlog.get_logger()


class BulkManager:
    """Manages bulk update operations.

    This class handles updating all stored data matching certain filters,
    using stored metadata to determine update parameters.

    Attributes:
        storage_manager: StorageManager for updates
        metadata_manager: MetadataManager for discovery

    Example:
        >>> bulk_mgr = BulkManager(storage_manager, metadata_manager)
        >>> results = bulk_mgr.update_all(provider="yahoo")
    """

    def __init__(
        self,
        storage_manager: StorageManager,
        metadata_manager: MetadataManager,
    ) -> None:
        """Initialize BulkManager.

        Args:
            storage_manager: StorageManager instance
            metadata_manager: MetadataManager instance
        """
        self.storage_manager = storage_manager
        self.metadata_manager = metadata_manager

    def update_all(
        self,
        provider: str | None = None,
        asset_class: str | None = None,
        exchange: str | None = None,
        _max_workers: int = 1,  # Kept for API compat; use async_batch_load for parallelism
    ) -> dict[str, str]:
        """Update all stored data matching the filters.

        Reads metadata for each symbol and calls update() using stored parameters.

        For parallel/concurrent updates, use async_batch_load() with the provider directly:
            from ml4t.data.managers.async_batch import async_batch_load
            async with YahooFinanceProvider() as provider:
                df = await async_batch_load(provider, symbols, start, end)

        Args:
            provider: Filter by provider
            asset_class: Filter by asset class
            exchange: Filter by exchange
            _max_workers: Reserved for API compatibility (use async_batch_load for parallelism)

        Returns:
            Dict mapping symbols to storage keys or error messages
        """
        storage = self.storage_manager.storage
        if not storage:
            raise ValueError("Storage not configured")

        # Collect symbols to update with their parameters
        symbols_to_update: list[tuple[str, str, str, str | None]] = []

        for key in storage.list_keys():
            try:
                metadata = self.metadata_manager.get_metadata_for_key(key)
                if metadata is None:
                    continue

                # Apply filters
                if provider and metadata.get("provider") != provider:
                    continue
                if asset_class and metadata.get("asset_class") != asset_class:
                    continue
                if exchange and metadata.get("exchange") != exchange:
                    continue

                symbols_to_update.append(
                    (
                        metadata.get("symbol"),
                        metadata.get("frequency", "daily"),
                        metadata.get("asset_class", "equities"),
                        metadata.get("provider"),
                    )
                )

            except Exception as e:
                logger.warning(f"Failed to read metadata for {key}: {e}")
                continue

        logger.info(f"Updating {len(symbols_to_update)} symbols")

        # Sequential updates - for parallel fetching, use async_batch_load()
        results: dict[str, str] = {}

        for symbol, freq, asset_cls, prov in symbols_to_update:
            if symbol is None:
                continue

            try:
                logger.info(f"Updating {symbol}")
                key = self.storage_manager.update(
                    symbol=symbol,
                    frequency=freq,
                    asset_class=asset_cls,
                    provider=prov,
                )
                results[symbol] = key

            except Exception as e:
                logger.error(f"Failed to update {symbol}: {e}")
                results[symbol] = f"ERROR: {e}"

        logger.info(f"Update completed: {len(results)} symbols processed")
        return results

    def update_symbols(
        self,
        symbols: list[str],
        frequency: str = "daily",
        asset_class: str = "equities",
        provider: str | None = None,
        lookback_days: int = 7,
        fill_gaps: bool = True,
        initial_start: str | None = None,
        initial_end: str | None = None,
        initial_load_days: int = 365,
    ) -> dict[str, str]:
        """Update specific symbols.

        Args:
            symbols: List of symbols to update
            frequency: Data frequency
            asset_class: Asset class
            provider: Optional provider override
            lookback_days: Days to look back for updates
            fill_gaps: If True, detect and fill gaps
            initial_start: Optional start date for first load when no data exists
            initial_end: Optional end date for first load when no data exists
            initial_load_days: Days to fetch on first load when initial_start is not set

        Returns:
            Dict mapping symbols to storage keys or error messages
        """
        results: dict[str, str] = {}

        for symbol in symbols:
            try:
                logger.info(f"Updating {symbol}")
                key = self.storage_manager.update(
                    symbol=symbol,
                    frequency=frequency,
                    asset_class=asset_class,
                    lookback_days=lookback_days,
                    fill_gaps=fill_gaps,
                    provider=provider,
                    initial_start=initial_start,
                    initial_end=initial_end,
                    initial_load_days=initial_load_days,
                )
                results[symbol] = key

            except Exception as e:
                logger.error(f"Failed to update {symbol}: {e}")
                results[symbol] = f"ERROR: {e}"

        return results

    def get_stale_symbols(
        self,
        max_age_days: int = 7,
        provider: str | None = None,
        asset_class: str | None = None,
    ) -> list[str]:
        """Get symbols that haven't been updated recently.

        Args:
            max_age_days: Maximum age in days before considered stale
            provider: Filter by provider
            asset_class: Filter by asset class

        Returns:
            List of stale symbol names
        """
        from datetime import datetime, timedelta

        cutoff = datetime.now() - timedelta(days=max_age_days)
        stale_symbols = []

        storage = self.storage_manager.storage
        if not storage:
            return []

        for key in storage.list_keys():
            try:
                metadata = self.metadata_manager.get_metadata_for_key(key)
                if metadata is None:
                    continue

                # Apply filters
                if provider and metadata.get("provider") != provider:
                    continue
                if asset_class and metadata.get("asset_class") != asset_class:
                    continue

                # Check last updated
                last_updated_str = metadata.get("attributes", {}).get("last_update")
                if last_updated_str:
                    last_updated = datetime.fromisoformat(last_updated_str.replace("Z", "+00:00"))
                    if last_updated.replace(tzinfo=None) < cutoff:
                        symbol = metadata.get("symbol")
                        if symbol:
                            stale_symbols.append(symbol)
                else:
                    # No last_update means it's stale
                    symbol = metadata.get("symbol")
                    if symbol:
                        stale_symbols.append(symbol)

            except Exception as e:
                logger.warning(f"Failed to check staleness for {key}: {e}")
                continue

        return sorted(set(stale_symbols))

    def get_update_summary(
        self,
        provider: str | None = None,
        asset_class: str | None = None,
    ) -> dict[str, Any]:
        """Get summary of stored data status.

        Args:
            provider: Filter by provider
            asset_class: Filter by asset class

        Returns:
            Summary dictionary with counts and statistics
        """
        storage = self.storage_manager.storage
        if not storage:
            return {"error": "Storage not configured"}

        total_symbols = 0
        stale_7d = 0
        stale_30d = 0
        providers: dict[str, int] = {}
        asset_classes: dict[str, int] = {}

        from datetime import datetime, timedelta

        cutoff_7d = datetime.now() - timedelta(days=7)
        cutoff_30d = datetime.now() - timedelta(days=30)

        for key in storage.list_keys():
            try:
                metadata = self.metadata_manager.get_metadata_for_key(key)
                if metadata is None:
                    continue

                # Apply filters
                if provider and metadata.get("provider") != provider:
                    continue
                if asset_class and metadata.get("asset_class") != asset_class:
                    continue

                total_symbols += 1

                # Count by provider
                prov = metadata.get("provider", "unknown")
                providers[prov] = providers.get(prov, 0) + 1

                # Count by asset class
                ac = metadata.get("asset_class", "unknown")
                asset_classes[ac] = asset_classes.get(ac, 0) + 1

                # Check staleness
                last_updated_str = metadata.get("attributes", {}).get("last_update")
                if last_updated_str:
                    last_updated = datetime.fromisoformat(last_updated_str.replace("Z", "+00:00"))
                    last_updated = last_updated.replace(tzinfo=None)

                    if last_updated < cutoff_30d:
                        stale_30d += 1
                    elif last_updated < cutoff_7d:
                        stale_7d += 1
                else:
                    stale_30d += 1

            except Exception as e:
                logger.warning(f"Failed to read metadata for {key}: {e}")
                continue

        return {
            "total_symbols": total_symbols,
            "stale_7d": stale_7d,
            "stale_30d": stale_30d,
            "fresh": total_symbols - stale_7d - stale_30d,
            "by_provider": providers,
            "by_asset_class": asset_classes,
        }
