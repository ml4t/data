"""Metadata and discovery operations for DataManager.

This module handles metadata management and symbol discovery:
- List symbols in storage
- Get metadata for symbols
- Session assignment and completion
"""

from __future__ import annotations

import json
from typing import Any, Literal

import polars as pl
import structlog

from ml4t.data.storage.backend import normalize_storage_metadata
from ml4t.data.storage.keys import storage_key_path
from ml4t.specs import UNSET_CALENDAR, CalendarUnset

logger = structlog.get_logger()


class MetadataManager:
    """Manages metadata and discovery operations.

    This class handles symbol discovery, metadata retrieval, and
    session management for stored data.

    Attributes:
        storage: Storage backend instance

    Example:
        >>> metadata_mgr = MetadataManager(storage)
        >>> symbols = metadata_mgr.list_symbols(provider="yahoo")
        >>> metadata = metadata_mgr.get_metadata("AAPL")
    """

    def __init__(self, storage: Any) -> None:
        """Initialize MetadataManager.

        Args:
            storage: Storage backend instance
        """
        self.storage = storage

    def list_symbols(
        self,
        provider: str | None = None,
        asset_class: str | None = None,
        exchange: str | None = None,
        bar_type: str | None = None,
    ) -> list[str]:
        """List all symbols in storage, optionally filtered by metadata.

        Args:
            provider: Filter by provider (e.g., "yahoo", "databento")
            asset_class: Filter by asset class (e.g., "equities", "crypto")
            exchange: Filter by exchange (e.g., "CME", "NYSE")
            bar_type: Filter by bar type (e.g., "time", "volume")

        Returns:
            List of symbol names matching the filters
        """
        if not self.storage:
            raise ValueError("Storage not configured")

        symbols = []
        for key in self.storage.list_keys():
            try:
                metadata = self.get_metadata_for_key(key)
                if metadata is None:
                    continue

                # Apply filters
                if provider and metadata.get("provider") != provider:
                    continue
                if asset_class and metadata.get("asset_class") != asset_class:
                    continue
                if exchange and metadata.get("exchange") != exchange:
                    continue
                if bar_type and metadata.get("bar_type") != bar_type:
                    continue

                symbol = metadata.get("symbol")
                if symbol:
                    symbols.append(symbol)

            except Exception as e:
                logger.warning(f"Failed to read metadata for {key}: {e}")
                continue

        return sorted(set(symbols))

    def get_metadata(
        self,
        symbol: str,
        asset_class: str = "equities",
        frequency: str = "daily",
    ) -> dict[str, Any] | None:
        """Get metadata for a specific symbol.

        Args:
            symbol: Symbol identifier
            asset_class: Asset class
            frequency: Data frequency

        Returns:
            Metadata dict or None if not found
        """
        if not self.storage:
            raise ValueError("Storage not configured")

        key = f"{asset_class}/{frequency}/{symbol}"
        return self.get_metadata_for_key(key)

    def get_metadata_for_key(self, key: str) -> dict[str, Any] | None:
        """Get metadata for a storage key.

        Args:
            key: Storage key

        Returns:
            Metadata dict or None if not found
        """
        try:
            get_metadata = getattr(self.storage, "get_metadata", None)
            metadata = get_metadata(key) if callable(get_metadata) else None
            if not metadata and hasattr(self.storage, "metadata_dir"):
                metadata_file = storage_key_path(self.storage.metadata_dir, key, ".json")
                if metadata_file.exists():
                    with open(metadata_file, encoding="utf-8") as metadata_handle:
                        metadata = json.load(metadata_handle)
            return normalize_storage_metadata(metadata, key)
        except Exception as e:
            logger.warning(f"Failed to read metadata for {key}: {e}")
            return None

    def get_all_metadata(self) -> dict[str, dict[str, Any]]:
        """Get metadata for all stored symbols.

        Returns:
            Dictionary mapping storage keys to metadata
        """
        if not self.storage:
            raise ValueError("Storage not configured")

        all_metadata = {}
        for key in self.storage.list_keys():
            metadata = self.get_metadata_for_key(key)
            if metadata:
                all_metadata[key] = metadata

        return all_metadata

    def assign_sessions(
        self,
        df: pl.DataFrame,
        exchange: str | None = None,
        calendar: str | None | CalendarUnset = UNSET_CALENDAR,
        bar_frequency: Literal["auto", "daily", "intraday"] = "auto",
        *,
        feed_spec: Any | None = None,
    ) -> pl.DataFrame:
        """Assign session_date column to DataFrame based on exchange calendar.

        Args:
            df: DataFrame with timestamp column
            exchange: Exchange code (e.g., "CME", "NYSE")
            calendar: Calendar name override
            bar_frequency: Daily period-label handling, intraday containment, or inference

        Returns:
            DataFrame with session_date column added

        Raises:
            ValueError: If neither exchange nor calendar provided
        """
        from ml4t.data.sessions import SessionAssigner

        assigner = SessionAssigner(calendar, exchange=exchange, feed_spec=feed_spec)

        return assigner.assign_sessions(df, bar_frequency=bar_frequency)

    def complete_sessions(
        self,
        df: pl.DataFrame,
        exchange: str | None = None,
        calendar: str | None | CalendarUnset = UNSET_CALENDAR,
        fill_gaps: bool = True,
        fill_method: str = "none",
        zero_volume: bool = False,
        *,
        feed_spec: Any | None = None,
    ) -> pl.DataFrame:
        """Expand the minute grid with null placeholders, never synthetic market values.

        Args:
            df: DataFrame with timestamp column
            exchange: Exchange code
            calendar: Calendar name override
            fill_gaps: Legacy name for grid expansion; False performs assignment only.
            fill_method: Only "none" is supported; missing market values remain null.
            zero_volume: Must be False; a missing observation has no known volume.

        Returns:
            DataFrame with complete minute sessions and an is_imputed column. Input must
            contain one symbol, unique minute-aligned timestamps, and no observations
            outside the inferred completion range.

        Raises:
            ValueError: If neither exchange nor calendar provided
        """
        from ml4t.data.sessions import SessionCompleter

        if not fill_gaps:
            from ml4t.data.sessions import SessionAssigner

            assigner = SessionAssigner(calendar, exchange=exchange, feed_spec=feed_spec)
            return assigner.assign_sessions(df)

        completer = SessionCompleter(calendar, exchange=exchange, feed_spec=feed_spec)
        return completer.complete_sessions(
            df,
            fill_method=fill_method,
            zero_volume=zero_volume,
        )

    def get_date_range(
        self,
        symbol: str,
        asset_class: str = "equities",
        frequency: str = "daily",
    ) -> tuple[str, str] | None:
        """Get the date range for a symbol in storage.

        Args:
            symbol: Symbol identifier
            asset_class: Asset class
            frequency: Data frequency

        Returns:
            Tuple of (start_date, end_date) or None if not found
        """
        metadata = self.get_metadata(symbol, asset_class, frequency)
        if metadata is None:
            return None

        data_range = metadata.get("data_range", {})
        start = data_range.get("start")
        end = data_range.get("end")

        if start and end:
            return (start, end)
        return None

    def get_row_count(
        self,
        symbol: str,
        asset_class: str = "equities",
        frequency: str = "daily",
    ) -> int | None:
        """Get the row count for a symbol in storage.

        Args:
            symbol: Symbol identifier
            asset_class: Asset class
            frequency: Data frequency

        Returns:
            Number of rows or None if not found
        """
        if not self.storage:
            return None

        key = f"{asset_class}/{frequency}/{symbol}"
        if not self.storage.exists(key):
            return None

        try:
            lazy_df = self.storage.read(key)
            return lazy_df.select(pl.len()).collect().item()
        except Exception as e:
            logger.warning(f"Failed to get row count for {key}: {e}")
            return None
