"""Storage protocols for incremental updates."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Protocol, runtime_checkable

import polars as pl


@runtime_checkable
class IncrementalStorageBackend(Protocol):
    """Protocol for storage backends that support incremental updates.

    This extends the basic StorageBackend with methods needed for
    incremental data updates and gap detection.
    """

    def get_latest_timestamp(self, symbol: str, provider: str) -> datetime | None:
        """Get the latest timestamp for a symbol from a provider.

        Args:
            symbol: Symbol identifier (e.g., "AAPL", "BTC-USD")
            provider: Data provider name (e.g., "yahoo", "polygon")

        Returns:
            Latest timestamp in the dataset, or None if no data exists
        """
        ...

    def save_chunk(
        self,
        data: pl.DataFrame,
        symbol: str,
        provider: str,
        start_time: datetime,
        end_time: datetime,
    ) -> Path:
        """Save an incremental data chunk.

        Chunks are stored separately for audit trail and debugging.

        Args:
            data: DataFrame with OHLCV data
            symbol: Symbol identifier
            provider: Data provider name
            start_time: Start time of this chunk
            end_time: End time of this chunk

        Returns:
            Path to the saved chunk file
        """
        ...

    def update_combined_file(
        self,
        data: pl.DataFrame,
        symbol: str,
        provider: str,
    ) -> int:
        """Update the main combined file with new data.

        This appends new records to the main dataset file,
        deduplicates, and sorts by timestamp.

        Args:
            data: New data to append
            symbol: Symbol identifier
            provider: Data provider name

        Returns:
            Number of new records added (after deduplication)
        """
        ...

    def get_combined_file_path(self, symbol: str, provider: str) -> Path:
        """Get path to the main combined data file.

        Args:
            symbol: Symbol identifier
            provider: Data provider name

        Returns:
            Path to combined data file
        """
        ...

    def read_data(
        self,
        symbol: str,
        provider: str,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
    ) -> pl.DataFrame:
        """Read data for a symbol with optional time filtering.

        Args:
            symbol: Symbol identifier
            provider: Data provider name
            start_time: Optional start time filter
            end_time: Optional end time filter

        Returns:
            DataFrame with filtered data
        """
        ...

    def update_metadata(
        self,
        symbol: str,
        provider: str,
        last_update: datetime,
        records_added: int,
        chunk_file: str,
    ) -> None:
        """Update metadata after incremental update.

        Args:
            symbol: Symbol identifier
            provider: Data provider name
            last_update: Timestamp of this update
            records_added: Number of records added
            chunk_file: Name of the chunk file saved

        Raises:
            KeyError: If no published dataset exists for the provider and symbol.
        """
        ...
