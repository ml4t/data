"""Storage backend interface and implementations for ML4T Data.

This module provides the abstract interface for storage backends and concrete
implementations for Hive partitioned and flat file storage strategies.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import polars as pl

# Type alias for partition granularity
PartitionGranularityType = Literal["year", "month", "day", "hour"]


@dataclass
class StorageConfig:
    """Configuration for storage backends.

    Attributes:
        base_path: Base directory for storage.
        strategy: Storage strategy ("hive" or "flat").
        compression: Compression type for Parquet files.
        partition_granularity: Time-based partition granularity for Hive storage.
            - "year": Best for daily data (~252 rows/partition for stocks)
            - "month": Best for hourly data (~720 rows/partition)
            - "day": Best for minute data (~1,440 rows/partition)
            - "hour": Best for second/tick data (~3,600 rows/partition)
        partition_cols: Deprecated. Use partition_granularity instead.
        atomic_writes: Use atomic writes with temp file rename.
        enable_locking: Enable file locking for concurrent access.
        metadata_tracking: Track metadata in manifest files.
    """

    base_path: Path
    strategy: str = "hive"  # "hive" or "flat"
    compression: str | None = "zstd"  # "zstd", "lz4", "snappy", None
    partition_granularity: PartitionGranularityType = "month"
    partition_cols: list[str] | None = None  # Deprecated, kept for backward compat
    atomic_writes: bool = True
    enable_locking: bool = True
    metadata_tracking: bool = True
    generate_profile: bool = True  # Generate column-level statistics on write

    def __post_init__(self) -> None:
        """Validate and set defaults."""
        self.base_path = Path(self.base_path)
        # Set partition_cols based on granularity for backward compatibility
        if self.partition_cols is None:
            if self.strategy == "hive":
                self.partition_cols = self._get_partition_cols_from_granularity()
            else:
                self.partition_cols = []

    def _get_partition_cols_from_granularity(self) -> list[str]:
        """Get partition columns based on granularity setting."""
        granularity_to_cols = {
            "year": ["year"],
            "month": ["year", "month"],
            "day": ["year", "month", "day"],
            "hour": ["year", "month", "day", "hour"],
        }
        return granularity_to_cols.get(self.partition_granularity, ["year", "month"])


class StorageBackend(ABC):
    """Abstract base class for storage backends."""

    def __init__(self, config: StorageConfig) -> None:
        """Initialize storage backend with configuration.

        Args:
            config: Storage configuration
        """
        self.config = config
        self.base_path = config.base_path
        self.base_path.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def write(self, data: pl.LazyFrame, key: str, metadata: dict[str, Any] | None = None) -> Path:
        """Write data to storage.

        Args:
            data: Polars LazyFrame to write
            key: Storage key (e.g., "BTC-USD", "SPY")
            metadata: Optional metadata to store alongside data

        Returns:
            Path to written file
        """

    @abstractmethod
    def read(
        self,
        key: str,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        columns: list[str] | None = None,
    ) -> pl.LazyFrame:
        """Read data from storage.

        Args:
            key: Storage key
            start_date: Optional start date filter
            end_date: Optional end date filter
            columns: Optional columns to select

        Returns:
            Polars LazyFrame with requested data
        """

    @abstractmethod
    def list_keys(self) -> list[str]:
        """List all available keys in storage.

        Returns:
            List of storage keys
        """

    @abstractmethod
    def exists(self, key: str) -> bool:
        """Check if a key exists in storage.

        Args:
            key: Storage key to check

        Returns:
            True if key exists
        """

    @abstractmethod
    def delete(self, key: str) -> bool:
        """Delete data for a key.

        Args:
            key: Storage key to delete

        Returns:
            True if deletion was successful
        """

    @abstractmethod
    def get_metadata(self, key: str) -> dict[str, Any] | None:
        """Get metadata for a key.

        Args:
            key: Storage key

        Returns:
            Metadata dictionary if exists, None otherwise
        """

    def _ensure_lazy(self, data: pl.DataFrame | pl.LazyFrame) -> pl.LazyFrame:
        """Ensure data is a LazyFrame for efficient processing.

        Args:
            data: DataFrame or LazyFrame

        Returns:
            LazyFrame
        """
        if isinstance(data, pl.DataFrame):
            return data.lazy()
        return data
