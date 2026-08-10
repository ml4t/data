"""ML4T Data storage module.

Provides configurable storage backends with Hive partitioning for
high-performance time-series data management.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .backend import StorageBackend
from .config import CompressionType, PartitionGranularity, StorageConfig, StorageStrategy
from .data_profile import (
    ColumnProfile,
    DatasetProfile,
    ProfileMixin,
    generate_profile,
    get_profile_path,
    load_profile,
    save_profile,
)
from .flat import FlatStorage
from .hive import HiveStorage, Partition
from .legacy_migration import (
    LegacyStorageEntry,
    LegacyStorageMigration,
    LegacyStorageMigrationError,
    find_legacy_storage_entries,
    migrate_legacy_storage,
)


def create_storage(
    base_path: StorageConfig | str | Path,
    strategy: str | StorageStrategy | None = None,
    **kwargs: Any,
) -> StorageBackend:
    """Create a storage backend with the specified strategy.

    Args:
        base_path: Base directory or a complete storage configuration
        strategy: Storage strategy ("hive" or "flat") when passing a directory
        **kwargs: Additional configuration options

    Returns:
        Configured storage backend

    Example:
        >>> storage = create_storage("/data", strategy="hive")
        >>> storage.write(df.lazy(), "BTC-USD")
    """
    if isinstance(base_path, StorageConfig):
        if strategy is not None or kwargs:
            raise TypeError("Do not pass strategy or options with a StorageConfig instance")
        config = base_path
    else:
        config = StorageConfig(
            base_path=Path(base_path),
            strategy=strategy or StorageStrategy.HIVE,
            **kwargs,
        )

    if config.strategy == StorageStrategy.HIVE:
        return HiveStorage(config)
    if config.strategy == StorageStrategy.FLAT:
        return FlatStorage(config)
    raise ValueError(f"Unknown storage strategy: {config.strategy}")


__all__ = [
    "ColumnProfile",
    "CompressionType",
    "DatasetProfile",
    "FlatStorage",
    "HiveStorage",
    "LegacyStorageEntry",
    "LegacyStorageMigration",
    "LegacyStorageMigrationError",
    "ProfileMixin",
    "Partition",
    "PartitionGranularity",
    "StorageBackend",
    "StorageConfig",
    "StorageStrategy",
    "create_storage",
    "find_legacy_storage_entries",
    "generate_profile",
    "get_profile_path",
    "load_profile",
    "migrate_legacy_storage",
    "save_profile",
]
