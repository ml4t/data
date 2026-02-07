"""ML4T Data storage module.

Provides configurable storage backends with Hive partitioning for
high-performance time-series data management.
"""

from __future__ import annotations

from pathlib import Path

from .backend import StorageBackend, StorageConfig
from .data_profile import (
    ColumnProfile,
    DatasetProfile,
    generate_profile,
    get_profile_path,
    load_profile,
    save_profile,
)
from .flat import FlatStorage
from .hive import HiveStorage


def create_storage(base_path: str | Path, strategy: str = "hive", **kwargs) -> StorageBackend:
    """Create a storage backend with the specified strategy.

    Args:
        base_path: Base directory for storage
        strategy: Storage strategy ("hive" or "flat")
        **kwargs: Additional configuration options

    Returns:
        Configured storage backend

    Example:
        >>> storage = create_storage("/data", strategy="hive")
        >>> storage.write(df.lazy(), "BTC-USD")
    """
    config = StorageConfig(base_path=Path(base_path), strategy=strategy, **kwargs)

    if strategy == "hive":
        return HiveStorage(config)
    if strategy == "flat":
        return FlatStorage(config)
    raise ValueError(f"Unknown storage strategy: {strategy}")


__all__ = [
    "ColumnProfile",
    "DatasetProfile",
    "FlatStorage",
    "HiveStorage",
    "StorageBackend",
    "StorageConfig",
    "create_storage",
    "generate_profile",
    "get_profile_path",
    "load_profile",
    "save_profile",
]
