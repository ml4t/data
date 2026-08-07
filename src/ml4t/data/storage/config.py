"""Canonical configuration for local Parquet storage."""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


def _default_base_path() -> Path:
    from ml4t.data.core.config import resolve_data_root

    return resolve_data_root()


class StorageStrategy(StrEnum):
    """Implemented storage layouts."""

    HIVE = "hive"
    FLAT = "flat"


class CompressionType(StrEnum):
    """Supported Parquet compression codecs."""

    ZSTD = "zstd"
    LZ4 = "lz4"
    SNAPPY = "snappy"
    GZIP = "gzip"


class PartitionGranularity(StrEnum):
    """Time components used by Hive storage partitions."""

    YEAR = "year"
    MONTH = "month"
    DAY = "day"
    HOUR = "hour"


_PARTITION_COLUMNS = {
    PartitionGranularity.YEAR: ["year"],
    PartitionGranularity.MONTH: ["year", "month"],
    PartitionGranularity.DAY: ["year", "month", "day"],
    PartitionGranularity.HOUR: ["year", "month", "day", "hour"],
}


class StorageConfig(BaseModel):
    """Configuration shared by every supported storage entry point."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    base_path: Path = Field(default_factory=_default_base_path)
    strategy: StorageStrategy = StorageStrategy.HIVE
    compression: CompressionType | None = CompressionType.ZSTD
    partition_granularity: PartitionGranularity = PartitionGranularity.MONTH
    lock_timeout: float = Field(default=30.0, gt=0)
    metadata_tracking: bool = True

    @model_validator(mode="before")
    @classmethod
    def migrate_beta_fields(cls, value: Any) -> Any:
        """Migrate beta fields only when their behavior is unambiguous."""
        if not isinstance(value, dict):
            return value
        data = dict(value)

        backend = data.pop("backend", None)
        if backend is not None:
            if str(backend).lower() != "filesystem":
                raise ValueError(f"Unsupported storage backend: {backend}")
            data.setdefault("strategy", StorageStrategy.HIVE)

        if "atomic_writes" in data:
            atomic_writes = data.pop("atomic_writes")
            if atomic_writes is not True:
                raise ValueError("Storage writes are always atomic")

        if "generate_profile" in data:
            raise ValueError(
                "generate_profile is not a storage write option; call generate_profile() explicitly"
            )

        partition_cols = data.pop("partition_cols", None)
        if partition_cols is not None:
            columns = list(partition_cols)
            matching = [
                granularity
                for granularity, expected_columns in _PARTITION_COLUMNS.items()
                if columns == expected_columns
            ]
            if not matching:
                raise ValueError(f"Unsupported partition_cols: {columns}")
            configured = data.get("partition_granularity")
            if configured is not None and str(configured).lower() != matching[0].value:
                raise ValueError("partition_cols conflicts with partition_granularity")
            data["partition_granularity"] = matching[0]

        return data

    @field_validator("base_path")
    @classmethod
    def normalize_base_path(cls, value: Path) -> Path:
        """Expand and resolve the storage root."""
        return value.expanduser().resolve()

    @field_validator("strategy", "partition_granularity", mode="before")
    @classmethod
    def normalize_enums(cls, value: Any) -> Any:
        """Accept case-insensitive string enum values."""
        return value.lower() if isinstance(value, str) else value

    @field_validator("compression", mode="before")
    @classmethod
    def normalize_compression(cls, value: Any) -> Any:
        """Accept case-insensitive codecs and explicit no-compression values."""
        if isinstance(value, str):
            return None if value.lower() in {"none", "null"} else value.lower()
        return value

    @property
    def partition_cols(self) -> list[str]:
        """Return the effective Hive partition columns for beta readers."""
        if self.strategy == StorageStrategy.FLAT:
            return []
        return list(_PARTITION_COLUMNS[self.partition_granularity])
