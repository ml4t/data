"""Configuration management for QLDM."""

from __future__ import annotations

import os
from enum import Enum
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from ml4t.data.paths import ML4T_DATA_ENV_VAR, default_ml4t_data_path, resolve_ml4t_data_path


class StorageBackendType(str, Enum):
    """Supported storage backend types."""

    FILESYSTEM = "filesystem"
    S3 = "s3"
    MEMORY = "memory"


class CompressionType(str, Enum):
    """Supported compression types."""

    NONE = "none"
    SNAPPY = "snappy"
    GZIP = "gzip"
    LZ4 = "lz4"


class StorageConfig(BaseModel):
    """Storage configuration."""

    backend: StorageBackendType = StorageBackendType.FILESYSTEM
    compression: CompressionType = CompressionType.SNAPPY

    @field_validator("backend", mode="before")
    @classmethod
    def validate_backend(cls, v: str) -> StorageBackendType:
        """Validate storage backend."""
        if isinstance(v, str):
            try:
                return StorageBackendType(v.lower())
            except ValueError as e:
                raise ValueError(f"Invalid storage backend: {v}") from e
        return v


class RetryConfig(BaseModel):
    """Retry configuration."""

    max_attempts: int = Field(default=3, ge=1)
    initial_wait: float = Field(default=1.0, gt=0)
    max_wait: float = Field(default=60.0, gt=0)
    exponential_base: float = Field(default=2.0, gt=1)


class CacheConfig(BaseModel):
    """Cache configuration."""

    enabled: bool = True
    ttl: int = Field(default=3600, ge=0)  # seconds
    max_size: int = Field(default=1000, ge=0)  # number of items


class Config(BaseModel):
    """Main configuration for QLDM."""

    data_root: Path = Field(
        default_factory=lambda: resolve_ml4t_data_path(".", default_path=default_ml4t_data_path())
    )
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    storage: StorageConfig = Field(default_factory=StorageConfig)
    retry: RetryConfig = Field(default_factory=RetryConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    # Add validation attribute for backward compatibility
    validation: dict[str, Any] = Field(default_factory=lambda: {"enabled": True, "strict": False})

    # Add base_dir as an alias for data_root for backward compatibility
    @property
    def base_dir(self) -> Path:
        """Alias for data_root for backward compatibility."""
        return self.data_root

    model_config = {"validate_assignment": True}

    def __init__(self, **data: Any) -> None:
        """Initialize config with environment variables."""
        if ML4T_DATA_ENV_VAR in os.environ:
            data["data_root"] = Path(os.environ[ML4T_DATA_ENV_VAR]).expanduser()
        if "QLDM_LOG_LEVEL" in os.environ:
            data["log_level"] = os.environ["QLDM_LOG_LEVEL"]

        super().__init__(**data)

    @field_validator("data_root", mode="before")
    @classmethod
    def validate_data_root(cls, v: str | Path) -> Path:
        """Validate and convert data root."""
        if isinstance(v, str):
            return Path(v)
        return v
