"""Configuration management for ML4T data storage and runtime defaults."""

from __future__ import annotations

import os
import warnings
from enum import Enum
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

PREFERRED_DATA_ENV_VAR = "ML4T_DATA_PATH"
LEGACY_DATA_ENV_VARS = ("ML4T_DATA_DIR", "QLDM_DATA_ROOT")
DEFAULT_RELATIVE_DATA_ROOT = Path("data")


def expand_path(path: str | Path) -> Path:
    """Expand and normalize a filesystem path."""
    return Path(path).expanduser().resolve()


def resolve_data_root(data_root: str | Path | None = None) -> Path:
    """Resolve the library data root.

    Resolution order:
    1. Explicit `data_root`
    2. `ML4T_DATA_PATH`
    3. Legacy env vars (`ML4T_DATA_DIR`, `QLDM_DATA_ROOT`) with warning
    4. Project-local `./data`
    """
    if data_root is not None:
        return expand_path(data_root)

    preferred = os.getenv(PREFERRED_DATA_ENV_VAR)
    if preferred:
        return expand_path(preferred)

    for env_var in LEGACY_DATA_ENV_VARS:
        legacy = os.getenv(env_var)
        if legacy:
            warnings.warn(
                f"{env_var} is deprecated; use {PREFERRED_DATA_ENV_VAR} instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            return expand_path(legacy)

    return (Path.cwd() / DEFAULT_RELATIVE_DATA_ROOT).resolve()


def resolve_data_path(*parts: str, data_root: str | Path | None = None) -> Path:
    """Resolve a subpath beneath the configured ML4T data root."""
    return resolve_data_root(data_root).joinpath(*parts)


def resolve_storage_path(
    path: str | Path | None,
    *default_parts: str,
    data_root: str | Path | None = None,
) -> Path:
    """Resolve an explicit path or fall back to a subpath under the data root."""
    if path is not None:
        return expand_path(path)
    return resolve_data_path(*default_parts, data_root=data_root)


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

    data_root: Path = Field(default_factory=resolve_data_root)
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
        # Override with environment variables
        if PREFERRED_DATA_ENV_VAR in os.environ:
            data["data_root"] = os.environ[PREFERRED_DATA_ENV_VAR]
        elif "QLDM_DATA_ROOT" in os.environ:
            data["data_root"] = os.environ["QLDM_DATA_ROOT"]
        if "QLDM_LOG_LEVEL" in os.environ:
            data["log_level"] = os.environ["QLDM_LOG_LEVEL"]

        super().__init__(**data)

    @field_validator("data_root", mode="before")
    @classmethod
    def validate_data_root(cls, v: str | Path) -> Path:
        """Validate and convert data root."""
        return expand_path(v)
