"""Configuration models for ML4T Data using Pydantic."""

from __future__ import annotations

from datetime import datetime
from datetime import time as datetime_time
from enum import Enum
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Import consolidated enums from core.models
from ml4t.data.core.models import AssetClass, Frequency


# Storage configuration enums
class StorageStrategy(str, Enum):
    """Storage backend strategy options."""

    HIVE = "hive"
    FLAT = "flat"


class CompressionType(str, Enum):
    """Supported compression types for Parquet."""

    ZSTD = "zstd"
    LZ4 = "lz4"
    SNAPPY = "snappy"
    NONE = "none"


class PartitionGranularity(str, Enum):
    """Partition granularity for Hive storage.

    Choose based on data frequency:
    - YEAR: Best for daily data (~252 rows/partition for stocks)
    - MONTH: Best for hourly data (~720 rows/partition)
    - DAY: Best for minute data (~1,440 rows/partition)
    - HOUR: Best for second/tick data (~3,600 rows/partition)
    """

    YEAR = "year"
    MONTH = "month"
    DAY = "day"
    HOUR = "hour"


class ProviderType(str, Enum):
    """Provider type enumeration."""

    YAHOO = "yahoo"
    BINANCE = "binance"
    CRYPTOCOMPARE = "cryptocompare"
    DATABENTO = "databento"
    OANDA = "oanda"
    POLYGON = "polygon"
    MOCK = "mock"


class ScheduleType(str, Enum):
    """Schedule type enumeration."""

    CRON = "cron"
    INTERVAL = "interval"
    DAILY = "daily"
    WEEKLY = "weekly"
    MARKET_HOURS = "market_hours"


class StorageConfig(BaseModel):
    """Storage backend configuration."""

    strategy: StorageStrategy = Field(
        default=StorageStrategy.HIVE,
        description="Storage strategy (hive for partitioned, flat for single file)",
    )
    base_path: Path = Field(default=Path("data"), description="Base directory for data storage")
    compression: CompressionType = Field(
        default=CompressionType.ZSTD, description="Compression type for Parquet files"
    )
    atomic_writes: bool = Field(default=True, description="Use atomic writes with temp file rename")
    enable_locking: bool = Field(
        default=True, description="Enable file locking for concurrent access safety"
    )
    lock_timeout: int = Field(default=30, ge=1, description="File lock timeout in seconds")
    metadata_tracking: bool = Field(default=True, description="Track metadata in manifest files")
    partition_granularity: PartitionGranularity = Field(
        default=PartitionGranularity.MONTH,
        description=(
            "Partition granularity for Hive storage. "
            "Use YEAR for daily, MONTH for hourly, DAY for minute, HOUR for second data."
        ),
    )
    partition_cols: list[str] = Field(
        default_factory=lambda: ["year", "month"],
        description="Deprecated: Use partition_granularity instead. Kept for backward compatibility.",
    )

    @field_validator("base_path")
    @classmethod
    def expand_path(cls, v: Path) -> Path:
        """Expand user home directory and make absolute."""
        return v.expanduser().resolve()

    @model_validator(mode="after")
    def validate_partitions(self) -> StorageConfig:
        """Ensure partition columns are valid for strategy."""
        if self.strategy == StorageStrategy.FLAT and self.partition_cols:
            self.partition_cols = []  # No partitions for flat storage
        return self


class RateLimitConfig(BaseModel):
    """Rate limiting configuration for API providers."""

    requests_per_second: float = Field(
        default=10.0, gt=0, description="Maximum requests per second"
    )
    burst_size: int = Field(default=1, ge=1, description="Burst size for rate limiter")
    retry_max_attempts: int = Field(default=3, ge=1, description="Maximum retry attempts")
    retry_backoff_factor: float = Field(
        default=2.0, gt=1.0, description="Exponential backoff factor"
    )
    circuit_breaker_threshold: int = Field(
        default=5, ge=1, description="Failures before circuit breaker opens"
    )
    circuit_breaker_timeout: int = Field(
        default=60, ge=1, description="Circuit breaker timeout in seconds"
    )


class ProviderConfig(BaseModel):
    """Base provider configuration."""

    name: str = Field(description="Provider name")
    type: ProviderType = Field(description="Provider type")
    enabled: bool = Field(default=True, description="Whether provider is enabled")
    api_key: str | None = Field(default=None, description="API key (can use ${ENV_VAR})")
    api_secret: str | None = Field(default=None, description="API secret (can use ${ENV_VAR})")
    rate_limit: RateLimitConfig = Field(
        default_factory=RateLimitConfig, description="Rate limiting configuration"
    )
    timeout: int = Field(default=30, description="Request timeout in seconds")
    retry_count: int = Field(default=3, description="Number of retries on failure")
    cache_enabled: bool = Field(default=True, description="Enable response caching")
    cache_ttl: int = Field(default=3600, ge=0, description="Cache TTL in seconds")
    extra: dict[str, Any] = Field(default_factory=dict, description="Provider-specific settings")

    @field_validator("rate_limit", mode="before")
    @classmethod
    def convert_rate_limit(cls, v):
        """Convert float rate_limit to RateLimitConfig for backward compatibility."""
        if isinstance(v, int | float):
            return RateLimitConfig(requests_per_second=float(v))
        return v

    @field_validator("api_key", "api_secret", mode="before")
    @classmethod
    def validate_secrets(cls, v):
        """Validate that secrets are not exposed in plain text."""
        if v and not v.startswith("${") and len(v) > 10:
            # Warn if it looks like a real API key not using env var
            import structlog

            logger = structlog.get_logger()
            logger.warning(
                "API credential appears to be in plain text. Consider using ${ENV_VAR} format"
            )
        return v


class SymbolUniverse(BaseModel):
    """Symbol universe definition for data collection."""

    name: str = Field(description="Universe name")
    symbols: list[str] = Field(default_factory=list, description="Symbol list")
    file: Path | None = Field(default=None, description="File with symbols (one per line)")
    provider: str | None = Field(default=None, description="Preferred provider")
    asset_class: AssetClass = Field(default=AssetClass.EQUITY, description="Asset class")

    @model_validator(mode="after")
    def load_from_file(self) -> SymbolUniverse:
        """Load symbols from file if specified."""
        if self.file and self.file.exists():
            with open(self.file) as f:
                file_symbols = [line.strip() for line in f if line.strip()]
                self.symbols.extend(file_symbols)
        # Remove duplicates while preserving order
        self.symbols = list(dict.fromkeys(self.symbols))
        return self


class DatasetConfig(BaseModel):
    """Dataset configuration."""

    name: str = Field(description="Dataset name")
    universe: str | None = Field(default=None, description="Symbol universe name")
    symbols: list[str] = Field(default_factory=list, description="Direct symbol list (legacy)")
    provider: str = Field(description="Provider name to use")
    frequency: Frequency = Field(default=Frequency.DAILY, description="Data frequency")
    asset_class: AssetClass = Field(default=AssetClass.EQUITY, description="Asset class")
    start_date: datetime | None = Field(default=None, description="Start date")
    end_date: datetime | None = Field(default=None, description="End date")
    update_mode: Literal["full", "incremental"] = Field(
        default="incremental", description="Update mode"
    )
    validation_enabled: bool = Field(default=True, description="Enable data validation")
    anomaly_detection: bool = Field(default=False, description="Enable anomaly detection")
    validation: dict[str, Any] = Field(default_factory=dict, description="Validation settings")
    storage: dict[str, Any] = Field(default_factory=dict, description="Storage settings")

    @field_validator("symbols", mode="before")
    @classmethod
    def expand_symbols(cls, v):
        """Convert single symbol string to list for convenience."""
        if isinstance(v, str):
            return [v]
        return v

    @model_validator(mode="after")
    def validate_universe_or_symbols(self) -> DatasetConfig:
        """Ensure either universe or symbols is provided with content."""
        # Reject empty symbols list (no actual symbols)
        if (not self.universe and not self.symbols) or (
            self.symbols is not None and len(self.symbols) == 0 and not self.universe
        ):
            raise ValueError(f"Dataset {self.name} has no symbols and no universe")
        return self


class ScheduleConfig(BaseModel):
    """Schedule configuration."""

    type: ScheduleType = Field(description="Schedule type")
    cron: str | None = Field(default=None, description="Cron expression (for cron type)")
    interval: int | None = Field(
        default=None, description="Interval in seconds (for interval type)"
    )
    time: datetime_time | None = Field(
        default=None, description="Time of day (for daily/weekly types)"
    )
    weekday: int | None = Field(default=None, description="Day of week 0-6 (for weekly type)")
    timezone: str = Field(default="UTC", description="Timezone for schedule")
    market_open_offset: int | None = Field(
        default=None, description="Minutes after market open (for market_hours type)"
    )
    market_close_offset: int | None = Field(
        default=None, description="Minutes before market close (for market_hours type)"
    )

    @model_validator(mode="after")
    def validate_schedule_fields(self):
        """Validate schedule fields based on schedule type."""
        if self.type == ScheduleType.CRON and not self.cron:
            raise ValueError("Cron expression required for cron schedule type")

        if self.type == ScheduleType.INTERVAL and (not self.interval or self.interval <= 0):
            raise ValueError("Positive interval required for interval schedule type")

        return self


class WorkflowConfig(BaseModel):
    """Workflow configuration."""

    name: str = Field(description="Workflow name")
    description: str | None = Field(default=None, description="Workflow description")
    datasets: list[str] = Field(description="List of dataset names to process")
    schedule: ScheduleConfig | None = Field(default=None, description="Workflow schedule")
    enabled: bool = Field(default=True, description="Whether workflow is enabled")
    pre_hooks: list[str] = Field(
        default_factory=list, description="Commands to run before workflow"
    )
    post_hooks: list[str] = Field(
        default_factory=list, description="Commands to run after workflow"
    )
    on_error: str = Field(default="stop", description="Error handling: stop, continue, or retry")
    notifications: dict[str, Any] = Field(default_factory=dict, description="Notification settings")


class DataConfig(BaseSettings):
    """Main ML4T Data configuration with environment variable support."""

    # Config metadata
    version: str = Field(default="1.0", description="Configuration file version")
    base_dir: Path = Field(default=Path("./data"), description="Base directory for project")

    # Storage configuration (supports both dict and StorageConfig)
    storage: StorageConfig | dict[str, Any] = Field(
        default_factory=dict, description="Storage settings"
    )

    # Defaults for datasets
    defaults: dict[str, Any] = Field(
        default_factory=dict, description="Default settings for datasets"
    )

    # Environment variables
    env: dict[str, Any] = Field(
        default_factory=dict, description="Environment variable definitions"
    )

    # Provider configurations
    providers: list[ProviderConfig] = Field(
        default_factory=list, description="Data provider configurations"
    )

    # Symbol universes
    universes: list[SymbolUniverse] = Field(
        default_factory=list, description="Symbol universe definitions"
    )

    # Datasets
    datasets: list[DatasetConfig] = Field(
        default_factory=list, description="Dataset configurations"
    )

    # Workflows
    workflows: list[WorkflowConfig] = Field(
        default_factory=list, description="Workflow configurations"
    )

    # Global settings
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(
        default="INFO", description="Logging level"
    )

    parallel_downloads: int = Field(
        default=4, ge=1, le=10, description="Max parallel provider requests"
    )

    default_start_date: datetime | None = Field(
        default=None, description="Default historical data start"
    )

    default_end_date: datetime | None = Field(
        default=None, description="Default historical data end"
    )

    # Validation settings
    validation: dict[str, Any] = Field(
        default_factory=dict, description="Global validation settings"
    )

    # Environment configuration for .env support
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        case_sensitive=False,
        extra="ignore",
    )

    @field_validator("storage", mode="before")
    @classmethod
    def convert_storage(cls, v):
        """Try to convert dict to StorageConfig, fallback to dict if incompatible."""
        if isinstance(v, dict):
            try:
                # Try to construct StorageConfig - if keys match, it will succeed
                return StorageConfig(**v)
            except Exception:
                # If it fails (e.g., unknown fields), keep as dict
                return v
        return v

    @classmethod
    def from_yaml(cls, path: str | Path) -> DataConfig:
        """Load configuration from YAML file with environment variable support."""
        import os
        import re

        import yaml

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")

        with open(path) as f:
            content = f.read()

        # Expand environment variables in format ${VAR_NAME}
        def expand_env_vars(match):
            var_name = match.group(1)
            return os.environ.get(var_name, match.group(0))

        content = re.sub(r"\$\{([^}]+)\}", expand_env_vars, content)
        data = yaml.safe_load(content)

        # Create instance with merged config
        return cls(**data)

    def to_yaml(self, path: str | Path) -> None:
        """Save configuration to YAML file."""
        import yaml

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Exclude None values for cleaner YAML, mode="json" for compatibility
        data = self.model_dump(exclude_none=True, exclude_defaults=False, mode="json")

        with open(path, "w") as f:
            yaml.safe_dump(data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

    def get_provider(self, name: str) -> ProviderConfig | None:
        """Get provider configuration by name."""
        for provider in self.providers:
            if provider.name == name:
                return provider
        return None

    def get_universe(self, name: str) -> SymbolUniverse | None:
        """Get symbol universe by name."""
        for universe in self.universes:
            if universe.name == name:
                return universe
        return None

    def get_dataset(self, name: str) -> DatasetConfig | None:
        """Get dataset configuration by name."""
        for dataset in self.datasets:
            if dataset.name == name:
                return dataset
        return None

    def get_workflow(self, name: str) -> WorkflowConfig | None:
        """Get workflow configuration by name."""
        for workflow in self.workflows:
            if workflow.name == name:
                return workflow
        return None

    def validate_config(self) -> list[str]:
        """Validate configuration and return list of issues."""
        issues = []

        # Check provider references in datasets
        for dataset in self.datasets:
            if not self.get_provider(dataset.provider):
                issues.append(
                    f"Dataset '{dataset.name}' references unknown provider '{dataset.provider}'"
                )

            if not self.get_universe(dataset.universe):
                issues.append(
                    f"Dataset '{dataset.name}' references unknown universe '{dataset.universe}'"
                )

        # Check dataset references in workflows
        for workflow in self.workflows:
            for dataset_name in workflow.datasets:
                if not self.get_dataset(dataset_name):
                    issues.append(
                        f"Workflow '{workflow.name}' references unknown dataset '{dataset_name}'"
                    )

        # Check for duplicate names
        provider_names = [p.name for p in self.providers]
        if len(provider_names) != len(set(provider_names)):
            issues.append("Duplicate provider names found")

        universe_names = [u.name for u in self.universes]
        if len(universe_names) != len(set(universe_names)):
            issues.append("Duplicate universe names found")

        return issues
