"""Configuration models for ML4T Data using Pydantic."""

from __future__ import annotations

from datetime import datetime
from datetime import time as datetime_time
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    SerializerFunctionWrapHandler,
    field_validator,
    model_serializer,
    model_validator,
)

from ml4t.data.assets.asset_class import AssetClass
from ml4t.data.core.config import resolve_data_root
from ml4t.data.core.models import Frequency
from ml4t.data.providers.registry import get_provider_spec
from ml4t.data.storage import config as _storage_config

CompressionType = _storage_config.CompressionType
PartitionGranularity = _storage_config.PartitionGranularity
StorageConfig = _storage_config.StorageConfig
StorageStrategy = _storage_config.StorageStrategy


def _is_environment_reference(value: str | None) -> bool:
    """Return whether a value is a complete environment-variable reference."""
    return value is not None and value.startswith("${") and value.endswith("}")


def _exclude_resolved_credential(value: str | None) -> bool:
    """Exclude runtime credentials while retaining environment references."""
    return value is not None and not _is_environment_reference(value)


class ProviderType(StrEnum):
    """Provider type enumeration."""

    YAHOO = "yahoo"
    ALPACA = "alpaca"
    TIINGO = "tiingo"
    FINNHUB = "finnhub"
    EODHD = "eodhd"
    FRED = "fred"
    FXMACRODATA = "fxmacrodata"
    AQR = "aqr"
    FAMA_FRENCH = "fama_french"
    KALSHI = "kalshi"
    POLYMARKET = "polymarket"
    COINGECKO = "coingecko"
    BINANCE = "binance"
    BINANCE_PUBLIC = "binance_public"
    OKX = "okx"
    CRYPTOCOMPARE = "cryptocompare"
    DATABENTO = "databento"
    MASSIVE = "massive"
    OANDA = "oanda"
    POLYGON = "polygon"
    TWELVE_DATA = "twelve_data"
    NASDAQ_ITCH = "nasdaq_itch"
    WIKI_PRICES = "wiki_prices"
    SYNTHETIC = "synthetic"
    LEARNED_SYNTHETIC = "learned_synthetic"
    MOCK = "mock"


class ScheduleType(StrEnum):
    """Schedule type enumeration."""

    CRON = "cron"
    INTERVAL = "interval"
    DAILY = "daily"
    WEEKLY = "weekly"
    MARKET_HOURS = "market_hours"


class RateLimitConfig(BaseModel):
    """Rate limiting configuration for API providers."""

    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    requests_per_second: float = Field(
        default=10.0, gt=0, description="Maximum requests per second"
    )
    burst_size: int = Field(default=1, ge=1, description="Burst size for rate limiter")


class ProviderConfig(BaseModel):
    """Base provider configuration."""

    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    _api_key_reference: str | None = PrivateAttr(default=None)
    _api_secret_reference: str | None = PrivateAttr(default=None)

    name: str = Field(description="Provider name")
    type: ProviderType = Field(description="Provider type")
    enabled: bool = Field(default=True, description="Whether provider is enabled")
    api_key: str | None = Field(
        default=None,
        description="API key (can use ${ENV_VAR})",
        repr=False,
        exclude_if=_exclude_resolved_credential,
    )
    api_secret: str | None = Field(
        default=None,
        description="API secret (can use ${ENV_VAR})",
        repr=False,
        exclude_if=_exclude_resolved_credential,
    )
    rate_limit: RateLimitConfig = Field(
        default_factory=RateLimitConfig, description="Rate limiting configuration"
    )
    extra: dict[str, Any] = Field(default_factory=dict, description="Provider-specific settings")

    @field_validator("type", mode="before")
    @classmethod
    def normalize_type(cls, value: Any) -> Any:
        """Accept enum member names without making configuration case-sensitive."""
        return value.lower() if isinstance(value, str) else value

    @model_validator(mode="after")
    def validate_extra_fields(self) -> ProviderConfig:
        """Keep structural and credential fields out of provider-specific settings."""
        reserved = {"name", "type", "enabled", "api_key", "api_secret"}
        conflicts = reserved.intersection(self.extra)
        if conflicts:
            names = ", ".join(sorted(conflicts))
            raise ValueError(f"Provider extra contains reserved fields: {names}")
        return self

    @model_serializer(mode="wrap")
    def _serialize_with_credential_references(
        self, handler: SerializerFunctionWrapHandler
    ) -> dict[str, Any]:
        """Serialize environment references without serializing resolved credentials."""
        data = handler(self)
        api_key_reference = (
            self.api_key if _is_environment_reference(self.api_key) else self._api_key_reference
        )
        api_secret_reference = (
            self.api_secret
            if _is_environment_reference(self.api_secret)
            else self._api_secret_reference
        )
        if api_key_reference is not None:
            data["api_key"] = api_key_reference
        if api_secret_reference is not None:
            data["api_secret"] = api_secret_reference
        return data

    def _set_credential_reference(self, field: str, reference: str) -> None:
        """Retain a raw environment reference for safe configuration serialization."""
        if field == "api_key":
            self._api_key_reference = reference
        elif field == "api_secret":
            self._api_secret_reference = reference
        else:
            raise ValueError(f"Unsupported credential field: {field}")

    @field_validator("rate_limit", mode="before")
    @classmethod
    def convert_rate_limit(cls, v):
        """Convert float rate_limit to RateLimitConfig for backward compatibility."""
        if isinstance(v, int | float):
            return RateLimitConfig(requests_per_second=float(v))
        if isinstance(v, tuple) and len(v) == 2:
            calls, period = v
            return RateLimitConfig(
                requests_per_second=float(calls) / float(period),
                burst_size=int(calls),
            )
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

    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    name: str = Field(description="Universe name")
    symbols: list[str] = Field(default_factory=list, description="Symbol list")
    file: Path | None = Field(default=None, description="File with symbols (one per line)")
    provider: str | None = Field(default=None, description="Preferred provider")
    asset_class: AssetClass = Field(default=AssetClass.EQUITY, description="Asset class")

    @field_validator("asset_class", mode="before")
    @classmethod
    def normalize_asset_class(cls, value: Any) -> Any:
        """Normalize legacy enum spellings at the configuration boundary."""
        if not isinstance(value, str):
            return value
        return {"fx": "forex"}.get(value.lower(), value.lower())

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

    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    name: str = Field(description="Dataset name")
    universe: str | None = Field(default=None, description="Symbol universe name")
    symbols: list[str] = Field(default_factory=list, description="Direct symbol list (legacy)")
    symbols_file: Path | None = Field(default=None, description="File containing symbols")
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
    lookback_days: int = Field(default=7, ge=0, description="Days to revisit during updates")
    fill_gaps: bool = Field(default=True, description="Fill gaps during updates")
    initial_load_days: int = Field(default=365, ge=1, description="Initial history length")
    extra: dict[str, Any] = Field(default_factory=dict, description="Dataset-specific settings")

    @model_validator(mode="before")
    @classmethod
    def normalize_beta_dates(cls, value: Any) -> Any:
        """Normalize beta start/end field names."""
        if not isinstance(value, dict):
            return value
        data = dict(value)
        if "start" in data and "start_date" not in data:
            data["start_date"] = data.pop("start")
        if "end" in data and "end_date" not in data:
            data["end_date"] = data.pop("end")
        return data

    @field_validator("symbols", mode="before")
    @classmethod
    def expand_symbols(cls, v):
        """Convert single symbol string to list for convenience."""
        if isinstance(v, str):
            return [v]
        return v

    @field_validator("frequency", "asset_class", mode="before")
    @classmethod
    def normalize_enums(cls, value: Any) -> Any:
        """Accept enum member names without making configuration case-sensitive."""
        return value.lower() if isinstance(value, str) else value

    @model_validator(mode="after")
    def validate_universe_or_symbols(self) -> DatasetConfig:
        """Ensure either universe or symbols is provided with content."""
        # Reject empty symbols list (no actual symbols)
        if (not self.universe and not self.symbols and not self.symbols_file) or (
            self.symbols is not None
            and len(self.symbols) == 0
            and not self.universe
            and not self.symbols_file
        ):
            raise ValueError(f"Dataset {self.name} has no symbols and no universe")
        return self


class ScheduleConfig(BaseModel):
    """Schedule configuration."""

    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

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

    @field_validator("type", mode="before")
    @classmethod
    def normalize_type(cls, value: Any) -> Any:
        """Accept enum member names without making configuration case-sensitive."""
        return value.lower() if isinstance(value, str) else value

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

    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

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


class DataConfig(BaseModel):
    """Main ML4T Data configuration with environment variable support."""

    # Config metadata
    version: str = Field(default="1.0", description="Configuration file version")
    base_dir: Path = Field(
        default_factory=resolve_data_root, description="Base directory for project"
    )

    storage: StorageConfig = Field(default_factory=StorageConfig, description="Storage settings")

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
    routing: dict[str, Any] = Field(default_factory=dict, description="Provider routing settings")

    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    @model_validator(mode="before")
    @classmethod
    def normalize_beta_mappings(cls, value: Any) -> Any:
        """Normalize beta mapping syntax into the canonical named-object lists."""
        if not isinstance(value, dict):
            return value
        data = dict(value)

        def named_list(field: str, *, provider: bool = False) -> None:
            entries = data.get(field)
            if not isinstance(entries, dict):
                return
            normalized = []
            for name, entry in entries.items():
                item = dict(entry or {})
                item.setdefault("name", name)
                if provider:
                    item.setdefault("type", name)
                normalized.append(item)
            data[field] = normalized

        named_list("providers", provider=True)
        named_list("universes")
        named_list("datasets")
        named_list("workflows")

        api_keys = data.pop("api_keys", None)
        if isinstance(api_keys, dict):
            providers = data.setdefault("providers", [])
            by_name = {item["name"]: item for item in providers if isinstance(item, dict)}
            for name, api_key in api_keys.items():
                provider_config = by_name.get(name)
                if provider_config is None:
                    provider_config = {"name": name, "type": name}
                    providers.append(provider_config)
                provider_config.setdefault("api_key", api_key)

        storage = data.get("storage")
        if isinstance(storage, dict) and "path" in storage:
            data["storage"] = {**storage, "base_path": storage["path"]}
            data["storage"].pop("path")
        return data

    def to_runtime_dict(self) -> dict[str, Any]:
        """Return the validated configuration shape consumed by DataManager."""
        providers: dict[str, dict[str, Any]] = {}
        for provider in self.providers:
            spec = get_provider_spec(provider.type.value)
            provider_config: dict[str, Any] = {
                "type": provider.type.value,
                "enabled": provider.enabled,
                **provider.extra,
            }
            for requirement in spec.credentials:
                value = getattr(provider, requirement.config_field, None)
                if value is not None:
                    provider_config[requirement.config_field] = value
            if (
                spec.optional_credential_environment
                and provider.api_key is not None
                and not _is_environment_reference(provider.api_key)
            ):
                provider_config["api_key"] = provider.api_key
            if "rate_limit" in provider.model_fields_set:
                rate = provider.rate_limit
                provider_config["rate_limit"] = (
                    rate.burst_size,
                    rate.burst_size / rate.requests_per_second,
                )
            providers[provider.name] = provider_config

        defaults = dict(self.defaults)
        defaults.setdefault("output_format", "polars")
        defaults.setdefault("frequency", "daily")
        defaults.setdefault("timezone", "UTC")
        return {
            "providers": providers,
            "routing": self.routing,
            "defaults": defaults,
        }

    @model_validator(mode="after")
    def validate_provider_credentials(self) -> DataConfig:
        """Reject credentials that the selected provider implementation cannot use."""
        for provider in self.providers:
            spec = get_provider_spec(provider.type.value)
            supported = {requirement.config_field for requirement in spec.credentials}
            if spec.optional_credential_environment:
                supported.add("api_key")
            for field in ("api_key", "api_secret"):
                if getattr(provider, field) is not None and field not in supported:
                    raise ValueError(
                        f"Provider '{provider.name}' of type '{provider.type.value}' "
                        f"does not accept {field}"
                    )
        return self

    @classmethod
    def from_yaml(cls, path: str | Path) -> DataConfig:
        """Load configuration from YAML file with environment variable support."""
        from ml4t.data.config.loader import ConfigLoader

        return ConfigLoader(Path(path)).load()

    def to_yaml(self, path: str | Path) -> None:
        """Save configuration to YAML file."""
        from ml4t.data.config._serialization import write_yaml

        path = Path(path)

        # Exclude None values for cleaner YAML, mode="json" for compatibility
        data = self.model_dump(exclude_none=True, exclude_defaults=False, mode="json")
        write_yaml(path, data)

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

            if dataset.universe is not None and not self.get_universe(dataset.universe):
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
