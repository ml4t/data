"""Configuration management for DataManager.

This module handles loading, merging, and applying configuration from multiple sources:
- YAML configuration files
- Environment variables
- Runtime parameters

Configuration hierarchy (lowest to highest priority):
    defaults < YAML file < environment variables < runtime parameters
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import structlog
import yaml

logger = structlog.get_logger()


class ConfigManager:
    """Manages configuration loading and merging from multiple sources.

    This class handles the configuration hierarchy for DataManager:
    1. Built-in defaults
    2. YAML configuration file
    3. Environment variables
    4. Runtime parameters

    Attributes:
        config: The merged configuration dictionary

    Example:
        >>> config_mgr = ConfigManager(config_path="ml4t-data.yaml")
        >>> config_mgr.apply_overrides(providers={"yahoo": {"timeout": 30}})
        >>> print(config_mgr.config["providers"]["yahoo"])
    """

    # Environment variable to provider config mapping. Later entries win when
    # two variables map to the same field, so the APCA_* aliases (Alpaca's own
    # SDK/CLI names) come before the ALPACA_* project-convention names.
    ENV_MAPPING = {
        "APCA_API_KEY_ID": ("alpaca", "api_key"),
        "APCA_API_SECRET_KEY": ("alpaca", "api_secret"),
        "ALPACA_API_KEY": ("alpaca", "api_key"),
        "ALPACA_API_SECRET": ("alpaca", "api_secret"),
        "CRYPTOCOMPARE_API_KEY": ("cryptocompare", "api_key"),
        "DATABENTO_API_KEY": ("databento", "api_key"),
        "POLYGON_API_KEY": ("massive", "api_key"),
        "MASSIVE_API_KEY": ("massive", "api_key"),
        "OANDA_API_KEY": ("oanda", "api_key"),
        "OANDA_ACCOUNT_ID": ("oanda", "account_id"),
    }

    # Default configuration
    DEFAULT_CONFIG: dict[str, Any] = {
        "providers": {},
        "routing": {"patterns": []},
        "defaults": {
            "output_format": "polars",
            "frequency": "daily",
            "timezone": "UTC",
        },
    }

    def __init__(
        self,
        config_path: str | None = None,
        output_format: str = "polars",
        providers: dict[str, dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize ConfigManager.

        Args:
            config_path: Path to YAML configuration file
            output_format: Default output format ('polars', 'pandas', 'lazy')
            providers: Provider-specific configuration overrides
            **kwargs: Additional configuration parameters
        """
        # Load base configuration
        self.config = self._load_config(config_path)

        # Apply configuration hierarchy
        self._apply_env_vars()
        self.apply_overrides(
            providers=providers,
            output_format=output_format,
            **kwargs,
        )

        # Validate output format
        actual_format = self.config.get("defaults", {}).get("output_format", output_format)
        if actual_format not in ["polars", "pandas", "lazy"]:
            raise ValueError(f"Invalid output format: {actual_format}")

        logger.debug(
            "ConfigManager initialized",
            config_path=config_path,
            output_format=actual_format,
        )

    @property
    def output_format(self) -> str:
        """Get the configured output format."""
        return self.config.get("defaults", {}).get("output_format", "polars")

    @property
    def default_frequency(self) -> str:
        """Get the default data frequency."""
        return self.config.get("defaults", {}).get("frequency", "daily")

    @property
    def timezone(self) -> str:
        """Get the configured timezone."""
        return self.config.get("defaults", {}).get("timezone", "UTC")

    def _load_config(self, config_path: str | None) -> dict[str, Any]:
        """Load configuration from YAML file.

        Args:
            config_path: Path to YAML configuration file

        Returns:
            Configuration dictionary with defaults applied
        """
        config = self._deep_copy_config(self.DEFAULT_CONFIG)

        if config_path and Path(config_path).exists():
            with open(config_path) as f:
                content = f.read()

                # Environment variable substitution: ${VAR_NAME}
                for env_var in re.findall(r"\$\{([^}]+)\}", content):
                    value = os.getenv(env_var, "")
                    content = content.replace(f"${{{env_var}}}", value)

                yaml_config = yaml.safe_load(content)
                if yaml_config:
                    config = self._merge_configs(config, yaml_config)
                    logger.info(f"Loaded configuration from {config_path}")

        return config

    def _deep_copy_config(self, config: dict[str, Any]) -> dict[str, Any]:
        """Create a deep copy of a configuration dictionary.

        Args:
            config: Configuration to copy

        Returns:
            Deep copy of the configuration
        """
        result = {}
        for key, value in config.items():
            if isinstance(value, dict):
                result[key] = self._deep_copy_config(value)
            elif isinstance(value, list):
                result[key] = value.copy()
            else:
                result[key] = value
        return result

    def _merge_configs(self, base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
        """Deep merge two configuration dictionaries.

        Args:
            base: Base configuration
            override: Override configuration (takes precedence)

        Returns:
            Merged configuration
        """
        result = base.copy()

        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_configs(result[key], value)
            else:
                result[key] = value

        return result

    def _apply_env_vars(self) -> None:
        """Apply environment variable configuration.

        Checks for known environment variables and applies them to the config.
        """
        for env_var, (provider, key) in self.ENV_MAPPING.items():
            value = os.getenv(env_var)
            if value:
                if provider not in self.config["providers"]:
                    self.config["providers"][provider] = {}
                self.config["providers"][provider][key] = value
                logger.debug(f"Applied {env_var} to {provider}.{key}")

    def apply_overrides(
        self,
        providers: dict[str, dict[str, Any]] | None = None,
        output_format: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Apply runtime parameter overrides.

        Args:
            providers: Provider-specific configuration
            output_format: Output format override
            **kwargs: Additional default overrides
        """
        if providers:
            self.config["providers"] = self._merge_configs(
                self.config.get("providers", {}),
                providers,
            )

        if output_format and output_format != "polars":  # Only override if not default
            self.config["defaults"]["output_format"] = output_format

        # Apply additional kwargs to defaults
        for key, value in kwargs.items():
            if key in self.config["defaults"]:
                self.config["defaults"][key] = value

    def get_provider_config(self, provider_name: str) -> dict[str, Any]:
        """Get configuration for a specific provider.

        Args:
            provider_name: Name of the provider

        Returns:
            Provider configuration dictionary (empty if not configured)
        """
        return self.config.get("providers", {}).get(provider_name, {})

    def get_routing_patterns(self) -> list[dict[str, str]]:
        """Get configured routing patterns.

        Returns:
            List of routing pattern configurations
        """
        return self.config.get("routing", {}).get("patterns", [])

    def has_api_key(self, provider_name: str) -> bool:
        """Check if a provider has an API key configured.

        Args:
            provider_name: Name of the provider

        Returns:
            True if API key is configured
        """
        provider_config = self.get_provider_config(provider_name)
        return bool(provider_config.get("api_key"))
