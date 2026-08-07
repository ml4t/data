"""Configuration loader with environment variable support."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import structlog
import yaml
from pydantic import ValidationError

from ml4t.data.config._serialization import write_yaml
from ml4t.data.config.models import DataConfig

logger = structlog.get_logger()


class ConfigLoader:
    """Load and process ML4T Data configuration files."""

    def __init__(self, config_path: Path | None = None):
        """
        Initialize configuration loader.

        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path or self._find_config_file()
        self.env_pattern = re.compile(r"\$\{([^}]+)\}")

    def _find_config_file(self) -> Path | None:
        """Find configuration file in standard locations."""
        search_paths = [
            Path.cwd() / "ml4t.data.yaml",
            Path.cwd() / "ml4t.data.yml",
            Path.cwd() / ".ml4t-data.yaml",
            Path.cwd() / ".ml4t-data.yml",
            Path.cwd() / "config" / "ml4t.data.yaml",
            Path.home() / ".config" / "ml4t-data" / "config.yaml",
        ]

        for path in search_paths:
            if path.exists():
                logger.info(f"Found configuration file: {path}")
                return path

        return None

    def _interpolate_env_vars(self, data: Any, environment: dict[str, str] | None = None) -> Any:
        """
        Recursively interpolate environment variables in configuration.

        Args:
            data: Configuration data (dict, list, or scalar)

        Returns:
            Data with environment variables interpolated
        """
        environment = environment or dict(os.environ)
        if isinstance(data, dict):
            return {
                key: self._interpolate_env_vars(value, environment) for key, value in data.items()
            }
        if isinstance(data, list):
            return [self._interpolate_env_vars(item, environment) for item in data]
        if isinstance(data, str):
            # Replace ${VAR} with environment variable value
            def replace_env(match: re.Match[str]) -> str:
                var_name = match.group(1)
                # Support default values: ${VAR:default}
                if ":" in var_name:
                    var_name, default = var_name.split(":", 1)
                    return environment.get(var_name, default)
                value = environment.get(var_name)
                if value is None:
                    logger.warning(f"Environment variable not found: {var_name}")
                    return match.group(0)  # Keep original if not found
                return value

            return self.env_pattern.sub(replace_env, data)
        return data

    def _merge_configs(self, base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
        """
        Recursively merge configuration dictionaries.

        Args:
            base: Base configuration
            override: Override configuration

        Returns:
            Merged configuration
        """
        result = base.copy()

        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                # Recursively merge dictionaries
                result[key] = self._merge_configs(result[key], value)
            elif key in result and isinstance(result[key], list) and isinstance(value, list):
                # Extend lists (or replace based on a flag)
                # For now, we'll replace lists entirely
                result[key] = value
            else:
                # Override value
                result[key] = value

        return result

    def _retain_credential_references(self, config: DataConfig, raw_data: dict[str, Any]) -> None:
        """Attach raw environment references without retaining resolved values."""
        raw_providers = raw_data.get("providers", [])
        if isinstance(raw_providers, dict):
            provider_entries = [
                {"name": name, **(entry if isinstance(entry, dict) else {})}
                for name, entry in raw_providers.items()
            ]
        elif isinstance(raw_providers, list):
            provider_entries = raw_providers
        else:
            return
        for provider, raw_provider in zip(config.providers, provider_entries, strict=False):
            if not isinstance(raw_provider, dict):
                continue
            for field in ("api_key", "api_secret"):
                value = raw_provider.get(field)
                if isinstance(value, str) and self.env_pattern.search(value):
                    provider._set_credential_reference(field, value)

    def _load_yaml(self, path: Path) -> dict[str, Any]:
        """
        Load YAML file.

        Args:
            path: Path to YAML file

        Returns:
            Parsed YAML data
        """
        try:
            with open(path) as f:
                data = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in {path}: {e}") from e
        except OSError as e:
            raise ValueError(f"Error loading {path}: {e}") from e
        if data is None:
            return {}
        if not isinstance(data, dict):
            raise ValueError(f"Configuration root in {path} must be a mapping")
        return data

    def _process_includes(self, data: dict[str, Any], base_dir: Path) -> dict[str, Any]:
        """
        Process include directives in configuration.

        Args:
            data: Configuration data
            base_dir: Base directory for relative includes

        Returns:
            Configuration with includes processed
        """
        if "include" in data:
            includes = data.pop("include")
            if isinstance(includes, str):
                includes = [includes]

            base_config: dict[str, Any] = {}
            for include_path in includes:
                # Resolve relative paths
                if not Path(include_path).is_absolute():
                    include_path = base_dir / include_path

                include_path = Path(include_path)
                if not include_path.exists():
                    logger.warning(f"Include file not found: {include_path}")
                    continue

                logger.debug(f"Including configuration from: {include_path}")
                include_data = self._load_yaml(include_path)

                # Process nested includes
                include_data = self._process_includes(include_data, include_path.parent)

                # Merge included configuration
                base_config = self._merge_configs(base_config, include_data)

            # Merge main configuration over includes
            data = self._merge_configs(base_config, data)

        return data

    @staticmethod
    def _resolve_path(value: str | Path, base_dir: Path) -> str:
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = base_dir / path
        return str(path.resolve())

    def _resolve_config_paths(self, data: dict[str, Any], base_dir: Path) -> None:
        """Resolve supported relative paths against the declaring config file."""
        if "base_dir" in data:
            data["base_dir"] = self._resolve_path(data["base_dir"], base_dir)
        storage = data.get("storage")
        if isinstance(storage, dict):
            for field in ("base_path", "path"):
                if field in storage:
                    storage[field] = self._resolve_path(storage[field], base_dir)
        for field in ("universes", "datasets"):
            entries = data.get(field, [])
            if isinstance(entries, dict):
                entries = entries.values()
            for entry in entries:
                if not isinstance(entry, dict):
                    continue
                path_field = "file" if field == "universes" else "symbols_file"
                if path_field in entry:
                    entry[path_field] = self._resolve_path(entry[path_field], base_dir)

    def load(self, override_path: Path | None = None) -> DataConfig:
        """
        Load and validate configuration.

        Args:
            override_path: Optional path to override configuration

        Returns:
            Validated DataConfig object

        Raises:
            ValueError: If configuration is invalid
            FileNotFoundError: If configuration file not found
        """
        # Use override path if provided
        config_path = override_path or self.config_path

        if not config_path:
            # Return default configuration if no file found
            logger.info("No configuration file found, using defaults")
            return DataConfig()

        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        logger.info(f"Loading configuration from: {config_path}")

        # Load YAML
        data = self._load_yaml(config_path)

        # Process includes
        data = self._process_includes(data, config_path.parent)

        raw_data = data

        # Interpolate environment variables
        local_environment = dict(os.environ)
        configured_environment = data.get("env", {})
        if isinstance(configured_environment, dict):
            for key, value in configured_environment.items():
                local_environment.setdefault(key, str(value))
        data = self._interpolate_env_vars(data, local_environment)
        self._resolve_config_paths(data, config_path.parent)

        # Apply defaults to datasets
        if isinstance(data.get("defaults"), dict) and isinstance(data.get("datasets"), list | dict):
            defaults = data["defaults"]
            datasets = data["datasets"]
            dataset_entries = datasets.values() if isinstance(datasets, dict) else datasets
            for dataset in dataset_entries:
                if not isinstance(dataset, dict):
                    continue
                # Apply defaults that aren't already set
                for key, value in defaults.items():
                    if key not in dataset:
                        dataset[key] = value

        # Validate and create configuration object
        try:
            config = DataConfig(**data)
            self._retain_credential_references(config, raw_data)
            logger.info(
                "Configuration loaded successfully",
                providers=len(config.providers),
                datasets=len(config.datasets),
                workflows=len(config.workflows),
            )
            return config
        except ValidationError as e:
            errors = e.errors(include_input=False, include_context=False)
            logger.error("Configuration validation failed", errors=errors)
            details = "; ".join(
                f"{'.'.join(str(part) for part in error['loc']) or 'configuration'}: {error['msg']}"
                for error in errors
            )
            raise ValueError(f"Invalid configuration: {details}") from e

    def save(self, config: DataConfig, path: Path | None = None) -> None:
        """
        Save configuration to file.

        Args:
            config: Configuration object to save
            path: Optional path to save to (defaults to original path)
        """
        save_path = path or self.config_path
        if not save_path:
            save_path = Path.cwd() / "mlquant.data.yaml"

        # Convert to dictionary with JSON-compatible types
        data = config.model_dump(exclude_none=True, exclude_defaults=False, mode="json")

        write_yaml(save_path, data)

        logger.info(f"Configuration saved to: {save_path}")


def load_config(path: Path | None = None) -> DataConfig:
    """
    Convenience function to load configuration.

    Args:
        path: Optional path to configuration file

    Returns:
        Loaded DataConfig object
    """
    loader = ConfigLoader(path)
    return loader.load()
