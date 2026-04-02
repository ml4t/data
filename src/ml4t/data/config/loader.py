"""Configuration loader with environment variable support."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import structlog
import yaml
from pydantic import ValidationError

from ml4t.data.config.models import DataConfig
from ml4t.data.paths import find_ml4t_data_config_path

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
        path = find_ml4t_data_config_path()
        if path is not None:
            logger.info(f"Found configuration file: {path}")
        return path

    def _interpolate_env_vars(self, data: Any) -> Any:
        """
        Recursively interpolate environment variables in configuration.

        Args:
            data: Configuration data (dict, list, or scalar)

        Returns:
            Data with environment variables interpolated
        """
        if isinstance(data, dict):
            return {key: self._interpolate_env_vars(value) for key, value in data.items()}
        if isinstance(data, list):
            return [self._interpolate_env_vars(item) for item in data]
        if isinstance(data, str):
            # Replace ${VAR} with environment variable value
            def replace_env(match):
                var_name = match.group(1)
                # Support default values: ${VAR:default}
                if ":" in var_name:
                    var_name, default = var_name.split(":", 1)
                    return os.environ.get(var_name, default)
                value = os.environ.get(var_name)
                if value is None:
                    logger.warning(f"Environment variable not found: {var_name}")
                    return match.group(0)  # Keep original if not found
                return value

            return self.env_pattern.sub(replace_env, data)
        return data

    def _merge_configs(self, base: dict, override: dict) -> dict:
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

    def _load_yaml(self, path: Path) -> dict:
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
                return data or {}
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in {path}: {e}") from e
        except Exception as e:
            raise ValueError(f"Error loading {path}: {e}") from e

    def _process_includes(self, data: dict, base_dir: Path) -> dict:
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

            base_config = {}
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

        # Set environment variables defined in config
        if "env" in data:
            for key, value in data["env"].items():
                if key not in os.environ:
                    os.environ[key] = str(value)
                    logger.debug(f"Set environment variable: {key}")

        # Interpolate environment variables
        data = self._interpolate_env_vars(data)

        # Apply defaults to datasets
        if "defaults" in data and "datasets" in data:
            defaults = data["defaults"]
            for dataset in data["datasets"]:
                # Apply defaults that aren't already set
                for key, value in defaults.items():
                    if key not in dataset:
                        dataset[key] = value

        # Validate and create configuration object
        try:
            config = DataConfig(**data)
            logger.info(
                "Configuration loaded successfully",
                providers=len(config.providers),
                datasets=len(config.datasets),
                workflows=len(config.workflows),
            )
            return config
        except ValidationError as e:
            logger.error("Configuration validation failed", errors=e.errors())
            raise ValueError(f"Invalid configuration: {e}") from e

    def save(self, config: DataConfig, path: Path | None = None) -> None:
        """
        Save configuration to file.

        Args:
            config: Configuration object to save
            path: Optional path to save to (defaults to original path)
        """
        save_path = path or self.config_path
        if not save_path:
            save_path = Path.cwd() / "ml4t.data.yaml"

        # Convert to dictionary with JSON-compatible types
        data = config.model_dump(exclude_none=True, exclude_defaults=False, mode="json")

        # Save to YAML
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

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
