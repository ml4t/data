"""Tests for configuration loader."""

import os
from pathlib import Path

import pytest
import yaml

from ml4t.data.config.loader import ConfigLoader
from ml4t.data.config.models import AssetClass, DataConfig, Frequency, ProviderType


class TestConfigLoader:
    """Test configuration loader."""

    def test_find_config_file(self, tmp_path):
        """Test finding configuration file in standard locations."""
        # Create a temporary config file
        config_file = tmp_path / "ml4t.data.yaml"
        config_file.write_text("version: '1.0'")

        # Change to temp directory
        original_cwd = Path.cwd()
        try:
            os.chdir(tmp_path)
            loader = ConfigLoader()
            assert loader.config_path == config_file
        finally:
            os.chdir(original_cwd)

    def test_no_config_file(self, tmp_path):
        """Test behavior when no config file exists."""
        original_cwd = Path.cwd()
        try:
            os.chdir(tmp_path)
            loader = ConfigLoader()
            assert loader.config_path is None
        finally:
            os.chdir(original_cwd)

    def test_load_basic_yaml(self, tmp_path):
        """Test loading basic YAML configuration."""
        config_file = tmp_path / "config.yaml"
        config_data = {
            "version": "1.0",
            "base_dir": "/data",
            "log_level": "DEBUG",
        }

        with open(config_file, "w") as f:
            yaml.dump(config_data, f)

        loader = ConfigLoader(config_file)
        config = loader.load()

        assert config.version == "1.0"
        assert config.base_dir == Path("/data")
        assert config.log_level == "DEBUG"

    def test_environment_variable_interpolation(self, tmp_path):
        """Test environment variable interpolation."""
        config_file = tmp_path / "config.yaml"
        config_data = {
            "version": "1.0",
            "base_dir": "${TEST_DATA_DIR}",
            "log_level": "${TEST_LOG_LEVEL:INFO}",  # With default
            "providers": [
                {
                    "name": "test",
                    "type": "yahoo",
                    "api_key": "${TEST_API_KEY}",
                }
            ],
        }

        with open(config_file, "w") as f:
            yaml.dump(config_data, f)

        # Set environment variables
        os.environ["TEST_DATA_DIR"] = "/test/data"
        os.environ["TEST_API_KEY"] = "secret_key_123"
        # TEST_LOG_LEVEL not set, should use default

        try:
            loader = ConfigLoader(config_file)
            config = loader.load()

            assert config.base_dir == Path("/test/data")
            assert config.log_level == "INFO"  # Default value
            assert config.providers[0].api_key == "secret_key_123"
        finally:
            # Clean up environment
            del os.environ["TEST_DATA_DIR"]
            del os.environ["TEST_API_KEY"]

    def test_nested_environment_variables(self, tmp_path):
        """Test environment variables in nested structures."""
        config_file = tmp_path / "config.yaml"
        config_data = {
            "version": "1.0",
            "datasets": [
                {
                    "name": "test",
                    "symbols": ["${SYMBOL1}", "${SYMBOL2:DEFAULT}"],
                    "provider": "yahoo",
                    "validation": {
                        "threshold": "${THRESHOLD:10}",
                    },
                }
            ],
        }

        with open(config_file, "w") as f:
            yaml.dump(config_data, f)

        os.environ["SYMBOL1"] = "AAPL"
        # SYMBOL2 not set, should use default

        try:
            loader = ConfigLoader(config_file)
            config = loader.load()

            dataset = config.datasets[0]
            assert dataset.symbols == ["AAPL", "DEFAULT"]
            assert dataset.validation["threshold"] == "10"
        finally:
            del os.environ["SYMBOL1"]

    def test_config_includes(self, tmp_path):
        """Test configuration includes."""
        # Create base config
        base_file = tmp_path / "base.yaml"
        base_data = {
            "version": "1.0",
            "base_dir": "/base/data",
            "log_level": "INFO",
            "providers": [{"name": "yahoo", "type": "yahoo"}],
        }

        with open(base_file, "w") as f:
            yaml.dump(base_data, f)

        # Create main config with include
        main_file = tmp_path / "main.yaml"
        main_data = {
            "include": ["base.yaml"],
            "log_level": "DEBUG",  # Override base
            "datasets": [
                {
                    "name": "stocks",
                    "symbols": ["AAPL"],
                    "provider": "yahoo",
                }
            ],
        }

        with open(main_file, "w") as f:
            yaml.dump(main_data, f)

        loader = ConfigLoader(main_file)
        config = loader.load()

        # Check that base is included
        assert config.version == "1.0"
        assert config.base_dir == Path("/base/data")
        # Check that override works
        assert config.log_level == "DEBUG"
        # Check that both base and main content are present
        assert len(config.providers) == 1
        assert len(config.datasets) == 1

    def test_multiple_includes(self, tmp_path):
        """Test multiple configuration includes."""
        # Create provider config
        providers_file = tmp_path / "providers.yaml"
        providers_data = {
            "providers": [
                {"name": "yahoo", "type": "yahoo"},
                {"name": "binance", "type": "binance"},
            ]
        }

        with open(providers_file, "w") as f:
            yaml.dump(providers_data, f)

        # Create dataset config
        datasets_file = tmp_path / "datasets.yaml"
        datasets_data = {
            "datasets": [
                {
                    "name": "stocks",
                    "symbols": ["AAPL"],
                    "provider": "yahoo",
                }
            ]
        }

        with open(datasets_file, "w") as f:
            yaml.dump(datasets_data, f)

        # Create main config with multiple includes
        main_file = tmp_path / "main.yaml"
        main_data = {
            "include": ["providers.yaml", "datasets.yaml"],
            "version": "1.0",
            "log_level": "INFO",
        }

        with open(main_file, "w") as f:
            yaml.dump(main_data, f)

        loader = ConfigLoader(main_file)
        config = loader.load()

        assert config.version == "1.0"
        assert len(config.providers) == 2
        assert len(config.datasets) == 1

    def test_apply_defaults(self, tmp_path):
        """Test applying defaults to datasets."""
        config_file = tmp_path / "config.yaml"
        config_data = {
            "version": "1.0",
            "defaults": {
                "update_mode": "incremental",
                "asset_class": "equity",
                "frequency": "daily",
            },
            "datasets": [
                {
                    "name": "stocks",
                    "symbols": ["AAPL"],
                    "provider": "yahoo",
                    # Will use defaults for update_mode, asset_class, frequency
                },
                {
                    "name": "crypto",
                    "symbols": ["BTC"],
                    "provider": "binance",
                    "asset_class": "crypto",  # Override default
                    # Will use defaults for update_mode and frequency
                },
            ],
        }

        with open(config_file, "w") as f:
            yaml.dump(config_data, f)

        loader = ConfigLoader(config_file)
        config = loader.load()

        stocks = config.datasets[0]
        assert stocks.update_mode == "incremental"  # From defaults
        assert stocks.asset_class == AssetClass.EQUITY  # From defaults
        assert stocks.frequency == Frequency.DAILY  # From defaults

        crypto = config.datasets[1]
        assert crypto.update_mode == "incremental"  # From defaults
        assert crypto.asset_class == AssetClass.CRYPTO  # Override
        assert crypto.frequency == Frequency.DAILY  # From defaults

    def test_set_environment_variables(self, tmp_path):
        """Test setting environment variables from config."""
        config_file = tmp_path / "config.yaml"
        config_data = {
            "version": "1.0",
            "env": {
                "TEST_ENV_VAR": "test_value",
                "TEST_NUMBER": "42",
            },
            "base_dir": "${TEST_ENV_VAR}",
        }

        with open(config_file, "w") as f:
            yaml.dump(config_data, f)

        # Ensure variables don't exist
        os.environ.pop("TEST_ENV_VAR", None)
        os.environ.pop("TEST_NUMBER", None)

        try:
            loader = ConfigLoader(config_file)
            config = loader.load()

            # Check that env vars were set
            assert os.environ["TEST_ENV_VAR"] == "test_value"
            assert os.environ["TEST_NUMBER"] == "42"
            # Check that interpolation worked
            assert config.base_dir == Path("test_value")
        finally:
            # Clean up
            os.environ.pop("TEST_ENV_VAR", None)
            os.environ.pop("TEST_NUMBER", None)

    def test_save_config(self, tmp_path):
        """Test saving configuration to file."""
        config = DataConfig(
            version="1.0",
            base_dir=Path("/data"),
            log_level="DEBUG",
            providers=[
                {
                    "name": "yahoo",
                    "type": ProviderType.YAHOO,
                    "rate_limit": 0.5,
                }
            ],
            datasets=[
                {
                    "name": "stocks",
                    "symbols": ["AAPL", "MSFT"],
                    "provider": "yahoo",
                    "frequency": Frequency.DAILY,
                    "asset_class": AssetClass.EQUITY,
                }
            ],
        )

        output_file = tmp_path / "output.yaml"
        loader = ConfigLoader()
        loader.save(config, output_file)

        assert output_file.exists()

        # Load the saved config
        with open(output_file) as f:
            saved_data = yaml.safe_load(f)

        assert saved_data["version"] == "1.0"
        assert saved_data["base_dir"] == "/data"
        assert saved_data["log_level"] == "DEBUG"
        assert len(saved_data["providers"]) == 1
        assert len(saved_data["datasets"]) == 1

    def test_invalid_yaml(self, tmp_path):
        """Test handling of invalid YAML."""
        config_file = tmp_path / "invalid.yaml"
        config_file.write_text("invalid: yaml: content: [")

        loader = ConfigLoader(config_file)

        with pytest.raises(ValueError) as exc_info:
            loader.load()
        assert "Invalid YAML" in str(exc_info.value)

    def test_missing_include_file(self, tmp_path):
        """Test handling of missing include file."""
        config_file = tmp_path / "config.yaml"
        config_data = {
            "include": ["nonexistent.yaml"],
            "version": "1.0",
        }

        with open(config_file, "w") as f:
            yaml.dump(config_data, f)

        loader = ConfigLoader(config_file)
        # Should continue with warning, not error
        config = loader.load()
        assert config.version == "1.0"

    def test_recursive_includes(self, tmp_path):
        """Test nested/recursive includes."""
        # Create base config
        base_file = tmp_path / "base.yaml"
        base_data = {"log_level": "INFO", "providers": [{"name": "base_provider", "type": "yahoo"}]}

        with open(base_file, "w") as f:
            yaml.dump(base_data, f)

        # Create middle config that includes base
        middle_file = tmp_path / "middle.yaml"
        middle_data = {
            "include": ["base.yaml"],
            "datasets": [
                {
                    "name": "middle_dataset",
                    "symbols": ["AAPL"],
                    "provider": "base_provider",
                }
            ],
        }

        with open(middle_file, "w") as f:
            yaml.dump(middle_data, f)

        # Create main config that includes middle
        main_file = tmp_path / "main.yaml"
        main_data = {
            "include": ["middle.yaml"],
            "version": "1.0",
            "workflows": [
                {
                    "name": "main_workflow",
                    "datasets": ["middle_dataset"],
                }
            ],
        }

        with open(main_file, "w") as f:
            yaml.dump(main_data, f)

        loader = ConfigLoader(main_file)
        config = loader.load()

        # Check that all levels are included
        assert config.version == "1.0"
        assert config.log_level == "INFO"  # From base
        assert len(config.providers) == 1  # From base
        assert len(config.datasets) == 1  # From middle
        assert len(config.workflows) == 1  # From main
