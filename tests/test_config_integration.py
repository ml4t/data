"""Integration tests for configuration system."""

import os

import yaml

from ml4t.data.config import (
    ConfigLoader,
    ConfigValidator,
    DataConfig,
    DatasetConfig,
    ProviderConfig,
    WorkflowConfig,
    load_config,
)
from ml4t.data.config.models import (
    AssetClass,
    Frequency,
    ProviderType,
    ScheduleConfig,
    ScheduleType,
)


class TestConfigIntegration:
    """Test configuration system integration."""

    def test_load_yaml_config_file(self, tmp_path):
        """Test loading a complete YAML configuration."""
        config_file = tmp_path / "test_config.yaml"
        config_data = {
            "version": "1.0",
            "base_dir": str(tmp_path / "data"),
            "log_level": "DEBUG",
            "providers": [
                {
                    "name": "test_provider",
                    "type": "yahoo",
                    "rate_limit": 1.0,
                }
            ],
            "datasets": [
                {
                    "name": "test_dataset",
                    "symbols": ["TEST1", "TEST2"],
                    "provider": "test_provider",
                    "frequency": "daily",
                    "asset_class": "equity",
                }
            ],
            "workflows": [
                {
                    "name": "test_workflow",
                    "datasets": ["test_dataset"],
                    "enabled": True,
                }
            ],
        }

        with open(config_file, "w") as f:
            yaml.dump(config_data, f)

        # Load configuration
        loader = ConfigLoader(config_file)
        config = loader.load()

        assert config.version == "1.0"
        assert config.log_level == "DEBUG"
        assert len(config.providers) == 1
        assert len(config.datasets) == 1
        assert len(config.workflows) == 1

        # Validate configuration
        validator = ConfigValidator(config)
        assert validator.validate() is True

    def test_environment_variable_in_config(self, tmp_path):
        """Test environment variable interpolation in configuration."""
        config_file = tmp_path / "env_config.yaml"

        # Set test environment variables
        os.environ["TEST_API_KEY"] = "secret_key_123"
        os.environ["TEST_LOG_LEVEL"] = "WARNING"

        try:
            config_data = {
                "version": "1.0",
                "log_level": "${TEST_LOG_LEVEL}",
                "providers": [
                    {
                        "name": "api_provider",
                        "type": "yahoo",
                        "api_key": "${TEST_API_KEY}",
                    }
                ],
            }

            with open(config_file, "w") as f:
                yaml.dump(config_data, f)

            # Load configuration
            config = load_config(config_file)

            assert config.log_level == "WARNING"
            assert config.providers[0].api_key == "secret_key_123"

        finally:
            # Clean up environment
            del os.environ["TEST_API_KEY"]
            del os.environ["TEST_LOG_LEVEL"]

    def test_config_with_includes(self, tmp_path):
        """Test configuration with include files."""
        # Create base configuration
        base_file = tmp_path / "base.yaml"
        base_data = {
            "version": "1.0",
            "log_level": "INFO",
            "providers": [{"name": "base_provider", "type": "yahoo"}],
        }

        with open(base_file, "w") as f:
            yaml.dump(base_data, f)

        # Create main configuration with include
        main_file = tmp_path / "main.yaml"
        main_data = {
            "include": ["base.yaml"],
            "datasets": [
                {
                    "name": "main_dataset",
                    "symbols": ["MAIN"],
                    "provider": "base_provider",
                }
            ],
        }

        with open(main_file, "w") as f:
            yaml.dump(main_data, f)

        # Load configuration
        config = load_config(main_file)

        assert config.version == "1.0"
        assert config.log_level == "INFO"
        assert len(config.providers) == 1
        assert len(config.datasets) == 1

    def test_config_with_defaults(self, tmp_path):
        """Test configuration with defaults applied to datasets."""
        config_file = tmp_path / "defaults_config.yaml"
        config_data = {
            "version": "1.0",
            "defaults": {
                "update_mode": "incremental",
                "asset_class": "equity",
                "frequency": "daily",
            },
            "providers": [{"name": "provider1", "type": "yahoo"}],
            "datasets": [
                {
                    "name": "dataset1",
                    "symbols": ["SYM1"],
                    "provider": "provider1",
                    # Should use defaults for other fields
                },
                {
                    "name": "dataset2",
                    "symbols": ["SYM2"],
                    "provider": "provider1",
                    "asset_class": "crypto",  # Override default
                },
            ],
        }

        with open(config_file, "w") as f:
            yaml.dump(config_data, f)

        # Load configuration
        config = load_config(config_file)

        # Check defaults were applied
        assert config.datasets[0].update_mode == "incremental"
        assert config.datasets[0].asset_class == AssetClass.EQUITY
        assert config.datasets[0].frequency == Frequency.DAILY

        # Check override worked
        assert config.datasets[1].asset_class == AssetClass.CRYPTO
        assert config.datasets[1].update_mode == "incremental"  # Still uses default

    def test_config_validation_errors(self, tmp_path):
        """Test configuration validation with errors."""
        config_file = tmp_path / "invalid_config.yaml"
        config_data = {
            "version": "1.0",
            "providers": [
                {"name": "provider1", "type": "yahoo"},
                {"name": "provider1", "type": "binance"},  # Duplicate name
            ],
            "datasets": [
                {
                    "name": "dataset1",
                    "symbols": ["SYM1"],
                    "provider": "nonexistent",  # Invalid provider reference
                }
            ],
        }

        with open(config_file, "w") as f:
            yaml.dump(config_data, f)

        # Load configuration
        config = load_config(config_file)

        # Validate configuration
        validator = ConfigValidator(config)
        assert validator.validate() is False
        assert len(validator.errors) > 0
        assert any("Duplicate provider name" in error for error in validator.errors)
        assert any("non-existent provider" in error for error in validator.errors)

    def test_schedule_configuration(self):
        """Test schedule configuration models."""
        # Test daily schedule
        daily_schedule = ScheduleConfig(
            type=ScheduleType.DAILY,
            time="09:30:00",
            timezone="America/New_York",
        )
        assert daily_schedule.type == ScheduleType.DAILY

        # Test interval schedule
        interval_schedule = ScheduleConfig(
            type=ScheduleType.INTERVAL,
            interval=3600,  # 1 hour
        )
        assert interval_schedule.interval == 3600

        # Test cron schedule
        cron_schedule = ScheduleConfig(
            type=ScheduleType.CRON,
            cron="0 9 * * 1-5",  # 9 AM weekdays
        )
        assert cron_schedule.cron == "0 9 * * 1-5"

    def test_workflow_with_hooks(self):
        """Test workflow configuration with pre and post hooks."""
        workflow = WorkflowConfig(
            name="complex_workflow",
            datasets=["dataset1", "dataset2"],
            pre_hooks=["echo 'Starting'", "python prepare.py"],
            post_hooks=["python validate.py", "echo 'Done'"],
            on_error="continue",
        )

        assert len(workflow.pre_hooks) == 2
        assert len(workflow.post_hooks) == 2
        assert workflow.on_error == "continue"

    def test_config_save_and_reload(self, tmp_path):
        """Test saving and reloading configuration."""
        # Create configuration programmatically
        config = DataConfig(
            version="1.0",
            base_dir=tmp_path / "data",
            log_level="INFO",
            providers=[
                ProviderConfig(
                    name="test_provider",
                    type=ProviderType.YAHOO,
                    rate_limit=0.5,
                )
            ],
            datasets=[
                DatasetConfig(
                    name="test_dataset",
                    symbols=["TEST"],
                    provider="test_provider",
                    frequency=Frequency.DAILY,
                    asset_class=AssetClass.EQUITY,
                )
            ],
        )

        # Save configuration
        save_path = tmp_path / "saved_config.yaml"
        loader = ConfigLoader()
        loader.save(config, save_path)

        assert save_path.exists()

        # Reload configuration
        reloaded_config = load_config(save_path)

        assert reloaded_config.version == config.version
        assert reloaded_config.log_level == config.log_level
        assert len(reloaded_config.providers) == len(config.providers)
        assert len(reloaded_config.datasets) == len(config.datasets)
