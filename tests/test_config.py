"""Tests for configuration management."""

from datetime import time as datetime_time
from pathlib import Path

import pytest
import yaml

from ml4t.data.config.loader import ConfigLoader
from ml4t.data.config.models import (
    CompressionType,
    DataConfig,
    DatasetConfig,
    ProviderConfig,
    ProviderType,
    RateLimitConfig,
    ScheduleConfig,
    ScheduleType,
    StorageConfig,
    StorageStrategy,
    SymbolUniverse,
    WorkflowConfig,
)
from ml4t.data.core.models import AssetClass, Frequency


class TestStorageConfig:
    """Test storage configuration."""

    def test_default_storage_config(self):
        """Test default storage configuration values."""
        config = StorageConfig()
        assert config.strategy == StorageStrategy.HIVE
        assert config.compression == CompressionType.ZSTD
        assert config.atomic_writes is True
        assert config.enable_locking is True
        assert config.metadata_tracking is True
        assert config.partition_cols == ["year", "month"]

    def test_flat_storage_no_partitions(self):
        """Test that flat storage removes partition columns."""
        config = StorageConfig(
            strategy=StorageStrategy.FLAT,
            partition_cols=["year", "month"],  # Should be cleared
        )
        assert config.partition_cols == []

    def test_path_expansion(self):
        """Test that paths are expanded correctly."""
        config = StorageConfig(base_path=Path("~/data"))
        assert config.base_path.is_absolute()
        assert str(config.base_path).startswith(str(Path.home()))


class TestRateLimitConfig:
    """Test rate limiting configuration."""

    def test_default_rate_limit(self):
        """Test default rate limit values."""
        config = RateLimitConfig()
        assert config.requests_per_second == 10.0
        assert config.burst_size == 1
        assert config.retry_max_attempts == 3
        assert config.retry_backoff_factor == 2.0
        assert config.circuit_breaker_threshold == 5
        assert config.circuit_breaker_timeout == 60

    def test_custom_rate_limit(self):
        """Test custom rate limit values."""
        config = RateLimitConfig(requests_per_second=5.0, burst_size=10, retry_max_attempts=5)
        assert config.requests_per_second == 5.0
        assert config.burst_size == 10
        assert config.retry_max_attempts == 5


class TestProviderConfig:
    """Test provider configuration."""

    def test_basic_provider(self):
        """Test basic provider configuration."""
        config = ProviderConfig(name="yahoo", type=ProviderType.YAHOO, api_key="test_key")
        assert config.name == "yahoo"
        assert config.type == ProviderType.YAHOO
        assert config.enabled is True
        assert config.api_key == "test_key"

    def test_provider_with_rate_limit(self):
        """Test provider with custom rate limiting."""
        config = ProviderConfig(
            name="cryptocompare",
            type=ProviderType.CRYPTOCOMPARE,
            rate_limit=RateLimitConfig(requests_per_second=2.0),
        )
        assert config.rate_limit.requests_per_second == 2.0


class TestSymbolUniverse:
    """Test symbol universe configuration."""

    def test_simple_universe(self):
        """Test simple symbol universe."""
        universe = SymbolUniverse(
            name="crypto", symbols=["BTC-USD", "ETH-USD"], asset_class=AssetClass.CRYPTO
        )
        assert universe.name == "crypto"
        assert universe.symbols == ["BTC-USD", "ETH-USD"]
        assert universe.asset_class == AssetClass.CRYPTO

    def test_universe_from_file(self, tmp_path):
        """Test loading symbols from file."""
        # Create symbol file
        symbol_file = tmp_path / "symbols.txt"
        symbol_file.write_text("AAPL\nMSFT\nGOOG\n")

        universe = SymbolUniverse(
            name="tech",
            file=symbol_file,
            symbols=["META"],  # Should be merged
        )
        assert "META" in universe.symbols
        assert "AAPL" in universe.symbols
        assert "MSFT" in universe.symbols
        assert "GOOG" in universe.symbols

    def test_universe_deduplication(self, tmp_path):
        """Test that duplicate symbols are removed."""
        symbol_file = tmp_path / "symbols.txt"
        symbol_file.write_text("AAPL\nAAPL\nMSFT\n")

        universe = SymbolUniverse(name="test", file=symbol_file, symbols=["AAPL", "GOOG"])
        # AAPL should appear only once
        assert universe.symbols.count("AAPL") == 1
        assert len(universe.symbols) == 3  # AAPL, MSFT, GOOG


class TestDataConfig:
    """Test main ML4T Data configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        config = DataConfig()
        assert config.storage.strategy == StorageStrategy.HIVE
        assert config.log_level == "INFO"
        assert config.parallel_downloads == 4
        assert len(config.providers) == 0
        assert len(config.universes) == 0

    def test_config_with_providers(self):
        """Test configuration with providers."""
        config = DataConfig(
            providers=[
                ProviderConfig(name="yahoo", type=ProviderType.YAHOO),
                ProviderConfig(name="crypto", type=ProviderType.CRYPTOCOMPARE),
            ]
        )
        assert len(config.providers) == 2
        assert config.get_provider("yahoo") is not None
        assert config.get_provider("crypto") is not None
        assert config.get_provider("nonexistent") is None

    def test_config_validation(self):
        """Test configuration validation."""
        config = DataConfig(
            providers=[ProviderConfig(name="yahoo", type=ProviderType.YAHOO)],
            universes=[SymbolUniverse(name="stocks", symbols=["AAPL"])],
            datasets=[
                DatasetConfig(
                    name="daily_stocks",
                    universe="stocks",
                    provider="yahoo",
                    frequency=Frequency.DAILY,
                )
            ],
        )

        # Should have no validation issues
        issues = config.validate_config()
        assert len(issues) == 0

        # Add dataset with invalid provider
        config.datasets.append(
            DatasetConfig(
                name="bad_dataset",
                universe="stocks",
                provider="nonexistent",
                frequency=Frequency.DAILY,
            )
        )

        issues = config.validate_config()
        assert len(issues) > 0
        assert any("unknown provider" in issue for issue in issues)

    def test_yaml_serialization(self, tmp_path):
        """Test saving and loading from YAML."""
        config = DataConfig(
            storage=StorageConfig(strategy=StorageStrategy.HIVE, base_path=tmp_path / "data"),
            providers=[ProviderConfig(name="yahoo", type=ProviderType.YAHOO, api_key="test_key")],
            universes=[SymbolUniverse(name="sp500", symbols=["AAPL", "MSFT"])],
        )

        # Save to YAML
        yaml_file = tmp_path / "config.yaml"
        config.to_yaml(yaml_file)
        assert yaml_file.exists()

        # Load from YAML
        loaded = DataConfig.from_yaml(yaml_file)
        assert loaded.storage.strategy == StorageStrategy.HIVE
        assert len(loaded.providers) == 1
        assert loaded.providers[0].name == "yahoo"
        assert len(loaded.universes) == 1
        assert loaded.universes[0].name == "sp500"


class TestConfigLoader:
    """Test configuration loader."""

    def test_load_default_config(self):
        """Test loading default configuration when no file exists."""
        # Use loader without specifying a path, so it searches in standard locations
        loader = ConfigLoader()
        config = loader.load()
        assert isinstance(config, DataConfig)
        assert config.log_level == "INFO"

    def test_load_from_yaml(self, tmp_path):
        """Test loading configuration from YAML file."""
        config_data = {
            "storage": {"strategy": "flat", "compression": "lz4"},
            "providers": [{"name": "yahoo", "type": "yahoo", "enabled": True}],
            "log_level": "DEBUG",
        }

        config_file = tmp_path / "config.yaml"
        with open(config_file, "w") as f:
            yaml.safe_dump(config_data, f)

        loader = ConfigLoader(config_file)
        config = loader.load()

        assert config.storage.strategy == StorageStrategy.FLAT
        assert config.storage.compression == CompressionType.LZ4
        assert config.log_level == "DEBUG"
        assert len(config.providers) == 1

    def test_environment_variable_expansion(self, tmp_path, monkeypatch):
        """Test environment variable expansion in configuration."""
        monkeypatch.setenv("TEST_API_KEY", "secret_key_123")
        monkeypatch.setenv("TEST_LOG_LEVEL", "WARNING")

        config_data = {
            "providers": [{"name": "test", "type": "yahoo", "api_key": "${TEST_API_KEY}"}],
            "log_level": "${TEST_LOG_LEVEL}",
        }

        config_file = tmp_path / "config.yaml"
        with open(config_file, "w") as f:
            yaml.safe_dump(config_data, f)

        loader = ConfigLoader(config_file)
        config = loader.load()

        assert config.providers[0].api_key == "secret_key_123"
        assert config.log_level == "WARNING"

    def test_environment_variable_with_default(self, tmp_path):
        """Test environment variables with default values."""
        config_data = {
            "providers": [
                {"name": "test", "type": "yahoo", "api_key": "${MISSING_VAR:default_value}"}
            ]
        }

        config_file = tmp_path / "config.yaml"
        with open(config_file, "w") as f:
            yaml.safe_dump(config_data, f)

        loader = ConfigLoader(config_file)
        config = loader.load()

        assert config.providers[0].api_key == "default_value"

    def test_config_includes(self, tmp_path):
        """Test configuration file includes."""
        # Create base configuration
        base_config = {"storage": {"strategy": "hive"}, "log_level": "INFO"}

        base_file = tmp_path / "base.yaml"
        with open(base_file, "w") as f:
            yaml.safe_dump(base_config, f)

        # Create main configuration with include
        main_config = {
            "include": str(base_file),
            "providers": [{"name": "yahoo", "type": "yahoo"}],
            "log_level": "DEBUG",  # Override base
        }

        main_file = tmp_path / "main.yaml"
        with open(main_file, "w") as f:
            yaml.safe_dump(main_config, f)

        loader = ConfigLoader(main_file)
        config = loader.load()

        # Should have storage from base
        assert config.storage.strategy == StorageStrategy.HIVE
        # Should have provider from main
        assert len(config.providers) == 1
        # Log level should be overridden
        assert config.log_level == "DEBUG"

    def test_save_config(self, tmp_path):
        """Test saving configuration."""
        config = DataConfig(
            log_level="WARNING", providers=[ProviderConfig(name="test", type=ProviderType.YAHOO)]
        )

        save_path = tmp_path / "saved.yaml"
        loader = ConfigLoader()
        loader.save(config, save_path)

        assert save_path.exists()

        # Load it back
        with open(save_path) as f:
            data = yaml.safe_load(f)

        assert data["log_level"] == "WARNING"
        assert len(data["providers"]) == 1
        assert data["providers"][0]["name"] == "test"


class TestScheduleConfig:
    """Test schedule configuration."""

    def test_cron_schedule(self):
        """Test cron schedule configuration."""
        schedule = ScheduleConfig(type=ScheduleType.CRON, cron="0 9 * * MON-FRI")
        assert schedule.type == ScheduleType.CRON
        assert schedule.cron == "0 9 * * MON-FRI"

    def test_interval_schedule(self):
        """Test interval schedule configuration."""
        schedule = ScheduleConfig(
            type=ScheduleType.INTERVAL,
            interval=3600,  # 1 hour
        )
        assert schedule.type == ScheduleType.INTERVAL
        assert schedule.interval == 3600

    def test_schedule_validation(self):
        """Test schedule validation."""
        # Cron without expression should fail
        with pytest.raises(ValueError, match="Cron expression required"):
            ScheduleConfig(type=ScheduleType.CRON)

        # Interval without value should fail
        with pytest.raises(ValueError, match="Positive interval required"):
            ScheduleConfig(type=ScheduleType.INTERVAL)


class TestWorkflowConfig:
    """Test workflow configuration."""

    def test_basic_workflow(self):
        """Test basic workflow configuration."""
        workflow = WorkflowConfig(name="daily_update", datasets=["stocks", "crypto"], enabled=True)
        assert workflow.name == "daily_update"
        assert workflow.datasets == ["stocks", "crypto"]
        assert workflow.enabled is True
        assert workflow.on_error == "stop"

    def test_workflow_with_schedule(self):
        """Test workflow with schedule."""
        workflow = WorkflowConfig(
            name="scheduled",
            datasets=["test"],
            schedule=ScheduleConfig(type=ScheduleType.DAILY, time=datetime_time(9, 30)),
        )
        assert workflow.schedule is not None
        assert workflow.schedule.type == ScheduleType.DAILY
