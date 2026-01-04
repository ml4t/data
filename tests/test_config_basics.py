"""Test configuration basic functionality."""

import tempfile
from pathlib import Path


class TestConfigBasics:
    """Test basic configuration functionality."""

    def test_config_loader_basic(self):
        """Test config loader basic functionality."""
        from ml4t.data.config.loader import ConfigLoader

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create simple config file
            config_file = Path(tmpdir) / "config.yaml"
            config_content = """
version: "1.0"
log_level: INFO
providers:
  - name: yahoo
    type: yahoo
"""
            config_file.write_text(config_content)

            # Test loading
            loader = ConfigLoader(config_file)
            config = loader.load()

            assert config.version == "1.0"
            assert config.log_level == "INFO"
            assert len(config.providers) == 1
            assert config.providers[0].name == "yahoo"

    def test_config_validator_basic(self):
        """Test config validator basic functionality."""
        from ml4t.data.config.models import DataConfig, ProviderConfig
        from ml4t.data.config.validator import ConfigValidator

        # Create valid config
        config = DataConfig(version="1.0", providers=[ProviderConfig(name="yahoo", type="yahoo")])

        # ConfigValidator requires config parameter
        validator = ConfigValidator(config)
        result = validator.validate()

        assert result is True
        assert len(validator.errors) == 0

    def test_config_models_basic(self):
        """Test config models basic functionality."""
        from ml4t.data.config.models import (
            DataConfig,
            DatasetConfig,
            ProviderConfig,
            ScheduleConfig,
            ScheduleType,
            WorkflowConfig,
        )

        # Test ProviderConfig
        provider = ProviderConfig(name="yahoo", type="yahoo")
        assert provider.name == "yahoo"
        assert provider.type == "yahoo"

        # Test DatasetConfig
        dataset = DatasetConfig(name="test_dataset", symbols=["TEST"], provider="yahoo")
        assert dataset.name == "test_dataset"
        assert "TEST" in dataset.symbols

        # Test ScheduleConfig
        schedule = ScheduleConfig(type=ScheduleType.INTERVAL, interval=3600)
        assert schedule.type == ScheduleType.INTERVAL
        assert schedule.interval == 3600

        # Test WorkflowConfig
        workflow = WorkflowConfig(name="test_workflow", datasets=["test_dataset"])
        assert workflow.name == "test_workflow"

        # Test DataConfig
        config = DataConfig(
            version="1.0", providers=[provider], datasets=[dataset], workflows=[workflow]
        )
        assert config.version == "1.0"
        assert len(config.providers) == 1
