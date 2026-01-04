"""Tests for configuration data models."""

from datetime import datetime, time
from pathlib import Path

import pytest
from pydantic import ValidationError

from ml4t.data.config.models import (
    AssetClass,
    CompressionType,
    DataConfig,
    DatasetConfig,
    Frequency,
    ProviderConfig,
    ProviderType,
    ScheduleConfig,
    ScheduleType,
    WorkflowConfig,
)


class TestProviderConfig:
    """Test provider configuration model."""

    def test_basic_provider(self):
        """Test basic provider configuration."""
        provider = ProviderConfig(
            name="test_provider",
            type=ProviderType.YAHOO,
        )

        assert provider.name == "test_provider"
        assert provider.type == ProviderType.YAHOO
        assert provider.timeout == 30  # Default
        assert provider.retry_count == 3  # Default

    def test_provider_with_api_key(self):
        """Test provider with API key."""
        provider = ProviderConfig(
            name="crypto_provider",
            type=ProviderType.BINANCE,
            api_key="${BINANCE_API_KEY}",
            api_secret="${BINANCE_API_SECRET}",
            rate_limit=10.0,
        )

        assert provider.api_key == "${BINANCE_API_KEY}"
        assert provider.api_secret == "${BINANCE_API_SECRET}"
        # rate_limit float auto-converts to RateLimitConfig
        assert provider.rate_limit.requests_per_second == 10.0

    def test_provider_extra_settings(self):
        """Test provider with extra settings."""
        provider = ProviderConfig(
            name="binance",
            type=ProviderType.BINANCE,
            extra={
                "market": "spot",
                "test_net": True,
            },
        )

        assert provider.extra["market"] == "spot"
        assert provider.extra["test_net"] is True


class TestDatasetConfig:
    """Test dataset configuration model."""

    def test_basic_dataset(self):
        """Test basic dataset configuration."""
        dataset = DatasetConfig(
            name="stocks",
            symbols=["AAPL", "MSFT", "GOOGL"],
            provider="yahoo",
        )

        assert dataset.name == "stocks"
        assert dataset.symbols == ["AAPL", "MSFT", "GOOGL"]
        assert dataset.provider == "yahoo"
        assert dataset.frequency == Frequency.DAILY  # Default
        assert dataset.asset_class == AssetClass.EQUITY  # Default
        assert dataset.update_mode == "incremental"  # Default

    def test_dataset_with_dates(self):
        """Test dataset with date range."""
        dataset = DatasetConfig(
            name="historical",
            symbols=["SPY"],
            provider="yahoo",
            start_date="2020-01-01",
            end_date="2023-12-31",
            update_mode="full",
        )

        # Pydantic parses date strings to datetime objects
        assert dataset.start_date == datetime(2020, 1, 1)
        assert dataset.end_date == datetime(2023, 12, 31)
        assert dataset.update_mode == "full"

    def test_dataset_symbol_expansion(self):
        """Test single symbol expansion to list."""
        dataset = DatasetConfig(
            name="single",
            symbols="AAPL",  # Single string
            provider="yahoo",
        )

        assert dataset.symbols == ["AAPL"]

    def test_dataset_validation_settings(self):
        """Test dataset with validation settings."""
        dataset = DatasetConfig(
            name="crypto",
            symbols=["BTC-USD", "ETH-USD"],
            provider="binance",
            asset_class=AssetClass.CRYPTO,
            frequency=Frequency.HOURLY,
            validation={
                "strict": False,
                "max_return_threshold": 2.0,
            },
            storage={
                "compression": "zstd",
                "partition_by": "day",
            },
        )

        assert dataset.asset_class == AssetClass.CRYPTO
        assert dataset.frequency == Frequency.HOURLY
        assert dataset.validation["strict"] is False
        assert dataset.storage["compression"] == "zstd"


class TestScheduleConfig:
    """Test schedule configuration model."""

    def test_cron_schedule(self):
        """Test cron schedule configuration."""
        schedule = ScheduleConfig(
            type=ScheduleType.CRON,
            cron="0 9 * * 1-5",  # 9 AM weekdays
        )

        assert schedule.type == ScheduleType.CRON
        assert schedule.cron == "0 9 * * 1-5"

    def test_interval_schedule(self):
        """Test interval schedule configuration."""
        schedule = ScheduleConfig(
            type=ScheduleType.INTERVAL,
            interval=900,  # 15 minutes
        )

        assert schedule.type == ScheduleType.INTERVAL
        assert schedule.interval == 900

    def test_daily_schedule(self):
        """Test daily schedule configuration."""
        schedule = ScheduleConfig(
            type=ScheduleType.DAILY,
            time=time(9, 30),  # 9:30 AM
            timezone="America/New_York",
        )

        assert schedule.type == ScheduleType.DAILY
        assert schedule.time == time(9, 30)
        assert schedule.timezone == "America/New_York"

    def test_weekly_schedule(self):
        """Test weekly schedule configuration."""
        schedule = ScheduleConfig(
            type=ScheduleType.WEEKLY,
            weekday=1,  # Monday
            time=time(6, 0),  # 6 AM
        )

        assert schedule.type == ScheduleType.WEEKLY
        assert schedule.weekday == 1
        assert schedule.time == time(6, 0)

    def test_market_hours_schedule(self):
        """Test market hours schedule configuration."""
        schedule = ScheduleConfig(
            type=ScheduleType.MARKET_HOURS,
            market_open_offset=30,  # 30 minutes after open
            timezone="America/New_York",
        )

        assert schedule.type == ScheduleType.MARKET_HOURS
        assert schedule.market_open_offset == 30

    def test_invalid_cron_schedule(self):
        """Test that cron schedule requires cron expression."""
        with pytest.raises(ValidationError):
            ScheduleConfig(
                type=ScheduleType.CRON,
                # Missing cron expression
            )

    def test_invalid_interval_schedule(self):
        """Test that interval schedule requires positive interval."""
        with pytest.raises(ValidationError):
            ScheduleConfig(
                type=ScheduleType.INTERVAL,
                interval=-10,  # Negative interval
            )


class TestWorkflowConfig:
    """Test workflow configuration model."""

    def test_basic_workflow(self):
        """Test basic workflow configuration."""
        workflow = WorkflowConfig(
            name="daily_update",
            datasets=["stocks", "crypto"],
        )

        assert workflow.name == "daily_update"
        assert workflow.datasets == ["stocks", "crypto"]
        assert workflow.enabled is True  # Default
        assert workflow.on_error == "stop"  # Default

    def test_workflow_with_schedule(self):
        """Test workflow with schedule."""
        schedule = ScheduleConfig(
            type=ScheduleType.DAILY,
            time=time(9, 0),
        )

        workflow = WorkflowConfig(
            name="morning_update",
            description="Update data every morning",
            datasets=["stocks"],
            schedule=schedule,
            enabled=True,
        )

        assert workflow.description == "Update data every morning"
        assert workflow.schedule.type == ScheduleType.DAILY
        assert workflow.schedule.time == time(9, 0)

    def test_workflow_with_hooks(self):
        """Test workflow with pre and post hooks."""
        workflow = WorkflowConfig(
            name="complex_workflow",
            datasets=["data1", "data2"],
            pre_hooks=[
                "python scripts/prepare.py",
                "echo 'Starting workflow'",
            ],
            post_hooks=[
                "python scripts/validate.py",
                "python scripts/notify.py",
            ],
            on_error="continue",
        )

        assert len(workflow.pre_hooks) == 2
        assert len(workflow.post_hooks) == 2
        assert workflow.on_error == "continue"

    def test_workflow_with_notifications(self):
        """Test workflow with notification settings."""
        workflow = WorkflowConfig(
            name="prod_workflow",
            datasets=["production"],
            notifications={
                "email": {
                    "to": ["alerts@example.com"],
                    "on_failure": True,
                },
                "slack": {
                    "webhook": "${SLACK_WEBHOOK}",
                    "channel": "#data-alerts",
                },
            },
        )

        assert "email" in workflow.notifications
        assert workflow.notifications["email"]["to"] == ["alerts@example.com"]
        assert "slack" in workflow.notifications


class TestDataConfig:
    """Test main QLDM configuration model."""

    def test_minimal_config(self):
        """Test minimal valid configuration."""
        config = DataConfig()

        assert config.version == "1.0"
        assert config.base_dir == Path("./data")
        assert config.log_level == "INFO"
        assert config.providers == []
        assert config.datasets == []
        assert config.workflows == []

    def test_complete_config(self):
        """Test complete configuration."""
        config = DataConfig(
            version="1.0",
            base_dir=Path("/var/lib/qldm"),
            log_level="DEBUG",
            providers=[
                ProviderConfig(
                    name="yahoo",
                    type=ProviderType.YAHOO,
                    rate_limit=0.5,
                )
            ],
            datasets=[
                DatasetConfig(
                    name="stocks",
                    symbols=["AAPL", "MSFT"],
                    provider="yahoo",
                )
            ],
            workflows=[
                WorkflowConfig(
                    name="daily",
                    datasets=["stocks"],
                )
            ],
            defaults={
                "update_mode": "incremental",
                "asset_class": "equity",
            },
            validation={
                "enabled": True,
                "strict": False,
            },
            storage={
                "backend": "filesystem",
                "compression": "zstd",
            },
            env={
                "QLDM_TEST": "value",
            },
        )

        assert config.version == "1.0"
        assert config.base_dir == Path("/var/lib/qldm")
        assert len(config.providers) == 1
        assert len(config.datasets) == 1
        assert len(config.workflows) == 1
        assert config.defaults["update_mode"] == "incremental"
        assert config.validation["enabled"] is True
        # Storage dict gets converted to StorageConfig when possible
        assert config.storage.compression == CompressionType.ZSTD
        assert config.env["QLDM_TEST"] == "value"

    def test_get_provider(self):
        """Test get_provider method."""
        config = DataConfig(
            providers=[
                ProviderConfig(name="yahoo", type=ProviderType.YAHOO),
                ProviderConfig(name="binance", type=ProviderType.BINANCE),
            ]
        )

        yahoo = config.get_provider("yahoo")
        assert yahoo is not None
        assert yahoo.name == "yahoo"

        binance = config.get_provider("binance")
        assert binance is not None
        assert binance.type == ProviderType.BINANCE

        missing = config.get_provider("nonexistent")
        assert missing is None

    def test_get_dataset(self):
        """Test get_dataset method."""
        config = DataConfig(
            datasets=[
                DatasetConfig(name="stocks", symbols=["AAPL"], provider="yahoo"),
                DatasetConfig(name="crypto", symbols=["BTC"], provider="binance"),
            ]
        )

        stocks = config.get_dataset("stocks")
        assert stocks is not None
        assert stocks.symbols == ["AAPL"]

        crypto = config.get_dataset("crypto")
        assert crypto is not None
        assert crypto.provider == "binance"

        missing = config.get_dataset("nonexistent")
        assert missing is None

    def test_get_workflow(self):
        """Test get_workflow method."""
        config = DataConfig(
            workflows=[
                WorkflowConfig(name="daily", datasets=["stocks"]),
                WorkflowConfig(name="hourly", datasets=["crypto"]),
            ]
        )

        daily = config.get_workflow("daily")
        assert daily is not None
        assert daily.datasets == ["stocks"]

        hourly = config.get_workflow("hourly")
        assert hourly is not None
        assert hourly.datasets == ["crypto"]

        missing = config.get_workflow("nonexistent")
        assert missing is None
