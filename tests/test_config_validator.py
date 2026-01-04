"""Tests for configuration validator."""

import pytest
from pydantic import ValidationError

from ml4t.data.config.models import (
    DataConfig,
    DatasetConfig,
    ProviderConfig,
    ProviderType,
    ScheduleConfig,
    ScheduleType,
    WorkflowConfig,
)
from ml4t.data.config.validator import ConfigValidator


class TestConfigValidator:
    """Test configuration validator."""

    def test_valid_configuration(self):
        """Test validation of valid configuration."""
        config = DataConfig(
            providers=[
                ProviderConfig(name="yahoo", type=ProviderType.YAHOO),
            ],
            datasets=[
                DatasetConfig(
                    name="stocks",
                    symbols=["AAPL"],
                    provider="yahoo",
                ),
            ],
            workflows=[
                WorkflowConfig(
                    name="daily",
                    datasets=["stocks"],
                ),
            ],
        )

        validator = ConfigValidator(config)
        assert validator.validate() is True
        assert len(validator.errors) == 0

    def test_duplicate_provider_names(self):
        """Test detection of duplicate provider names."""
        config = DataConfig(
            providers=[
                ProviderConfig(name="yahoo", type=ProviderType.YAHOO),
                ProviderConfig(name="yahoo", type=ProviderType.BINANCE),
            ],
        )

        validator = ConfigValidator(config)
        assert validator.validate() is False
        assert any("Duplicate provider name" in error for error in validator.errors)

    def test_duplicate_dataset_names(self):
        """Test detection of duplicate dataset names."""
        config = DataConfig(
            providers=[
                ProviderConfig(name="yahoo", type=ProviderType.YAHOO),
            ],
            datasets=[
                DatasetConfig(name="data", symbols=["A"], provider="yahoo"),
                DatasetConfig(name="data", symbols=["B"], provider="yahoo"),
            ],
        )

        validator = ConfigValidator(config)
        assert validator.validate() is False
        assert any("Duplicate dataset name" in error for error in validator.errors)

    def test_dataset_invalid_provider(self):
        """Test detection of dataset referencing non-existent provider."""
        config = DataConfig(
            providers=[
                ProviderConfig(name="yahoo", type=ProviderType.YAHOO),
            ],
            datasets=[
                DatasetConfig(
                    name="stocks",
                    symbols=["AAPL"],
                    provider="nonexistent",
                ),
            ],
        )

        validator = ConfigValidator(config)
        assert validator.validate() is False
        assert any("non-existent provider" in error for error in validator.errors)

    def test_workflow_invalid_dataset(self):
        """Test detection of workflow referencing non-existent dataset."""
        config = DataConfig(
            providers=[
                ProviderConfig(name="yahoo", type=ProviderType.YAHOO),
            ],
            datasets=[
                DatasetConfig(name="stocks", symbols=["AAPL"], provider="yahoo"),
            ],
            workflows=[
                WorkflowConfig(
                    name="daily",
                    datasets=["stocks", "nonexistent"],
                ),
            ],
        )

        validator = ConfigValidator(config)
        assert validator.validate() is False
        assert any("non-existent dataset" in error for error in validator.errors)

    def test_invalid_rate_limit(self):
        """Test detection of invalid rate limit."""
        # Invalid rate limit is now caught at model creation time by Pydantic
        with pytest.raises(ValidationError) as exc_info:
            ProviderConfig(
                name="yahoo",
                type=ProviderType.YAHOO,
                rate_limit=-1.0,
            )

        # Verify the error is about requests_per_second being <= 0
        assert "requests_per_second" in str(exc_info.value)
        assert "greater than 0" in str(exc_info.value)

    def test_dataset_no_symbols(self):
        """Test detection of dataset with no symbols."""
        # Empty symbols list is now caught at model creation time
        with pytest.raises(ValidationError) as exc_info:
            DatasetConfig(
                name="empty",
                symbols=[],
                provider="yahoo",
            )

        # Verify the error mentions no symbols
        assert "no symbols" in str(exc_info.value).lower()

    def test_invalid_date_range(self):
        """Test detection of invalid date range."""
        config = DataConfig(
            providers=[
                ProviderConfig(name="yahoo", type=ProviderType.YAHOO),
            ],
            datasets=[
                DatasetConfig(
                    name="stocks",
                    symbols=["AAPL"],
                    provider="yahoo",
                    start_date="2023-12-31",
                    end_date="2023-01-01",
                ),
            ],
        )

        validator = ConfigValidator(config)
        assert validator.validate() is False
        assert any("invalid date range" in error for error in validator.errors)

    def test_invalid_cron_schedule(self):
        """Test detection of invalid cron schedule."""
        config = DataConfig(
            workflows=[
                WorkflowConfig(
                    name="cron_job",
                    datasets=[],
                    schedule=ScheduleConfig(
                        type=ScheduleType.CRON,
                        cron="invalid",
                    ),
                ),
            ],
        )

        validator = ConfigValidator(config)
        assert validator.validate() is False
        assert any("invalid cron expression" in error for error in validator.errors)

    def test_orphaned_provider_warning(self):
        """Test warning for unused provider."""
        config = DataConfig(
            providers=[
                ProviderConfig(name="yahoo", type=ProviderType.YAHOO),
                ProviderConfig(name="unused", type=ProviderType.BINANCE),
            ],
            datasets=[
                DatasetConfig(
                    name="stocks",
                    symbols=["AAPL"],
                    provider="yahoo",
                ),
            ],
        )

        validator = ConfigValidator(config)
        assert validator.validate() is True  # Warnings don't fail validation
        assert any("not used" in warning for warning in validator.warnings)

    def test_orphaned_dataset_warning(self):
        """Test warning for unused dataset."""
        config = DataConfig(
            providers=[
                ProviderConfig(name="yahoo", type=ProviderType.YAHOO),
            ],
            datasets=[
                DatasetConfig(name="used", symbols=["A"], provider="yahoo"),
                DatasetConfig(name="unused", symbols=["B"], provider="yahoo"),
            ],
            workflows=[
                WorkflowConfig(name="daily", datasets=["used"]),
            ],
        )

        validator = ConfigValidator(config)
        assert validator.validate() is True  # Warnings don't fail validation
        assert any("not used in any workflow" in warning for warning in validator.warnings)

    def test_get_summary(self):
        """Test getting validation summary."""
        config = DataConfig(
            providers=[
                ProviderConfig(name="yahoo", type=ProviderType.YAHOO),
            ],
            datasets=[
                DatasetConfig(
                    name="stocks",
                    symbols=["AAPL"],
                    provider="nonexistent",  # Error
                ),
            ],
        )

        validator = ConfigValidator(config)
        validator.validate()

        summary = validator.get_summary()
        assert summary["valid"] is False
        assert summary["error_count"] > 0
        assert len(summary["errors"]) > 0
