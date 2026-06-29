"""Configuration validation utilities."""

from __future__ import annotations

from typing import Any

import structlog

from ml4t.data.config.models import DataConfig, ScheduleType

logger = structlog.get_logger()


class ConfigValidator:
    """Validate ML4T Data configuration for correctness and consistency."""

    def __init__(self, config: DataConfig):
        """
        Initialize configuration validator.

        Args:
            config: Configuration to validate
        """
        self.config = config
        self.errors: list[str] = []
        self.warnings: list[str] = []

    def validate(self) -> bool:
        """
        Validate the configuration.

        Returns:
            True if configuration is valid, False otherwise
        """
        self.errors = []
        self.warnings = []

        # Run validation checks
        self._validate_providers()
        self._validate_datasets()
        self._validate_workflows()
        self._validate_schedules()
        self._validate_paths()
        self._validate_references()

        # Log results
        if self.errors:
            for error in self.errors:
                logger.error(f"Validation error: {error}")

        if self.warnings:
            for warning in self.warnings:
                logger.warning(f"Validation warning: {warning}")

        return len(self.errors) == 0

    def _validate_providers(self) -> None:
        """Validate provider configurations."""
        provider_names = set()

        for provider in self.config.providers:
            # Check for duplicate names
            if provider.name in provider_names:
                self.errors.append(f"Duplicate provider name: {provider.name}")
            provider_names.add(provider.name)

            # Check API keys for providers that need them
            if provider.type in ["massive", "polygon", "cryptocompare"] and not provider.api_key:
                self.warnings.append(
                    f"Provider {provider.name} ({provider.type}) may require an API key"
                )

            # Validate rate limits
            if provider.rate_limit and provider.rate_limit.requests_per_second <= 0:
                self.errors.append(
                    f"Invalid rate limit for provider {provider.name}: "
                    f"{provider.rate_limit.requests_per_second}"
                )

    def _validate_datasets(self) -> None:
        """Validate dataset configurations."""
        dataset_names = set()

        for dataset in self.config.datasets:
            # Check for duplicate names
            if dataset.name in dataset_names:
                self.errors.append(f"Duplicate dataset name: {dataset.name}")
            dataset_names.add(dataset.name)

            # Check that provider exists
            if not self.config.get_provider(dataset.provider):
                self.errors.append(
                    f"Dataset {dataset.name} references non-existent provider: {dataset.provider}"
                )

            # Check universe exists
            if hasattr(dataset, "universe") and dataset.universe:
                if not self.config.get_universe(dataset.universe):
                    self.errors.append(
                        f"Dataset {dataset.name} references non-existent universe: {dataset.universe}"
                    )

            # Validate date ranges
            if dataset.start_date and dataset.end_date and dataset.start_date >= dataset.end_date:
                self.errors.append(
                    f"Dataset {dataset.name} has invalid date range: "
                    f"{dataset.start_date} >= {dataset.end_date}"
                )

            # Check update mode
            if dataset.update_mode not in ["full", "incremental"]:
                self.errors.append(
                    f"Dataset {dataset.name} has invalid update_mode: {dataset.update_mode}"
                )

    def _validate_workflows(self) -> None:
        """Validate workflow configurations."""
        workflow_names = set()

        for workflow in self.config.workflows:
            # Check for duplicate names
            if workflow.name in workflow_names:
                self.errors.append(f"Duplicate workflow name: {workflow.name}")
            workflow_names.add(workflow.name)

            # Check that referenced datasets exist
            for dataset_name in workflow.datasets:
                if not self.config.get_dataset(dataset_name):
                    self.errors.append(
                        f"Workflow {workflow.name} references non-existent dataset: {dataset_name}"
                    )

            # Validate error handling
            if workflow.on_error not in ["stop", "continue", "retry"]:
                self.errors.append(
                    f"Workflow {workflow.name} has invalid on_error: {workflow.on_error}"
                )

            # Check hooks are valid commands
            for hook in workflow.pre_hooks + workflow.post_hooks:
                if not hook.strip():
                    self.warnings.append(f"Workflow {workflow.name} has empty hook command")

    def _validate_schedules(self) -> None:
        """Validate schedule configurations."""
        for workflow in self.config.workflows:
            if not workflow.schedule:
                continue

            schedule = workflow.schedule

            # Validate based on schedule type
            if schedule.type == ScheduleType.CRON:
                if not schedule.cron:
                    self.errors.append(
                        f"Workflow {workflow.name} has cron schedule but no cron expression"
                    )
                else:
                    # Basic cron validation (5 or 6 fields)
                    fields = schedule.cron.split()
                    if len(fields) not in [5, 6]:
                        self.errors.append(
                            f"Workflow {workflow.name} has invalid cron expression: {schedule.cron}"
                        )

            elif schedule.type == ScheduleType.INTERVAL:
                if not schedule.interval or schedule.interval <= 0:
                    self.errors.append(
                        f"Workflow {workflow.name} has interval schedule but invalid interval"
                    )

            elif schedule.type in [ScheduleType.DAILY, ScheduleType.WEEKLY]:
                if not schedule.time:
                    self.warnings.append(
                        f"Workflow {workflow.name} has {schedule.type} schedule but no time specified"
                    )

                if schedule.type == ScheduleType.WEEKLY and schedule.weekday is None:
                    self.errors.append(
                        f"Workflow {workflow.name} has weekly schedule but no weekday specified"
                    )

            elif (
                schedule.type == ScheduleType.MARKET_HOURS
                and schedule.market_open_offset is None
                and schedule.market_close_offset is None
            ):
                self.errors.append(
                    f"Workflow {workflow.name} has market_hours schedule but no offset specified"
                )

    def _validate_paths(self) -> None:
        """Validate file paths and directories."""
        # Check base directory
        if not self.config.base_dir.exists():
            self.warnings.append(f"Base directory does not exist: {self.config.base_dir}")

        # Check for symbol files referenced in universes
        for universe in self.config.universes:
            if universe.file and not universe.file.exists():
                self.errors.append(
                    f"Universe {universe.name} references non-existent symbol file: {universe.file}"
                )

    def _validate_references(self) -> None:
        """Validate internal references between configuration elements."""
        # Check for orphaned providers (not used by any dataset)
        used_providers = {dataset.provider for dataset in self.config.datasets}
        for provider in self.config.providers:
            if provider.name not in used_providers:
                self.warnings.append(f"Provider {provider.name} is defined but not used")

        # Check for orphaned datasets (not used by any workflow)
        used_datasets = set()
        for workflow in self.config.workflows:
            used_datasets.update(workflow.datasets)

        for dataset in self.config.datasets:
            if dataset.name not in used_datasets:
                self.warnings.append(
                    f"Dataset {dataset.name} is defined but not used in any workflow"
                )

    def get_summary(self) -> dict[str, Any]:
        """
        Get validation summary.

        Returns:
            Dictionary with validation results
        """
        return {
            "valid": len(self.errors) == 0,
            "errors": self.errors,
            "warnings": self.warnings,
            "error_count": len(self.errors),
            "warning_count": len(self.warnings),
        }
