"""Base validation classes and interfaces."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

import polars as pl


class Severity(StrEnum):
    """Validation issue severity levels."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class ValidationIssue:
    """Represents a single validation issue."""

    severity: Severity
    check: str
    message: str
    details: dict[str, Any] = field(default_factory=dict)
    row_count: int | None = None
    sample_rows: list[int] | None = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))


@dataclass
class ValidationResult:
    """Result of validation checks."""

    passed: bool
    issues: list[ValidationIssue] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    duration_seconds: float | None = None

    @property
    def error_count(self) -> int:
        """Count of error-level issues."""
        return sum(1 for issue in self.issues if issue.severity == Severity.ERROR)

    @property
    def warning_count(self) -> int:
        """Count of warning-level issues."""
        return sum(1 for issue in self.issues if issue.severity == Severity.WARNING)

    @property
    def critical_count(self) -> int:
        """Count of critical-level issues."""
        return sum(1 for issue in self.issues if issue.severity == Severity.CRITICAL)

    def add_issue(self, issue: ValidationIssue) -> None:
        """Add a validation issue to the result."""
        self.issues.append(issue)
        # Update passed status based on critical and error issues
        if issue.severity in (Severity.CRITICAL, Severity.ERROR):
            self.passed = False


class Validator(ABC):
    """Abstract base class for data validators."""

    @abstractmethod
    def validate(self, df: pl.DataFrame, **kwargs: Any) -> ValidationResult:
        """
        Validate a DataFrame.

        Args:
            df: DataFrame to validate
            **kwargs: Additional validation parameters

        Returns:
            ValidationResult with issues found
        """

    @abstractmethod
    def name(self) -> str:
        """Return the validator name."""
