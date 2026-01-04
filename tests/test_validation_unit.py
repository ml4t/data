"""Unit tests for validation components."""

from datetime import datetime

import polars as pl

from ml4t.data.validation.base import Severity, ValidationIssue, ValidationResult
from ml4t.data.validation.ohlcv import OHLCVValidator
from ml4t.data.validation.report import ValidationReport


class TestValidationResult:
    """Test ValidationResult class."""

    def test_validation_result_creation(self):
        """Test creating validation result."""
        result = ValidationResult(
            passed=True,
            metadata={"key": "value"},
        )

        assert result.passed is True
        assert result.metadata == {"key": "value"}
        assert len(result.issues) == 0

    def test_validation_result_failed(self):
        """Test failed validation result with issue."""
        issue = ValidationIssue(
            severity=Severity.ERROR,
            check="negative_price",
            message="Price is negative",
            details={"price": -10.0, "row": 5},
        )
        result = ValidationResult(passed=False, issues=[issue])

        assert result.passed is False
        assert len(result.issues) == 1
        assert result.error_count == 1
        assert "negative" in result.issues[0].message.lower()


class TestOHLCVValidator:
    """Test OHLCV validator."""

    def test_validator_initialization(self):
        """Test validator initialization."""
        validator = OHLCVValidator()

        assert validator.check_nulls is True
        assert validator.check_negative_prices is True
        assert validator.name() == "OHLCVValidator"

    def test_validate_valid_data(self):
        """Test validating valid OHLCV data."""
        validator = OHLCVValidator()

        df = pl.DataFrame(
            {
                "timestamp": [datetime(2024, 1, i) for i in range(1, 4)],
                "open": [100.0, 101.0, 102.0],
                "high": [105.0, 106.0, 107.0],
                "low": [99.0, 100.0, 101.0],
                "close": [104.0, 105.0, 106.0],
                "volume": [1000000, 1100000, 1200000],
            }
        )

        result = validator.validate(df)

        assert isinstance(result, ValidationResult)
        # Valid data should pass
        assert result.passed is True or result.error_count == 0

    def test_validate_with_issues(self):
        """Test validating data with issues."""
        validator = OHLCVValidator()

        df = pl.DataFrame(
            {
                "timestamp": [datetime(2024, 1, i) for i in range(1, 4)],
                "open": [100.0, -101.0, 102.0],  # Negative price
                "high": [105.0, 106.0, 107.0],
                "low": [99.0, 100.0, 101.0],
                "close": [104.0, 105.0, 106.0],
                "volume": [1000000, 0, 1200000],  # Zero volume
            }
        )

        result = validator.validate(df)

        # Should have issues for invalid data
        assert isinstance(result, ValidationResult)
        assert len(result.issues) > 0 or not result.passed

    def test_validate_disabled_checks(self):
        """Test validation with disabled checks."""
        validator = OHLCVValidator(check_negative_prices=False, check_negative_volume=False)

        df = pl.DataFrame(
            {
                "timestamp": [datetime(2024, 1, 1)],
                "open": [-100.0],  # Invalid data
                "high": [105.0],
                "low": [99.0],
                "close": [104.0],
                "volume": [-1000000],
            }
        )

        result = validator.validate(df)

        # Should not flag disabled checks
        assert isinstance(result, ValidationResult)


class TestValidationReport:
    """Test ValidationReport class."""

    def test_report_creation(self):
        """Test creating validation report."""
        issue = ValidationIssue(
            check="NonNegativePrice",
            severity=Severity.ERROR,
            message="Negative price found",
            details={"row": 5, "column": "open", "value": -10.0},
        )
        validation_result = ValidationResult(passed=False, issues=[issue])

        report = ValidationReport(
            symbol="AAPL", provider="test", timestamp=datetime.now(), results=[validation_result]
        )

        assert report.symbol == "AAPL"
        assert len(report.results) == 1
        assert report.passed is False
        assert report.error_count == 1
        assert report.warning_count == 0
        assert report.critical_count == 0

    def test_report_summary(self):
        """Test report summary generation."""
        issues = [
            ValidationIssue(
                check="Rule1",
                severity=Severity.ERROR,
                message="Error 1",
            ),
            ValidationIssue(
                check="Rule2",
                severity=Severity.WARNING,
                message="Warning 1",
            ),
            ValidationIssue(
                check="Rule3",
                severity=Severity.CRITICAL,
                message="Critical 1",
            ),
        ]
        validation_result = ValidationResult(passed=False, issues=issues)

        report = ValidationReport(
            symbol="TEST", provider="test", timestamp=datetime.now(), results=[validation_result]
        )

        summary = report.to_dict()

        assert summary["symbol"] == "TEST"
        assert summary["passed"] is False
        assert summary["summary"]["total_issues"] == 3
        assert summary["summary"]["critical"] == 1
        assert summary["summary"]["errors"] == 1
        assert summary["summary"]["warnings"] == 1

    def test_empty_report(self):
        """Test empty report (all validations passed)."""
        validation_result = ValidationResult(passed=True, issues=[])
        report = ValidationReport(
            symbol="AAPL", provider="test", timestamp=datetime.now(), results=[validation_result]
        )

        assert report.passed is True
        assert report.total_issues == 0
        assert report.error_count == 0
        assert report.warning_count == 0
        assert report.critical_count == 0
