"""Comprehensive tests for ValidationReport.

Tests cover:
- Report creation and properties
- Issue counting and aggregation
- JSON serialization and file I/O
- Print summary output
- DataFrame conversion
"""

from datetime import UTC, datetime
from pathlib import Path

import polars as pl

from ml4t.data.validation.base import Severity, ValidationIssue, ValidationResult
from ml4t.data.validation.report import ValidationReport


class TestValidationReportInitialization:
    """Test report initialization."""

    def test_basic_initialization(self):
        """Report initializes with required fields."""
        report = ValidationReport(symbol="AAPL", provider="yahoo")

        assert report.symbol == "AAPL"
        assert report.provider == "yahoo"
        assert isinstance(report.timestamp, datetime)
        assert len(report.results) == 0
        assert report.metadata == {}

    def test_custom_timestamp(self):
        """Report accepts custom timestamp."""
        ts = datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC)
        report = ValidationReport(
            symbol="AAPL",
            provider="yahoo",
            timestamp=ts,
        )

        assert report.timestamp == ts

    def test_custom_metadata(self):
        """Report accepts custom metadata."""
        report = ValidationReport(
            symbol="AAPL",
            provider="yahoo",
            metadata={"source": "test", "version": "1.0"},
        )

        assert report.metadata["source"] == "test"
        assert report.metadata["version"] == "1.0"


class TestIssueCountProperties:
    """Test issue counting properties."""

    def test_total_issues_empty(self):
        """Empty report has zero issues."""
        report = ValidationReport(symbol="AAPL", provider="yahoo")
        assert report.total_issues == 0

    def test_total_issues_with_results(self):
        """Total issues counts all issues."""
        report = ValidationReport(symbol="AAPL", provider="yahoo")

        # Add result with 2 issues
        result1 = ValidationResult(passed=True)
        result1.add_issue(
            ValidationIssue(severity=Severity.WARNING, check="test1", message="Warning 1")
        )
        result1.add_issue(
            ValidationIssue(severity=Severity.ERROR, check="test2", message="Error 1")
        )
        report.add_result(result1, "Validator1")

        # Add result with 1 issue
        result2 = ValidationResult(passed=True)
        result2.add_issue(ValidationIssue(severity=Severity.INFO, check="test3", message="Info 1"))
        report.add_result(result2, "Validator2")

        assert report.total_issues == 3

    def test_critical_count(self):
        """Critical count sums across results."""
        report = ValidationReport(symbol="AAPL", provider="yahoo")

        result = ValidationResult(passed=True)
        result.add_issue(
            ValidationIssue(severity=Severity.CRITICAL, check="test1", message="Critical 1")
        )
        result.add_issue(
            ValidationIssue(severity=Severity.CRITICAL, check="test2", message="Critical 2")
        )
        result.add_issue(
            ValidationIssue(severity=Severity.WARNING, check="test3", message="Warning 1")
        )
        report.add_result(result, "Validator")

        assert report.critical_count == 2

    def test_error_count(self):
        """Error count sums across results."""
        report = ValidationReport(symbol="AAPL", provider="yahoo")

        result = ValidationResult(passed=True)
        result.add_issue(ValidationIssue(severity=Severity.ERROR, check="test1", message="Error 1"))
        result.add_issue(ValidationIssue(severity=Severity.ERROR, check="test2", message="Error 2"))
        report.add_result(result, "Validator")

        assert report.error_count == 2

    def test_warning_count(self):
        """Warning count sums across results."""
        report = ValidationReport(symbol="AAPL", provider="yahoo")

        result = ValidationResult(passed=True)
        result.add_issue(
            ValidationIssue(severity=Severity.WARNING, check="test1", message="Warning 1")
        )
        report.add_result(result, "Validator")

        assert report.warning_count == 1


class TestPassedProperty:
    """Test passed property."""

    def test_passed_with_no_results(self):
        """Report with no results is passed."""
        report = ValidationReport(symbol="AAPL", provider="yahoo")
        assert report.passed is True

    def test_passed_with_all_passing_results(self):
        """Report passes when all results pass."""
        report = ValidationReport(symbol="AAPL", provider="yahoo")

        result1 = ValidationResult(passed=True)
        result2 = ValidationResult(passed=True)
        report.add_result(result1, "Validator1")
        report.add_result(result2, "Validator2")

        assert report.passed is True

    def test_failed_with_one_failing_result(self):
        """Report fails when any result fails."""
        report = ValidationReport(symbol="AAPL", provider="yahoo")

        result1 = ValidationResult(passed=True)
        result2 = ValidationResult(passed=False)
        report.add_result(result1, "Validator1")
        report.add_result(result2, "Validator2")

        assert report.passed is False


class TestAddResult:
    """Test add_result method."""

    def test_add_result_stores_validator_name(self):
        """add_result stores validator name in metadata."""
        report = ValidationReport(symbol="AAPL", provider="yahoo")
        result = ValidationResult(passed=True)

        report.add_result(result, "MyValidator")

        assert result.metadata["validator"] == "MyValidator"
        assert report.results[0].metadata["validator"] == "MyValidator"

    def test_add_multiple_results(self):
        """Multiple results can be added."""
        report = ValidationReport(symbol="AAPL", provider="yahoo")

        report.add_result(ValidationResult(passed=True), "V1")
        report.add_result(ValidationResult(passed=True), "V2")
        report.add_result(ValidationResult(passed=True), "V3")

        assert len(report.results) == 3


class TestToDict:
    """Test to_dict serialization."""

    def test_to_dict_basic_fields(self):
        """to_dict includes basic fields."""
        ts = datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC)
        report = ValidationReport(
            symbol="AAPL",
            provider="yahoo",
            timestamp=ts,
            metadata={"extra": "data"},
        )

        data = report.to_dict()

        assert data["symbol"] == "AAPL"
        assert data["provider"] == "yahoo"
        assert "2024-01-01" in data["timestamp"]
        assert data["metadata"]["extra"] == "data"

    def test_to_dict_summary(self):
        """to_dict includes summary counts."""
        report = ValidationReport(symbol="AAPL", provider="yahoo")
        result = ValidationResult(passed=True)
        result.add_issue(ValidationIssue(severity=Severity.ERROR, check="test", message="Error"))
        report.add_result(result, "Validator")

        data = report.to_dict()

        assert data["summary"]["total_issues"] == 1
        assert data["summary"]["errors"] == 1

    def test_to_dict_results(self):
        """to_dict includes result details."""
        report = ValidationReport(symbol="AAPL", provider="yahoo")
        result = ValidationResult(passed=True, duration_seconds=1.5)
        result.add_issue(
            ValidationIssue(
                severity=Severity.WARNING,
                check="test_check",
                message="Test message",
                details={"key": "value"},
                row_count=5,
                sample_rows=[0, 1, 2],
            )
        )
        report.add_result(result, "TestValidator")

        data = report.to_dict()

        assert len(data["results"]) == 1
        result_data = data["results"][0]
        assert result_data["validator"] == "TestValidator"
        assert result_data["passed"] is True
        assert result_data["duration_seconds"] == 1.5

        issue_data = result_data["issues"][0]
        assert issue_data["severity"] == "warning"
        assert issue_data["check"] == "test_check"
        assert issue_data["message"] == "Test message"
        assert issue_data["details"]["key"] == "value"
        assert issue_data["row_count"] == 5
        assert issue_data["sample_rows"] == [0, 1, 2]


class TestToJson:
    """Test to_json serialization."""

    def test_to_json_returns_string(self):
        """to_json returns a valid JSON string."""
        report = ValidationReport(symbol="AAPL", provider="yahoo")
        json_str = report.to_json()

        assert isinstance(json_str, str)
        # Verify it's valid JSON by parsing
        import json

        parsed = json.loads(json_str)
        assert parsed["symbol"] == "AAPL"

    def test_to_json_with_indent(self):
        """to_json respects indent parameter."""
        report = ValidationReport(symbol="AAPL", provider="yahoo")

        json_str = report.to_json(indent=4)

        # Indented JSON has multiple lines
        assert "\n" in json_str


class TestSaveAndLoad:
    """Test file I/O operations."""

    def test_save_and_load_roundtrip(self, tmp_path: Path):
        """Save and load preserves all data."""
        ts = datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC)
        report = ValidationReport(
            symbol="AAPL",
            provider="yahoo",
            timestamp=ts,
            metadata={"source": "test"},
        )

        result = ValidationResult(passed=True, duration_seconds=1.5)
        result.add_issue(
            ValidationIssue(
                severity=Severity.WARNING,
                check="test_check",
                message="Test warning",
                details={"key": "value"},
                row_count=10,
                sample_rows=[1, 2, 3],
            )
        )
        report.add_result(result, "TestValidator")

        # Save
        path = tmp_path / "report.json"
        report.save(path)
        assert path.exists()

        # Load
        loaded = ValidationReport.load(path)

        assert loaded.symbol == "AAPL"
        assert loaded.provider == "yahoo"
        assert loaded.timestamp == ts
        assert loaded.metadata["source"] == "test"
        assert len(loaded.results) == 1
        assert loaded.results[0].passed is True
        assert len(loaded.results[0].issues) == 1
        assert loaded.results[0].issues[0].severity == Severity.WARNING
        assert loaded.results[0].issues[0].check == "test_check"

    def test_save_creates_directory(self, tmp_path: Path):
        """Save creates parent directories."""
        report = ValidationReport(symbol="AAPL", provider="yahoo")
        path = tmp_path / "nested" / "dir" / "report.json"

        report.save(path)

        assert path.exists()

    def test_load_empty_metadata(self, tmp_path: Path):
        """Load handles missing metadata gracefully."""
        report = ValidationReport(symbol="AAPL", provider="yahoo")
        path = tmp_path / "report.json"
        report.save(path)

        loaded = ValidationReport.load(path)

        assert loaded.metadata == {}


class TestPrintSummary:
    """Test print_summary output."""

    def test_print_summary_passed(self, capsys):
        """print_summary shows passed status."""
        report = ValidationReport(symbol="AAPL", provider="yahoo")
        result = ValidationResult(passed=True)
        report.add_result(result, "Validator")

        report.print_summary()

        captured = capsys.readouterr()
        assert "AAPL" in captured.out
        assert "yahoo" in captured.out
        assert "PASSED" in captured.out

    def test_print_summary_failed(self, capsys):
        """print_summary shows failed status."""
        report = ValidationReport(symbol="AAPL", provider="yahoo")
        result = ValidationResult(passed=False)
        result.add_issue(
            ValidationIssue(severity=Severity.ERROR, check="test", message="Error occurred")
        )
        report.add_result(result, "Validator")

        report.print_summary()

        captured = capsys.readouterr()
        assert "FAILED" in captured.out
        assert "Error occurred" in captured.out

    def test_print_summary_counts(self, capsys):
        """print_summary shows issue counts."""
        report = ValidationReport(symbol="AAPL", provider="yahoo")
        result = ValidationResult(passed=True)
        result.add_issue(
            ValidationIssue(severity=Severity.CRITICAL, check="c1", message="Critical")
        )
        result.add_issue(ValidationIssue(severity=Severity.ERROR, check="e1", message="Error"))
        result.add_issue(ValidationIssue(severity=Severity.WARNING, check="w1", message="Warning"))
        report.add_result(result, "Validator")

        report.print_summary()

        captured = capsys.readouterr()
        assert "Critical: 1" in captured.out
        assert "Errors: 1" in captured.out
        assert "Warnings: 1" in captured.out

    def test_print_summary_truncates_issues(self, capsys):
        """print_summary truncates at 5 issues."""
        report = ValidationReport(symbol="AAPL", provider="yahoo")
        result = ValidationResult(passed=True)
        for i in range(10):
            result.add_issue(
                ValidationIssue(
                    severity=Severity.WARNING, check=f"check{i}", message=f"Warning {i}"
                )
            )
        report.add_result(result, "Validator")

        report.print_summary()

        captured = capsys.readouterr()
        assert "and 5 more issues" in captured.out

    def test_print_summary_shows_row_count(self, capsys):
        """print_summary shows affected row count."""
        report = ValidationReport(symbol="AAPL", provider="yahoo")
        result = ValidationResult(passed=True)
        result.add_issue(
            ValidationIssue(
                severity=Severity.ERROR,
                check="test",
                message="Error",
                row_count=42,
            )
        )
        report.add_result(result, "Validator")

        report.print_summary()

        captured = capsys.readouterr()
        assert "42" in captured.out

    def test_print_summary_shows_duration(self, capsys):
        """print_summary shows validation duration."""
        report = ValidationReport(symbol="AAPL", provider="yahoo")
        result = ValidationResult(passed=True, duration_seconds=2.5)
        report.add_result(result, "Validator")

        report.print_summary()

        captured = capsys.readouterr()
        assert "2.50s" in captured.out


class TestGetSeverityIcon:
    """Test severity icon getter."""

    def test_info_icon(self):
        """INFO severity returns info icon."""
        report = ValidationReport(symbol="AAPL", provider="yahoo")
        icon = report._get_severity_icon(Severity.INFO)
        assert "ℹ" in icon

    def test_warning_icon(self):
        """WARNING severity returns warning icon."""
        report = ValidationReport(symbol="AAPL", provider="yahoo")
        icon = report._get_severity_icon(Severity.WARNING)
        assert "⚠" in icon

    def test_error_icon(self):
        """ERROR severity returns error icon."""
        report = ValidationReport(symbol="AAPL", provider="yahoo")
        icon = report._get_severity_icon(Severity.ERROR)
        assert "❌" in icon

    def test_critical_icon(self):
        """CRITICAL severity returns critical icon."""
        report = ValidationReport(symbol="AAPL", provider="yahoo")
        icon = report._get_severity_icon(Severity.CRITICAL)
        assert "🚨" in icon


class TestToDataFrame:
    """Test DataFrame conversion."""

    def test_to_dataframe_with_issues(self):
        """to_dataframe returns DataFrame with issues."""
        report = ValidationReport(symbol="AAPL", provider="yahoo")
        result = ValidationResult(passed=True)
        result.add_issue(
            ValidationIssue(
                severity=Severity.WARNING,
                check="test_check",
                message="Test warning",
                row_count=5,
            )
        )
        result.add_issue(
            ValidationIssue(
                severity=Severity.ERROR,
                check="test_check2",
                message="Test error",
                row_count=10,
            )
        )
        report.add_result(result, "TestValidator")

        df = report.to_dataframe()

        assert isinstance(df, pl.DataFrame)
        assert len(df) == 2
        assert "validator" in df.columns
        assert "severity" in df.columns
        assert "check" in df.columns
        assert "message" in df.columns
        assert "row_count" in df.columns
        assert "timestamp" in df.columns

    def test_to_dataframe_empty(self):
        """to_dataframe returns empty DataFrame with schema."""
        report = ValidationReport(symbol="AAPL", provider="yahoo")

        df = report.to_dataframe()

        assert isinstance(df, pl.DataFrame)
        assert len(df) == 0
        assert "validator" in df.columns
        assert "severity" in df.columns

    def test_to_dataframe_null_row_count(self):
        """to_dataframe handles null row_count."""
        report = ValidationReport(symbol="AAPL", provider="yahoo")
        result = ValidationResult(passed=True)
        result.add_issue(
            ValidationIssue(
                severity=Severity.INFO,
                check="test",
                message="Info",
                row_count=None,
            )
        )
        report.add_result(result, "Validator")

        df = report.to_dataframe()

        assert df["row_count"][0] == 0  # None converted to 0

    def test_to_dataframe_multiple_validators(self):
        """to_dataframe includes all validators."""
        report = ValidationReport(symbol="AAPL", provider="yahoo")

        result1 = ValidationResult(passed=True)
        result1.add_issue(ValidationIssue(severity=Severity.WARNING, check="c1", message="W1"))
        report.add_result(result1, "Validator1")

        result2 = ValidationResult(passed=True)
        result2.add_issue(ValidationIssue(severity=Severity.ERROR, check="c2", message="E1"))
        report.add_result(result2, "Validator2")

        df = report.to_dataframe()

        validators = df["validator"].unique().to_list()
        assert "Validator1" in validators
        assert "Validator2" in validators
