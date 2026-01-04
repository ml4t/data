"""Comprehensive tests for anomaly detection base classes.

Tests cover:
- AnomalySeverity enum
- AnomalyType enum
- Anomaly data model
- AnomalyDetector abstract base class
- AnomalyReport aggregation and filtering
"""

from __future__ import annotations

from datetime import datetime

import polars as pl
import pytest

from ml4t.data.anomaly.base import (
    Anomaly,
    AnomalyDetector,
    AnomalyReport,
    AnomalySeverity,
    AnomalyType,
)


class TestAnomalySeverity:
    """Test AnomalySeverity enum."""

    def test_all_severity_levels_exist(self):
        """All expected severity levels exist."""
        assert AnomalySeverity.INFO.value == "info"
        assert AnomalySeverity.WARNING.value == "warning"
        assert AnomalySeverity.ERROR.value == "error"
        assert AnomalySeverity.CRITICAL.value == "critical"

    def test_severity_is_string_enum(self):
        """Severity is a string enum."""
        assert isinstance(AnomalySeverity.INFO, str)
        assert AnomalySeverity.INFO == "info"

    def test_severity_from_string(self):
        """Severity can be created from string."""
        assert AnomalySeverity("info") == AnomalySeverity.INFO
        assert AnomalySeverity("critical") == AnomalySeverity.CRITICAL

    def test_invalid_severity_raises(self):
        """Invalid severity raises ValueError."""
        with pytest.raises(ValueError):
            AnomalySeverity("invalid")


class TestAnomalyType:
    """Test AnomalyType enum."""

    def test_all_anomaly_types_exist(self):
        """All expected anomaly types exist."""
        assert AnomalyType.RETURN_OUTLIER.value == "return_outlier"
        assert AnomalyType.VOLUME_SPIKE.value == "volume_spike"
        assert AnomalyType.PRICE_STALE.value == "price_stale"
        assert AnomalyType.DATA_GAP.value == "data_gap"
        assert AnomalyType.PRICE_SPIKE.value == "price_spike"
        assert AnomalyType.ZERO_VOLUME.value == "zero_volume"
        assert AnomalyType.NEGATIVE_PRICE.value == "negative_price"

    def test_type_is_string_enum(self):
        """Type is a string enum."""
        assert isinstance(AnomalyType.RETURN_OUTLIER, str)
        assert AnomalyType.RETURN_OUTLIER == "return_outlier"

    def test_type_from_string(self):
        """Type can be created from string."""
        assert AnomalyType("volume_spike") == AnomalyType.VOLUME_SPIKE
        assert AnomalyType("price_stale") == AnomalyType.PRICE_STALE


class TestAnomaly:
    """Test Anomaly data model."""

    @pytest.fixture
    def sample_anomaly(self) -> Anomaly:
        """Create a sample anomaly."""
        return Anomaly(
            timestamp=datetime(2024, 1, 15, 10, 30, 0),
            symbol="AAPL",
            type=AnomalyType.RETURN_OUTLIER,
            severity=AnomalySeverity.WARNING,
            value=5.5,
            message="Unusual return of 5.5%",
        )

    def test_required_fields(self, sample_anomaly: Anomaly):
        """Required fields are set correctly."""
        assert sample_anomaly.timestamp == datetime(2024, 1, 15, 10, 30, 0)
        assert sample_anomaly.symbol == "AAPL"
        assert sample_anomaly.type == AnomalyType.RETURN_OUTLIER
        assert sample_anomaly.severity == AnomalySeverity.WARNING
        assert sample_anomaly.value == 5.5
        assert sample_anomaly.message == "Unusual return of 5.5%"

    def test_optional_fields_default(self):
        """Optional fields have correct defaults."""
        anomaly = Anomaly(
            timestamp=datetime.now(),
            symbol="TEST",
            type=AnomalyType.VOLUME_SPIKE,
            severity=AnomalySeverity.INFO,
            value=1000000,
            message="High volume",
        )

        assert anomaly.expected_range is None
        assert anomaly.threshold is None
        assert anomaly.metadata == {}

    def test_optional_fields_set(self):
        """Optional fields can be set."""
        anomaly = Anomaly(
            timestamp=datetime.now(),
            symbol="TEST",
            type=AnomalyType.RETURN_OUTLIER,
            severity=AnomalySeverity.ERROR,
            value=10.0,
            expected_range=(-3.0, 3.0),
            threshold=3.0,
            message="Extreme return",
            metadata={"z_score": 4.5, "method": "mad"},
        )

        assert anomaly.expected_range == (-3.0, 3.0)
        assert anomaly.threshold == 3.0
        assert anomaly.metadata["z_score"] == 4.5

    def test_str_representation(self, sample_anomaly: Anomaly):
        """String representation includes key info."""
        result = str(sample_anomaly)

        assert "[WARNING]" in result
        assert "AAPL" in result
        assert "Unusual return" in result

    def test_critical_str_representation(self):
        """Critical anomaly string representation."""
        anomaly = Anomaly(
            timestamp=datetime(2024, 1, 15),
            symbol="GME",
            type=AnomalyType.VOLUME_SPIKE,
            severity=AnomalySeverity.CRITICAL,
            value=50000000,
            message="Massive volume spike",
        )

        result = str(anomaly)

        assert "[CRITICAL]" in result
        assert "GME" in result

    def test_model_dump(self, sample_anomaly: Anomaly):
        """Anomaly can be serialized."""
        data = sample_anomaly.model_dump()

        assert data["symbol"] == "AAPL"
        assert data["value"] == 5.5
        assert data["type"] == "return_outlier"
        assert data["severity"] == "warning"

    def test_model_from_dict(self):
        """Anomaly can be created from dict."""
        data = {
            "timestamp": datetime(2024, 1, 15),
            "symbol": "TSLA",
            "type": "volume_spike",
            "severity": "error",
            "value": 1000000,
            "message": "Volume anomaly",
        }

        anomaly = Anomaly(**data)

        assert anomaly.symbol == "TSLA"
        assert anomaly.type == AnomalyType.VOLUME_SPIKE
        assert anomaly.severity == AnomalySeverity.ERROR


class TestAnomalyDetectorABC:
    """Test AnomalyDetector abstract base class."""

    def test_cannot_instantiate_directly(self):
        """Cannot instantiate abstract class directly."""
        with pytest.raises(TypeError):
            AnomalyDetector()

    def test_concrete_implementation_works(self):
        """Concrete implementation can be instantiated."""

        class TestDetector(AnomalyDetector):
            @property
            def name(self) -> str:
                return "test_detector"

            def detect(self, df: pl.DataFrame, symbol: str) -> list[Anomaly]:
                return []

        detector = TestDetector(enabled=True)
        assert detector.is_enabled() is True
        assert detector.name == "test_detector"

    def test_enabled_state(self):
        """Enabled state can be controlled."""

        class TestDetector(AnomalyDetector):
            @property
            def name(self) -> str:
                return "test"

            def detect(self, df: pl.DataFrame, symbol: str) -> list[Anomaly]:
                return []

        enabled = TestDetector(enabled=True)
        disabled = TestDetector(enabled=False)

        assert enabled.is_enabled() is True
        assert disabled.is_enabled() is False

    def test_default_enabled(self):
        """Default enabled state is True."""

        class TestDetector(AnomalyDetector):
            @property
            def name(self) -> str:
                return "test"

            def detect(self, df: pl.DataFrame, symbol: str) -> list[Anomaly]:
                return []

        detector = TestDetector()
        assert detector.is_enabled() is True


class TestAnomalyReport:
    """Test AnomalyReport aggregation class."""

    @pytest.fixture
    def sample_report(self) -> AnomalyReport:
        """Create a sample report."""
        return AnomalyReport(
            symbol="AAPL",
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 12, 31),
            total_rows=252,
        )

    @pytest.fixture
    def sample_anomalies(self) -> list[Anomaly]:
        """Create sample anomalies of different types and severities."""
        return [
            Anomaly(
                timestamp=datetime(2024, 1, 15),
                symbol="AAPL",
                type=AnomalyType.RETURN_OUTLIER,
                severity=AnomalySeverity.INFO,
                value=2.5,
                message="Minor return anomaly",
            ),
            Anomaly(
                timestamp=datetime(2024, 3, 20),
                symbol="AAPL",
                type=AnomalyType.VOLUME_SPIKE,
                severity=AnomalySeverity.WARNING,
                value=5000000,
                message="Volume spike",
            ),
            Anomaly(
                timestamp=datetime(2024, 6, 15),
                symbol="AAPL",
                type=AnomalyType.RETURN_OUTLIER,
                severity=AnomalySeverity.ERROR,
                value=8.0,
                message="Major return anomaly",
            ),
            Anomaly(
                timestamp=datetime(2024, 9, 10),
                symbol="AAPL",
                type=AnomalyType.PRICE_STALE,
                severity=AnomalySeverity.CRITICAL,
                value=150.0,
                message="Price unchanged for 10 days",
            ),
        ]

    def test_initialization(self, sample_report: AnomalyReport):
        """Report is initialized correctly."""
        assert sample_report.symbol == "AAPL"
        assert sample_report.start_date == datetime(2024, 1, 1)
        assert sample_report.end_date == datetime(2024, 12, 31)
        assert sample_report.total_rows == 252
        assert sample_report.anomalies == []
        assert sample_report.summary == {}
        assert sample_report.detectors_used == []

    def test_add_anomaly(self, sample_report: AnomalyReport, sample_anomalies: list[Anomaly]):
        """Adding anomalies updates report."""
        sample_report.add_anomaly(sample_anomalies[0])

        assert len(sample_report.anomalies) == 1
        assert sample_report.anomalies[0].type == AnomalyType.RETURN_OUTLIER

    def test_add_anomaly_updates_summary(
        self, sample_report: AnomalyReport, sample_anomalies: list[Anomaly]
    ):
        """Adding anomalies updates summary statistics."""
        sample_report.add_anomaly(sample_anomalies[0])  # INFO, RETURN_OUTLIER

        assert sample_report.summary["info_count"] == 1
        assert sample_report.summary["return_outlier_count"] == 1

    def test_summary_accumulates(
        self, sample_report: AnomalyReport, sample_anomalies: list[Anomaly]
    ):
        """Summary accumulates across multiple anomalies."""
        for anomaly in sample_anomalies:
            sample_report.add_anomaly(anomaly)

        assert sample_report.summary["info_count"] == 1
        assert sample_report.summary["warning_count"] == 1
        assert sample_report.summary["error_count"] == 1
        assert sample_report.summary["critical_count"] == 1
        assert sample_report.summary["return_outlier_count"] == 2
        assert sample_report.summary["volume_spike_count"] == 1
        assert sample_report.summary["price_stale_count"] == 1

    def test_get_critical_anomalies(
        self, sample_report: AnomalyReport, sample_anomalies: list[Anomaly]
    ):
        """Get only critical anomalies."""
        for anomaly in sample_anomalies:
            sample_report.add_anomaly(anomaly)

        critical = sample_report.get_critical_anomalies()

        assert len(critical) == 1
        assert critical[0].severity == AnomalySeverity.CRITICAL
        assert critical[0].type == AnomalyType.PRICE_STALE

    def test_get_critical_anomalies_empty(self, sample_report: AnomalyReport):
        """Get critical anomalies when none exist."""
        # Add only INFO anomaly
        sample_report.add_anomaly(
            Anomaly(
                timestamp=datetime.now(),
                symbol="AAPL",
                type=AnomalyType.VOLUME_SPIKE,
                severity=AnomalySeverity.INFO,
                value=100,
                message="Minor spike",
            )
        )

        critical = sample_report.get_critical_anomalies()

        assert len(critical) == 0

    def test_get_by_type(self, sample_report: AnomalyReport, sample_anomalies: list[Anomaly]):
        """Get anomalies by type."""
        for anomaly in sample_anomalies:
            sample_report.add_anomaly(anomaly)

        return_outliers = sample_report.get_by_type(AnomalyType.RETURN_OUTLIER)
        volume_spikes = sample_report.get_by_type(AnomalyType.VOLUME_SPIKE)
        price_stale = sample_report.get_by_type(AnomalyType.PRICE_STALE)

        assert len(return_outliers) == 2
        assert len(volume_spikes) == 1
        assert len(price_stale) == 1

    def test_get_by_type_not_found(self, sample_report: AnomalyReport):
        """Get anomalies by type when none exist."""
        result = sample_report.get_by_type(AnomalyType.DATA_GAP)
        assert len(result) == 0

    def test_has_critical_issues_true(
        self, sample_report: AnomalyReport, sample_anomalies: list[Anomaly]
    ):
        """has_critical_issues returns True when critical exists."""
        for anomaly in sample_anomalies:
            sample_report.add_anomaly(anomaly)

        assert sample_report.has_critical_issues() is True

    def test_has_critical_issues_false(self, sample_report: AnomalyReport):
        """has_critical_issues returns False when no critical."""
        sample_report.add_anomaly(
            Anomaly(
                timestamp=datetime.now(),
                symbol="AAPL",
                type=AnomalyType.VOLUME_SPIKE,
                severity=AnomalySeverity.WARNING,
                value=100,
                message="Warning only",
            )
        )

        assert sample_report.has_critical_issues() is False

    def test_to_dataframe_empty(self, sample_report: AnomalyReport):
        """Empty report returns empty DataFrame."""
        df = sample_report.to_dataframe()

        assert isinstance(df, pl.DataFrame)
        assert len(df) == 0

    def test_to_dataframe_with_anomalies(
        self, sample_report: AnomalyReport, sample_anomalies: list[Anomaly]
    ):
        """Report with anomalies converts to DataFrame."""
        for anomaly in sample_anomalies:
            sample_report.add_anomaly(anomaly)

        df = sample_report.to_dataframe()

        assert len(df) == 4
        assert "timestamp" in df.columns
        assert "symbol" in df.columns
        assert "type" in df.columns
        assert "severity" in df.columns
        assert "value" in df.columns
        assert "message" in df.columns

    def test_to_dataframe_values(
        self, sample_report: AnomalyReport, sample_anomalies: list[Anomaly]
    ):
        """DataFrame contains correct values."""
        sample_report.add_anomaly(sample_anomalies[0])

        df = sample_report.to_dataframe()

        assert df["symbol"][0] == "AAPL"
        assert df["type"][0] == "return_outlier"
        assert df["severity"][0] == "info"
        assert df["value"][0] == 2.5

    def test_detectors_used_tracking(self, sample_report: AnomalyReport):
        """Detectors used are tracked."""
        sample_report.detectors_used.append("return_outliers")
        sample_report.detectors_used.append("volume_spikes")

        assert len(sample_report.detectors_used) == 2
        assert "return_outliers" in sample_report.detectors_used
        assert "volume_spikes" in sample_report.detectors_used


class TestAnomalyReportSerialization:
    """Test AnomalyReport serialization."""

    def test_empty_report_serialization(self):
        """Empty report can be serialized."""
        report = AnomalyReport(
            symbol="TEST",
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 12, 31),
            total_rows=100,
        )

        data = report.model_dump()

        assert data["symbol"] == "TEST"
        assert data["total_rows"] == 100
        assert data["anomalies"] == []

    def test_report_with_anomalies_serialization(self):
        """Report with anomalies can be serialized."""
        report = AnomalyReport(
            symbol="TEST",
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 12, 31),
            total_rows=100,
        )

        report.add_anomaly(
            Anomaly(
                timestamp=datetime(2024, 6, 15),
                symbol="TEST",
                type=AnomalyType.RETURN_OUTLIER,
                severity=AnomalySeverity.WARNING,
                value=5.0,
                message="Test anomaly",
            )
        )

        data = report.model_dump()

        assert len(data["anomalies"]) == 1
        assert data["anomalies"][0]["symbol"] == "TEST"
        assert data["anomalies"][0]["type"] == "return_outlier"

    def test_json_serialization(self):
        """Report can be serialized to JSON."""
        report = AnomalyReport(
            symbol="AAPL",
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 12, 31),
            total_rows=252,
            detectors_used=["return_outliers", "volume_spikes"],
        )

        json_str = report.model_dump_json()

        assert "AAPL" in json_str
        assert "252" in json_str
        assert "return_outliers" in json_str
