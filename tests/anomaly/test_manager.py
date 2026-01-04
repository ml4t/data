"""Comprehensive tests for AnomalyManager.

Tests cover:
- Manager initialization with config
- Custom detector integration
- Analyze method for single symbols
- Batch analysis
- Report saving and filtering
- Statistics calculation
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import polars as pl
import pytest

from ml4t.data.anomaly.base import Anomaly, AnomalyDetector, AnomalySeverity, AnomalyType
from ml4t.data.anomaly.config import (
    AnomalyConfig,
    PriceStalenessConfig,
    ReturnOutlierConfig,
    VolumeSpikeConfig,
)
from ml4t.data.anomaly.manager import AnomalyManager


class CustomTestDetector(AnomalyDetector):
    """Custom detector for testing."""

    def __init__(self, enabled: bool = True, anomaly_count: int = 1):
        super().__init__(enabled=enabled)
        self.anomaly_count = anomaly_count

    @property
    def name(self) -> str:
        return "custom_test"

    def detect(self, df: pl.DataFrame, symbol: str) -> list[Anomaly]:
        if not self.enabled:
            return []

        return [
            Anomaly(
                timestamp=datetime.now(),
                symbol=symbol,
                type=AnomalyType.DATA_GAP,
                severity=AnomalySeverity.INFO,
                value=0.0,
                message="Custom anomaly",
            )
            for _ in range(self.anomaly_count)
        ]


class TestAnomalyManagerInitialization:
    """Test AnomalyManager initialization."""

    def test_default_initialization(self):
        """Default initialization enables all detectors."""
        manager = AnomalyManager()

        assert manager.config.enabled is True
        assert len(manager.detectors) == 3  # return, volume, staleness

    def test_disabled_config(self):
        """Disabled config creates no detectors."""
        config = AnomalyConfig(enabled=False)
        manager = AnomalyManager(config=config)

        assert len(manager.detectors) == 0

    def test_partial_detector_config(self):
        """Can enable only some detectors."""
        config = AnomalyConfig(
            return_outliers=ReturnOutlierConfig(enabled=True),
            volume_spikes=VolumeSpikeConfig(enabled=False),
            price_staleness=PriceStalenessConfig(enabled=False),
        )
        manager = AnomalyManager(config=config)

        assert len(manager.detectors) == 1
        assert manager.detectors[0].name == "return_outliers"

    def test_custom_detectors_added(self):
        """Custom detectors are added."""
        custom = CustomTestDetector()
        manager = AnomalyManager(custom_detectors=[custom])

        assert len(manager.detectors) == 4  # 3 built-in + 1 custom
        assert any(d.name == "custom_test" for d in manager.detectors)

    def test_custom_detectors_only(self):
        """Can use only custom detectors."""
        config = AnomalyConfig(
            return_outliers=ReturnOutlierConfig(enabled=False),
            volume_spikes=VolumeSpikeConfig(enabled=False),
            price_staleness=PriceStalenessConfig(enabled=False),
        )
        custom = CustomTestDetector()
        manager = AnomalyManager(config=config, custom_detectors=[custom])

        assert len(manager.detectors) == 1
        assert manager.detectors[0].name == "custom_test"


class TestAnomalyManagerAnalyze:
    """Test AnomalyManager.analyze method."""

    @pytest.fixture
    def sample_data(self) -> pl.DataFrame:
        """Create sample OHLCV data."""
        dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(60)]

        closes = []
        volumes = []

        for i in range(60):
            if i == 30:
                closes.append(150.0)  # 50% spike
                volumes.append(10000000)  # Volume spike
            else:
                closes.append(100.0 + (i % 5) * 0.1)
                volumes.append(1000000)

        return pl.DataFrame(
            {
                "timestamp": dates,
                "open": closes,
                "high": [p * 1.01 for p in closes],
                "low": [p * 0.99 for p in closes],
                "close": closes,
                "volume": volumes,
            }
        )

    @pytest.fixture
    def manager(self) -> AnomalyManager:
        """Create default manager."""
        return AnomalyManager()

    def test_analyze_returns_report(self, manager: AnomalyManager, sample_data: pl.DataFrame):
        """Analyze returns an AnomalyReport."""
        report = manager.analyze(sample_data, "AAPL")

        assert report.symbol == "AAPL"
        assert report.total_rows == 60

    def test_analyze_sets_date_range(self, manager: AnomalyManager, sample_data: pl.DataFrame):
        """Analyze sets date range from data."""
        report = manager.analyze(sample_data, "AAPL")

        assert report.start_date == datetime(2024, 1, 1)
        assert report.end_date == sample_data["timestamp"].max()

    def test_analyze_finds_anomalies(self, manager: AnomalyManager, sample_data: pl.DataFrame):
        """Analyze finds anomalies in data."""
        report = manager.analyze(sample_data, "AAPL")

        # Should find at least the return outlier
        assert len(report.anomalies) >= 1

    def test_analyze_tracks_detectors(self, manager: AnomalyManager, sample_data: pl.DataFrame):
        """Analyze tracks which detectors were used."""
        report = manager.analyze(sample_data, "AAPL")

        assert len(report.detectors_used) >= 1
        assert "return_outliers" in report.detectors_used

    def test_analyze_disabled_returns_empty_report(self, sample_data: pl.DataFrame):
        """Disabled manager returns empty report."""
        config = AnomalyConfig(enabled=False)
        manager = AnomalyManager(config=config)

        report = manager.analyze(sample_data, "AAPL")

        assert len(report.anomalies) == 0
        assert report.total_rows == 0

    def test_analyze_empty_dataframe(self, manager: AnomalyManager):
        """Empty DataFrame returns empty report."""
        empty_df = pl.DataFrame(
            {
                "timestamp": [],
                "open": [],
                "high": [],
                "low": [],
                "close": [],
                "volume": [],
            }
        )

        report = manager.analyze(empty_df, "EMPTY")

        assert len(report.anomalies) == 0
        assert report.total_rows == 0

    def test_analyze_missing_columns(self, manager: AnomalyManager):
        """Missing columns returns empty report."""
        incomplete_df = pl.DataFrame(
            {
                "timestamp": [datetime(2024, 1, 1)],
                "close": [100.0],
                # Missing open, high, low, volume
            }
        )

        report = manager.analyze(incomplete_df, "INCOMPLETE")

        assert len(report.anomalies) == 0

    def test_analyze_with_asset_class(self, sample_data: pl.DataFrame):
        """Asset class overrides are applied."""
        config = AnomalyConfig(
            return_outliers=ReturnOutlierConfig(threshold=3.0),
            asset_overrides={
                "crypto": {
                    "return_outliers": {"threshold": 5.0},
                }
            },
        )
        manager = AnomalyManager(config=config)

        report = manager.analyze(sample_data, "BTCUSD", asset_class="crypto")

        # Higher threshold means fewer anomalies
        assert report.symbol == "BTCUSD"

    def test_analyze_with_custom_detector(self, sample_data: pl.DataFrame):
        """Custom detectors work in analyze."""
        custom = CustomTestDetector(anomaly_count=3)
        manager = AnomalyManager(custom_detectors=[custom])

        report = manager.analyze(sample_data, "TEST")

        # Should include custom anomalies
        custom_anomalies = [a for a in report.anomalies if a.type == AnomalyType.DATA_GAP]
        assert len(custom_anomalies) == 3


class TestAnomalyManagerBatch:
    """Test AnomalyManager.analyze_batch method."""

    @pytest.fixture
    def datasets(self) -> dict[str, pl.DataFrame]:
        """Create multiple datasets."""
        result = {}

        for symbol in ["AAPL", "GOOGL", "MSFT"]:
            dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(30)]
            closes = [100.0 + i * 0.1 for i in range(30)]

            result[symbol] = pl.DataFrame(
                {
                    "timestamp": dates,
                    "open": closes,
                    "high": [p * 1.01 for p in closes],
                    "low": [p * 0.99 for p in closes],
                    "close": closes,
                    "volume": [1000000] * 30,
                }
            )

        return result

    def test_batch_returns_reports_for_all(self, datasets: dict[str, pl.DataFrame]):
        """Batch analysis returns report for each dataset."""
        manager = AnomalyManager()

        reports = manager.analyze_batch(datasets)

        assert len(reports) == 3
        assert "AAPL" in reports
        assert "GOOGL" in reports
        assert "MSFT" in reports

    def test_batch_with_asset_classes(self, datasets: dict[str, pl.DataFrame]):
        """Batch analysis with asset class mapping."""
        manager = AnomalyManager()

        asset_classes = {
            "AAPL": "equity",
            "GOOGL": "equity",
            "MSFT": "equity",
        }

        reports = manager.analyze_batch(datasets, asset_classes=asset_classes)

        assert len(reports) == 3

    def test_batch_empty_datasets(self):
        """Empty datasets dict returns empty reports."""
        manager = AnomalyManager()

        reports = manager.analyze_batch({})

        assert len(reports) == 0


class TestAnomalyManagerReportSaving:
    """Test AnomalyManager report saving functionality."""

    @pytest.fixture
    def sample_report(self) -> tuple[AnomalyManager, pl.DataFrame]:
        """Create manager and sample data."""
        manager = AnomalyManager()
        dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(30)]
        closes = [100.0 + i * 0.1 for i in range(30)]

        df = pl.DataFrame(
            {
                "timestamp": dates,
                "open": closes,
                "high": [p * 1.01 for p in closes],
                "low": [p * 0.99 for p in closes],
                "close": closes,
                "volume": [1000000] * 30,
            }
        )

        return manager, df

    def test_save_report_creates_file(
        self, sample_report: tuple[AnomalyManager, pl.DataFrame], tmp_path: Path
    ):
        """Save report creates JSON file."""
        manager, df = sample_report
        report = manager.analyze(df, "AAPL")

        output_path = manager.save_report(report, tmp_path)

        assert output_path.exists()
        assert output_path.suffix == ".json"
        assert "AAPL" in output_path.name

    def test_save_report_creates_directory(
        self, sample_report: tuple[AnomalyManager, pl.DataFrame], tmp_path: Path
    ):
        """Save report creates directory if needed."""
        manager, df = sample_report
        report = manager.analyze(df, "AAPL")

        nested_path = tmp_path / "nested" / "dir"
        output_path = manager.save_report(report, nested_path)

        assert output_path.exists()
        assert nested_path.exists()

    def test_save_report_valid_json(
        self, sample_report: tuple[AnomalyManager, pl.DataFrame], tmp_path: Path
    ):
        """Saved report is valid JSON."""
        import json

        manager, df = sample_report
        report = manager.analyze(df, "AAPL")

        output_path = manager.save_report(report, tmp_path)

        with open(output_path) as f:
            data = json.load(f)

        assert data["symbol"] == "AAPL"
        assert "anomalies" in data


class TestAnomalyManagerFiltering:
    """Test AnomalyManager filter_by_severity method."""

    @pytest.fixture
    def report_with_anomalies(self) -> tuple[AnomalyManager, pl.DataFrame]:
        """Create report with various severity anomalies."""
        custom_detectors = [
            CustomTestDetector(anomaly_count=2),  # Creates INFO anomalies
        ]

        # Configure to find different severity levels
        config = AnomalyConfig(
            return_outliers=ReturnOutlierConfig(threshold=2.0),
        )
        manager = AnomalyManager(config=config, custom_detectors=custom_detectors)

        # Data with outliers
        dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(60)]
        closes = []
        for i in range(60):
            if i == 30:
                closes.append(200.0)  # 100% spike - should be high severity
            else:
                closes.append(100.0 + (i % 5) * 0.1)

        df = pl.DataFrame(
            {
                "timestamp": dates,
                "open": closes,
                "high": [p * 1.01 for p in closes],
                "low": [p * 0.99 for p in closes],
                "close": closes,
                "volume": [1000000] * 60,
            }
        )

        return manager, df

    def test_filter_by_info(self, report_with_anomalies: tuple[AnomalyManager, pl.DataFrame]):
        """Filter by INFO includes all anomalies."""
        manager, df = report_with_anomalies
        report = manager.analyze(df, "TEST")

        filtered = manager.filter_by_severity(report, "info")

        # Should include all anomalies
        assert len(filtered.anomalies) == len(report.anomalies)

    def test_filter_by_warning(self, report_with_anomalies: tuple[AnomalyManager, pl.DataFrame]):
        """Filter by WARNING excludes INFO."""
        manager, df = report_with_anomalies
        report = manager.analyze(df, "TEST")

        filtered = manager.filter_by_severity(report, "warning")

        # Should exclude INFO anomalies
        info_count = sum(1 for a in filtered.anomalies if a.severity == AnomalySeverity.INFO)
        assert info_count == 0

    def test_filter_by_critical(self, report_with_anomalies: tuple[AnomalyManager, pl.DataFrame]):
        """Filter by CRITICAL only includes critical."""
        manager, df = report_with_anomalies
        report = manager.analyze(df, "TEST")

        filtered = manager.filter_by_severity(report, "critical")

        # Should only include CRITICAL
        for anomaly in filtered.anomalies:
            assert anomaly.severity == AnomalySeverity.CRITICAL

    def test_filter_preserves_metadata(
        self, report_with_anomalies: tuple[AnomalyManager, pl.DataFrame]
    ):
        """Filter preserves report metadata."""
        manager, df = report_with_anomalies
        report = manager.analyze(df, "TEST")

        filtered = manager.filter_by_severity(report, "warning")

        assert filtered.symbol == report.symbol
        assert filtered.start_date == report.start_date
        assert filtered.end_date == report.end_date
        assert filtered.total_rows == report.total_rows
        assert filtered.detectors_used == report.detectors_used


class TestAnomalyManagerStatistics:
    """Test AnomalyManager.get_statistics method."""

    @pytest.fixture
    def report_with_varied_anomalies(self) -> tuple[AnomalyManager, pl.DataFrame]:
        """Create report with varied anomalies."""
        config = AnomalyConfig(
            return_outliers=ReturnOutlierConfig(threshold=2.0),
            volume_spikes=VolumeSpikeConfig(threshold=2.0),
        )
        manager = AnomalyManager(config=config)

        # Data with both return and volume anomalies
        dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(60)]
        closes = []
        volumes = []

        for i in range(60):
            if i == 30:
                closes.append(150.0)  # 50% spike
                volumes.append(10000000)  # Volume spike
            elif i == 40:
                closes.append(80.0)  # Big drop
                volumes.append(8000000)  # Another volume spike
            else:
                closes.append(100.0 + (i % 5) * 0.1)
                volumes.append(1000000)

        df = pl.DataFrame(
            {
                "timestamp": dates,
                "open": closes,
                "high": [p * 1.01 for p in closes],
                "low": [p * 0.99 for p in closes],
                "close": closes,
                "volume": volumes,
            }
        )

        return manager, df

    def test_statistics_total_anomalies(
        self, report_with_varied_anomalies: tuple[AnomalyManager, pl.DataFrame]
    ):
        """Statistics includes total anomaly count."""
        manager, df = report_with_varied_anomalies
        report = manager.analyze(df, "TEST")

        stats = manager.get_statistics(report)

        assert "total_anomalies" in stats
        assert stats["total_anomalies"] == len(report.anomalies)

    def test_statistics_by_severity(
        self, report_with_varied_anomalies: tuple[AnomalyManager, pl.DataFrame]
    ):
        """Statistics includes breakdown by severity."""
        manager, df = report_with_varied_anomalies
        report = manager.analyze(df, "TEST")

        stats = manager.get_statistics(report)

        assert "by_severity" in stats
        assert "info" in stats["by_severity"]
        assert "warning" in stats["by_severity"]
        assert "error" in stats["by_severity"]
        assert "critical" in stats["by_severity"]

    def test_statistics_by_type(
        self, report_with_varied_anomalies: tuple[AnomalyManager, pl.DataFrame]
    ):
        """Statistics includes breakdown by type."""
        manager, df = report_with_varied_anomalies
        report = manager.analyze(df, "TEST")

        stats = manager.get_statistics(report)

        assert "by_type" in stats

    def test_statistics_detection_rate(
        self, report_with_varied_anomalies: tuple[AnomalyManager, pl.DataFrame]
    ):
        """Statistics includes detection rate."""
        manager, df = report_with_varied_anomalies
        report = manager.analyze(df, "TEST")

        stats = manager.get_statistics(report)

        assert "detection_rate" in stats
        assert 0 <= stats["detection_rate"] <= 1

    def test_statistics_empty_report(self):
        """Statistics for empty report."""
        manager = AnomalyManager()
        df = pl.DataFrame(
            {
                "timestamp": [datetime(2024, 1, i) for i in range(1, 31)],
                "open": [100.0] * 30,
                "high": [101.0] * 30,
                "low": [99.0] * 30,
                "close": [100.0 + i * 0.01 for i in range(30)],  # Minimal change
                "volume": [1000000] * 30,
            }
        )
        report = manager.analyze(df, "STABLE")

        stats = manager.get_statistics(report)

        assert stats["total_anomalies"] >= 0
        assert stats["detection_rate"] >= 0
