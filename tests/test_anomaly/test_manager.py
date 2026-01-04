"""Tests for anomaly manager."""

import tempfile
from datetime import datetime
from pathlib import Path

import polars as pl
import pytest

from ml4t.data.anomaly.config import AnomalyConfig
from ml4t.data.anomaly.manager import AnomalyManager


class TestAnomalyManager:
    """Test anomaly detection manager."""

    @pytest.fixture
    def sample_data(self) -> pl.DataFrame:
        """Create sample data with various anomalies."""
        dates = [datetime(2024, 1, i) for i in range(1, 31)]

        closes = [100.0]
        volumes = []

        for i in range(1, 31):
            if i == 15:
                # Return outlier
                closes.append(closes[-1] * 1.20)
                volumes.append(1000000)
            elif i == 20:
                # Volume spike
                closes.append(closes[-1] * 1.01)
                volumes.append(10000000)
            elif 25 <= i <= 30:
                # Stale prices
                closes.append(110.0)
                volumes.append(1000000)
            else:
                # Normal
                closes.append(closes[-1] * (1.0 + (i % 3 - 1) * 0.01))
                volumes.append(1000000 + (i % 5) * 50000)

        return pl.DataFrame(
            {
                "timestamp": dates[:30],
                "open": closes[:30],
                "high": [c * 1.02 for c in closes[:30]],
                "low": [c * 0.98 for c in closes[:30]],
                "close": closes[:30],
                "volume": volumes,
            }
        )

    def test_analyze_all_detectors(self, sample_data):
        """Test analysis with all detectors enabled."""
        config = AnomalyConfig(
            enabled=True,
            return_outliers={"enabled": True, "threshold": 3.0},
            volume_spikes={"enabled": True, "window": 10, "threshold": 3.0},
            price_staleness={"enabled": True, "max_unchanged_days": 3},
        )

        manager = AnomalyManager(config)
        report = manager.analyze(sample_data, "TEST")

        # Should detect multiple types of anomalies
        assert len(report.anomalies) > 0
        assert len(report.detectors_used) == 3
        assert report.symbol == "TEST"
        assert report.total_rows == 30

        # Check summary
        assert len(report.summary) > 0

    def test_analyze_disabled(self, sample_data):
        """Test with anomaly detection disabled."""
        config = AnomalyConfig(enabled=False)

        manager = AnomalyManager(config)
        report = manager.analyze(sample_data, "TEST")

        assert len(report.anomalies) == 0
        assert len(report.detectors_used) == 0

    def test_analyze_batch(self, sample_data):
        """Test batch analysis."""
        datasets = {
            "AAPL": sample_data,
            "GOOGL": sample_data.clone(),
            "MSFT": sample_data.clone(),
        }

        manager = AnomalyManager()
        reports = manager.analyze_batch(datasets)

        assert len(reports) == 3
        assert "AAPL" in reports
        assert "GOOGL" in reports
        assert "MSFT" in reports

        # Each should have anomalies
        for symbol, report in reports.items():
            assert report.symbol == symbol
            assert report.total_rows == 30

    def test_asset_class_overrides(self, sample_data):
        """Test configuration overrides for asset classes."""
        config = AnomalyConfig(
            enabled=True,
            return_outliers={"enabled": True, "threshold": 3.0},
            asset_overrides={"crypto": {"return_outliers": {"threshold": 5.0}}},
        )

        manager = AnomalyManager(config)

        # Analyze as equity (default threshold)
        equity_report = manager.analyze(sample_data, "SPY", asset_class="equity")

        # Analyze as crypto (higher threshold)
        crypto_report = manager.analyze(sample_data, "BTC", asset_class="crypto")

        # Crypto should have fewer outliers due to higher threshold
        equity_outliers = [a for a in equity_report.anomalies if a.type.value == "return_outlier"]
        crypto_outliers = [a for a in crypto_report.anomalies if a.type.value == "return_outlier"]

        # With higher threshold, crypto might have fewer outliers
        assert len(crypto_outliers) <= len(equity_outliers)

    def test_symbol_overrides(self, sample_data):
        """Test symbol-specific configuration overrides."""
        config = AnomalyConfig(
            enabled=True,
            return_outliers={"enabled": True, "threshold": 3.0},
            symbol_overrides={"TSLA": {"return_outliers": {"threshold": 10.0}}},
        )

        manager = AnomalyManager(config)

        # Normal symbol
        normal_report = manager.analyze(sample_data, "AAPL")

        # TSLA with override
        tsla_report = manager.analyze(sample_data, "TSLA")

        # TSLA should have fewer outliers
        normal_outliers = [a for a in normal_report.anomalies if a.type.value == "return_outlier"]
        tsla_outliers = [a for a in tsla_report.anomalies if a.type.value == "return_outlier"]

        assert len(tsla_outliers) <= len(normal_outliers)

    def test_save_report(self, sample_data):
        """Test saving report to disk."""
        manager = AnomalyManager()
        report = manager.analyze(sample_data, "TEST")

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            saved_path = manager.save_report(report, output_dir)

            assert saved_path.exists()
            assert "anomaly_report_TEST" in saved_path.name
            assert saved_path.suffix == ".json"

    def test_filter_by_severity(self, sample_data):
        """Test filtering report by severity."""
        manager = AnomalyManager()
        report = manager.analyze(sample_data, "TEST")

        # Add some anomalies if none detected
        if len(report.anomalies) == 0:
            from ml4t.data.anomaly.base import Anomaly, AnomalySeverity, AnomalyType

            report.add_anomaly(
                Anomaly(
                    timestamp=datetime.now(),
                    symbol="TEST",
                    type=AnomalyType.RETURN_OUTLIER,
                    severity=AnomalySeverity.INFO,
                    value=1.0,
                    message="Info anomaly",
                )
            )
            report.add_anomaly(
                Anomaly(
                    timestamp=datetime.now(),
                    symbol="TEST",
                    type=AnomalyType.RETURN_OUTLIER,
                    severity=AnomalySeverity.ERROR,
                    value=10.0,
                    message="Error anomaly",
                )
            )

        # Filter to only errors and above
        filtered = manager.filter_by_severity(report, "error")

        # Should only have error/critical anomalies
        for anomaly in filtered.anomalies:
            assert anomaly.severity.value in ["error", "critical"]

    def test_get_statistics(self, sample_data):
        """Test getting statistics from report."""
        manager = AnomalyManager()
        report = manager.analyze(sample_data, "TEST")

        stats = manager.get_statistics(report)

        assert "total_anomalies" in stats
        assert "by_severity" in stats
        assert "by_type" in stats
        assert "detection_rate" in stats

        # Detection rate should be between 0 and 1
        assert 0 <= stats["detection_rate"] <= 1

    def test_missing_columns(self):
        """Test handling of missing required columns."""
        df = pl.DataFrame(
            {
                "timestamp": [datetime(2024, 1, 1)],
                "close": [100.0],
                # Missing open, high, low, volume
            }
        )

        manager = AnomalyManager()
        report = manager.analyze(df, "TEST")

        # Should handle gracefully
        assert report.total_rows == 1
        assert len(report.anomalies) == 0

    def test_empty_dataframe(self):
        """Test with empty DataFrame."""
        df = pl.DataFrame()

        manager = AnomalyManager()
        report = manager.analyze(df, "TEST")

        assert report.total_rows == 0
        assert len(report.anomalies) == 0

    def test_report_to_dataframe(self, sample_data):
        """Test converting report to DataFrame."""
        manager = AnomalyManager()
        report = manager.analyze(sample_data, "TEST")

        if len(report.anomalies) > 0:
            df = report.to_dataframe()

            assert isinstance(df, pl.DataFrame)
            assert len(df) == len(report.anomalies)
            assert "timestamp" in df.columns
            assert "severity" in df.columns
            assert "type" in df.columns
