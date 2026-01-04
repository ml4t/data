"""Tests for anomaly detectors."""

from datetime import datetime

import polars as pl
import pytest

from ml4t.data.anomaly.base import AnomalySeverity, AnomalyType
from ml4t.data.anomaly.config import (
    PriceStalenessConfig,
    ReturnOutlierConfig,
    VolumeSpikeConfig,
)
from ml4t.data.anomaly.detectors import (
    PriceStalenessDetector,
    ReturnOutlierDetector,
    VolumeSpikeDetector,
)


class TestReturnOutlierDetector:
    """Test return outlier detection."""

    @pytest.fixture
    def sample_data(self) -> pl.DataFrame:
        """Create sample OHLCV data with outlier."""
        dates = [datetime(2024, 1, i) for i in range(1, 31)]

        # Normal returns around 1% with one outlier
        closes = [100.0]
        for i in range(1, 30):
            if i == 15:
                # Add outlier: 20% return
                closes.append(closes[-1] * 1.20)
            else:
                # Normal ~1% returns
                closes.append(closes[-1] * (1.0 + (i % 3 - 1) * 0.01))

        return pl.DataFrame(
            {
                "timestamp": dates,
                "open": closes,
                "high": [c * 1.02 for c in closes],
                "low": [c * 0.98 for c in closes],
                "close": closes,
                "volume": [1000000] * 30,
            }
        )

    def test_mad_detection(self, sample_data):
        """Test MAD-based outlier detection."""
        config = ReturnOutlierConfig(method="mad", threshold=3.0)
        detector = ReturnOutlierDetector(config)

        anomalies = detector.detect(sample_data, "TEST")

        # Should detect the 20% outlier
        assert len(anomalies) >= 1
        assert any(a.type == AnomalyType.RETURN_OUTLIER for a in anomalies)

        # Check the outlier details
        outlier = next(a for a in anomalies if abs(a.value) > 15)
        assert outlier.severity in [AnomalySeverity.ERROR, AnomalySeverity.CRITICAL]
        assert "MAD" in outlier.message

    def test_zscore_detection(self, sample_data):
        """Test z-score based outlier detection."""
        config = ReturnOutlierConfig(method="zscore", threshold=3.0)
        detector = ReturnOutlierDetector(config)

        anomalies = detector.detect(sample_data, "TEST")

        # Should detect the outlier
        assert len(anomalies) >= 1
        assert any(abs(a.value) > 15 for a in anomalies)

    def test_iqr_detection(self, sample_data):
        """Test IQR-based outlier detection."""
        config = ReturnOutlierConfig(method="iqr", threshold=1.5)
        detector = ReturnOutlierDetector(config)

        anomalies = detector.detect(sample_data, "TEST")

        # Should detect outliers
        assert len(anomalies) >= 1
        assert all(a.type == AnomalyType.RETURN_OUTLIER for a in anomalies)

    def test_no_outliers(self):
        """Test with data containing no outliers."""
        # Create steady growth data
        dates = [datetime(2024, 1, i) for i in range(1, 31)]
        closes = [100.0 * (1.001**i) for i in range(30)]

        df = pl.DataFrame(
            {
                "timestamp": dates,
                "open": closes,
                "high": closes,
                "low": closes,
                "close": closes,
                "volume": [1000000] * 30,
            }
        )

        config = ReturnOutlierConfig(method="mad", threshold=3.0)
        detector = ReturnOutlierDetector(config)

        anomalies = detector.detect(df, "TEST")

        # Should not detect any outliers
        assert len(anomalies) == 0

    def test_disabled_detector(self, sample_data):
        """Test disabled detector returns no anomalies."""
        config = ReturnOutlierConfig(enabled=False)
        detector = ReturnOutlierDetector(config)

        anomalies = detector.detect(sample_data, "TEST")
        assert len(anomalies) == 0

    def test_insufficient_data(self):
        """Test with insufficient data."""
        df = pl.DataFrame(
            {
                "timestamp": [datetime(2024, 1, 1)],
                "open": [100.0],
                "high": [101.0],
                "low": [99.0],
                "close": [100.5],
                "volume": [1000000],
            }
        )

        config = ReturnOutlierConfig(min_samples=20)
        detector = ReturnOutlierDetector(config)

        anomalies = detector.detect(df, "TEST")
        assert len(anomalies) == 0


class TestVolumeSpikeDetector:
    """Test volume spike detection."""

    @pytest.fixture
    def sample_data(self) -> pl.DataFrame:
        """Create sample data with volume spike."""
        dates = [datetime(2024, 1, i) for i in range(1, 31)]

        # Normal volume around 1M with spike
        volumes = []
        for i in range(30):
            if i == 20:
                # Volume spike: 10x normal
                volumes.append(10000000)
            else:
                # Normal volume with some variation
                volumes.append(1000000 + (i % 5) * 50000)

        return pl.DataFrame(
            {
                "timestamp": dates,
                "open": [100.0] * 30,
                "high": [101.0] * 30,
                "low": [99.0] * 30,
                "close": [100.0] * 30,
                "volume": volumes,
            }
        )

    def test_volume_spike_detection(self, sample_data):
        """Test basic volume spike detection."""
        config = VolumeSpikeConfig(window=10, threshold=2.5)
        detector = VolumeSpikeDetector(config)

        anomalies = detector.detect(sample_data, "TEST")

        # Should detect the spike
        assert len(anomalies) >= 1
        assert any(a.type == AnomalyType.VOLUME_SPIKE for a in anomalies)

        # Check spike details
        spike = next(a for a in anomalies if a.value > 5000000)
        assert spike.value == 10000000
        assert "spike" in spike.message.lower()

    def test_min_volume_filter(self):
        """Test minimum volume filtering."""
        dates = [datetime(2024, 1, i) for i in range(1, 21)]

        # Mix of low and normal volumes
        volumes = [100 if i < 10 else 1000000 for i in range(20)]
        volumes[15] = 5000000  # Spike in normal volume

        df = pl.DataFrame(
            {
                "timestamp": dates,
                "open": [100.0] * 20,
                "high": [101.0] * 20,
                "low": [99.0] * 20,
                "close": [100.0] * 20,
                "volume": volumes,
            }
        )

        config = VolumeSpikeConfig(window=5, threshold=2.0, min_volume=1000)
        detector = VolumeSpikeDetector(config)

        anomalies = detector.detect(df, "TEST")

        # Should only detect spikes above min_volume
        assert all(a.value >= 1000 for a in anomalies)

    def test_no_volume_spikes(self):
        """Test with steady volume."""
        dates = [datetime(2024, 1, i) for i in range(1, 31)]

        df = pl.DataFrame(
            {
                "timestamp": dates,
                "open": [100.0] * 30,
                "high": [101.0] * 30,
                "low": [99.0] * 30,
                "close": [100.0] * 30,
                "volume": [1000000] * 30,  # Constant volume
            }
        )

        config = VolumeSpikeConfig(window=10, threshold=3.0)
        detector = VolumeSpikeDetector(config)

        anomalies = detector.detect(df, "TEST")
        assert len(anomalies) == 0


class TestPriceStalenessDetector:
    """Test price staleness detection."""

    @pytest.fixture
    def stale_data(self) -> pl.DataFrame:
        """Create data with stale prices."""
        dates = [datetime(2024, 1, i) for i in range(1, 31)]

        # Prices unchanged for days 10-20
        closes = []
        for i in range(30):
            if 10 <= i < 20:
                closes.append(100.0)  # Stale price
            else:
                closes.append(100.0 + i * 0.1)

        return pl.DataFrame(
            {
                "timestamp": dates,
                "open": closes,
                "high": [c + 0.1 if i < 10 or i >= 20 else c for i, c in enumerate(closes)],
                "low": [c - 0.1 if i < 10 or i >= 20 else c for i, c in enumerate(closes)],
                "close": closes,
                "volume": [1000000] * 30,
            }
        )

    def test_stale_price_detection(self, stale_data):
        """Test detection of stale prices."""
        config = PriceStalenessConfig(max_unchanged_days=5, check_close_only=True)
        detector = PriceStalenessDetector(config)

        anomalies = detector.detect(stale_data, "TEST")

        # Should detect the stale period
        assert len(anomalies) >= 1
        assert any(a.type == AnomalyType.PRICE_STALE for a in anomalies)

        # Check staleness details
        stale = anomalies[0]
        assert stale.metadata["days_unchanged"] > 5
        assert stale.value == 100.0

    def test_all_prices_unchanged(self):
        """Test detection when all OHLC prices are unchanged."""
        dates = [datetime(2024, 1, i) for i in range(1, 16)]

        df = pl.DataFrame(
            {
                "timestamp": dates,
                "open": [100.0] * 15,
                "high": [100.0] * 15,
                "low": [100.0] * 15,
                "close": [100.0] * 15,
                "volume": [1000000] * 15,
            }
        )

        config = PriceStalenessConfig(max_unchanged_days=5, check_close_only=False)
        detector = PriceStalenessDetector(config)

        anomalies = detector.detect(df, "TEST")

        # Should detect staleness
        assert len(anomalies) >= 1
        assert anomalies[0].type == AnomalyType.PRICE_STALE
        assert anomalies[0].severity in [AnomalySeverity.WARNING, AnomalySeverity.ERROR]

    def test_no_stale_prices(self):
        """Test with constantly changing prices."""
        dates = [datetime(2024, 1, i) for i in range(1, 31)]

        # Prices change every day
        closes = [100.0 + i * 0.1 for i in range(30)]

        df = pl.DataFrame(
            {
                "timestamp": dates,
                "open": closes,
                "high": [c + 0.5 for c in closes],
                "low": [c - 0.5 for c in closes],
                "close": closes,
                "volume": [1000000] * 30,
            }
        )

        config = PriceStalenessConfig(max_unchanged_days=5)
        detector = PriceStalenessDetector(config)

        anomalies = detector.detect(df, "TEST")
        assert len(anomalies) == 0

    def test_severity_levels(self):
        """Test different severity levels based on staleness duration."""
        dates = [datetime(2024, 1, i) for i in range(1, 26)]

        # 25 days of unchanged prices
        df = pl.DataFrame(
            {
                "timestamp": dates,
                "open": [100.0] * 25,
                "high": [100.0] * 25,
                "low": [100.0] * 25,
                "close": [100.0] * 25,
                "volume": [1000000] * 25,
            }
        )

        config = PriceStalenessConfig(max_unchanged_days=5)
        detector = PriceStalenessDetector(config)

        anomalies = detector.detect(df, "TEST")

        # Should be critical for 25 days unchanged
        assert len(anomalies) >= 1
        assert anomalies[0].severity == AnomalySeverity.CRITICAL
