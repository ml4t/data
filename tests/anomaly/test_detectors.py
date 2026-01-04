"""Comprehensive tests for anomaly detectors.

Tests cover:
- ReturnOutlierDetector with MAD, z-score, and IQR methods
- VolumeSpikeDetector with rolling window statistics
- PriceStalenessDetector with consecutive unchanged days detection
- Severity level assignment
- Edge cases and boundary conditions
"""

from __future__ import annotations

from datetime import datetime, timedelta

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
    """Test ReturnOutlierDetector."""

    @pytest.fixture
    def detector(self) -> ReturnOutlierDetector:
        """Create default detector."""
        return ReturnOutlierDetector()

    @pytest.fixture
    def normal_data(self) -> pl.DataFrame:
        """Create normal OHLCV data with no outliers."""
        dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(50)]
        base_price = 100.0

        # Generate prices with small daily changes (normal volatility)
        closes = [base_price]
        for i in range(1, 50):
            # Small random-like changes (deterministic for testing)
            change = 0.01 * (((i * 7) % 11) - 5) / 5  # -1% to +1%
            closes.append(closes[-1] * (1 + change))

        return pl.DataFrame(
            {
                "timestamp": dates,
                "open": closes,
                "high": [p * 1.01 for p in closes],
                "low": [p * 0.99 for p in closes],
                "close": closes,
                "volume": [1000000] * 50,
            }
        )

    @pytest.fixture
    def outlier_data(self) -> pl.DataFrame:
        """Create OHLCV data with return outliers."""
        dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(50)]
        base_price = 100.0

        # Generate prices with one extreme spike
        closes = [base_price]
        for i in range(1, 50):
            if i == 25:
                # Extreme 20% jump
                closes.append(closes[-1] * 1.20)
            elif i == 26:
                # Extreme 15% drop
                closes.append(closes[-1] * 0.85)
            else:
                # Normal small changes
                change = 0.005 * (((i * 7) % 11) - 5) / 5
                closes.append(closes[-1] * (1 + change))

        return pl.DataFrame(
            {
                "timestamp": dates,
                "open": closes,
                "high": [p * 1.01 for p in closes],
                "low": [p * 0.99 for p in closes],
                "close": closes,
                "volume": [1000000] * 50,
            }
        )

    def test_default_initialization(self, detector: ReturnOutlierDetector):
        """Default initialization uses MAD method."""
        assert detector.name == "return_outliers"
        assert detector.config.method == "mad"
        assert detector.config.threshold == 3.0
        assert detector.is_enabled() is True

    def test_custom_config(self):
        """Custom configuration is applied."""
        config = ReturnOutlierConfig(method="zscore", threshold=2.5, enabled=True)
        detector = ReturnOutlierDetector(config=config)

        assert detector.config.method == "zscore"
        assert detector.config.threshold == 2.5

    def test_disabled_detector_returns_empty(self, outlier_data: pl.DataFrame):
        """Disabled detector returns no anomalies."""
        config = ReturnOutlierConfig(enabled=False)
        detector = ReturnOutlierDetector(config=config)

        anomalies = detector.detect(outlier_data, "TEST")

        assert len(anomalies) == 0

    def test_insufficient_samples(self, detector: ReturnOutlierDetector):
        """Insufficient samples returns no anomalies."""
        small_data = pl.DataFrame(
            {
                "timestamp": [datetime(2024, 1, i) for i in range(1, 6)],
                "open": [100, 101, 102, 103, 104],
                "high": [101, 102, 103, 104, 105],
                "low": [99, 100, 101, 102, 103],
                "close": [100, 101, 102, 103, 104],
                "volume": [1000] * 5,
            }
        )

        anomalies = detector.detect(small_data, "TEST")

        assert len(anomalies) == 0

    def test_detects_return_outliers_mad(self, outlier_data: pl.DataFrame):
        """MAD method detects return outliers."""
        config = ReturnOutlierConfig(method="mad", threshold=3.0)
        detector = ReturnOutlierDetector(config=config)

        anomalies = detector.detect(outlier_data, "TEST")

        assert len(anomalies) > 0
        assert all(a.type == AnomalyType.RETURN_OUTLIER for a in anomalies)

    def test_detects_return_outliers_zscore(self, outlier_data: pl.DataFrame):
        """Z-score method detects return outliers."""
        config = ReturnOutlierConfig(method="zscore", threshold=2.0)
        detector = ReturnOutlierDetector(config=config)

        anomalies = detector.detect(outlier_data, "TEST")

        assert len(anomalies) > 0
        for anomaly in anomalies:
            assert anomaly.type == AnomalyType.RETURN_OUTLIER
            assert "z-score" in anomaly.message

    def test_detects_return_outliers_iqr(self, outlier_data: pl.DataFrame):
        """IQR method detects return outliers."""
        config = ReturnOutlierConfig(method="iqr", threshold=1.5)
        detector = ReturnOutlierDetector(config=config)

        anomalies = detector.detect(outlier_data, "TEST")

        assert len(anomalies) > 0
        for anomaly in anomalies:
            assert anomaly.type == AnomalyType.RETURN_OUTLIER
            assert "IQR" in anomaly.message

    def test_severity_assignment_mad(self, outlier_data: pl.DataFrame):
        """Severity is assigned based on z-score magnitude."""
        config = ReturnOutlierConfig(method="mad", threshold=2.0)
        detector = ReturnOutlierDetector(config=config)

        anomalies = detector.detect(outlier_data, "TEST")

        # With extreme outliers, we should have high severity
        severities = {a.severity for a in anomalies}
        assert len(severities) > 0

    def test_expected_range_in_output(self, outlier_data: pl.DataFrame):
        """Expected range is included in anomaly."""
        detector = ReturnOutlierDetector()

        anomalies = detector.detect(outlier_data, "TEST")

        if anomalies:
            assert anomalies[0].expected_range is not None
            assert len(anomalies[0].expected_range) == 2

    def test_metadata_includes_method(self, outlier_data: pl.DataFrame):
        """Metadata includes detection method."""
        detector = ReturnOutlierDetector()

        anomalies = detector.detect(outlier_data, "TEST")

        if anomalies:
            assert "method" in anomalies[0].metadata
            assert anomalies[0].metadata["method"] == "mad"

    def test_no_variation_returns_empty(self, detector: ReturnOutlierDetector):
        """Data with no variation returns no anomalies."""
        flat_data = pl.DataFrame(
            {
                "timestamp": [datetime(2024, 1, i) for i in range(1, 31)],
                "open": [100.0] * 30,
                "high": [100.0] * 30,
                "low": [100.0] * 30,
                "close": [100.0] * 30,
                "volume": [1000] * 30,
            }
        )

        anomalies = detector.detect(flat_data, "TEST")

        assert len(anomalies) == 0

    def test_normal_data_no_anomalies(
        self, detector: ReturnOutlierDetector, normal_data: pl.DataFrame
    ):
        """Normal data with low volatility has few or no anomalies."""
        anomalies = detector.detect(normal_data, "TEST")

        # Should have very few anomalies with normal volatility
        assert len(anomalies) <= 2


class TestVolumeSpikeDetector:
    """Test VolumeSpikeDetector."""

    @pytest.fixture
    def detector(self) -> VolumeSpikeDetector:
        """Create default detector."""
        return VolumeSpikeDetector()

    @pytest.fixture
    def normal_data(self) -> pl.DataFrame:
        """Create normal OHLCV data with no spikes."""
        dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(50)]

        # Normal volume around 1 million
        volumes = [1000000 + (i % 10) * 10000 for i in range(50)]

        return pl.DataFrame(
            {
                "timestamp": dates,
                "open": [100.0] * 50,
                "high": [101.0] * 50,
                "low": [99.0] * 50,
                "close": [100.0] * 50,
                "volume": volumes,
            }
        )

    @pytest.fixture
    def spike_data(self) -> pl.DataFrame:
        """Create OHLCV data with volume spikes."""
        dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(50)]

        # Normal volume with extreme spikes at positions 30 and 35
        volumes = [1000000] * 50
        volumes[30] = 10000000  # 10x spike
        volumes[35] = 8000000  # 8x spike

        return pl.DataFrame(
            {
                "timestamp": dates,
                "open": [100.0] * 50,
                "high": [101.0] * 50,
                "low": [99.0] * 50,
                "close": [100.0] * 50,
                "volume": volumes,
            }
        )

    def test_default_initialization(self, detector: VolumeSpikeDetector):
        """Default initialization."""
        assert detector.name == "volume_spikes"
        assert detector.config.window == 20
        assert detector.config.threshold == 3.0
        assert detector.is_enabled() is True

    def test_custom_config(self):
        """Custom configuration is applied."""
        config = VolumeSpikeConfig(window=30, threshold=4.0, min_volume=100000)
        detector = VolumeSpikeDetector(config=config)

        assert detector.config.window == 30
        assert detector.config.threshold == 4.0
        assert detector.config.min_volume == 100000

    def test_disabled_detector_returns_empty(self, spike_data: pl.DataFrame):
        """Disabled detector returns no anomalies."""
        config = VolumeSpikeConfig(enabled=False)
        detector = VolumeSpikeDetector(config=config)

        anomalies = detector.detect(spike_data, "TEST")

        assert len(anomalies) == 0

    def test_insufficient_data_returns_empty(self, detector: VolumeSpikeDetector):
        """Insufficient data returns no anomalies."""
        small_data = pl.DataFrame(
            {
                "timestamp": [datetime(2024, 1, i) for i in range(1, 11)],
                "open": [100.0] * 10,
                "high": [101.0] * 10,
                "low": [99.0] * 10,
                "close": [100.0] * 10,
                "volume": [1000000] * 10,
            }
        )

        anomalies = detector.detect(small_data, "TEST")

        assert len(anomalies) == 0

    def test_detects_volume_spikes(self, detector: VolumeSpikeDetector, spike_data: pl.DataFrame):
        """Detects volume spikes."""
        anomalies = detector.detect(spike_data, "TEST")

        assert len(anomalies) >= 1
        assert all(a.type == AnomalyType.VOLUME_SPIKE for a in anomalies)

    def test_spike_value_is_volume(self, detector: VolumeSpikeDetector, spike_data: pl.DataFrame):
        """Anomaly value is the actual volume."""
        anomalies = detector.detect(spike_data, "TEST")

        if anomalies:
            # One of the spikes should be 10 million
            volumes = [a.value for a in anomalies]
            assert 10000000 in volumes or 8000000 in volumes

    def test_metadata_includes_zscore(
        self, detector: VolumeSpikeDetector, spike_data: pl.DataFrame
    ):
        """Metadata includes z-score."""
        anomalies = detector.detect(spike_data, "TEST")

        if anomalies:
            assert "z_score" in anomalies[0].metadata
            assert "average_volume" in anomalies[0].metadata

    def test_severity_based_on_zscore(self, spike_data: pl.DataFrame):
        """Severity is based on z-score magnitude."""
        config = VolumeSpikeConfig(threshold=2.0)
        detector = VolumeSpikeDetector(config=config)

        anomalies = detector.detect(spike_data, "TEST")

        # With 10x spike, z-score should be high
        if anomalies:
            # High z-scores should have ERROR or WARNING severity
            severities = {a.severity for a in anomalies}
            assert severities  # Should have at least one

    def test_min_volume_filter(self, spike_data: pl.DataFrame):
        """Min volume filter excludes low volume."""
        config = VolumeSpikeConfig(min_volume=20000000)  # Higher than all volumes
        detector = VolumeSpikeDetector(config=config)

        anomalies = detector.detect(spike_data, "TEST")

        assert len(anomalies) == 0

    def test_normal_data_no_spikes(self, detector: VolumeSpikeDetector, normal_data: pl.DataFrame):
        """Normal data has no volume spikes."""
        anomalies = detector.detect(normal_data, "TEST")

        assert len(anomalies) == 0

    def test_expected_range_in_output(
        self, detector: VolumeSpikeDetector, spike_data: pl.DataFrame
    ):
        """Expected range is included in anomaly."""
        anomalies = detector.detect(spike_data, "TEST")

        if anomalies:
            assert anomalies[0].expected_range is not None
            lower, upper = anomalies[0].expected_range
            assert lower < upper


class TestPriceStalenessDetector:
    """Test PriceStalenessDetector."""

    @pytest.fixture
    def detector(self) -> PriceStalenessDetector:
        """Create default detector."""
        return PriceStalenessDetector()

    @pytest.fixture
    def normal_data(self) -> pl.DataFrame:
        """Create normal OHLCV data with changing prices."""
        dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(30)]

        # Prices change each day
        closes = [100.0 + i * 0.1 for i in range(30)]

        return pl.DataFrame(
            {
                "timestamp": dates,
                "open": closes,
                "high": [p * 1.01 for p in closes],
                "low": [p * 0.99 for p in closes],
                "close": closes,
                "volume": [1000000] * 30,
            }
        )

    @pytest.fixture
    def stale_data(self) -> pl.DataFrame:
        """Create OHLCV data with stale prices."""
        dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(30)]

        # Prices are constant for days 10-20 (10 unchanged days)
        closes = []
        for i in range(30):
            if 10 <= i < 20:
                closes.append(100.0)
            else:
                closes.append(100.0 + i * 0.1)

        return pl.DataFrame(
            {
                "timestamp": dates,
                "open": closes,
                "high": closes,  # All OHLC same for stale period
                "low": closes,
                "close": closes,
                "volume": [1000000] * 30,
            }
        )

    def test_default_initialization(self, detector: PriceStalenessDetector):
        """Default initialization."""
        assert detector.name == "price_staleness"
        assert detector.config.max_unchanged_days == 5
        assert detector.config.check_close_only is False
        assert detector.is_enabled() is True

    def test_custom_config(self):
        """Custom configuration is applied."""
        config = PriceStalenessConfig(max_unchanged_days=10, check_close_only=True)
        detector = PriceStalenessDetector(config=config)

        assert detector.config.max_unchanged_days == 10
        assert detector.config.check_close_only is True

    def test_disabled_detector_returns_empty(self, stale_data: pl.DataFrame):
        """Disabled detector returns no anomalies."""
        config = PriceStalenessConfig(enabled=False)
        detector = PriceStalenessDetector(config=config)

        anomalies = detector.detect(stale_data, "TEST")

        assert len(anomalies) == 0

    def test_insufficient_data_returns_empty(self, detector: PriceStalenessDetector):
        """Insufficient data returns no anomalies."""
        single_row = pl.DataFrame(
            {
                "timestamp": [datetime(2024, 1, 1)],
                "open": [100.0],
                "high": [101.0],
                "low": [99.0],
                "close": [100.0],
                "volume": [1000000],
            }
        )

        anomalies = detector.detect(single_row, "TEST")

        assert len(anomalies) == 0

    def test_detects_stale_prices(self, detector: PriceStalenessDetector, stale_data: pl.DataFrame):
        """Detects stale prices."""
        anomalies = detector.detect(stale_data, "TEST")

        assert len(anomalies) >= 1
        assert all(a.type == AnomalyType.PRICE_STALE for a in anomalies)

    def test_stale_value_is_price(self, detector: PriceStalenessDetector, stale_data: pl.DataFrame):
        """Anomaly value is the stale price."""
        anomalies = detector.detect(stale_data, "TEST")

        if anomalies:
            assert anomalies[0].value == 100.0

    def test_metadata_includes_days_unchanged(
        self, detector: PriceStalenessDetector, stale_data: pl.DataFrame
    ):
        """Metadata includes days unchanged."""
        anomalies = detector.detect(stale_data, "TEST")

        if anomalies:
            assert "days_unchanged" in anomalies[0].metadata
            assert anomalies[0].metadata["days_unchanged"] >= 5

    def test_severity_based_on_duration(self, stale_data: pl.DataFrame):
        """Severity is based on staleness duration."""
        config = PriceStalenessConfig(max_unchanged_days=3)  # Lower threshold
        detector = PriceStalenessDetector(config=config)

        anomalies = detector.detect(stale_data, "TEST")

        if anomalies:
            # With 10 unchanged days, should be high severity
            assert anomalies[0].severity in [
                AnomalySeverity.WARNING,
                AnomalySeverity.ERROR,
                AnomalySeverity.CRITICAL,
            ]

    def test_normal_data_no_staleness(
        self, detector: PriceStalenessDetector, normal_data: pl.DataFrame
    ):
        """Normal data has no staleness."""
        anomalies = detector.detect(normal_data, "TEST")

        assert len(anomalies) == 0

    def test_check_close_only_mode(self):
        """Close-only mode only checks close prices."""
        dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(30)]

        # Close is stale but other prices change
        df = pl.DataFrame(
            {
                "timestamp": dates,
                "open": [100.0 + i * 0.1 for i in range(30)],  # Changing
                "high": [101.0 + i * 0.1 for i in range(30)],  # Changing
                "low": [99.0 + i * 0.1 for i in range(30)],  # Changing
                "close": [100.0] * 30,  # Stale
                "volume": [1000000] * 30,
            }
        )

        config = PriceStalenessConfig(max_unchanged_days=5, check_close_only=True)
        detector = PriceStalenessDetector(config=config)

        anomalies = detector.detect(df, "TEST")

        # Should detect staleness in close-only mode
        assert len(anomalies) >= 1

    def test_check_all_prices_mode(self):
        """All-prices mode checks all OHLC."""
        dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(30)]

        # Close is stale but other prices change
        df = pl.DataFrame(
            {
                "timestamp": dates,
                "open": [100.0 + i * 0.1 for i in range(30)],  # Changing
                "high": [101.0 + i * 0.1 for i in range(30)],  # Changing
                "low": [99.0 + i * 0.1 for i in range(30)],  # Changing
                "close": [100.0] * 30,  # Stale
                "volume": [1000000] * 30,
            }
        )

        config = PriceStalenessConfig(max_unchanged_days=5, check_close_only=False)
        detector = PriceStalenessDetector(config=config)

        anomalies = detector.detect(df, "TEST")

        # Should not detect staleness because other prices change
        assert len(anomalies) == 0

    def test_severity_levels(self):
        """Test all severity levels based on duration."""
        dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(50)]

        # Create data with long stale period (25 days)
        closes = []
        for i in range(50):
            if 10 <= i < 35:  # 25 unchanged days
                closes.append(100.0)
            else:
                closes.append(100.0 + i * 0.1)

        df = pl.DataFrame(
            {
                "timestamp": dates,
                "open": closes,
                "high": closes,
                "low": closes,
                "close": closes,
                "volume": [1000000] * 50,
            }
        )

        config = PriceStalenessConfig(max_unchanged_days=3)
        detector = PriceStalenessDetector(config=config)

        anomalies = detector.detect(df, "TEST")

        if anomalies:
            # 25 days should be CRITICAL (> 20 days)
            assert anomalies[0].severity == AnomalySeverity.CRITICAL


class TestDetectorIntegration:
    """Integration tests for multiple detectors."""

    @pytest.fixture
    def problematic_data(self) -> pl.DataFrame:
        """Create data with multiple issues."""
        dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(60)]

        closes = []
        volumes = []

        for i in range(60):
            if 20 <= i < 30:
                # Stale period
                closes.append(100.0)
                volumes.append(1000000)
            elif i == 40:
                # Price spike
                closes.append(150.0)  # 50% jump
                volumes.append(10000000)  # Volume spike too
            elif i == 41:
                # Price drop
                closes.append(100.0)  # Back to normal
                volumes.append(1000000)
            else:
                closes.append(100.0 + (i % 5) * 0.1)
                volumes.append(1000000 + (i % 10) * 10000)

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

    def test_multiple_detectors_find_different_issues(self, problematic_data: pl.DataFrame):
        """Multiple detectors find different types of issues."""
        return_detector = ReturnOutlierDetector(ReturnOutlierConfig(threshold=2.0))
        volume_detector = VolumeSpikeDetector(VolumeSpikeConfig(threshold=3.0))
        stale_detector = PriceStalenessDetector(PriceStalenessConfig(max_unchanged_days=5))

        return_anomalies = return_detector.detect(problematic_data, "TEST")
        volume_anomalies = volume_detector.detect(problematic_data, "TEST")
        stale_anomalies = stale_detector.detect(problematic_data, "TEST")

        # Should find return outliers (50% jump)
        assert len(return_anomalies) >= 1

        # Should find volume spike
        assert len(volume_anomalies) >= 1

        # Should find stale period
        assert len(stale_anomalies) >= 1

        # All types should be different
        return_types = {a.type for a in return_anomalies}
        volume_types = {a.type for a in volume_anomalies}
        stale_types = {a.type for a in stale_anomalies}

        assert AnomalyType.RETURN_OUTLIER in return_types
        assert AnomalyType.VOLUME_SPIKE in volume_types
        assert AnomalyType.PRICE_STALE in stale_types
