"""Comprehensive tests for anomaly detection configuration.

Tests cover:
- DetectorConfig base class
- ReturnOutlierConfig with detection methods
- VolumeSpikeConfig with window settings
- PriceStalenessConfig with staleness thresholds
- AnomalyConfig with detector management and overrides
"""

from __future__ import annotations

from ml4t.data.anomaly.config import (
    AnomalyConfig,
    DetectorConfig,
    PriceStalenessConfig,
    ReturnOutlierConfig,
    VolumeSpikeConfig,
)


class TestDetectorConfig:
    """Test DetectorConfig base class."""

    def test_default_initialization(self):
        """Default config has all checks enabled."""
        config = DetectorConfig()
        assert config.enabled is True
        assert config.threshold == 3.0
        assert config.window is None
        assert config.min_samples == 20

    def test_custom_values(self):
        """Custom values are stored correctly."""
        config = DetectorConfig(
            enabled=False,
            threshold=2.5,
            window=10,
            min_samples=50,
        )
        assert config.enabled is False
        assert config.threshold == 2.5
        assert config.window == 10
        assert config.min_samples == 50

    def test_model_dump(self):
        """Config can be serialized to dict."""
        config = DetectorConfig(threshold=4.0)
        data = config.model_dump()

        assert data["threshold"] == 4.0
        assert "enabled" in data
        assert "min_samples" in data

    def test_model_from_dict(self):
        """Config can be created from dict."""
        data = {
            "enabled": False,
            "threshold": 2.0,
            "min_samples": 30,
        }
        config = DetectorConfig(**data)

        assert config.enabled is False
        assert config.threshold == 2.0
        assert config.min_samples == 30


class TestReturnOutlierConfig:
    """Test ReturnOutlierConfig for return outlier detection."""

    def test_default_initialization(self):
        """Default config uses MAD method."""
        config = ReturnOutlierConfig()
        assert config.enabled is True
        assert config.method == "mad"
        assert config.threshold == 3.0
        assert config.min_samples == 20

    def test_valid_methods(self):
        """All valid detection methods are accepted."""
        for method in ["mad", "zscore", "iqr"]:
            config = ReturnOutlierConfig(method=method)
            assert config.method == method

    def test_custom_threshold(self):
        """Custom threshold is stored correctly."""
        config = ReturnOutlierConfig(threshold=2.5)
        assert config.threshold == 2.5

    def test_zscore_method_config(self):
        """Z-score method configuration works."""
        config = ReturnOutlierConfig(
            method="zscore",
            threshold=2.0,
            min_samples=30,
        )
        assert config.method == "zscore"
        assert config.threshold == 2.0
        assert config.min_samples == 30

    def test_iqr_method_config(self):
        """IQR method configuration works."""
        config = ReturnOutlierConfig(
            method="iqr",
            threshold=1.5,
        )
        assert config.method == "iqr"
        assert config.threshold == 1.5

    def test_disabled_detector(self):
        """Detector can be disabled."""
        config = ReturnOutlierConfig(enabled=False)
        assert config.enabled is False

    def test_high_min_samples(self):
        """High min_samples value works."""
        config = ReturnOutlierConfig(min_samples=100)
        assert config.min_samples == 100


class TestVolumeSpikeConfig:
    """Test VolumeSpikeConfig for volume spike detection."""

    def test_default_initialization(self):
        """Default config has 20-day window."""
        config = VolumeSpikeConfig()
        assert config.enabled is True
        assert config.window == 20
        assert config.threshold == 3.0
        assert config.min_volume == 0

    def test_custom_window(self):
        """Custom window size is stored."""
        config = VolumeSpikeConfig(window=50)
        assert config.window == 50

    def test_custom_threshold(self):
        """Custom threshold is stored."""
        config = VolumeSpikeConfig(threshold=4.0)
        assert config.threshold == 4.0

    def test_min_volume_filter(self):
        """Minimum volume filter is stored."""
        config = VolumeSpikeConfig(min_volume=1000000)
        assert config.min_volume == 1000000

    def test_combined_settings(self):
        """Combined settings work correctly."""
        config = VolumeSpikeConfig(
            window=30,
            threshold=2.5,
            min_volume=500000,
            min_samples=10,
        )
        assert config.window == 30
        assert config.threshold == 2.5
        assert config.min_volume == 500000
        assert config.min_samples == 10


class TestPriceStalenessConfig:
    """Test PriceStalenessConfig for stale price detection."""

    def test_default_initialization(self):
        """Default config checks all prices for 5 days."""
        config = PriceStalenessConfig()
        assert config.enabled is True
        assert config.max_unchanged_days == 5
        assert config.check_close_only is False

    def test_custom_max_unchanged_days(self):
        """Custom max unchanged days is stored."""
        config = PriceStalenessConfig(max_unchanged_days=10)
        assert config.max_unchanged_days == 10

    def test_check_close_only_mode(self):
        """Close-only mode can be enabled."""
        config = PriceStalenessConfig(check_close_only=True)
        assert config.check_close_only is True

    def test_strict_staleness_config(self):
        """Strict staleness configuration (low threshold)."""
        config = PriceStalenessConfig(
            max_unchanged_days=2,
            check_close_only=False,
        )
        assert config.max_unchanged_days == 2
        assert config.check_close_only is False

    def test_relaxed_staleness_config(self):
        """Relaxed staleness configuration (high threshold)."""
        config = PriceStalenessConfig(
            max_unchanged_days=20,
            check_close_only=True,
        )
        assert config.max_unchanged_days == 20
        assert config.check_close_only is True


class TestAnomalyConfig:
    """Test AnomalyConfig main configuration."""

    def test_default_initialization(self):
        """Default config enables all detectors."""
        config = AnomalyConfig()
        assert config.enabled is True
        assert config.return_outliers.enabled is True
        assert config.volume_spikes.enabled is True
        assert config.price_staleness.enabled is True
        assert config.report_severity_threshold == "info"
        assert config.save_reports is True

    def test_disabled_detection(self):
        """Detection can be globally disabled."""
        config = AnomalyConfig(enabled=False)
        assert config.enabled is False

    def test_custom_detector_configs(self):
        """Custom detector configurations are stored."""
        config = AnomalyConfig(
            return_outliers=ReturnOutlierConfig(method="zscore", threshold=2.0),
            volume_spikes=VolumeSpikeConfig(window=30),
            price_staleness=PriceStalenessConfig(max_unchanged_days=10),
        )

        assert config.return_outliers.method == "zscore"
        assert config.return_outliers.threshold == 2.0
        assert config.volume_spikes.window == 30
        assert config.price_staleness.max_unchanged_days == 10

    def test_asset_overrides_empty(self):
        """Empty asset overrides by default."""
        config = AnomalyConfig()
        assert config.asset_overrides == {}

    def test_asset_overrides_configured(self):
        """Asset class overrides can be configured."""
        config = AnomalyConfig(
            asset_overrides={
                "crypto": {
                    "return_outliers": {"threshold": 5.0},
                    "volume_spikes": {"window": 10},
                },
                "forex": {
                    "return_outliers": {"threshold": 2.0},
                },
            }
        )

        assert "crypto" in config.asset_overrides
        assert config.asset_overrides["crypto"]["return_outliers"]["threshold"] == 5.0
        assert config.asset_overrides["crypto"]["volume_spikes"]["window"] == 10
        assert config.asset_overrides["forex"]["return_outliers"]["threshold"] == 2.0

    def test_symbol_overrides_empty(self):
        """Empty symbol overrides by default."""
        config = AnomalyConfig()
        assert config.symbol_overrides == {}

    def test_symbol_overrides_configured(self):
        """Symbol-specific overrides can be configured."""
        config = AnomalyConfig(
            symbol_overrides={
                "TSLA": {
                    "return_outliers": {"threshold": 4.0},
                },
                "GME": {
                    "volume_spikes": {"threshold": 10.0},
                },
            }
        )

        assert "TSLA" in config.symbol_overrides
        assert config.symbol_overrides["TSLA"]["return_outliers"]["threshold"] == 4.0
        assert config.symbol_overrides["GME"]["volume_spikes"]["threshold"] == 10.0

    def test_report_severity_threshold_levels(self):
        """All severity threshold levels work."""
        for level in ["info", "warning", "error", "critical"]:
            config = AnomalyConfig(report_severity_threshold=level)
            assert config.report_severity_threshold == level

    def test_save_reports_disabled(self):
        """Report saving can be disabled."""
        config = AnomalyConfig(save_reports=False)
        assert config.save_reports is False


class TestAnomalyConfigGetDetectorConfig:
    """Test get_detector_config method."""

    def test_get_base_config_return_outliers(self):
        """Get base return outliers config."""
        config = AnomalyConfig(return_outliers=ReturnOutlierConfig(method="zscore", threshold=2.5))

        detector_config = config.get_detector_config("return_outliers")

        assert isinstance(detector_config, ReturnOutlierConfig)
        assert detector_config.method == "zscore"
        assert detector_config.threshold == 2.5

    def test_get_base_config_volume_spikes(self):
        """Get base volume spikes config."""
        config = AnomalyConfig(volume_spikes=VolumeSpikeConfig(window=30))

        detector_config = config.get_detector_config("volume_spikes")

        assert isinstance(detector_config, VolumeSpikeConfig)
        assert detector_config.window == 30

    def test_get_base_config_price_staleness(self):
        """Get base price staleness config."""
        config = AnomalyConfig(price_staleness=PriceStalenessConfig(max_unchanged_days=7))

        detector_config = config.get_detector_config("price_staleness")

        assert isinstance(detector_config, PriceStalenessConfig)
        assert detector_config.max_unchanged_days == 7

    def test_get_unknown_detector_returns_default(self):
        """Unknown detector returns default config."""
        config = AnomalyConfig()

        detector_config = config.get_detector_config("unknown_detector")

        assert isinstance(detector_config, DetectorConfig)

    def test_asset_class_override_applied(self):
        """Asset class override is applied."""
        config = AnomalyConfig(
            return_outliers=ReturnOutlierConfig(threshold=3.0),
            asset_overrides={
                "crypto": {
                    "return_outliers": {"threshold": 5.0},
                }
            },
        )

        detector_config = config.get_detector_config("return_outliers", asset_class="crypto")

        assert detector_config.threshold == 5.0

    def test_asset_class_override_not_found(self):
        """Base config returned when no asset override."""
        config = AnomalyConfig(
            return_outliers=ReturnOutlierConfig(threshold=3.0),
            asset_overrides={
                "crypto": {
                    "return_outliers": {"threshold": 5.0},
                }
            },
        )

        detector_config = config.get_detector_config("return_outliers", asset_class="forex")

        assert detector_config.threshold == 3.0

    def test_symbol_override_applied(self):
        """Symbol override is applied (highest priority)."""
        config = AnomalyConfig(
            return_outliers=ReturnOutlierConfig(threshold=3.0),
            symbol_overrides={
                "TSLA": {
                    "return_outliers": {"threshold": 4.0},
                }
            },
        )

        detector_config = config.get_detector_config("return_outliers", symbol="TSLA")

        assert detector_config.threshold == 4.0

    def test_symbol_override_takes_priority_over_asset(self):
        """Symbol override takes priority over asset class."""
        config = AnomalyConfig(
            return_outliers=ReturnOutlierConfig(threshold=3.0),
            asset_overrides={
                "equity": {
                    "return_outliers": {"threshold": 2.0},
                }
            },
            symbol_overrides={
                "TSLA": {
                    "return_outliers": {"threshold": 5.0},
                }
            },
        )

        detector_config = config.get_detector_config(
            "return_outliers", asset_class="equity", symbol="TSLA"
        )

        assert detector_config.threshold == 5.0

    def test_multiple_overrides_combined(self):
        """Multiple override fields are combined correctly."""
        config = AnomalyConfig(
            volume_spikes=VolumeSpikeConfig(window=20, threshold=3.0, min_volume=0),
            asset_overrides={
                "crypto": {
                    "volume_spikes": {"window": 10},
                }
            },
            symbol_overrides={
                "BTCUSD": {
                    "volume_spikes": {"min_volume": 1000000},
                }
            },
        )

        # Only asset override
        crypto_config = config.get_detector_config("volume_spikes", asset_class="crypto")
        assert crypto_config.window == 10
        assert crypto_config.min_volume == 0

        # Both overrides - symbol override adds min_volume
        btc_config = config.get_detector_config(
            "volume_spikes", asset_class="crypto", symbol="BTCUSD"
        )
        assert btc_config.window == 10  # From asset override
        assert btc_config.min_volume == 1000000  # From symbol override


class TestConfigSerialization:
    """Test configuration serialization."""

    def test_full_config_serialization(self):
        """Full config can be serialized and deserialized."""
        config = AnomalyConfig(
            enabled=True,
            return_outliers=ReturnOutlierConfig(method="zscore", threshold=2.0),
            volume_spikes=VolumeSpikeConfig(window=30),
            price_staleness=PriceStalenessConfig(max_unchanged_days=7),
            report_severity_threshold="warning",
        )

        data = config.model_dump()
        restored = AnomalyConfig(**data)

        assert restored.enabled == config.enabled
        assert restored.return_outliers.method == config.return_outliers.method
        assert restored.volume_spikes.window == config.volume_spikes.window
        assert (
            restored.price_staleness.max_unchanged_days == config.price_staleness.max_unchanged_days
        )

    def test_json_serialization(self):
        """Config can be serialized to JSON."""
        config = AnomalyConfig(
            return_outliers=ReturnOutlierConfig(threshold=2.5),
        )

        json_str = config.model_dump_json()

        assert "2.5" in json_str
        assert "return_outliers" in json_str

    def test_config_with_overrides_serialization(self):
        """Config with overrides serializes correctly."""
        config = AnomalyConfig(
            asset_overrides={
                "crypto": {"return_outliers": {"threshold": 5.0}},
            },
            symbol_overrides={
                "TSLA": {"volume_spikes": {"window": 50}},
            },
        )

        data = config.model_dump()

        assert data["asset_overrides"]["crypto"]["return_outliers"]["threshold"] == 5.0
        assert data["symbol_overrides"]["TSLA"]["volume_spikes"]["window"] == 50
