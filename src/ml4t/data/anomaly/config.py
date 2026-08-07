"""Configuration for anomaly detection."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class DetectorConfig(BaseModel):
    """Configuration for individual detector."""

    enabled: bool = Field(default=True, description="Whether detector is enabled")
    threshold: float = Field(default=3.0, description="Detection threshold")
    window: int | None = Field(default=None, description="Window size for rolling calculations")
    min_samples: int = Field(default=20, description="Minimum samples required")


class ReturnOutlierConfig(DetectorConfig):
    """Configuration for return outlier detector."""

    method: str = Field(default="mad", description="Detection method (mad, zscore, iqr)")
    threshold: float = Field(default=3.0, description="MAD threshold for outliers")


class VolumeSpikeConfig(DetectorConfig):
    """Configuration for volume spike detector."""

    window: int = Field(default=20, description="Rolling window for baseline")
    threshold: float = Field(default=3.0, description="Z-score threshold")
    min_volume: float = Field(default=0, description="Minimum volume to consider")


class PriceStalenessConfig(DetectorConfig):
    """Configuration for price staleness detector."""

    max_unchanged_days: int = Field(default=5, description="Max days with unchanged price")
    check_close_only: bool = Field(default=False, description="Only check close prices")


class AnomalyConfig(BaseModel):
    """Main anomaly detection configuration."""

    enabled: bool = Field(default=True, description="Enable anomaly detection")

    # Detector configurations
    return_outliers: ReturnOutlierConfig = Field(
        default_factory=ReturnOutlierConfig, description="Return outlier detection config"
    )
    volume_spikes: VolumeSpikeConfig = Field(
        default_factory=VolumeSpikeConfig, description="Volume spike detection config"
    )
    price_staleness: PriceStalenessConfig = Field(
        default_factory=PriceStalenessConfig, description="Price staleness detection config"
    )

    # Asset class overrides
    asset_overrides: dict[str, dict[str, dict[str, Any]]] = Field(
        default_factory=dict, description="Asset-specific configuration overrides"
    )

    # Symbol-specific overrides
    symbol_overrides: dict[str, dict[str, dict[str, Any]]] = Field(
        default_factory=dict, description="Symbol-specific configuration overrides"
    )

    # Reporting
    report_severity_threshold: str = Field(
        default="info", description="Minimum severity to include in reports"
    )
    save_reports: bool = Field(default=True, description="Save anomaly reports to disk")

    def get_detector_config(
        self, detector_name: str, asset_class: str | None = None, symbol: str | None = None
    ) -> DetectorConfig:
        """
        Get configuration for a specific detector with overrides.

        Args:
            detector_name: Name of the detector
            asset_class: Optional asset class for overrides
            symbol: Optional symbol for specific overrides

        Returns:
            Detector configuration with overrides applied
        """
        # Start with base config
        base_config = getattr(self, detector_name, None)
        if not base_config:
            return DetectorConfig()

        config_dict = base_config.model_dump()

        # Apply asset class overrides
        if (
            asset_class
            and asset_class in self.asset_overrides
            and detector_name in self.asset_overrides[asset_class]
        ):
            config_dict.update(self.asset_overrides[asset_class][detector_name])

        # Apply symbol-specific overrides (highest priority)
        if (
            symbol
            and symbol in self.symbol_overrides
            and detector_name in self.symbol_overrides[symbol]
        ):
            config_dict.update(self.symbol_overrides[symbol][detector_name])

        # Return appropriate config type
        if detector_name == "return_outliers":
            return ReturnOutlierConfig(**config_dict)
        if detector_name == "volume_spikes":
            return VolumeSpikeConfig(**config_dict)
        if detector_name == "price_staleness":
            return PriceStalenessConfig(**config_dict)
        return DetectorConfig(**config_dict)
