"""Base classes for anomaly detection."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from enum import StrEnum
from typing import Any

import polars as pl
from pydantic import BaseModel, Field


class AnomalySeverity(StrEnum):
    """Anomaly severity levels."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AnomalyType(StrEnum):
    """Types of anomalies."""

    RETURN_OUTLIER = "return_outlier"
    VOLUME_SPIKE = "volume_spike"
    PRICE_STALE = "price_stale"
    DATA_GAP = "data_gap"
    PRICE_SPIKE = "price_spike"
    ZERO_VOLUME = "zero_volume"
    NEGATIVE_PRICE = "negative_price"


class Anomaly(BaseModel):
    """Represents a detected anomaly."""

    timestamp: datetime = Field(description="When the anomaly occurred")
    symbol: str = Field(description="Symbol with the anomaly")
    type: AnomalyType = Field(description="Type of anomaly")
    severity: AnomalySeverity = Field(description="Severity level")
    value: float = Field(description="The anomalous value")
    expected_range: tuple[float, float] | None = Field(
        default=None, description="Expected value range"
    )
    threshold: float | None = Field(default=None, description="Detection threshold used")
    message: str = Field(description="Human-readable description")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional context")

    def __str__(self) -> str:
        """String representation."""
        return f"[{self.severity.value.upper()}] {self.symbol} @ {self.timestamp}: {self.message}"


class AnomalyDetector(ABC):
    """Abstract base class for anomaly detectors."""

    def __init__(self, enabled: bool = True):
        """
        Initialize detector.

        Args:
            enabled: Whether this detector is enabled
        """
        self.enabled = enabled

    @abstractmethod
    def detect(self, df: pl.DataFrame, symbol: str) -> list[Anomaly]:
        """
        Detect anomalies in the data.

        Args:
            df: DataFrame with OHLCV data
            symbol: Symbol being analyzed

        Returns:
            List of detected anomalies
        """

    @property
    @abstractmethod
    def name(self) -> str:
        """Return detector name."""

    def is_enabled(self) -> bool:
        """Check if detector is enabled."""
        return self.enabled


class AnomalyReport(BaseModel):
    """Anomaly detection report."""

    symbol: str = Field(description="Symbol analyzed")
    start_date: datetime = Field(description="Start of analysis period")
    end_date: datetime = Field(description="End of analysis period")
    total_rows: int = Field(description="Total data points analyzed")
    anomalies: list[Anomaly] = Field(default_factory=list, description="Detected anomalies")
    summary: dict[str, int] = Field(default_factory=dict, description="Summary statistics")
    detectors_used: list[str] = Field(default_factory=list, description="Detectors applied")

    def add_anomaly(self, anomaly: Anomaly) -> None:
        """Add an anomaly to the report."""
        self.anomalies.append(anomaly)

        # Update summary
        severity_key = f"{anomaly.severity.value}_count"
        self.summary[severity_key] = self.summary.get(severity_key, 0) + 1

        type_key = f"{anomaly.type.value}_count"
        self.summary[type_key] = self.summary.get(type_key, 0) + 1

    def get_critical_anomalies(self) -> list[Anomaly]:
        """Get only critical anomalies."""
        return [a for a in self.anomalies if a.severity == AnomalySeverity.CRITICAL]

    def get_by_type(self, anomaly_type: AnomalyType) -> list[Anomaly]:
        """Get anomalies of specific type."""
        return [a for a in self.anomalies if a.type == anomaly_type]

    def has_critical_issues(self) -> bool:
        """Check if there are any critical issues."""
        return any(a.severity == AnomalySeverity.CRITICAL for a in self.anomalies)

    def to_dataframe(self) -> pl.DataFrame:
        """Convert anomalies to DataFrame for analysis."""
        if not self.anomalies:
            return pl.DataFrame()

        records = []
        for anomaly in self.anomalies:
            records.append(
                {
                    "timestamp": anomaly.timestamp,
                    "symbol": anomaly.symbol,
                    "type": anomaly.type.value,
                    "severity": anomaly.severity.value,
                    "value": anomaly.value,
                    "message": anomaly.message,
                }
            )

        return pl.DataFrame(records)
