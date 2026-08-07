"""Anomaly detection manager."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import polars as pl
import structlog

from ml4t.data.anomaly.base import AnomalyDetector, AnomalyReport
from ml4t.data.anomaly.config import (
    AnomalyConfig,
    PriceStalenessConfig,
    ReturnOutlierConfig,
    VolumeSpikeConfig,
)
from ml4t.data.anomaly.detectors import (
    PriceStalenessDetector,
    ReturnOutlierDetector,
    VolumeSpikeDetector,
)

logger = structlog.get_logger()


class AnomalyManager:
    """Manages anomaly detection across multiple detectors."""

    def __init__(
        self,
        config: AnomalyConfig | None = None,
        custom_detectors: list[AnomalyDetector] | None = None,
    ):
        """
        Initialize anomaly manager.

        Args:
            config: Anomaly detection configuration
            custom_detectors: Additional custom detectors
        """
        self.config = config or AnomalyConfig()
        self.detectors: list[AnomalyDetector] = []

        if self.config.enabled:
            # All built-ins remain available so per-symbol overrides can enable them.
            self.detectors.extend(
                [
                    ReturnOutlierDetector(self.config.return_outliers),
                    VolumeSpikeDetector(self.config.volume_spikes),
                    PriceStalenessDetector(self.config.price_staleness),
                ]
            )

            # Add custom detectors
            if custom_detectors:
                self.detectors.extend(custom_detectors)

    def analyze(
        self,
        df: pl.DataFrame,
        symbol: str,
        asset_class: str | None = None,
    ) -> AnomalyReport:
        """
        Analyze data for anomalies.

        Args:
            df: DataFrame with OHLCV data
            symbol: Symbol being analyzed
            asset_class: Optional asset class for configuration overrides

        Returns:
            Anomaly detection report
        """
        if not self.config.enabled or df.is_empty():
            return AnomalyReport(
                symbol=symbol,
                start_date=datetime.now(),
                end_date=datetime.now(),
                total_rows=0,
            )

        # Ensure required columns exist
        required_cols = ["timestamp", "open", "high", "low", "close", "volume"]
        missing_cols = set(required_cols) - set(df.columns)
        if missing_cols:
            logger.warning(f"Missing required columns for anomaly detection: {missing_cols}")
            return AnomalyReport(
                symbol=symbol,
                start_date=datetime.now(),
                end_date=datetime.now(),
                total_rows=len(df),
            )

        timestamp_dtype = df.schema["timestamp"]
        try:
            if timestamp_dtype == pl.Date:
                df = df.with_columns(pl.col("timestamp").cast(pl.Datetime("us")))
            elif timestamp_dtype == pl.String:
                df = df.with_columns(pl.col("timestamp").str.to_datetime(strict=True))
            elif not isinstance(timestamp_dtype, pl.Datetime):
                raise TypeError(f"unsupported timestamp dtype {timestamp_dtype}")
        except (pl.exceptions.PolarsError, TypeError) as exc:
            logger.warning(
                "Cannot analyze data with invalid timestamps",
                symbol=symbol,
                error=str(exc),
            )
            return AnomalyReport(
                symbol=symbol,
                start_date=datetime.now(),
                end_date=datetime.now(),
                total_rows=len(df),
            )

        # Sort by timestamp
        df = df.sort("timestamp")

        start_date = df["timestamp"].min()
        end_date = df["timestamp"].max()
        if not isinstance(start_date, datetime) or not isinstance(end_date, datetime):
            logger.warning("Cannot analyze data without valid timestamps", symbol=symbol)
            return AnomalyReport(
                symbol=symbol,
                start_date=datetime.now(),
                end_date=datetime.now(),
                total_rows=len(df),
            )

        # Create report
        report = AnomalyReport(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            total_rows=len(df),
        )

        # Apply each detector
        for detector in self.detectors:
            try:
                # Get detector config with overrides
                detector_config = self.config.get_detector_config(
                    detector.name, asset_class=asset_class, symbol=symbol
                )
                if isinstance(detector, ReturnOutlierDetector):
                    config = ReturnOutlierConfig(**detector_config.model_dump())
                    detector.config = config
                    detector.enabled = config.enabled
                elif isinstance(detector, VolumeSpikeDetector):
                    config = VolumeSpikeConfig(**detector_config.model_dump())
                    detector.config = config
                    detector.enabled = config.enabled
                elif isinstance(detector, PriceStalenessDetector):
                    config = PriceStalenessConfig(**detector_config.model_dump())
                    detector.config = config
                    detector.enabled = config.enabled

                if not detector.is_enabled():
                    continue

                # Detect anomalies
                anomalies = detector.detect(df, symbol)

                # Add to report
                for anomaly in anomalies:
                    report.add_anomaly(anomaly)

                report.detectors_used.append(detector.name)

                logger.debug(
                    f"Detector {detector.name} found {len(anomalies)} anomalies",
                    symbol=symbol,
                    count=len(anomalies),
                )

            except Exception as e:
                logger.error(
                    f"Error in detector {detector.name}",
                    symbol=symbol,
                    error=str(e),
                )

        # Log summary
        if report.anomalies:
            logger.info(
                "Anomaly detection complete",
                symbol=symbol,
                total_anomalies=len(report.anomalies),
                critical=len(report.get_critical_anomalies()),
                summary=report.summary,
            )

        return report

    def analyze_batch(
        self,
        datasets: dict[str, pl.DataFrame],
        asset_classes: dict[str, str] | None = None,
    ) -> dict[str, AnomalyReport]:
        """
        Analyze multiple datasets for anomalies.

        Args:
            datasets: Dictionary of symbol -> DataFrame
            asset_classes: Optional mapping of symbol -> asset class

        Returns:
            Dictionary of symbol -> AnomalyReport
        """
        reports = {}

        for symbol, df in datasets.items():
            asset_class = asset_classes.get(symbol) if asset_classes else None
            reports[symbol] = self.analyze(df, symbol, asset_class)

        return reports

    def save_report(self, report: AnomalyReport, output_dir: Path) -> Path:
        """
        Save anomaly report to disk.

        Args:
            report: Anomaly report to save
            output_dir: Directory to save report

        Returns:
            Path to saved report
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"anomaly_report_{report.symbol}_{timestamp}.json"
        output_path = output_dir / filename

        # Save report
        with open(output_path, "w") as f:
            f.write(report.model_dump_json(indent=2))

        logger.info("Anomaly report saved", path=str(output_path))

        return output_path

    def filter_by_severity(self, report: AnomalyReport, min_severity: str) -> AnomalyReport:
        """
        Filter report to only include anomalies above minimum severity.

        Args:
            report: Original report
            min_severity: Minimum severity level

        Returns:
            Filtered report
        """
        severity_order = ["info", "warning", "error", "critical"]
        min_level = severity_order.index(min_severity.lower())

        filtered = AnomalyReport(
            symbol=report.symbol,
            start_date=report.start_date,
            end_date=report.end_date,
            total_rows=report.total_rows,
            detectors_used=report.detectors_used,
        )

        for anomaly in report.anomalies:
            anomaly_level = severity_order.index(anomaly.severity.value)
            if anomaly_level >= min_level:
                filtered.add_anomaly(anomaly)

        return filtered

    def get_statistics(self, report: AnomalyReport) -> dict[str, Any]:
        """
        Get statistics from anomaly report.

        Args:
            report: Anomaly report

        Returns:
            Dictionary of statistics
        """
        stats = {
            "total_anomalies": len(report.anomalies),
            "by_severity": {},
            "by_type": {},
            "detection_rate": 0,
        }

        # Count by severity
        for severity in ["info", "warning", "error", "critical"]:
            count = sum(1 for a in report.anomalies if a.severity.value == severity)
            stats["by_severity"][severity] = count

        # Count by type
        for anomaly in report.anomalies:
            type_name = anomaly.type.value
            stats["by_type"][type_name] = stats["by_type"].get(type_name, 0) + 1

        # Calculate detection rate
        if report.total_rows > 0:
            stats["detection_rate"] = len(report.anomalies) / report.total_rows

        return stats
