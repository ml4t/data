"""Export manager for coordinating data exports."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any, ClassVar

import structlog

from ml4t.data.export.formats import (
    CSVExporter,
    ExcelExporter,
    ExportConfig,
    JSONExporter,
)
from ml4t.data.export.formats.base import ExportResult

logger = structlog.get_logger()


class ExportManager:
    """Manages data export operations."""

    EXPORTERS: ClassVar[dict[str, type]] = {
        "csv": CSVExporter,
        "excel": ExcelExporter,
        "xlsx": ExcelExporter,
        "json": JSONExporter,
    }

    def __init__(
        self,
        storage: Any,
        progress_callback: Callable[[str, float], None] | None = None,
    ) -> None:
        """
        Initialize export manager.

        Args:
            storage: Storage backend for reading data
            progress_callback: Optional callback for progress updates
        """
        self.storage = storage
        self.progress_callback = progress_callback

    def export(
        self,
        key: str,
        output_path: str | Path,
        format_type: str = "csv",
        **options: Any,
    ) -> ExportResult:
        """
        Export a single dataset.

        Args:
            key: Storage key for the dataset
            output_path: Output file or directory path
            format_type: Export format (csv, excel, json)
            **options: Additional export options

        Returns:
            Export result
        """
        # Validate format
        format_lower = format_type.lower()
        if format_lower not in self.EXPORTERS:
            raise ValueError(
                f"Unsupported format: {format_type}. Supported: {list(self.EXPORTERS.keys())}"
            )

        # Read data
        self._report_progress(f"Reading {key}", 0.1)

        if not self.storage.exists(key):
            logger.error(f"Dataset not found: {key}")
            return ExportResult(
                success=False,
                output_path=Path(output_path),
                rows_exported=0,
                file_size=0,
                duration_seconds=0,
                error=f"Dataset not found: {key}",
            )

        data_obj = self.storage.read(key)

        # Create export config
        config = ExportConfig(
            output_path=Path(output_path),
            format=format_lower,
            **options,
        )

        # Get exporter
        exporter_class = self.EXPORTERS[format_lower]
        exporter = exporter_class(config)

        # Export data
        self._report_progress(f"Exporting to {format_type}", 0.5)

        symbol = data_obj.metadata.symbol
        result = exporter.export(data_obj.data, symbol)

        self._report_progress("Export complete", 1.0)

        return result

    def export_batch(
        self,
        keys: list[str],
        output_path: str | Path,
        format_type: str = "excel",
        **options: Any,
    ) -> list[ExportResult]:
        """
        Export multiple datasets.

        Args:
            keys: List of storage keys
            output_path: Output file or directory path
            format_type: Export format
            **options: Additional export options

        Returns:
            List of export results
        """
        # Validate format
        format_lower = format_type.lower()
        if format_lower not in self.EXPORTERS:
            raise ValueError(
                f"Unsupported format: {format_type}. Supported: {list(self.EXPORTERS.keys())}"
            )

        # Read all datasets
        datasets = {}
        total = len(keys)

        for idx, key in enumerate(keys):
            self._report_progress(f"Reading {key}", (idx + 1) / total * 0.5)

            if self.storage.exists(key):
                data_obj = self.storage.read(key)
                symbol = data_obj.metadata.symbol
                datasets[symbol] = data_obj.data
            else:
                logger.warning(f"Dataset not found: {key}")

        if not datasets:
            logger.error("No valid datasets found")
            return [
                ExportResult(
                    success=False,
                    output_path=Path(output_path),
                    rows_exported=0,
                    file_size=0,
                    duration_seconds=0,
                    error="No valid datasets found",
                )
            ]

        # Create export config
        config = ExportConfig(
            output_path=Path(output_path),
            format=format_lower,
            **options,
        )

        # Get exporter
        exporter_class = self.EXPORTERS[format_lower]
        exporter = exporter_class(config)

        # Export data
        self._report_progress(f"Exporting {len(datasets)} datasets to {format_type}", 0.75)

        results = exporter.export_batch(datasets)

        self._report_progress("Batch export complete", 1.0)

        return results

    def export_pattern(
        self,
        pattern: str,
        output_path: str | Path,
        format_type: str = "excel",
        **options: Any,
    ) -> list[ExportResult]:
        """
        Export datasets matching a pattern.

        Args:
            pattern: Pattern to match (e.g., "equities/daily/*")
            output_path: Output file or directory path
            format_type: Export format
            **options: Additional export options

        Returns:
            List of export results
        """
        # List matching keys
        # Convert pattern to prefix for list_keys
        prefix = pattern.replace("*", "")
        keys = self.storage.list_keys(prefix)

        if not keys:
            logger.warning(f"No datasets match pattern: {pattern}")
            return []

        logger.info(f"Found {len(keys)} datasets matching {pattern}")

        # Export batch
        return self.export_batch(keys, output_path, format_type, **options)

    def _report_progress(self, message: str, progress: float) -> None:
        """Report progress if callback is configured."""
        if self.progress_callback:
            self.progress_callback(message, progress)
        logger.info(message, progress=f"{progress:.0%}")

    @classmethod
    def list_formats(cls) -> list[str]:
        """Get list of supported export formats."""
        return list(cls.EXPORTERS.keys())
