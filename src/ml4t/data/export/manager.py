"""Export manager for coordinating data exports."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from fnmatch import fnmatchcase
from inspect import signature
from pathlib import Path
from typing import Any, ClassVar, Protocol

import polars as pl
import structlog

from ml4t.data.core.models import DataObject
from ml4t.data.export.formats import (
    CSVExporter,
    ExcelExporter,
    ExportConfig,
    JSONExporter,
)
from ml4t.data.export.formats.base import ExportResult
from ml4t.data.storage.backend import normalize_storage_metadata

logger = structlog.get_logger()


class ExportStorage(Protocol):
    """Storage operations required by the export manager."""

    def exists(self, key: str) -> bool: ...

    def read(
        self,
        key: str,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        columns: list[str] | None = None,
    ) -> pl.LazyFrame | pl.DataFrame | DataObject: ...

    def list_keys(self) -> list[str]: ...


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
        storage: ExportStorage,
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

        data, symbol, pushed_date_filter = self._read_dataset(key, options)

        # Create export config
        export_options = dict(options)
        if pushed_date_filter:
            export_options.pop("date_filter", None)
        config = ExportConfig(
            output_path=Path(output_path),
            format=format_lower,
            **export_options,
        )

        # Get exporter
        exporter_class = self.EXPORTERS[format_lower]
        exporter = exporter_class(config)

        # Export data
        self._report_progress(f"Exporting to {format_type}", 0.5)

        result = exporter.export(data, symbol)

        completion = "Export complete" if result.success else "Export failed"
        self._report_progress(completion, 1.0)

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
        preflight_results: list[ExportResult] = []
        pushed_date_filters: list[bool] = []
        total = len(keys)

        for idx, key in enumerate(keys):
            self._report_progress(f"Reading {key}", (idx + 1) / total * 0.5)

            if self.storage.exists(key):
                data, symbol, pushed_date_filter = self._read_dataset(key, options)
                if symbol in datasets:
                    preflight_results.append(
                        ExportResult(
                            success=False,
                            output_path=Path(output_path),
                            rows_exported=0,
                            file_size=0,
                            duration_seconds=0,
                            error=f"Multiple storage keys resolve to export symbol '{symbol}'",
                        )
                    )
                    continue
                datasets[symbol] = data
                pushed_date_filters.append(pushed_date_filter)
            else:
                logger.warning(f"Dataset not found: {key}")
                preflight_results.append(
                    ExportResult(
                        success=False,
                        output_path=Path(output_path),
                        rows_exported=0,
                        file_size=0,
                        duration_seconds=0,
                        error=f"Dataset not found: {key}",
                    )
                )

        if not datasets:
            logger.error("No valid datasets found")
            return preflight_results or [
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
        export_options = dict(options)
        if pushed_date_filters and all(pushed_date_filters):
            export_options.pop("date_filter", None)
        config = ExportConfig(
            output_path=Path(output_path),
            format=format_lower,
            **export_options,
        )

        # Get exporter
        exporter_class = self.EXPORTERS[format_lower]
        exporter = exporter_class(config)

        # Export data
        self._report_progress(f"Exporting {len(datasets)} datasets to {format_type}", 0.75)

        results = exporter.export_batch(datasets)

        completion = (
            "Batch export complete"
            if all(result.success for result in results)
            else "Batch export failed"
        )
        self._report_progress(completion, 1.0)

        return [*preflight_results, *results]

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
        keys = [key for key in self.storage.list_keys() if fnmatchcase(key, pattern)]

        if not keys:
            logger.warning(f"No datasets match pattern: {pattern}")
            return []

        logger.info(f"Found {len(keys)} datasets matching {pattern}")

        # Export batch
        return self.export_batch(keys, output_path, format_type, **options)

    def _read_dataset(
        self, key: str, options: dict[str, Any] | None = None
    ) -> tuple[pl.DataFrame, str, bool]:
        """Read one dataset through the canonical frame-key storage protocol."""
        options = options or {}
        stored = self.storage.read(key)
        if isinstance(stored, DataObject):
            data = stored.data
            if "timestamp" in data.columns:
                data = data.sort("timestamp")
            return data, stored.metadata.symbol, False

        pushed_date_filter = False
        read_parameters = signature(self.storage.read).parameters
        supports_pushdown = {"start_date", "end_date", "columns"}.issubset(read_parameters)
        requested_columns = options.get("columns")
        date_filter = options.get("date_filter")
        if (
            supports_pushdown
            and isinstance(stored, pl.LazyFrame)
            and (requested_columns is not None or date_filter is not None)
        ):
            schema = stored.collect_schema()
            missing_columns = (
                [column for column in requested_columns if column not in schema]
                if isinstance(requested_columns, list)
                else []
            )
            start_date, end_date = self._storage_date_bounds(date_filter)
            pushed_date_filter = date_filter is not None
            stored = self.storage.read(
                key,
                start_date=start_date,
                end_date=end_date,
                columns=None if missing_columns else requested_columns,
            )

        if isinstance(stored, pl.LazyFrame):
            data = stored.collect()
        elif isinstance(stored, pl.DataFrame):
            data = stored
        else:
            raise TypeError(
                f"Storage read for '{key}' returned unsupported type {type(stored).__name__}"
            )

        metadata_getter = getattr(self.storage, "get_metadata", None)
        metadata = metadata_getter(key) if callable(metadata_getter) else None
        symbol = self._symbol_from_metadata(metadata) or key.rsplit("/", 1)[-1]
        if not symbol:
            raise ValueError(f"No export symbol is available for storage key '{key}'")
        if "timestamp" in data.columns:
            data = data.sort("timestamp")
        return data, symbol, pushed_date_filter

    @staticmethod
    def _storage_date_bounds(
        date_filter: object,
    ) -> tuple[datetime | None, datetime | None]:
        if not isinstance(date_filter, tuple) or len(date_filter) != 2:
            return None, None

        start_value, end_value = date_filter

        def parse(value: object) -> datetime:
            parsed = value if isinstance(value, datetime) else datetime.fromisoformat(str(value))
            return parsed.replace(tzinfo=UTC) if parsed.tzinfo is None else parsed.astimezone(UTC)

        start = parse(start_value)
        end = parse(end_value)
        if isinstance(end_value, str) and len(end_value) == 10:
            end += timedelta(days=1)
        else:
            end += timedelta(microseconds=1)
        return start, end

    @staticmethod
    def _symbol_from_metadata(metadata: Any) -> str | None:
        """Read the explicit symbol from canonical or legacy metadata shapes."""
        normalized = normalize_storage_metadata(metadata)
        if normalized is None:
            return None
        symbol = normalized.get("symbol")
        return symbol if isinstance(symbol, str) else None

    def _report_progress(self, message: str, progress: float) -> None:
        """Report progress if callback is configured."""
        if self.progress_callback:
            self.progress_callback(message, progress)
        logger.info(message, progress=f"{progress:.0%}")

    @classmethod
    def list_formats(cls) -> list[str]:
        """Get list of supported export formats."""
        return list(cls.EXPORTERS.keys())
