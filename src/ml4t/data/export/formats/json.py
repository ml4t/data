"""JSON export functionality."""

from __future__ import annotations

import gzip
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import polars as pl
import structlog

from ml4t.data.export.formats.base import BaseExporter, ExportResult

logger = structlog.get_logger()


class JSONExporter(BaseExporter):
    """Export data to JSON format."""

    def export(self, data: pl.DataFrame, symbol: str) -> ExportResult:
        """
        Export data to JSON file.

        Args:
            data: Data to export
            symbol: Symbol being exported

        Returns:
            Export result
        """
        start_time = time.time()

        try:
            # Apply transformations
            df = self._apply_transformations(data)

            # Determine output path
            output_file = self._get_output_path(symbol)
            self._ensure_output_dir()

            # Convert to JSON-serializable format
            json_data = self._prepare_json_data(df, symbol)

            # Export based on compression
            if self.config.compression == "gzip":
                self._export_compressed(json_data, output_file)
            else:
                self._export_uncompressed(json_data, output_file)

            # Get file size
            file_size = output_file.stat().st_size

            duration = time.time() - start_time

            logger.info(
                f"Exported {symbol} to JSON",
                rows=len(df),
                file=str(output_file),
                size_mb=file_size / 1024 / 1024,
                duration=duration,
            )

            return ExportResult(
                success=True,
                output_path=output_file,
                rows_exported=len(df),
                file_size=file_size,
                duration_seconds=duration,
            )

        except Exception as e:
            logger.error(f"Failed to export {symbol} to JSON: {e}")
            return ExportResult(
                success=False,
                output_path=self.config.output_path,
                rows_exported=0,
                file_size=0,
                duration_seconds=time.time() - start_time,
                error=str(e),
            )

    def export_batch(self, datasets: dict[str, pl.DataFrame]) -> list[ExportResult]:
        """
        Export multiple datasets to a single JSON file.

        Args:
            datasets: Dict of symbol -> data mappings

        Returns:
            List with single export result
        """
        start_time = time.time()
        total_rows = 0

        try:
            # Determine output path
            output_file = self._get_batch_output_path()
            self._ensure_output_dir()

            # Prepare combined JSON data
            combined_data = {}

            for symbol, data in datasets.items():
                df = self._apply_transformations(data)
                combined_data[symbol] = self._dataframe_to_dict(df)
                total_rows += len(df)

            # Add metadata if configured
            if self.config.include_metadata:
                combined_data["_metadata"] = self._get_metadata(datasets)

            # Export
            if self.config.compression == "gzip":
                self._export_compressed(combined_data, output_file)
            else:
                self._export_uncompressed(combined_data, output_file)

            file_size = output_file.stat().st_size
            duration = time.time() - start_time

            logger.info(
                f"Exported {len(datasets)} datasets to JSON",
                total_rows=total_rows,
                file=str(output_file),
                size_mb=file_size / 1024 / 1024,
                duration=duration,
            )

            return [
                ExportResult(
                    success=True,
                    output_path=output_file,
                    rows_exported=total_rows,
                    file_size=file_size,
                    duration_seconds=duration,
                )
            ]

        except Exception as e:
            logger.error(f"Failed to export batch to JSON: {e}")
            return [
                ExportResult(
                    success=False,
                    output_path=self.config.output_path,
                    rows_exported=0,
                    file_size=0,
                    duration_seconds=time.time() - start_time,
                    error=str(e),
                )
            ]

    def _get_output_path(self, symbol: str) -> Path:
        """Get output file path for a symbol."""
        if self.config.output_path.is_dir():
            filename = f"{symbol}.json"
            if self.config.compression == "gzip":
                filename += ".gz"
            return self.config.output_path / filename
        return self.config.output_path

    def _get_batch_output_path(self) -> Path:
        """Get output file path for batch export."""
        if self.config.output_path.is_dir():
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"export_{timestamp}.json"
            if self.config.compression == "gzip":
                filename += ".gz"
            return self.config.output_path / filename
        return self.config.output_path

    def _prepare_json_data(self, df: pl.DataFrame, symbol: str) -> dict[str, Any]:
        """
        Prepare JSON-serializable data structure.

        Args:
            df: DataFrame to convert
            symbol: Symbol name

        Returns:
            JSON-serializable dictionary
        """
        result: dict[str, Any] = {
            "symbol": symbol,
            "data": self._dataframe_to_dict(df),
        }

        if self.config.include_metadata:
            metadata: dict[str, Any] = {
                "exported_at": datetime.now().isoformat(),
                "rows": len(df),
                "columns": df.columns,
            }

            if "timestamp" in df.columns and len(df) > 0:
                metadata["date_range"] = {
                    "start": self._serialize_datetime(df["timestamp"].min()),
                    "end": self._serialize_datetime(df["timestamp"].max()),
                }
            result["metadata"] = metadata

        return result

    def _dataframe_to_dict(self, df: pl.DataFrame) -> Any:
        """
        Convert DataFrame to dictionary format.

        Args:
            df: DataFrame to convert

        Returns:
            Dictionary representation
        """
        # Convert to records format (list of dicts)
        # This is more natural for JSON consumption
        records = []

        for row in df.iter_rows(named=True):
            record = {}
            for key, value in row.items():
                # Handle datetime serialization
                if hasattr(value, "isoformat"):
                    record[key] = value.isoformat()
                elif value is None:
                    record[key] = None
                else:
                    record[key] = value
            records.append(record)

        return records

    def _serialize_datetime(self, dt: Any) -> str:
        """Serialize datetime object to ISO format string."""
        if hasattr(dt, "isoformat"):
            return dt.isoformat()
        return str(dt)

    def _export_uncompressed(self, data: Any, output_file: Path) -> None:
        """Export data to uncompressed JSON file."""
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)

    def _export_compressed(self, data: Any, output_file: Path) -> None:
        """Export data to compressed JSON file."""
        json_string = json.dumps(data, indent=2, default=str)

        with gzip.open(output_file, "wt", encoding="utf-8") as f:
            f.write(json_string)

    def _get_metadata(self, datasets: dict[str, pl.DataFrame]) -> dict[str, Any]:
        """
        Get metadata for batch export.

        Args:
            datasets: Dict of symbol -> data mappings

        Returns:
            Metadata dictionary
        """
        metadata = {
            "exported_at": datetime.now().isoformat(),
            "datasets": {},
        }

        for symbol, data in datasets.items():
            df = self._apply_transformations(data)
            metadata["datasets"][symbol] = {
                "rows": len(df),
                "columns": df.columns,
            }

            if "timestamp" in df.columns and len(df) > 0:
                metadata["datasets"][symbol]["date_range"] = {
                    "start": self._serialize_datetime(df["timestamp"].min()),
                    "end": self._serialize_datetime(df["timestamp"].max()),
                }

        return metadata
