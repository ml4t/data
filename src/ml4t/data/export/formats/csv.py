"""CSV export functionality."""

from __future__ import annotations

import gzip
import time
from pathlib import Path

import polars as pl
import structlog

from ml4t.data.export.formats.base import BaseExporter, ExportResult

logger = structlog.get_logger()


class CSVExporter(BaseExporter):
    """Export data to CSV format."""

    def export(self, data: pl.DataFrame, symbol: str) -> ExportResult:
        """
        Export data to CSV file.

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

            # Export based on compression
            if self.config.compression == "gzip":
                self._export_compressed(df, output_file)
            else:
                self._export_uncompressed(df, output_file)

            # Get file size
            file_size = output_file.stat().st_size

            duration = time.time() - start_time

            logger.info(
                f"Exported {symbol} to CSV",
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
            logger.error(f"Failed to export {symbol} to CSV: {e}")
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
        Export multiple datasets to separate CSV files.

        Args:
            datasets: Dict of symbol -> data mappings

        Returns:
            List of export results
        """
        results = []

        for symbol, data in datasets.items():
            result = self.export(data, symbol)
            results.append(result)

        return results

    def _get_output_path(self, symbol: str) -> Path:
        """
        Get output file path for a symbol.

        Args:
            symbol: Symbol name

        Returns:
            Output file path
        """
        if self.config.output_path.is_dir():
            filename = f"{symbol}.csv"
            if self.config.compression == "gzip":
                filename += ".gz"
            return self.config.output_path / filename
        # Single file specified
        return self.config.output_path

    def _export_uncompressed(self, df: pl.DataFrame, output_file: Path) -> None:
        """
        Export DataFrame to uncompressed CSV.

        Args:
            df: Data to export
            output_file: Output file path
        """
        # Use Polars native CSV writer for efficiency
        df.write_csv(output_file)

    def _export_compressed(self, df: pl.DataFrame, output_file: Path) -> None:
        """
        Export DataFrame to compressed CSV.

        Args:
            df: Data to export
            output_file: Output file path
        """
        # Convert to CSV string and compress
        csv_string = df.write_csv()

        with gzip.open(output_file, "wt", encoding="utf-8") as f:
            f.write(csv_string)
