"""Base exporter interface and configuration."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import polars as pl
from pydantic import BaseModel


class ExportConfig(BaseModel):
    """Configuration for data export."""

    output_path: Path
    format: str
    include_metadata: bool = True
    compression: str | None = None  # gzip, zip, etc.
    batch_size: int = 100000  # For chunked processing

    # Transformation options
    resample: str | None = None  # Target frequency
    date_filter: tuple[str, str] | None = None  # (start, end)
    columns: list[str] | None = None  # Specific columns to export
    add_returns: bool = False
    add_volatility: bool = False

    model_config = {"arbitrary_types_allowed": True}


@dataclass
class ExportResult:
    """Result of an export operation."""

    success: bool
    output_path: Path
    rows_exported: int
    file_size: int
    duration_seconds: float
    error: str | None = None


class BaseExporter(ABC):
    """Abstract base class for data exporters."""

    def __init__(self, config: ExportConfig) -> None:
        """
        Initialize exporter.

        Args:
            config: Export configuration
        """
        self.config = config

    @abstractmethod
    def export(self, data: pl.DataFrame, symbol: str) -> ExportResult:
        """
        Export data to the target format.

        Args:
            data: Data to export
            symbol: Symbol being exported

        Returns:
            Export result with status and metadata
        """

    @abstractmethod
    def export_batch(self, datasets: dict[str, pl.DataFrame]) -> list[ExportResult]:
        """
        Export multiple datasets.

        Args:
            datasets: Dict of symbol -> data mappings

        Returns:
            List of export results
        """

    def _apply_transformations(self, data: pl.DataFrame) -> pl.DataFrame:
        """
        Apply configured transformations to data.

        Args:
            data: Input data

        Returns:
            Transformed data
        """
        df = data

        # Apply date filter
        if self.config.date_filter:
            start, end = self.config.date_filter
            df = df.filter((pl.col("timestamp") >= start) & (pl.col("timestamp") <= end))

        # Select specific columns
        if self.config.columns:
            missing = [column for column in self.config.columns if column not in df.columns]
            if missing:
                raise ValueError(f"Requested export columns are missing: {missing}")
            df = df.select(self.config.columns)

        # Add calculated fields
        if self.config.add_returns and "close" in df.columns:
            df = df.with_columns(pl.col("close").pct_change().alias("returns"))

        if self.config.add_volatility and "returns" in df.columns:
            # 20-day rolling volatility
            df = df.with_columns(pl.col("returns").rolling_std(20).alias("volatility_20d"))

        return df

    def _ensure_output_dir(self) -> None:
        """Ensure output directory exists."""
        if self.config.output_path.is_dir():
            self.config.output_path.mkdir(parents=True, exist_ok=True)
        else:
            self.config.output_path.parent.mkdir(parents=True, exist_ok=True)
