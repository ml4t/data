"""Data profiling for dataset documentation.

Generates column-level statistics and metadata for datasets,
stored alongside the data for documentation and quality monitoring.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import polars as pl
import structlog

logger = structlog.get_logger()


@dataclass
class ColumnProfile:
    """Profile of a single column."""

    name: str
    dtype: str
    total_count: int
    null_count: int
    unique_count: int
    # Numeric statistics (None for non-numeric columns)
    min: float | int | str | None = None
    max: float | int | str | None = None
    mean: float | None = None
    std: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ColumnProfile:
        """Create from dictionary."""
        return cls(**data)


@dataclass
class DatasetProfile:
    """Profile of an entire dataset."""

    total_rows: int
    total_columns: int
    columns: list[ColumnProfile]
    generated_at: str
    source: str = ""  # e.g., "ETFDataManager", "CryptoDataManager"
    date_range_start: str | None = None
    date_range_end: str | None = None
    symbols: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "total_rows": self.total_rows,
            "total_columns": self.total_columns,
            "columns": [c.to_dict() for c in self.columns],
            "generated_at": self.generated_at,
            "source": self.source,
            "date_range_start": self.date_range_start,
            "date_range_end": self.date_range_end,
            "symbols": self.symbols,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DatasetProfile:
        """Create from dictionary."""
        columns = [ColumnProfile.from_dict(c) for c in data["columns"]]
        return cls(
            total_rows=data["total_rows"],
            total_columns=data["total_columns"],
            columns=columns,
            generated_at=data["generated_at"],
            source=data.get("source", ""),
            date_range_start=data.get("date_range_start"),
            date_range_end=data.get("date_range_end"),
            symbols=data.get("symbols", []),
        )

    def to_dataframe(self) -> pl.DataFrame:
        """Convert column profiles to a DataFrame for display."""
        return pl.DataFrame([c.to_dict() for c in self.columns])

    def summary(self) -> str:
        """Generate a human-readable summary."""
        lines = [
            f"Dataset Profile (generated: {self.generated_at})",
            f"  Rows: {self.total_rows:,}",
            f"  Columns: {self.total_columns}",
        ]
        if self.date_range_start and self.date_range_end:
            lines.append(f"  Date range: {self.date_range_start} to {self.date_range_end}")
        if self.symbols:
            lines.append(f"  Symbols: {len(self.symbols)}")
        lines.append("")
        lines.append("  Column Details:")
        for col in self.columns:
            null_pct = (col.null_count / col.total_count * 100) if col.total_count > 0 else 0
            line = f"    {col.name}: {col.dtype} ({col.unique_count} unique, {null_pct:.1f}% null)"
            if col.mean is not None:
                line += f" [mean={col.mean:.4g}, std={col.std:.4g}]"
            lines.append(line)
        return "\n".join(lines)


def generate_profile(
    df: pl.DataFrame,
    source: str = "",
    timestamp_col: str = "timestamp",
    symbol_col: str | None = "symbol",
) -> DatasetProfile:
    """Generate a profile for a DataFrame.

    Args:
        df: DataFrame to profile
        source: Source identifier (e.g., "ETFDataManager")
        timestamp_col: Name of timestamp column for date range extraction
        symbol_col: Name of symbol column for symbol list extraction (None to skip)

    Returns:
        DatasetProfile with column-level statistics
    """
    columns = []

    for col_name in df.columns:
        col = df[col_name]
        dtype_str = str(col.dtype)
        total_count = len(col)
        null_count = col.null_count()
        unique_count = col.n_unique()

        # Compute numeric statistics for numeric types
        min_val = None
        max_val = None
        mean_val = None
        std_val = None

        if col.dtype.is_numeric():
            # Filter out nulls for statistics
            non_null = col.drop_nulls()
            if len(non_null) > 0:
                min_val = float(non_null.min())  # type: ignore
                max_val = float(non_null.max())  # type: ignore
                mean_val = float(non_null.mean())  # type: ignore
                std_val = float(non_null.std()) if len(non_null) > 1 else 0.0  # type: ignore
        elif col.dtype == pl.Datetime or str(col.dtype).startswith("Datetime"):
            # For datetime, store min/max as strings
            non_null = col.drop_nulls()
            if len(non_null) > 0:
                min_val = str(non_null.min())
                max_val = str(non_null.max())
        elif col.dtype == pl.Date:
            non_null = col.drop_nulls()
            if len(non_null) > 0:
                min_val = str(non_null.min())
                max_val = str(non_null.max())

        columns.append(
            ColumnProfile(
                name=col_name,
                dtype=dtype_str,
                total_count=total_count,
                null_count=null_count,
                unique_count=unique_count,
                min=min_val,
                max=max_val,
                mean=mean_val,
                std=std_val,
            )
        )

    # Extract date range
    date_range_start = None
    date_range_end = None
    if timestamp_col in df.columns:
        ts_col = df[timestamp_col].drop_nulls()
        if len(ts_col) > 0:
            date_range_start = str(ts_col.min())
            date_range_end = str(ts_col.max())

    # Extract symbols
    symbols: list[str] = []
    if symbol_col and symbol_col in df.columns:
        symbols = sorted(df[symbol_col].unique().drop_nulls().to_list())

    return DatasetProfile(
        total_rows=len(df),
        total_columns=len(df.columns),
        columns=columns,
        generated_at=datetime.now().isoformat(),
        source=source,
        date_range_start=date_range_start,
        date_range_end=date_range_end,
        symbols=symbols,
    )


def save_profile(profile: DatasetProfile, path: Path) -> None:
    """Save a profile to JSON file.

    Args:
        profile: DatasetProfile to save
        path: Path to save to (typically ending in _profile.json)
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(profile.to_dict(), f, indent=2)
    logger.debug("Saved data profile", path=str(path))


def load_profile(path: Path) -> DatasetProfile | None:
    """Load a profile from JSON file.

    Args:
        path: Path to load from

    Returns:
        DatasetProfile if file exists, None otherwise
    """
    if not path.exists():
        return None
    with open(path) as f:
        data = json.load(f)
    return DatasetProfile.from_dict(data)


def get_profile_path(data_path: Path) -> Path:
    """Get the profile path for a data file or directory.

    For a file like `data.parquet`, returns `data_profile.json`.
    For a directory, returns `_profile.json` inside it.
    """
    if data_path.is_dir():
        return data_path / "_profile.json"
    else:
        return data_path.with_name(data_path.stem + "_profile.json")
