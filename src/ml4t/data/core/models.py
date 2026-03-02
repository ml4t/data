"""Core data models for ml4t-data."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum
from typing import Any

import polars as pl
from pydantic import BaseModel, Field, model_validator

from ml4t.data.assets.asset_class import AssetClass  # noqa: F401 — re-exported


class SchemaVersion(str, Enum):
    """Schema version for data storage."""

    V1_0 = "1.0"


class Frequency(str, Enum):
    """Data frequency."""

    # High-frequency data
    TICK = "tick"
    SECOND = "second"
    MINUTE = "minute"
    FIVE_MINUTE = "5minute"
    FIFTEEN_MINUTE = "15minute"
    THIRTY_MINUTE = "30minute"
    HOUR = "hour"
    HOURLY = "hourly"  # Alias for HOUR

    # Daily and longer frequencies
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"


class BarType(str, Enum):
    """Types of bars for OHLCV data."""

    TIME = "time"  # Time-based bars (1min, 5min, daily, etc.)
    VOLUME = "volume"  # Volume bars (every N volume)
    TRADE = "trade"  # Trade bars (every N trades)
    DOLLAR = "dollar"  # Dollar bars (every $N traded)
    TICK = "tick"  # Tick bars (every N ticks)


class Metadata(BaseModel):
    """Metadata for a data object.

    Enhanced to support:
    - Alternative bar types (volume, trade, dollar, tick)
    - Exchange calendar information for session-aware operations
    - Date range tracking for incremental updates
    """

    # Core identification
    provider: str
    symbol: str
    asset_class: str

    # Bar specification
    bar_type: str = "time"  # Use BarType enum values
    bar_params: dict[str, Any] = Field(default_factory=dict)
    # For time bars: {"frequency": "1min"}
    # For volume bars: {"threshold": 1000}
    # For trade bars: {"threshold": 500}
    # For dollar bars: {"threshold": 100000}

    # Exchange and calendar (for session-aware operations)
    exchange: str = "UNKNOWN"
    calendar: str | None = None  # pandas_market_calendars name (e.g., "CME_Globex_Crypto")

    # Tracking
    start_date: datetime | None = None
    end_date: datetime | None = None
    last_updated: datetime | None = None

    # Legacy/compatibility
    schema_version: SchemaVersion = SchemaVersion.V1_0
    download_utc_timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    data_range: dict[str, str] | None = None
    provider_params: dict[str, Any] = Field(default_factory=dict)

    model_config = {"use_enum_values": True}

    @property
    def frequency(self) -> str:
        """Get frequency string for backward compatibility.

        For time bars, returns the frequency (e.g., "1min", "daily").
        For other bar types, returns a descriptive string (e.g., "volume_1000").
        """
        if self.bar_type == "time":
            return self.bar_params.get("frequency", "unknown")
        threshold = self.bar_params.get("threshold", "")
        return f"{self.bar_type}_{threshold}"


class DataObject(BaseModel):
    """Container for data and metadata."""

    data: pl.DataFrame
    metadata: Metadata

    model_config = {"arbitrary_types_allowed": True}

    @model_validator(mode="after")
    def validate_schema(self) -> DataObject:
        """Validate DataFrame schema matches expected format."""
        required_columns = {
            "timestamp": pl.Datetime,
            "open": pl.Float64,
            "high": pl.Float64,
            "low": pl.Float64,
            "close": pl.Float64,
            "volume": pl.Float64,
        }

        # Check required columns exist
        for col, dtype in required_columns.items():
            if col not in self.data.columns:
                raise ValueError(f"Missing required column: {col}")

            # Check data type (allow compatible types)
            actual_dtype = self.data[col].dtype
            if col == "timestamp":
                if not isinstance(actual_dtype, pl.Datetime):
                    raise ValueError(f"Column {col} must be Datetime, got {actual_dtype}")
            elif actual_dtype not in (dtype, pl.Float32, pl.Int64, pl.Int32):
                # Allow numeric types that can be converted to Float64
                raise ValueError(f"Column {col} must be numeric, got {actual_dtype}")

        return self

    def to_canonical_schema(self) -> pl.DataFrame:
        """Convert DataFrame to canonical schema."""
        df = self.data.clone()

        # Ensure proper data types
        type_mapping = {
            "timestamp": pl.Datetime,
            "open": pl.Float64,
            "high": pl.Float64,
            "low": pl.Float64,
            "close": pl.Float64,
            "volume": pl.Float64,
            "dividends": pl.Float64,
            "splits": pl.Float64,
        }

        for col, dtype in type_mapping.items():
            if col in df.columns:
                if col == "timestamp" and not isinstance(df[col].dtype, pl.Datetime):
                    # Convert to datetime if needed
                    df = df.with_columns(pl.col(col).cast(dtype))
                elif col != "timestamp" and df[col].dtype != dtype:
                    # Cast numeric columns
                    df = df.with_columns(pl.col(col).cast(dtype))
            elif col in ("dividends", "splits"):
                # Add missing optional columns with defaults
                if col == "dividends":
                    df = df.with_columns(pl.lit(0.0).alias(col).cast(dtype))
                else:  # splits
                    df = df.with_columns(pl.lit(1.0).alias(col).cast(dtype))

        # Select columns in canonical order
        columns = ["timestamp", "open", "high", "low", "close", "volume"]
        if "dividends" in df.columns:
            columns.append("dividends")
        if "splits" in df.columns:
            columns.append("splits")

        return df.select(columns)
