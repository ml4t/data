"""Hive partitioned storage implementation.

Provides efficient time-series data storage using Hive-style partitioning
with measured 7x query performance improvement.
"""

from __future__ import annotations

import json
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import polars as pl
from filelock import FileLock

from .backend import StorageBackend, StorageConfig

if TYPE_CHECKING:
    from ml4t.data.core.models import DataObject


class HiveStorage(StorageBackend):
    """Hive partitioned storage with configurable time-based partitioning.

    This implementation provides:
    - 7x query performance improvement for time-based queries
    - Configurable partition granularity (year, month, day, hour)
    - Atomic writes with temp file pattern
    - Metadata tracking in JSON manifests
    - File locking for concurrent access safety
    - Polars lazy evaluation throughout

    Partition Granularity:
        Configure via StorageConfig.partition_granularity:
        - "year": Best for daily data (~252 rows/partition)
        - "month": Best for hourly data (~720 rows/partition) [default]
        - "day": Best for minute data (~1,440 rows/partition)
        - "hour": Best for second/tick data (~3,600 rows/partition)

    Example:
        >>> from ml4t.data.storage import HiveStorage, StorageConfig
        >>> # For minute data, use day-level partitioning
        >>> config = StorageConfig(base_path="./data", partition_granularity="day")
        >>> storage = HiveStorage(config)
    """

    def __init__(self, config: StorageConfig):
        """Initialize Hive storage backend.

        Args:
            config: Storage configuration
        """
        super().__init__(config)
        self.metadata_dir = self.base_path / ".metadata"
        self.metadata_dir.mkdir(exist_ok=True)

    def _get_partition_columns(self) -> list[str]:
        """Get partition column names based on configured granularity.

        Returns:
            List of partition column names (e.g., ["year", "month"])
        """
        granularity = getattr(self.config, "partition_granularity", "month")
        granularity_to_cols = {
            "year": ["year"],
            "month": ["year", "month"],
            "day": ["year", "month", "day"],
            "hour": ["year", "month", "day", "hour"],
        }
        return granularity_to_cols.get(granularity, ["year", "month"])

    def _add_partition_columns(self, df: pl.DataFrame, partition_cols: list[str]) -> pl.DataFrame:
        """Add partition columns to DataFrame based on timestamp.

        Args:
            df: DataFrame with timestamp column
            partition_cols: List of partition columns to add

        Returns:
            DataFrame with partition columns added
        """
        col_exprs = []
        if "year" in partition_cols:
            col_exprs.append(pl.col("timestamp").dt.year().alias("year"))
        if "month" in partition_cols:
            col_exprs.append(pl.col("timestamp").dt.month().alias("month"))
        if "day" in partition_cols:
            col_exprs.append(pl.col("timestamp").dt.day().alias("day"))
        if "hour" in partition_cols:
            col_exprs.append(pl.col("timestamp").dt.hour().alias("hour"))

        if col_exprs:
            return df.with_columns(col_exprs)
        return df

    def _build_partition_path(
        self, base_path: Path, partition_cols: list[str], values: tuple
    ) -> Path:
        """Build partition directory path from column names and values.

        Args:
            base_path: Base directory for the key
            partition_cols: List of partition column names
            values: Tuple of partition values (from group_by)

        Returns:
            Path to partition directory
        """
        # Handle both tuple and single value from group_by
        if not isinstance(values, tuple):
            values = (values,)

        path = base_path
        for col, val in zip(partition_cols, values, strict=False):
            path = path / f"{col}={val}"
        return path

    def _find_partition_paths(
        self,
        key_path: Path,
        partition_cols: list[str],
        start_date: datetime | None,
        end_date: datetime | None,
    ) -> list[Path]:
        """Find partition paths with optional date pruning.

        Args:
            key_path: Base path for the key's data
            partition_cols: List of partition column names
            start_date: Optional start date for pruning
            end_date: Optional end date for pruning

        Returns:
            List of paths to data.parquet files
        """
        # Build glob pattern based on partition columns
        glob_pattern = "/".join(f"{col}=*" for col in partition_cols) + "/data.parquet"

        if not (start_date or end_date):
            # No filtering, return all partitions
            return list(key_path.glob(glob_pattern))

        # With date filtering, we need to prune partitions
        partition_paths = []

        for parquet_path in sorted(key_path.glob(glob_pattern)):
            # Extract partition values from path
            partition_values = self._extract_partition_values(parquet_path, partition_cols)

            # Check if partition is within date range
            if self._partition_in_range(partition_values, start_date, end_date):
                partition_paths.append(parquet_path)

        return partition_paths

    def _extract_partition_values(self, path: Path, partition_cols: list[str]) -> dict[str, int]:
        """Extract partition values from a partition path.

        Args:
            path: Path containing partition directories (e.g., .../year=2024/month=1/data.parquet)
            partition_cols: Expected partition column names

        Returns:
            Dictionary mapping column names to their integer values
        """
        values = {}
        for part in path.parts:
            if "=" in part:
                col, val = part.split("=", 1)
                if col in partition_cols:
                    values[col] = int(val)
        return values

    def _partition_in_range(
        self,
        partition_values: dict[str, int],
        start_date: datetime | None,
        end_date: datetime | None,
    ) -> bool:
        """Check if a partition is within the date range.

        Args:
            partition_values: Dictionary of partition column values
            start_date: Optional start date
            end_date: Optional end date

        Returns:
            True if partition may contain data in range
        """
        year = partition_values.get("year")
        month = partition_values.get("month")
        day = partition_values.get("day")
        hour = partition_values.get("hour")

        # Year-level pruning
        if year is not None:
            if start_date and year < start_date.year:
                return False
            if end_date and year > end_date.year:
                return False

        # Month-level pruning (only if year matches boundary)
        if month is not None:
            if start_date and year == start_date.year and month < start_date.month:
                return False
            if end_date and year == end_date.year and month > end_date.month:
                return False

        # Day-level pruning (only if year/month match boundary)
        if day is not None:
            if (
                start_date
                and year == start_date.year
                and month == start_date.month
                and day < start_date.day
            ):
                return False
            if (
                end_date
                and year == end_date.year
                and month == end_date.month
                and day > end_date.day
            ):
                return False

        # Hour-level pruning (only if year/month/day match boundary)
        if hour is not None:
            if (
                start_date
                and year == start_date.year
                and month == start_date.month
                and day == start_date.day
                and hour < start_date.hour
            ):
                return False
            if (
                end_date
                and year == end_date.year
                and month == end_date.month
                and day == end_date.day
                and hour > end_date.hour
            ):
                return False

        return True

    def write(
        self,
        data: pl.LazyFrame | pl.DataFrame | DataObject,
        key: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Path | str:
        """Write data using Hive partitioning.

        Args:
            data: Data to write (DataFrame, LazyFrame, or DataObject)
            key: Storage key (e.g., "BTC-USD" or "equities/daily/AAPL"). Optional if data is DataObject.
            metadata: Optional metadata dict

        Returns:
            Path to base directory (old API) or storage key string (new DataObject API)
        """
        # Handle DataObject input (new API)
        from ml4t.data.core.models import DataObject

        if isinstance(data, DataObject):
            # Extract components from DataObject
            df_data = data.data
            data_metadata = data.metadata
            # Construct storage key from metadata
            storage_key = (
                f"{data_metadata.asset_class}/{data_metadata.frequency}/{data_metadata.symbol}"
            )
            # Use the old API internally
            self.write(df_data, storage_key, None)
            return storage_key

        # Old API: data is DataFrame/LazyFrame, key is required
        if key is None:
            raise ValueError("key is required when data is not a DataObject")

        # Ensure LazyFrame for efficiency
        lazy_data = self._ensure_lazy(data)

        # Collect minimal data for partitioning info
        df = lazy_data.collect()

        # Ensure timestamp column exists
        if "timestamp" not in df.columns:
            raise ValueError("Data must have 'timestamp' column for Hive partitioning")

        # Get partition columns based on granularity
        partition_cols = self._get_partition_columns()

        # Add partition columns dynamically based on granularity
        df = self._add_partition_columns(df, partition_cols)

        # Create key directory
        key_path = self.base_path / key.replace("/", "_")
        key_path.mkdir(exist_ok=True)

        # Group by partitions and write
        partitions_written = []

        for partition_values, partition_df in df.group_by(partition_cols, maintain_order=True):
            # Create partition path dynamically
            partition_path = self._build_partition_path(key_path, partition_cols, partition_values)
            partition_path.mkdir(parents=True, exist_ok=True)

            # Remove partition columns from data
            partition_df = partition_df.drop(partition_cols)

            # Write with atomic pattern
            file_path = partition_path / "data.parquet"
            self._atomic_write(partition_df, file_path)
            partitions_written.append(str(partition_path.relative_to(self.base_path)))

        # Update metadata
        if self.config.metadata_tracking:
            self._update_metadata(
                key,
                {
                    "last_updated": datetime.now().isoformat(),
                    "partitions": partitions_written,
                    "row_count": len(df),
                    "schema": list(df.columns),
                    "custom": metadata or {},
                },
            )

        return key_path

    def read(
        self,
        key: str,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        columns: list[str] | None = None,
    ) -> pl.LazyFrame:
        """Read data from Hive partitions.

        Args:
            key: Storage key
            start_date: Optional start date filter
            end_date: Optional end date filter
            columns: Optional columns to select

        Returns:
            LazyFrame with requested data
        """
        key_path = self.base_path / key.replace("/", "_")

        if not key_path.exists():
            raise KeyError(f"Key '{key}' not found in storage")

        # Get partition columns based on granularity
        partition_cols = self._get_partition_columns()

        # Build list of partition paths to read with pruning
        partition_paths = self._find_partition_paths(key_path, partition_cols, start_date, end_date)

        if not partition_paths:
            return pl.LazyFrame()

        # Use Polars lazy reading with predicate pushdown
        lazy_frames = []
        for path in partition_paths:
            lf = pl.scan_parquet(path)

            # Apply column selection
            if columns:
                lf = lf.select(columns)

            # Apply date filters
            if start_date:
                lf = lf.filter(pl.col("timestamp") >= start_date)
            if end_date:
                lf = lf.filter(pl.col("timestamp") < end_date)

            lazy_frames.append(lf)

        # Concatenate all partitions
        if len(lazy_frames) == 1:
            return lazy_frames[0]
        return pl.concat(lazy_frames, how="vertical_relaxed")

    def list_keys(self) -> list[str]:
        """List all keys in storage.

        Returns:
            List of storage keys
        """
        keys = []
        for path in self.base_path.iterdir():
            if path.is_dir() and not path.name.startswith("."):
                # Convert back from filesystem-safe name
                keys.append(path.name.replace("_", "/"))
        return sorted(keys)

    def exists(self, key: str) -> bool:
        """Check if key exists.

        Args:
            key: Storage key

        Returns:
            True if key exists
        """
        key_path = self.base_path / key.replace("/", "_")
        return key_path.exists()

    def delete(self, key: str) -> bool:
        """Delete all data for a key.

        Args:
            key: Storage key

        Returns:
            True if successful
        """
        key_path = self.base_path / key.replace("/", "_")
        if key_path.exists():
            shutil.rmtree(key_path)

            # Remove metadata
            metadata_file = self.metadata_dir / f"{key.replace('/', '_')}.json"
            if metadata_file.exists():
                metadata_file.unlink()

            return True
        return False

    def get_metadata(self, key: str) -> dict[str, Any] | None:
        """Get metadata for a key.

        Args:
            key: Storage key

        Returns:
            Metadata dict or None
        """
        metadata_file = self.metadata_dir / f"{key.replace('/', '_')}.json"
        if metadata_file.exists():
            with open(metadata_file) as f:
                return json.load(f)
        return None

    def _atomic_write(self, df: pl.DataFrame, target_path: Path) -> None:
        """Write DataFrame atomically using temp file pattern.

        Args:
            df: DataFrame to write
            target_path: Target file path
        """
        # Write to temp file first
        with tempfile.NamedTemporaryFile(
            dir=target_path.parent, suffix=".parquet.tmp", delete=False
        ) as tmp_file:
            tmp_path = Path(tmp_file.name)

            # Write with compression
            df.write_parquet(tmp_path, compression=self.config.compression or "zstd")

            # Atomic rename
            tmp_path.replace(target_path)

    def _update_metadata(self, key: str, metadata: dict[str, Any]) -> None:
        """Update metadata for a key.

        Args:
            key: Storage key
            metadata: Metadata to store
        """
        metadata_file = self.metadata_dir / f"{key.replace('/', '_')}.json"

        if self.config.enable_locking:
            lock_file = self.metadata_dir / f"{key.replace('/', '_')}.lock"
            lock = FileLock(lock_file, timeout=10)

            with lock:
                self._write_metadata_file(metadata_file, metadata)
        else:
            self._write_metadata_file(metadata_file, metadata)

    def _write_metadata_file(self, path: Path, metadata: dict[str, Any]) -> None:
        """Write metadata to file.

        Args:
            path: Metadata file path
            metadata: Metadata to write
        """
        with tempfile.NamedTemporaryFile(
            dir=path.parent, mode="w", suffix=".json.tmp", delete=False
        ) as tmp_file:
            json.dump(metadata, tmp_file, indent=2, default=str)
            tmp_path = Path(tmp_file.name)
            tmp_path.replace(path)

    # Incremental update methods for IncrementalStorageBackend protocol

    def get_latest_timestamp(self, symbol: str, provider: str) -> datetime | None:
        """Get the latest timestamp for a symbol from a provider.

        Args:
            symbol: Symbol identifier
            provider: Data provider name

        Returns:
            Latest timestamp in the dataset, or None if no data exists
        """
        key = f"{provider}/{symbol}"

        if not self.exists(key):
            return None

        try:
            df = self.read(key).select("timestamp").collect()
            if df.is_empty():
                return None
            return df["timestamp"].max()
        except Exception:
            return None

    def save_chunk(
        self,
        data: pl.DataFrame,
        symbol: str,
        provider: str,
        start_time: datetime,
        end_time: datetime,
    ) -> Path:
        """Save an incremental data chunk.

        Args:
            data: DataFrame with OHLCV data
            symbol: Symbol identifier
            provider: Data provider name
            start_time: Start time of this chunk
            end_time: End time of this chunk

        Returns:
            Path to the saved chunk file
        """
        # Create chunks directory
        chunks_dir = self.base_path / ".chunks" / provider / symbol
        chunks_dir.mkdir(parents=True, exist_ok=True)

        # Create chunk filename with timestamp range
        chunk_name = (
            f"{start_time.strftime('%Y%m%d_%H%M')}_{end_time.strftime('%Y%m%d_%H%M')}.parquet"
        )
        chunk_path = chunks_dir / chunk_name

        # Save chunk
        data.write_parquet(chunk_path, compression=self.config.compression or "zstd")

        return chunk_path

    def update_combined_file(
        self,
        data: pl.DataFrame,
        symbol: str,
        provider: str,
    ) -> int:
        """Update the main combined file with new data.

        Args:
            data: New data to append
            symbol: Symbol identifier
            provider: Data provider name

        Returns:
            Number of new records added (after deduplication)
        """
        key = f"{provider}/{symbol}"

        # Read existing data
        if self.exists(key):
            existing_df = self.read(key).collect()
            combined = pl.concat([existing_df, data])
        else:
            combined = data

        # Deduplicate by timestamp, keeping latest
        rows_before = len(combined)
        combined = combined.unique(subset=["timestamp"], keep="last").sort("timestamp")
        rows_after = len(combined)

        # Write back to storage (correct parameter order: data, key, metadata)
        self.write(combined, key)

        return rows_after - rows_before

    def get_combined_file_path(self, symbol: str, provider: str) -> Path:
        """Get path to the main combined data directory.

        Args:
            symbol: Symbol identifier
            provider: Data provider name

        Returns:
            Path to combined data directory
        """
        key = f"{provider}/{symbol}"
        return self.base_path / key.replace("/", "_")

    def read_data(
        self,
        symbol: str,
        provider: str,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
    ) -> pl.DataFrame:
        """Read data for a symbol with optional time filtering.

        Args:
            symbol: Symbol identifier
            provider: Data provider name
            start_time: Optional start time filter
            end_time: Optional end time filter

        Returns:
            DataFrame with filtered data
        """
        key = f"{provider}/{symbol}"

        if not self.exists(key):
            return pl.DataFrame()

        return self.read(key, start_date=start_time, end_date=end_time).collect()

    def update_metadata(
        self,
        symbol: str,
        provider: str,
        last_update: datetime,
        records_added: int,
        chunk_file: str,
    ) -> None:
        """Update metadata after incremental update.

        Args:
            symbol: Symbol identifier
            provider: Data provider name
            last_update: Timestamp of this update
            records_added: Number of records added
            chunk_file: Name of the chunk file saved
        """
        key = f"{provider}/{symbol}"
        metadata_file = self.metadata_dir / f"{key.replace('/', '_')}.json"

        # Load existing metadata or create new
        if metadata_file.exists():
            with open(metadata_file) as f:
                metadata = json.load(f)
        else:
            metadata = {
                "symbol": symbol,
                "provider": provider,
                "first_update": last_update.isoformat(),
                "update_history": [],
            }

        # Update metadata
        metadata["last_update"] = last_update.isoformat()

        # Ensure update_history exists
        if "update_history" not in metadata:
            metadata["update_history"] = []

        metadata["update_history"].append(
            {
                "timestamp": last_update.isoformat(),
                "records_added": records_added,
                "chunk_file": chunk_file,
            }
        )

        # Keep only last 100 updates in history
        if len(metadata["update_history"]) > 100:
            metadata["update_history"] = metadata["update_history"][-100:]

        # Write metadata
        self._update_metadata(key, metadata)
