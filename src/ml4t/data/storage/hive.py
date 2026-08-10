"""Hive partitioned storage implementation.

Provides efficient time-series data storage using Hive-style partitioning
with measured 7x query performance improvement.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl
import structlog

from ml4t.data.core.schemas import align_frames_for_concat

from .backend import StorageBackend, StorageConfig
from .keys import (
    KEY_ENCODING_PREFIX,
    contained_path,
    decode_storage_key,
    encode_storage_key,
)

logger = structlog.get_logger()


@dataclass(frozen=True)
class Partition:
    """One Parquet partition backing a storage key.

    ``values`` maps each partition column to its value, so a caller reports what a
    partition covers without parsing the path. The key directory is an encoded name and
    the generation directory is an implementation detail; neither is meant to be read.
    """

    path: Path
    values: dict[str, int]
    size_bytes: int

    @property
    def label(self) -> str:
        """Partition columns joined as they are conventionally displayed, e.g. ``2023-01``."""
        return "-".join(
            f"{value:02d}" if column != "year" else str(value)
            for column, value in self.values.items()
        )


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
        self, base_path: Path, partition_cols: list[str], values: tuple[Any, ...]
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
            return sorted(key_path.glob(glob_pattern))

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
        data: pl.LazyFrame | pl.DataFrame,
        key: str,
        metadata: dict[str, Any] | None = None,
    ) -> Path:
        """Write data using Hive partitioning.

        Args:
            data: DataFrame or LazyFrame to write
            key: Storage key (e.g., "BTC-USD" or "equities/daily/AAPL")
            metadata: Optional metadata dict

        Returns:
            Path to the committed data generation
        """
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

        with self._key_lock(key):
            staging_path, generation_id = self._prepare_generation(key)
            try:
                partitions_written = []
                for partition_values, partition_df in df.group_by(
                    partition_cols, maintain_order=True
                ):
                    partition_path = self._build_partition_path(
                        staging_path, partition_cols, partition_values
                    )
                    partition_path.mkdir(parents=True, exist_ok=True)
                    partition_df = partition_df.drop(partition_cols)
                    self._atomic_write(partition_df, partition_path / "data.parquet")
                    partitions_written.append(partition_path.relative_to(staging_path).as_posix())

                commit_metadata = (
                    {
                        "last_updated": datetime.now().isoformat(),
                        "partitions": partitions_written,
                        "row_count": len(df),
                        "schema": list(df.columns),
                        "custom": metadata or {},
                    }
                    if self.config.metadata_tracking
                    else {}
                )
                commit = self._publish_generation(key, staging_path, generation_id, commit_metadata)
            except BaseException:
                self._cleanup_staging(staging_path)
                raise

        return commit.generation_path

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
        key_path = self._current_commit(key).generation_path

        # Get partition columns based on granularity
        partition_cols = self._get_partition_columns()

        # Build list of partition paths to read with pruning
        partition_paths = self._find_partition_paths(key_path, partition_cols, start_date, end_date)

        if not partition_paths:
            return pl.LazyFrame()

        # Use Polars lazy reading with predicate pushdown
        lazy_frames = []
        normalized_start = (
            start_date.replace(tzinfo=UTC)
            if start_date is not None and start_date.tzinfo is None
            else start_date.astimezone(UTC)
            if start_date is not None
            else None
        )
        normalized_end = (
            end_date.replace(tzinfo=UTC)
            if end_date is not None and end_date.tzinfo is None
            else end_date.astimezone(UTC)
            if end_date is not None
            else None
        )
        for path in partition_paths:
            lf = pl.scan_parquet(path)
            timestamp_type = lf.collect_schema().get("timestamp")
            if isinstance(timestamp_type, pl.Datetime):
                timestamp = pl.col("timestamp")
                if timestamp_type.time_zone is None:
                    timestamp = timestamp.dt.replace_time_zone("UTC")
                else:
                    timestamp = timestamp.dt.convert_time_zone("UTC")
                lf = lf.with_columns(timestamp.cast(pl.Datetime("us", "UTC")))

            # Apply date filters
            if normalized_start:
                lf = lf.filter(pl.col("timestamp") >= normalized_start)
            if normalized_end:
                lf = lf.filter(pl.col("timestamp") < normalized_end)

            # Apply column selection after timestamp filtering.
            if columns:
                lf = lf.select(columns)

            lazy_frames.append(lf)

        # Concatenate all partitions
        if len(lazy_frames) == 1:
            return lazy_frames[0]
        return pl.concat(lazy_frames, how="vertical_relaxed")

    def partitions(
        self,
        key: str,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> list[Partition]:
        """Describe the partitions currently backing a key.

        The on-disk layout is not addressable from outside: the key directory is an
        encoded name and writes land in a generation directory that changes on every
        write. Anything that needs to report what is stored - how many partitions, what
        each covers, how large it is - has to ask, and this is the way to ask.

        Args:
            key: Storage key
            start_date: Optional start date filter, pruning as ``read`` does
            end_date: Optional end date filter

        Returns:
            One ``Partition`` per Parquet file, ordered by partition value

        Example:
            >>> for part in storage.partitions(key):  # doctest: +SKIP
            ...     print(part.label, part.size_bytes)
        """
        partition_cols = self._get_partition_columns()
        generation_path = self._current_commit(key).generation_path
        found = [
            Partition(
                path=path,
                values=self._extract_partition_values(path, partition_cols),
                size_bytes=path.stat().st_size,
            )
            for path in self._find_partition_paths(
                generation_path, partition_cols, start_date, end_date
            )
        ]
        # Sorted by value, not by path: the directory names are unpadded, so a path sort
        # puts month=10 immediately after month=1.
        return sorted(found, key=lambda part: tuple(part.values[col] for col in partition_cols))

    def list_keys(self) -> list[str]:
        """List all keys in storage.

        Returns:
            List of storage keys
        """
        keys = []
        for path in self.base_path.glob(f"{KEY_ENCODING_PREFIX}*"):
            if path.is_dir() and (path / "CURRENT").is_file():
                try:
                    keys.append(decode_storage_key(path.name))
                except ValueError as error:
                    logger.warning("Ignoring invalid storage entry", path=path, error=str(error))
        legacy_entries = [
            path
            for path in self.base_path.iterdir()
            if path.is_dir()
            and not path.name.startswith((".", KEY_ENCODING_PREFIX))
            and any(path.glob("year=*/**/data.parquet"))
        ]
        if legacy_entries:
            logger.warning(
                "Legacy Hive storage entries require explicit migration",
                base_path=self.base_path,
                entries=len(legacy_entries),
            )
        return sorted(keys)

    def exists(self, key: str) -> bool:
        """Check if key exists.

        Args:
            key: Storage key

        Returns:
            True if key exists
        """
        try:
            self._current_commit(key)
        except KeyError:
            return False
        return True

    def delete(self, key: str) -> bool:
        """Delete all data for a key.

        Args:
            key: Storage key

        Returns:
            True if successful
        """
        return self._delete_key(key)

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

        stored = self.read(key)
        if "timestamp" not in stored.collect_schema().names():
            return None
        df = stored.select("timestamp").collect()
        if df.is_empty():
            return None
        latest = df["timestamp"].max()
        if not isinstance(latest, datetime):
            raise TypeError(f"Expected datetime timestamp for '{key}', got {type(latest).__name__}")
        return latest

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
        chunks_dir = contained_path(
            self.base_path,
            ".chunks",
            encode_storage_key(provider),
            encode_storage_key(symbol),
        )
        chunks_dir.mkdir(parents=True, exist_ok=True)

        # Create chunk filename with timestamp range
        chunk_name = (
            f"{start_time.strftime('%Y%m%d_%H%M')}_{end_time.strftime('%Y%m%d_%H%M')}.parquet"
        )
        chunk_path = chunks_dir / chunk_name

        # Save chunk
        data.write_parquet(chunk_path, compression=self._parquet_compression())

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
            existing_df, data = align_frames_for_concat(existing_df, data)
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
        try:
            return self._current_commit(key).generation_path
        except KeyError:
            return self._key_path(key)

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

        Raises:
            KeyError: If no published dataset exists for the provider and symbol.
        """
        key = f"{provider}/{symbol}"
        metadata = self.get_metadata(key) or {
            "symbol": symbol,
            "provider": provider,
            "first_update": last_update.isoformat(),
            "update_history": [],
        }
        metadata.setdefault("symbol", symbol)
        metadata.setdefault("provider", provider)
        metadata.setdefault("first_update", last_update.isoformat())

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
