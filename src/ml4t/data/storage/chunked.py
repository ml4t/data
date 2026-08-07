"""Chunk-based storage strategy for large datasets."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import polars as pl
import structlog

from ml4t.data.core.models import DataObject, Metadata
from ml4t.data.core.schemas import align_frames_for_concat
from ml4t.data.storage.keys import (
    KEY_ENCODING_PREFIX,
    decode_storage_key,
    storage_key_path,
)
from ml4t.data.utils.locking import file_lock

logger = structlog.get_logger()


@dataclass
class ChunkInfo:
    """Information about a data chunk."""

    chunk_id: str
    start_date: datetime
    end_date: datetime
    row_count: int
    file_path: Path
    size_bytes: int

    @property
    def date_range_str(self) -> str:
        """Get date range as string."""
        return f"{self.start_date.date()} to {self.end_date.date()}"


class ChunkedStorage:
    """
    Storage backend that splits data into time-based chunks.

    Useful for:
    - Large datasets that would be inefficient as single files
    - Incremental updates without rewriting entire dataset
    - Parallel processing of data chunks
    - Efficient querying of specific time ranges
    """

    DEFAULT_CHUNK_SIZE_DAYS = 30  # Monthly chunks by default
    MAX_CHUNK_SIZE_MB = 100  # Maximum chunk size in MB

    def __init__(
        self,
        base_path: Path,
        chunk_size_days: int = DEFAULT_CHUNK_SIZE_DAYS,
        compression: str = "snappy",
    ) -> None:
        """
        Initialize chunked storage.

        Args:
            base_path: Base directory for storage
            chunk_size_days: Number of days per chunk
            compression: Compression algorithm for Parquet files
        """
        self.base_path = Path(base_path).expanduser().resolve()
        self.chunk_size_days = chunk_size_days
        self.compression = compression

        # Chunk storage directory
        self.chunks_path = self.base_path / "chunks"
        self.chunks_path.mkdir(parents=True, exist_ok=True)

        # Metadata storage
        self.metadata_path = self.base_path / "metadata"
        self.metadata_path.mkdir(parents=True, exist_ok=True)

    def _chunk_path(self, key: str, chunk_id: str) -> Path:
        """Return a chunk path without repeating the full key in one filename."""
        key_directory = storage_key_path(self.chunks_path, key)
        return storage_key_path(key_directory, chunk_id, ".parquet")

    def _get_chunk_id(
        self,
        symbol: str,
        frequency: str,
        start_date: datetime,
    ) -> str:
        """
        Generate chunk ID based on symbol, frequency, and date.

        Args:
            symbol: Symbol name
            frequency: Data frequency
            start_date: Start date of chunk

        Returns:
            Unique chunk identifier
        """
        year = start_date.year
        month = start_date.month

        if self.chunk_size_days <= 7:
            # Weekly chunks
            week = start_date.isocalendar()[1]
            return f"{symbol}_{frequency}_{year}_W{week:02d}"
        if self.chunk_size_days <= 31:
            # Monthly chunks
            return f"{symbol}_{frequency}_{year}_{month:02d}"
        if self.chunk_size_days <= 93:
            # Quarterly chunks
            quarter = (month - 1) // 3 + 1
            return f"{symbol}_{frequency}_{year}_Q{quarter}"
        # Yearly chunks
        return f"{symbol}_{frequency}_{year}"

    def _get_chunk_boundaries(
        self,
        start_date: datetime,
    ) -> tuple[datetime, datetime]:
        """
        Get the inclusive start and exclusive end for a chunk.

        Args:
            start_date: Reference date

        Returns:
            Tuple of (chunk_start, exclusive_chunk_end)
        """
        day_start = start_date.replace(hour=0, minute=0, second=0, microsecond=0)
        if self.chunk_size_days <= 7:
            # Weekly: Monday through the following Monday.
            chunk_start = day_start - timedelta(days=day_start.weekday())
            chunk_end = chunk_start + timedelta(days=7)
        elif self.chunk_size_days <= 31:
            chunk_start = day_start.replace(day=1)
            if start_date.month == 12:
                chunk_end = chunk_start.replace(year=start_date.year + 1, month=1)
            else:
                chunk_end = chunk_start.replace(month=start_date.month + 1)
        elif self.chunk_size_days <= 93:
            quarter = (start_date.month - 1) // 3
            chunk_start = day_start.replace(
                month=quarter * 3 + 1,
                day=1,
            )
            if quarter == 3:
                chunk_end = chunk_start.replace(year=start_date.year + 1, month=1)
            else:
                chunk_end = chunk_start.replace(month=(quarter + 1) * 3 + 1)
        else:
            chunk_start = day_start.replace(month=1, day=1)
            chunk_end = chunk_start.replace(year=start_date.year + 1)

        return chunk_start, chunk_end

    def _split_into_chunks(
        self,
        df: pl.DataFrame,
        metadata: Metadata,
    ) -> list[tuple[pl.DataFrame, str]]:
        """
        Split DataFrame into time-based chunks.

        Args:
            df: DataFrame to split
            metadata: Data metadata

        Returns:
            List of (chunk_df, chunk_id) tuples
        """
        if df.is_empty():
            return []

        # Ensure data is sorted by timestamp
        df = df.sort("timestamp")

        chunks = []
        min_ts = df["timestamp"].min()
        max_ts = df["timestamp"].max()

        # Generate chunk boundaries
        current = min_ts
        while current <= max_ts:
            chunk_start, chunk_end = self._get_chunk_boundaries(current)

            # Filter data for this chunk
            chunk_df = df.filter(
                (pl.col("timestamp") >= chunk_start) & (pl.col("timestamp") < chunk_end)
            )

            if not chunk_df.is_empty():
                chunk_id = self._get_chunk_id(
                    metadata.symbol,
                    metadata.frequency,
                    chunk_start,
                )
                chunks.append((chunk_df, chunk_id))

            # Move to next chunk period
            current = chunk_end

        logger.info(
            f"Split data into {len(chunks)} chunks",
            symbol=metadata.symbol,
            total_rows=len(df),
            chunk_size_days=self.chunk_size_days,
        )

        return chunks

    def _load_chunk_index(self, key: str) -> dict[str, ChunkInfo]:
        """
        Load chunk index for a data key.

        Args:
            key: Storage key

        Returns:
            Dictionary mapping chunk_id to ChunkInfo
        """
        index_file = storage_key_path(self.metadata_path, key, "_index.json")

        if not index_file.exists():
            return {}

        with file_lock(index_file):
            import json

            with open(index_file) as f:
                index_data = json.load(f)

        # Convert to ChunkInfo objects
        chunks = {}
        for chunk_id, info in index_data.items():
            chunks[chunk_id] = ChunkInfo(
                chunk_id=chunk_id,
                start_date=datetime.fromisoformat(info["start_date"]),
                end_date=datetime.fromisoformat(info["end_date"]),
                row_count=info["row_count"],
                file_path=Path(info["file_path"]),
                size_bytes=info["size_bytes"],
            )

        return chunks

    def _save_chunk_index(
        self,
        key: str,
        chunks: dict[str, ChunkInfo],
    ) -> None:
        """
        Save chunk index for a data key.

        Args:
            key: Storage key
            chunks: Chunk information dictionary
        """
        index_file = storage_key_path(self.metadata_path, key, "_index.json")

        # Convert to JSON-serializable format
        index_data = {}
        for chunk_id, info in chunks.items():
            index_data[chunk_id] = {
                "chunk_id": info.chunk_id,
                "start_date": info.start_date.isoformat(),
                "end_date": info.end_date.isoformat(),
                "row_count": info.row_count,
                "file_path": str(info.file_path),
                "size_bytes": info.size_bytes,
            }

        with file_lock(index_file):
            import json

            with open(index_file, "w") as f:
                json.dump(index_data, f, indent=2)

    def exists(self, key: str) -> bool:
        """
        Check if data exists for the given key.

        Args:
            key: Storage key

        Returns:
            True if data exists
        """
        index_file = storage_key_path(self.metadata_path, key, "_index.json")
        return index_file.exists()

    def read(
        self,
        key: str,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> DataObject:
        """
        Read data from chunked storage.

        Args:
            key: Storage key
            start_date: Optional start date filter
            end_date: Optional exclusive end date filter

        Returns:
            DataObject with combined data from chunks

        Raises:
            KeyError: If key doesn't exist
        """
        if not self.exists(key):
            raise KeyError(f"Key {key} not found")

        # Load chunk index
        chunks = self._load_chunk_index(key)

        if not chunks:
            raise ValueError(f"No chunks found for {key}")

        # Filter chunks by date range if specified
        relevant_chunks = []
        for _chunk_id, info in chunks.items():
            if start_date and info.end_date < start_date:
                continue
            if end_date and info.start_date >= end_date:
                continue
            relevant_chunks.append(info)

        if not relevant_chunks:
            logger.warning(
                "No chunks match date range",
                key=key,
                start_date=start_date,
                end_date=end_date,
            )
            return DataObject(
                data=self._empty_frame(key, chunks), metadata=self._metadata_for_key(key)
            )

        # Sort chunks by start date
        relevant_chunks.sort(key=lambda c: c.start_date)

        logger.info(
            f"Reading {len(relevant_chunks)} chunks",
            key=key,
            total_chunks=len(chunks),
            date_range=f"{start_date} to {end_date}" if start_date or end_date else "all",
        )

        # Read and combine chunks
        dfs = []
        for chunk_info in relevant_chunks:
            chunk_path = self._chunk_path(key, chunk_info.chunk_id)

            # Read Parquet file with file locking
            with file_lock(chunk_path):
                chunk_df = pl.read_parquet(chunk_path)

            # Apply date filter if needed
            if start_date or end_date:
                if start_date:
                    chunk_df = chunk_df.filter(pl.col("timestamp") >= start_date)
                if end_date:
                    chunk_df = chunk_df.filter(pl.col("timestamp") < end_date)
            dfs.append(chunk_df)

        # Combine all chunks
        combined_df = pl.concat(dfs) if dfs else pl.DataFrame()

        metadata = self._metadata_for_key(key)

        # Update metadata with actual data range
        if not combined_df.is_empty():
            metadata.data_range = {
                "start": str(combined_df["timestamp"].min()),
                "end": str(combined_df["timestamp"].max()),
            }

        return DataObject(data=combined_df, metadata=metadata)

    def _metadata_for_key(self, key: str) -> Metadata:
        """Reconstruct the metadata encoded in a chunked-storage key."""
        parts = key.split("/")
        if len(parts) == 3:
            asset_class, frequency, symbol = parts
        else:
            asset_class, frequency, symbol = "", "", ""
        return Metadata(
            provider="",
            symbol=symbol,
            asset_class=asset_class,
            bar_params={"frequency": frequency},
        )

    def _empty_frame(self, key: str, chunks: dict[str, ChunkInfo]) -> pl.DataFrame:
        """Return an empty frame with the stored data schema."""
        first_chunk = min(chunks.values(), key=lambda chunk: chunk.start_date)
        chunk_path = self._chunk_path(key, first_chunk.chunk_id)
        return pl.DataFrame(schema=pl.read_parquet_schema(chunk_path))

    def write(self, data_object: DataObject) -> str:
        """
        Write data to chunked storage.

        Args:
            data_object: Data object to store

        Returns:
            Storage key
        """
        metadata = data_object.metadata
        key = f"{metadata.asset_class}/{metadata.frequency}/{metadata.symbol}"

        # Split data into chunks
        chunks_data = self._split_into_chunks(data_object.data, metadata)

        if not chunks_data:
            logger.warning("No data to write", key=key)
            return key

        # Load existing chunk index
        existing_chunks = self._load_chunk_index(key)

        # Write each chunk
        chunk_index = {}
        for chunk_df, chunk_id in chunks_data:
            # Check if chunk exists and merge if needed
            if chunk_id in existing_chunks:
                logger.info(
                    f"Merging with existing chunk {chunk_id}",
                    existing_rows=existing_chunks[chunk_id].row_count,
                    new_rows=len(chunk_df),
                )

                # Read existing chunk
                chunk_path = self._chunk_path(key, chunk_id)
                with file_lock(chunk_path):
                    existing_df = pl.read_parquet(chunk_path)

                # Merge data
                existing_df, chunk_df = align_frames_for_concat(existing_df, chunk_df)
                merged_df = (
                    pl.concat([existing_df, chunk_df])
                    .unique(
                        subset=["timestamp"],
                        keep="last",
                    )
                    .sort("timestamp")
                )

                chunk_df = merged_df

            # Prepare chunk data for writing
            # (chunk_df is already the DataFrame to write)

            # Write chunk directly using Parquet
            chunk_path = self._chunk_path(key, chunk_id)
            chunk_path.parent.mkdir(parents=True, exist_ok=True)

            # Write Parquet file with file locking
            with file_lock(chunk_path):
                chunk_df.write_parquet(chunk_path, compression=self.compression)

            # Create chunk info
            chunk_info = ChunkInfo(
                chunk_id=chunk_id,
                start_date=chunk_df["timestamp"].min(),
                end_date=chunk_df["timestamp"].max(),
                row_count=len(chunk_df),
                file_path=chunk_path,
                size_bytes=chunk_path.stat().st_size if chunk_path.exists() else 0,
            )

            chunk_index[chunk_id] = chunk_info

            logger.info(
                f"Wrote chunk {chunk_id}",
                rows=chunk_info.row_count,
                size_mb=chunk_info.size_bytes / (1024 * 1024),
                date_range=chunk_info.date_range_str,
            )

        # Merge with existing chunks not updated
        for chunk_id, info in existing_chunks.items():
            if chunk_id not in chunk_index:
                chunk_index[chunk_id] = info

        # Save chunk index
        self._save_chunk_index(key, chunk_index)

        logger.info(
            "Chunked storage complete",
            key=key,
            total_chunks=len(chunk_index),
            new_chunks=len(chunks_data),
        )

        return key

    def delete(self, key: str) -> None:
        """
        Delete all chunks for a key.

        Args:
            key: Storage key
        """
        # Load chunk index
        chunks = self._load_chunk_index(key)

        # Delete each chunk file
        for chunk_id in chunks:
            chunk_path = self._chunk_path(key, chunk_id)
            try:
                if chunk_path.exists():
                    chunk_path.unlink()
            except Exception as e:
                logger.warning(
                    f"Failed to delete chunk {chunk_id}",
                    error=str(e),
                )

        # Delete index file
        index_file = storage_key_path(self.metadata_path, key, "_index.json")
        if index_file.exists():
            index_file.unlink()

        logger.info(f"Deleted {len(chunks)} chunks for {key}")

    def list_keys(self, prefix: str = "") -> list[str]:
        """
        List all keys with optional prefix filter.

        Args:
            prefix: Key prefix to filter by

        Returns:
            List of matching keys
        """
        keys = []

        # List all index files
        for index_file in self.metadata_path.glob(f"{KEY_ENCODING_PREFIX}*_index.json"):
            encoded_key = index_file.name.removesuffix("_index.json")
            try:
                key = decode_storage_key(encoded_key)
            except ValueError as error:
                logger.warning("Ignoring invalid chunk index", path=index_file, error=str(error))
                continue

            if not prefix or key.startswith(prefix):
                keys.append(key)

        if any(
            not path.name.startswith(KEY_ENCODING_PREFIX)
            for path in self.metadata_path.glob("*_index.json")
        ):
            logger.warning(
                "Legacy chunked storage entries require explicit migration",
                base_path=self.base_path,
            )

        return sorted(keys)

    def get_chunk_info(self, key: str) -> list[ChunkInfo]:
        """
        Get information about all chunks for a key.

        Args:
            key: Storage key

        Returns:
            List of ChunkInfo objects
        """
        chunks = self._load_chunk_index(key)
        return sorted(chunks.values(), key=lambda c: c.start_date)
