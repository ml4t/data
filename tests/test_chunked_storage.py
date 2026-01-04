"""Tests for chunked storage implementation."""

from datetime import datetime
from pathlib import Path

import polars as pl
import pytest

from ml4t.data.core.models import DataObject, Metadata
from ml4t.data.storage.chunked import ChunkedStorage, ChunkInfo


class TestChunkedStorage:
    """Test chunked storage functionality."""

    def test_basic_write_and_read(self, tmp_path: Path) -> None:
        """Test basic chunked storage operations."""
        storage = ChunkedStorage(base_path=tmp_path, chunk_size_days=30)

        # Create test data spanning multiple months
        dates = []
        for month in range(1, 4):  # Jan, Feb, Mar
            for day in range(1, 11):  # 10 days per month
                dates.append(datetime(2024, month, day))

        df = pl.DataFrame(
            {
                "timestamp": dates,
                "open": [100.0 + i for i in range(len(dates))],
                "high": [105.0 + i for i in range(len(dates))],
                "low": [95.0 + i for i in range(len(dates))],
                "close": [102.0 + i for i in range(len(dates))],
                "volume": [1000000 + i * 10000 for i in range(len(dates))],
            }
        )

        metadata = Metadata(
            provider="test",
            symbol="TEST",
            asset_class="equities",
            bar_type="time",
            bar_params={"frequency": "daily"},
        )

        data_obj = DataObject(data=df, metadata=metadata)

        # Write data
        key = storage.write(data_obj)
        assert key == "equities/daily/TEST"

        # Check chunks were created
        chunks = storage.get_chunk_info(key)
        assert len(chunks) == 3  # One chunk per month

        # Read all data back
        result = storage.read(key)
        assert len(result.data) == len(df)
        assert result.data["close"].to_list() == df["close"].to_list()

    def test_date_range_filtering(self, tmp_path: Path) -> None:
        """Test reading specific date ranges from chunks."""
        storage = ChunkedStorage(base_path=tmp_path, chunk_size_days=30)

        # Create test data for entire year
        dates = []
        for month in range(1, 13):
            for day in [1, 15]:  # Two days per month
                dates.append(datetime(2024, month, day))

        df = pl.DataFrame(
            {
                "timestamp": dates,
                "open": [100.0 + i for i in range(len(dates))],
                "high": [105.0 + i for i in range(len(dates))],
                "low": [95.0 + i for i in range(len(dates))],
                "close": [102.0 + i for i in range(len(dates))],
                "volume": [1000000 + i * 10000 for i in range(len(dates))],
            }
        )

        metadata = Metadata(
            provider="test",
            symbol="TEST",
            asset_class="equities",
            bar_type="time",
            bar_params={"frequency": "daily"},
        )

        # Write data
        storage.write(DataObject(data=df, metadata=metadata))
        key = "equities/daily/TEST"

        # Read specific month
        result = storage.read(
            key,
            start_date=datetime(2024, 3, 1),
            end_date=datetime(2024, 3, 31),
        )

        # Should only have March data
        assert len(result.data) == 2
        assert result.data["timestamp"].min().month == 3
        assert result.data["timestamp"].max().month == 3

    def test_incremental_updates(self, tmp_path: Path) -> None:
        """Test incremental updates to existing chunks."""
        storage = ChunkedStorage(base_path=tmp_path, chunk_size_days=30)

        # Initial data for January
        initial_dates = [
            datetime(2024, 1, 1),
            datetime(2024, 1, 5),
            datetime(2024, 1, 10),
        ]

        initial_df = pl.DataFrame(
            {
                "timestamp": initial_dates,
                "open": [100.0, 105.0, 110.0],
                "high": [105.0, 110.0, 115.0],
                "low": [95.0, 100.0, 105.0],
                "close": [102.0, 107.0, 112.0],
                "volume": [1000000, 1010000, 1020000],
            }
        )

        metadata = Metadata(
            provider="test",
            symbol="TEST",
            asset_class="equities",
            bar_type="time",
            bar_params={"frequency": "daily"},
        )

        # Write initial data
        storage.write(DataObject(data=initial_df, metadata=metadata))
        key = "equities/daily/TEST"

        # Add more January data
        new_dates = [
            datetime(2024, 1, 15),
            datetime(2024, 1, 20),
            datetime(2024, 1, 25),
        ]

        new_df = pl.DataFrame(
            {
                "timestamp": new_dates,
                "open": [115.0, 120.0, 125.0],
                "high": [120.0, 125.0, 130.0],
                "low": [110.0, 115.0, 120.0],
                "close": [117.0, 122.0, 127.0],
                "volume": [1030000, 1040000, 1050000],
            }
        )

        # Write additional data
        storage.write(DataObject(data=new_df, metadata=metadata))

        # Read all data
        result = storage.read(key)

        # Should have all 6 data points
        assert len(result.data) == 6
        assert result.data["close"].min() == 102.0
        assert result.data["close"].max() == 127.0

    def test_duplicate_handling(self, tmp_path: Path) -> None:
        """Test that duplicates are handled correctly."""
        storage = ChunkedStorage(base_path=tmp_path, chunk_size_days=30)

        dates = [
            datetime(2024, 1, 1),
            datetime(2024, 1, 2),
            datetime(2024, 1, 3),
        ]

        # Initial data
        df1 = pl.DataFrame(
            {
                "timestamp": dates,
                "open": [100.0, 101.0, 102.0],
                "high": [105.0, 106.0, 107.0],
                "low": [95.0, 96.0, 97.0],
                "close": [102.0, 103.0, 104.0],
                "volume": [1000000, 1010000, 1020000],
            }
        )

        metadata = Metadata(
            provider="test",
            symbol="TEST",
            asset_class="equities",
            bar_type="time",
            bar_params={"frequency": "daily"},
        )

        # Write initial data
        storage.write(DataObject(data=df1, metadata=metadata))
        key = "equities/daily/TEST"

        # Overlapping data with updates
        df2 = pl.DataFrame(
            {
                "timestamp": [*dates[1:], datetime(2024, 1, 4)],
                "open": [201.0, 202.0, 203.0],
                "high": [206.0, 207.0, 208.0],
                "low": [196.0, 197.0, 198.0],
                "close": [203.0, 204.0, 205.0],
                "volume": [2010000, 2020000, 2030000],
            }
        )

        # Write overlapping data
        storage.write(DataObject(data=df2, metadata=metadata))

        # Read all data
        result = storage.read(key)

        # Should have 4 unique timestamps
        assert len(result.data) == 4

        # Check that newer values were kept
        jan2_value = result.data.filter(pl.col("timestamp") == datetime(2024, 1, 2))["close"][0]
        assert jan2_value == 203.0  # Updated value

    def test_weekly_chunks(self, tmp_path: Path) -> None:
        """Test weekly chunk size."""
        storage = ChunkedStorage(base_path=tmp_path, chunk_size_days=7)

        # Create data spanning multiple weeks
        dates = []
        for day in range(1, 22):  # 3 weeks
            dates.append(datetime(2024, 1, day))

        df = pl.DataFrame(
            {
                "timestamp": dates,
                "open": [100.0 + i for i in range(len(dates))],
                "high": [105.0 + i for i in range(len(dates))],
                "low": [95.0 + i for i in range(len(dates))],
                "close": [102.0 + i for i in range(len(dates))],
                "volume": [1000000 + i * 10000 for i in range(len(dates))],
            }
        )

        metadata = Metadata(
            provider="test",
            symbol="TEST",
            asset_class="equities",
            bar_type="time",
            bar_params={"frequency": "daily"},
        )

        # Write data
        key = storage.write(DataObject(data=df, metadata=metadata))

        # Check chunks
        chunks = storage.get_chunk_info(key)
        assert len(chunks) >= 3  # At least 3 weekly chunks

        # Verify chunk IDs contain week numbers
        assert any("_W" in chunk.chunk_id for chunk in chunks)

    def test_quarterly_chunks(self, tmp_path: Path) -> None:
        """Test quarterly chunk size."""
        storage = ChunkedStorage(base_path=tmp_path, chunk_size_days=90)

        # Create data spanning multiple quarters
        dates = []
        for month in range(1, 10):  # Q1, Q2, Q3
            dates.append(datetime(2024, month, 15))

        df = pl.DataFrame(
            {
                "timestamp": dates,
                "open": [100.0 + i for i in range(len(dates))],
                "high": [105.0 + i for i in range(len(dates))],
                "low": [95.0 + i for i in range(len(dates))],
                "close": [102.0 + i for i in range(len(dates))],
                "volume": [1000000 + i * 10000 for i in range(len(dates))],
            }
        )

        metadata = Metadata(
            provider="test",
            symbol="TEST",
            asset_class="equities",
            bar_type="time",
            bar_params={"frequency": "daily"},
        )

        # Write data
        key = storage.write(DataObject(data=df, metadata=metadata))

        # Check chunks
        chunks = storage.get_chunk_info(key)
        assert len(chunks) == 3  # Q1, Q2, Q3

        # Verify chunk IDs contain quarter numbers
        assert any("_Q" in chunk.chunk_id for chunk in chunks)

    def test_yearly_chunks(self, tmp_path: Path) -> None:
        """Test yearly chunk size."""
        storage = ChunkedStorage(base_path=tmp_path, chunk_size_days=365)

        # Create data spanning multiple years
        dates = [
            datetime(2023, 6, 15),
            datetime(2024, 1, 15),
            datetime(2024, 6, 15),
            datetime(2025, 1, 15),
        ]

        df = pl.DataFrame(
            {
                "timestamp": dates,
                "open": [100.0, 200.0, 300.0, 400.0],
                "high": [105.0, 205.0, 305.0, 405.0],
                "low": [95.0, 195.0, 295.0, 395.0],
                "close": [102.0, 202.0, 302.0, 402.0],
                "volume": [1000000, 2000000, 3000000, 4000000],
            }
        )

        metadata = Metadata(
            provider="test",
            symbol="TEST",
            asset_class="equities",
            bar_type="time",
            bar_params={"frequency": "daily"},
        )

        # Write data
        key = storage.write(DataObject(data=df, metadata=metadata))

        # Check chunks
        chunks = storage.get_chunk_info(key)
        assert len(chunks) == 3  # 2023, 2024, 2025

        # Verify chunk IDs contain years
        chunk_ids = [chunk.chunk_id for chunk in chunks]
        assert any("2023" in cid for cid in chunk_ids)
        assert any("2024" in cid for cid in chunk_ids)
        assert any("2025" in cid for cid in chunk_ids)

    def test_empty_data_handling(self, tmp_path: Path) -> None:
        """Test handling of empty data."""
        storage = ChunkedStorage(base_path=tmp_path)

        # Empty DataFrame with proper schema
        df = pl.DataFrame(
            {
                "timestamp": pl.Series([], dtype=pl.Datetime),
                "open": pl.Series([], dtype=pl.Float64),
                "high": pl.Series([], dtype=pl.Float64),
                "low": pl.Series([], dtype=pl.Float64),
                "close": pl.Series([], dtype=pl.Float64),
                "volume": pl.Series([], dtype=pl.Int64),
            }
        )

        metadata = Metadata(
            provider="test",
            symbol="TEST",
            asset_class="equities",
            bar_type="time",
            bar_params={"frequency": "daily"},
        )

        # Write empty data
        key = storage.write(DataObject(data=df, metadata=metadata))

        # Should not create any chunks
        assert not storage.exists(key)

    def test_delete_chunks(self, tmp_path: Path) -> None:
        """Test deleting all chunks for a key."""
        storage = ChunkedStorage(base_path=tmp_path, chunk_size_days=30)

        # Create test data
        dates = []
        for month in range(1, 4):
            for day in [1, 15]:
                dates.append(datetime(2024, month, day))

        df = pl.DataFrame(
            {
                "timestamp": dates,
                "open": [100.0 + i for i in range(len(dates))],
                "high": [105.0 + i for i in range(len(dates))],
                "low": [95.0 + i for i in range(len(dates))],
                "close": [102.0 + i for i in range(len(dates))],
                "volume": [1000000 + i * 10000 for i in range(len(dates))],
            }
        )

        metadata = Metadata(
            provider="test",
            symbol="TEST",
            asset_class="equities",
            bar_type="time",
            bar_params={"frequency": "daily"},
        )

        # Write data
        key = storage.write(DataObject(data=df, metadata=metadata))
        assert storage.exists(key)

        # Delete data
        storage.delete(key)

        # Should no longer exist
        assert not storage.exists(key)

        # Should raise error when trying to read
        with pytest.raises(KeyError):
            storage.read(key)

    def test_list_keys(self, tmp_path: Path) -> None:
        """Test listing storage keys."""
        storage = ChunkedStorage(base_path=tmp_path)

        # Write multiple datasets
        metadata1 = Metadata(
            provider="test",
            symbol="AAPL",
            asset_class="equities",
            bar_type="time",
            bar_params={"frequency": "daily"},
        )

        metadata2 = Metadata(
            provider="test",
            symbol="GOOGL",
            asset_class="equities",
            bar_type="time",
            bar_params={"frequency": "daily"},
        )

        metadata3 = Metadata(
            provider="test",
            symbol="BTC",
            asset_class="crypto",
            bar_type="time",
            bar_params={"frequency": "hourly"},
        )

        df = pl.DataFrame(
            {
                "timestamp": [datetime(2024, 1, 1)],
                "open": [100.0],
                "high": [105.0],
                "low": [95.0],
                "close": [102.0],
                "volume": [1000000],
            }
        )

        storage.write(DataObject(data=df, metadata=metadata1))
        storage.write(DataObject(data=df, metadata=metadata2))
        storage.write(DataObject(data=df, metadata=metadata3))

        # List all keys
        keys = storage.list_keys()
        assert len(keys) == 3
        assert "equities/daily/AAPL" in keys
        assert "equities/daily/GOOGL" in keys
        assert "crypto/hourly/BTC" in keys

        # List with prefix
        equity_keys = storage.list_keys(prefix="equities/")
        assert len(equity_keys) == 2
        assert all(k.startswith("equities/") for k in equity_keys)

    def test_chunk_info_properties(self) -> None:
        """Test ChunkInfo class properties."""
        chunk = ChunkInfo(
            chunk_id="TEST_daily_2024_01",
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 31),
            row_count=100,
            file_path=Path("/tmp/test.parquet"),
            size_bytes=1024 * 1024,  # 1 MB
        )

        assert chunk.date_range_str == "2024-01-01 to 2024-01-31"
        assert chunk.row_count == 100
        assert chunk.size_bytes == 1024 * 1024
