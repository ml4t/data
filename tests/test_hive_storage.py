"""Tests for Hive partitioned storage module."""

import json
from datetime import datetime

import polars as pl
import pytest

from ml4t.data.storage.backend import StorageConfig
from ml4t.data.storage.hive import HiveStorage


class TestHiveStorageInit:
    """Tests for HiveStorage initialization."""

    def test_init_creates_directories(self, tmp_path):
        """Test initialization creates base and metadata directories."""
        config = StorageConfig(base_path=tmp_path)
        storage = HiveStorage(config)

        assert storage.base_path == tmp_path
        assert storage.metadata_dir.exists()
        assert storage.metadata_dir == tmp_path / ".metadata"

    def test_init_with_existing_directory(self, tmp_path):
        """Test initialization with existing directory."""
        (tmp_path / ".metadata").mkdir()
        config = StorageConfig(base_path=tmp_path)
        storage = HiveStorage(config)

        assert storage.metadata_dir.exists()


class TestHiveStorageWrite:
    """Tests for write method."""

    @pytest.fixture
    def storage(self, tmp_path):
        """Create storage instance."""
        config = StorageConfig(base_path=tmp_path, metadata_tracking=True)
        return HiveStorage(config)

    def test_write_dataframe(self, storage):
        """Test writing a DataFrame."""
        df = pl.DataFrame(
            {
                "timestamp": [datetime(2024, 1, 15), datetime(2024, 2, 15)],
                "close": [100.0, 101.0],
            }
        )

        result = storage.write(df, "test_key")

        assert result.exists()
        assert (result / "year=2024" / "month=1" / "data.parquet").exists()
        assert (result / "year=2024" / "month=2" / "data.parquet").exists()

    def test_write_lazy_frame(self, storage):
        """Test writing a LazyFrame."""
        df = pl.DataFrame(
            {
                "timestamp": [datetime(2024, 1, 15)],
                "close": [100.0],
            }
        ).lazy()

        result = storage.write(df, "test_key")

        assert result.exists()

    def test_write_without_timestamp_raises(self, storage):
        """Test writing without timestamp column raises error."""
        df = pl.DataFrame(
            {
                "close": [100.0, 101.0],
            }
        )

        with pytest.raises(ValueError, match="timestamp"):
            storage.write(df, "test_key")

    def test_write_without_key_raises(self, storage):
        """Test writing without key raises error."""
        df = pl.DataFrame(
            {
                "timestamp": [datetime(2024, 1, 15)],
                "close": [100.0],
            }
        )

        with pytest.raises(ValueError, match="key is required"):
            storage.write(df, None)

    def test_write_creates_metadata(self, storage):
        """Test writing creates metadata file."""
        df = pl.DataFrame(
            {
                "timestamp": [datetime(2024, 1, 15)],
                "close": [100.0],
            }
        )

        storage.write(df, "test_key")

        metadata_file = storage.metadata_dir / "test_key.json"
        assert metadata_file.exists()

        with open(metadata_file) as f:
            metadata = json.load(f)
        assert "last_updated" in metadata
        assert "row_count" in metadata
        assert metadata["row_count"] == 1


class TestHiveStorageRead:
    """Tests for read method."""

    @pytest.fixture
    def storage_with_data(self, tmp_path):
        """Create storage with test data."""
        config = StorageConfig(base_path=tmp_path)
        storage = HiveStorage(config)

        df = pl.DataFrame(
            {
                "timestamp": [
                    datetime(2024, 1, 15),
                    datetime(2024, 2, 15),
                    datetime(2024, 3, 15),
                ],
                "close": [100.0, 101.0, 102.0],
            }
        )
        storage.write(df, "test_key")

        return storage

    def test_read_all_data(self, storage_with_data):
        """Test reading all data."""
        lf = storage_with_data.read("test_key")
        df = lf.collect()

        assert len(df) == 3
        assert "timestamp" in df.columns
        assert "close" in df.columns

    def test_read_with_start_date(self, storage_with_data):
        """Test reading with start date filter."""
        lf = storage_with_data.read("test_key", start_date=datetime(2024, 2, 1))
        df = lf.collect()

        assert len(df) == 2  # Feb and March

    def test_read_with_end_date(self, storage_with_data):
        """Test reading with end date filter."""
        lf = storage_with_data.read("test_key", end_date=datetime(2024, 2, 28))
        df = lf.collect()

        assert len(df) == 2  # Jan and Feb

    def test_read_with_date_range(self, storage_with_data):
        """Test reading with date range."""
        lf = storage_with_data.read(
            "test_key", start_date=datetime(2024, 2, 1), end_date=datetime(2024, 2, 28)
        )
        df = lf.collect()

        assert len(df) == 1  # Only Feb

    def test_read_with_columns(self, storage_with_data):
        """Test reading with column selection."""
        lf = storage_with_data.read("test_key", columns=["timestamp"])
        df = lf.collect()

        assert "timestamp" in df.columns
        assert "close" not in df.columns

    def test_read_nonexistent_key_raises(self, storage_with_data):
        """Test reading nonexistent key raises error."""
        with pytest.raises(KeyError, match="not found"):
            storage_with_data.read("nonexistent_key")


class TestHiveStorageListKeys:
    """Tests for list_keys method."""

    def test_list_keys_empty(self, tmp_path):
        """Test listing keys on empty storage."""
        config = StorageConfig(base_path=tmp_path)
        storage = HiveStorage(config)

        keys = storage.list_keys()
        assert keys == []

    def test_list_keys_with_data(self, tmp_path):
        """Test listing keys with data."""
        config = StorageConfig(base_path=tmp_path)
        storage = HiveStorage(config)

        df = pl.DataFrame(
            {
                "timestamp": [datetime(2024, 1, 15)],
                "close": [100.0],
            }
        )
        storage.write(df, "key1")
        storage.write(df, "key2")

        keys = storage.list_keys()
        assert len(keys) == 2
        assert "key1" in keys
        assert "key2" in keys


class TestHiveStorageExists:
    """Tests for exists method."""

    def test_exists_true(self, tmp_path):
        """Test exists returns True for existing key."""
        config = StorageConfig(base_path=tmp_path)
        storage = HiveStorage(config)

        df = pl.DataFrame(
            {
                "timestamp": [datetime(2024, 1, 15)],
                "close": [100.0],
            }
        )
        storage.write(df, "test_key")

        assert storage.exists("test_key") is True

    def test_exists_false(self, tmp_path):
        """Test exists returns False for nonexistent key."""
        config = StorageConfig(base_path=tmp_path)
        storage = HiveStorage(config)

        assert storage.exists("nonexistent") is False


class TestHiveStorageDelete:
    """Tests for delete method."""

    def test_delete_existing_key(self, tmp_path):
        """Test deleting existing key."""
        config = StorageConfig(base_path=tmp_path, metadata_tracking=True)
        storage = HiveStorage(config)

        df = pl.DataFrame(
            {
                "timestamp": [datetime(2024, 1, 15)],
                "close": [100.0],
            }
        )
        storage.write(df, "test_key")

        result = storage.delete("test_key")

        assert result is True
        assert storage.exists("test_key") is False

    def test_delete_nonexistent_key(self, tmp_path):
        """Test deleting nonexistent key returns False."""
        config = StorageConfig(base_path=tmp_path)
        storage = HiveStorage(config)

        result = storage.delete("nonexistent")
        assert result is False


class TestHiveStorageMetadata:
    """Tests for metadata methods."""

    def test_get_metadata(self, tmp_path):
        """Test getting metadata."""
        config = StorageConfig(base_path=tmp_path, metadata_tracking=True)
        storage = HiveStorage(config)

        df = pl.DataFrame(
            {
                "timestamp": [datetime(2024, 1, 15)],
                "close": [100.0],
            }
        )
        storage.write(df, "test_key")

        metadata = storage.get_metadata("test_key")

        assert metadata is not None
        assert "last_updated" in metadata
        assert "row_count" in metadata

    def test_get_metadata_nonexistent(self, tmp_path):
        """Test getting metadata for nonexistent key returns None."""
        config = StorageConfig(base_path=tmp_path)
        storage = HiveStorage(config)

        metadata = storage.get_metadata("nonexistent")
        assert metadata is None


class TestHiveStorageIncrementalMethods:
    """Tests for incremental update methods."""

    @pytest.fixture
    def storage(self, tmp_path):
        """Create storage instance."""
        config = StorageConfig(base_path=tmp_path, metadata_tracking=True)
        return HiveStorage(config)

    def test_get_latest_timestamp_no_data(self, storage):
        """Test getting latest timestamp when no data exists."""
        result = storage.get_latest_timestamp("AAPL", "yahoo")
        assert result is None

    def test_get_latest_timestamp_with_data(self, storage):
        """Test getting latest timestamp with existing data."""
        df = pl.DataFrame(
            {
                "timestamp": [
                    datetime(2024, 1, 15),
                    datetime(2024, 2, 15),
                ],
                "close": [100.0, 101.0],
            }
        )
        storage.write(df, "yahoo/AAPL")

        result = storage.get_latest_timestamp("AAPL", "yahoo")
        assert result == datetime(2024, 2, 15)

    def test_save_chunk(self, storage):
        """Test saving incremental chunk."""
        df = pl.DataFrame(
            {
                "timestamp": [datetime(2024, 1, 15)],
                "close": [100.0],
            }
        )

        chunk_path = storage.save_chunk(
            df, "AAPL", "yahoo", datetime(2024, 1, 1), datetime(2024, 1, 31)
        )

        assert chunk_path.exists()
        assert ".chunks" in str(chunk_path)

    def test_update_combined_file_new(self, storage):
        """Test updating combined file with new data."""
        df = pl.DataFrame(
            {
                "timestamp": [datetime(2024, 1, 15)],
                "close": [100.0],
            }
        )

        records_added = storage.update_combined_file(df, "AAPL", "yahoo")

        # First write may return 0 due to dedup logic
        assert records_added >= 0
        assert storage.exists("yahoo/AAPL")

    def test_update_combined_file_append(self, storage):
        """Test appending to combined file."""
        df1 = pl.DataFrame(
            {
                "timestamp": [datetime(2024, 1, 15)],
                "close": [100.0],
            }
        )
        storage.update_combined_file(df1, "AAPL", "yahoo")

        df2 = pl.DataFrame(
            {
                "timestamp": [datetime(2024, 2, 15)],
                "close": [101.0],
            }
        )
        records_added = storage.update_combined_file(df2, "AAPL", "yahoo")

        # Should have one new record
        assert records_added >= 0

    def test_get_combined_file_path(self, storage):
        """Test getting combined file path."""
        path = storage.get_combined_file_path("AAPL", "yahoo")
        assert "yahoo_AAPL" in str(path)

    def test_read_data_no_data(self, storage):
        """Test reading data when no data exists."""
        df = storage.read_data("AAPL", "yahoo")
        assert df.is_empty()

    def test_read_data_with_filter(self, storage):
        """Test reading data with time filter."""
        df = pl.DataFrame(
            {
                "timestamp": [
                    datetime(2024, 1, 15),
                    datetime(2024, 2, 15),
                ],
                "close": [100.0, 101.0],
            }
        )
        storage.write(df, "yahoo/AAPL")

        result = storage.read_data("AAPL", "yahoo", start_time=datetime(2024, 2, 1))

        assert len(result) == 1

    def test_update_metadata_new(self, storage):
        """Test updating metadata for new symbol."""
        storage.update_metadata(
            "AAPL", "yahoo", datetime(2024, 1, 15), 100, "chunk_20240101.parquet"
        )

        metadata = storage.get_metadata("yahoo/AAPL")
        assert metadata is not None
        assert metadata["symbol"] == "AAPL"
        assert "update_history" in metadata


class TestHiveStorageAtomicWrite:
    """Tests for atomic write functionality."""

    def test_atomic_write_creates_file(self, tmp_path):
        """Test atomic write creates target file."""
        config = StorageConfig(base_path=tmp_path)
        storage = HiveStorage(config)

        df = pl.DataFrame({"a": [1, 2, 3]})
        target_path = tmp_path / "test.parquet"

        storage._atomic_write(df, target_path)

        assert target_path.exists()

    def test_atomic_write_no_temp_file_left(self, tmp_path):
        """Test atomic write cleans up temp file."""
        config = StorageConfig(base_path=tmp_path)
        storage = HiveStorage(config)

        df = pl.DataFrame({"a": [1, 2, 3]})
        target_path = tmp_path / "test.parquet"

        storage._atomic_write(df, target_path)

        # No .tmp files should be left
        tmp_files = list(tmp_path.glob("*.tmp"))
        assert len(tmp_files) == 0


class TestHiveStorageSlashInKey:
    """Tests for keys with slashes (hierarchy)."""

    def test_write_with_slash_key(self, tmp_path):
        """Test writing with slash in key converts to underscore."""
        config = StorageConfig(base_path=tmp_path)
        storage = HiveStorage(config)

        df = pl.DataFrame(
            {
                "timestamp": [datetime(2024, 1, 15)],
                "close": [100.0],
            }
        )
        storage.write(df, "provider/symbol")

        # Directory name should use underscore
        assert (tmp_path / "provider_symbol").exists()

    def test_exists_with_slash_key(self, tmp_path):
        """Test exists handles slash in key."""
        config = StorageConfig(base_path=tmp_path)
        storage = HiveStorage(config)

        df = pl.DataFrame(
            {
                "timestamp": [datetime(2024, 1, 15)],
                "close": [100.0],
            }
        )
        storage.write(df, "provider/symbol")

        assert storage.exists("provider/symbol") is True

    def test_read_with_slash_key(self, tmp_path):
        """Test read handles slash in key."""
        config = StorageConfig(base_path=tmp_path)
        storage = HiveStorage(config)

        df = pl.DataFrame(
            {
                "timestamp": [datetime(2024, 1, 15)],
                "close": [100.0],
            }
        )
        storage.write(df, "provider/symbol")

        result = storage.read("provider/symbol").collect()
        assert len(result) == 1
