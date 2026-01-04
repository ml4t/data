"""Tests for ML4T Data Hive and Flat storage backends."""

import shutil
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import polars as pl
import pytest

from ml4t.data.storage import (
    FlatStorage,
    HiveStorage,
    StorageBackend,
    StorageConfig,
    create_storage,
)


@pytest.fixture
def sample_data():
    """Create sample time-series data."""
    dates = [datetime(2023, 1, 1) + timedelta(days=i) for i in range(365)]
    return pl.DataFrame(
        {"timestamp": dates, "price": list(range(365)), "volume": list(range(365, 730))}
    )


@pytest.fixture
def temp_dir():
    """Create a temporary directory."""
    path = tempfile.mkdtemp(prefix="ml4t_data_test_")
    yield Path(path)
    shutil.rmtree(path, ignore_errors=True)


class TestStorageBackends:
    """Test storage backend implementations."""

    @pytest.mark.parametrize("strategy", ["hive", "flat"])
    def test_create_storage(self, temp_dir, strategy):
        """Test storage creation."""
        storage = create_storage(temp_dir, strategy=strategy)
        assert isinstance(storage, StorageBackend)
        if strategy == "hive":
            assert isinstance(storage, HiveStorage)
        else:
            assert isinstance(storage, FlatStorage)

    @pytest.mark.parametrize("strategy", ["hive", "flat"])
    def test_write_read_cycle(self, temp_dir, sample_data, strategy):
        """Test basic write and read operations."""
        storage = create_storage(temp_dir, strategy=strategy)

        # Write data
        path = storage.write(sample_data.lazy(), "test_key")
        assert path.exists()

        # Read data back
        df = storage.read("test_key").collect()
        assert len(df) == len(sample_data)
        assert set(df.columns) == set(sample_data.columns)

    @pytest.mark.parametrize("strategy", ["hive", "flat"])
    def test_date_filtering(self, temp_dir, sample_data, strategy):
        """Test reading with date filters."""
        storage = create_storage(temp_dir, strategy=strategy)
        storage.write(sample_data.lazy(), "test_key")

        # Read specific month
        start = datetime(2023, 6, 1)
        end = datetime(2023, 7, 1)
        df = storage.read("test_key", start_date=start, end_date=end).collect()

        # Should have June data only
        assert len(df) == 30
        assert df["timestamp"].min() >= start
        assert df["timestamp"].max() < end

    @pytest.mark.parametrize("strategy", ["hive", "flat"])
    def test_column_selection(self, temp_dir, sample_data, strategy):
        """Test reading specific columns."""
        storage = create_storage(temp_dir, strategy=strategy)
        storage.write(sample_data.lazy(), "test_key")

        # Read only price column
        df = storage.read("test_key", columns=["timestamp", "price"]).collect()
        assert set(df.columns) == {"timestamp", "price"}

    @pytest.mark.parametrize("strategy", ["hive", "flat"])
    def test_metadata_tracking(self, temp_dir, sample_data, strategy):
        """Test metadata storage and retrieval."""
        storage = create_storage(temp_dir, strategy=strategy)

        # Write with metadata
        custom_meta = {"source": "test", "version": 1}
        storage.write(sample_data.lazy(), "test_key", metadata=custom_meta)

        # Retrieve metadata
        metadata = storage.get_metadata("test_key")
        assert metadata is not None
        assert "last_updated" in metadata
        assert "row_count" in metadata
        assert metadata["row_count"] == len(sample_data)
        assert "custom" in metadata
        assert metadata["custom"]["source"] == "test"

    @pytest.mark.parametrize("strategy", ["hive", "flat"])
    def test_list_keys(self, temp_dir, sample_data, strategy):
        """Test listing stored keys."""
        storage = create_storage(temp_dir, strategy=strategy)

        # Initially empty
        assert storage.list_keys() == []

        # Add some keys
        storage.write(sample_data.lazy(), "key1")
        storage.write(sample_data.lazy(), "key2")

        keys = storage.list_keys()
        assert len(keys) == 2
        assert "key1" in keys
        assert "key2" in keys

    @pytest.mark.parametrize("strategy", ["hive", "flat"])
    def test_exists(self, temp_dir, sample_data, strategy):
        """Test key existence check."""
        storage = create_storage(temp_dir, strategy=strategy)

        assert not storage.exists("test_key")
        storage.write(sample_data.lazy(), "test_key")
        assert storage.exists("test_key")

    @pytest.mark.parametrize("strategy", ["hive", "flat"])
    def test_delete(self, temp_dir, sample_data, strategy):
        """Test data deletion."""
        storage = create_storage(temp_dir, strategy=strategy)

        storage.write(sample_data.lazy(), "test_key")
        assert storage.exists("test_key")

        success = storage.delete("test_key")
        assert success
        assert not storage.exists("test_key")

        # Deleting non-existent key returns False
        assert not storage.delete("non_existent")

    def test_hive_partitioning(self, temp_dir, sample_data):
        """Test Hive-specific partitioning structure."""
        storage = create_storage(temp_dir, strategy="hive")
        storage.write(sample_data.lazy(), "test_key")

        # Check partition structure
        key_path = temp_dir / "test_key"
        assert key_path.exists()

        # Should have year directories
        year_dirs = list(key_path.glob("year=*"))
        assert len(year_dirs) == 1
        assert year_dirs[0].name == "year=2023"

        # Should have month directories
        month_dirs = list(year_dirs[0].glob("month=*"))
        assert len(month_dirs) == 12  # All 12 months

    def test_atomic_writes(self, temp_dir, sample_data):
        """Test atomic write behavior."""
        storage = create_storage(temp_dir, strategy="flat")

        # Write should not leave partial files on failure
        # This is hard to test directly, but we can verify temp files are cleaned
        storage.write(sample_data.lazy(), "test_key")

        # No temp files should remain
        temp_files = list(temp_dir.glob("*.tmp"))
        assert len(temp_files) == 0

    def test_lazy_evaluation(self, temp_dir, sample_data):
        """Test that lazy evaluation is preserved."""
        storage = create_storage(temp_dir, strategy="hive")
        storage.write(sample_data.lazy(), "test_key")

        # Read should return LazyFrame
        lf = storage.read("test_key")
        assert isinstance(lf, pl.LazyFrame)

        # With filters
        lf_filtered = storage.read(
            "test_key", start_date=datetime(2023, 6, 1), columns=["timestamp", "price"]
        )
        assert isinstance(lf_filtered, pl.LazyFrame)


class TestStorageConfig:
    """Test storage configuration."""

    def test_default_config(self, temp_dir):
        """Test default configuration values."""
        config = StorageConfig(base_path=temp_dir)
        assert config.strategy == "hive"
        assert config.compression == "zstd"
        assert config.atomic_writes
        assert config.enable_locking
        assert config.metadata_tracking
        assert config.partition_cols == ["year", "month"]

    def test_flat_config(self, temp_dir):
        """Test flat storage configuration."""
        config = StorageConfig(base_path=temp_dir, strategy="flat")
        assert config.strategy == "flat"
        assert config.partition_cols == []  # No partitions for flat

    def test_custom_config(self, temp_dir):
        """Test custom configuration."""
        config = StorageConfig(
            base_path=temp_dir, compression="lz4", atomic_writes=False, enable_locking=False
        )
        assert config.compression == "lz4"
        assert not config.atomic_writes
        assert not config.enable_locking
