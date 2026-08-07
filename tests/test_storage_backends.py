"""Tests for ML4T Data Hive and Flat storage backends."""

import shutil
import tempfile
from concurrent.futures import ThreadPoolExecutor
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
    def test_relative_base_path_is_resolved(self, tmp_path, monkeypatch, sample_data, strategy):
        """Relative storage roots support a complete write and read cycle."""
        monkeypatch.chdir(tmp_path)
        storage = create_storage("relative-data", strategy=strategy)

        storage.write(sample_data.lazy(), "test_key")

        assert storage.base_path == (tmp_path / "relative-data").resolve()
        assert len(storage.read("test_key").collect()) == len(sample_data)

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
    def test_list_keys_ignores_malformed_encoded_entry(self, temp_dir, sample_data, strategy):
        """One malformed physical entry does not hide valid logical keys."""
        storage = create_storage(temp_dir, strategy=strategy)
        storage.write(sample_data.lazy(), "valid_key")
        malformed = storage.base_path / "k1_invalid$"
        malformed.mkdir()
        (malformed / "CURRENT").write_text("not-a-commit\n", encoding="utf-8")

        assert storage.list_keys() == ["valid_key"]

    @pytest.mark.parametrize("strategy", ["hive", "flat"])
    def test_logical_keys_do_not_alias(self, temp_dir, strategy):
        """Separators and underscores retain distinct physical identities."""
        storage = create_storage(temp_dir, strategy=strategy)
        first = pl.DataFrame({"timestamp": [datetime(2024, 1, 1)], "close": [1.0]})
        second = pl.DataFrame({"timestamp": [datetime(2024, 1, 1)], "close": [2.0]})

        first_path = storage.write(first, "a/b_c")
        second_path = storage.write(second, "a_b/c")

        assert first_path != second_path
        assert storage.read("a/b_c").collect()["close"].item() == 1.0
        assert storage.read("a_b/c").collect()["close"].item() == 2.0
        assert storage.list_keys() == ["a/b_c", "a_b/c"]

    @pytest.mark.parametrize("strategy", ["hive", "flat"])
    @pytest.mark.parametrize("key", ["../../../escaped", "a/../escaped", "a\\..\\escaped"])
    def test_storage_keys_cannot_escape_base_path(self, temp_dir, sample_data, strategy, key):
        storage = create_storage(temp_dir, strategy=strategy)

        with pytest.raises(ValueError):
            storage.write(sample_data, key)

        assert not (temp_dir.parent / "escaped").exists()

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
        key_path = storage.write(sample_data.lazy(), "test_key")

        # Check partition structure
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

    def test_hive_repeated_write_is_full_replacement(self, temp_dir):
        storage = create_storage(temp_dir, strategy="hive")
        original = pl.DataFrame(
            {
                "timestamp": [datetime(2024, 1, 15), datetime(2024, 2, 15)],
                "close": [1.0, 2.0],
            }
        )
        replacement = pl.DataFrame({"timestamp": [datetime(2024, 1, 20)], "close": [3.0]})
        storage.write(original, "prices")

        storage.write(replacement, "prices")

        assert storage.read("prices").collect().equals(replacement)
        metadata = storage.get_metadata("prices")
        assert metadata is not None
        assert metadata["row_count"] == 1
        assert metadata["partitions"] == ["year=2024/month=1"]

    def test_hive_partition_failure_preserves_previous_commit(self, temp_dir, monkeypatch):
        storage = create_storage(temp_dir, strategy="hive")
        original = pl.DataFrame(
            {
                "timestamp": [datetime(2024, 1, 15), datetime(2024, 2, 15)],
                "close": [1.0, 2.0],
            }
        )
        replacement = pl.DataFrame(
            {
                "timestamp": [datetime(2024, 3, 15), datetime(2024, 4, 15)],
                "close": [3.0, 4.0],
            }
        )
        storage.write(original, "prices")
        previous_commit = storage._current_commit("prices")
        original_atomic_write = storage._atomic_write
        call_count = 0

        def fail_second_partition(df, path):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise OSError("injected partition failure")
            original_atomic_write(df, path)

        monkeypatch.setattr(storage, "_atomic_write", fail_second_partition)

        with pytest.raises(OSError, match="injected partition failure"):
            storage.write(replacement, "prices")

        assert storage.read("prices").collect().equals(original)
        assert storage._current_commit("prices").commit_id == previous_commit.commit_id
        assert storage.get_metadata("prices") == previous_commit.metadata
        assert not list(storage._key_path("prices").glob(".staging-*"))
        restarted = create_storage(temp_dir, strategy="hive")
        assert restarted.read("prices").collect().equals(original)

    def test_pointer_failure_preserves_previous_commit(self, temp_dir, monkeypatch):
        storage = create_storage(temp_dir, strategy="flat")
        original = pl.DataFrame({"timestamp": [datetime(2024, 1, 1)], "value": [1]})
        replacement = pl.DataFrame({"timestamp": [datetime(2024, 1, 2)], "value": [2]})
        storage.write(original, "prices")
        previous_commit = storage._current_commit("prices")

        def fail_pointer(path, content):
            raise OSError("injected pointer failure")

        monkeypatch.setattr(storage, "_atomic_write_text", fail_pointer)

        with pytest.raises(OSError, match="injected pointer failure"):
            storage.write(replacement, "prices")

        assert storage.read("prices").collect().equals(original)
        assert storage._current_commit("prices").commit_id == previous_commit.commit_id
        restarted = create_storage(temp_dir, strategy="flat")
        assert restarted.read("prices").collect().equals(original)

    def test_next_write_removes_unpublished_staging_directory(self, temp_dir):
        storage = create_storage(temp_dir, strategy="flat")
        storage.write(pl.DataFrame({"value": [1]}), "prices")
        staging = storage._key_path("prices") / ".staging-interrupted"
        staging.mkdir()
        (staging / "partial.parquet").write_bytes(b"partial")

        restarted = create_storage(temp_dir, strategy="flat")
        assert staging.exists()
        restarted.write(pl.DataFrame({"value": [2]}), "prices")

        assert not staging.exists()

    def test_constructor_does_not_delete_another_writer_staging(self, temp_dir):
        storage = create_storage(temp_dir, strategy="flat")
        key_path = storage._key_path("prices")
        key_path.mkdir()
        staging = key_path / ".staging-active"
        staging.mkdir()
        (staging / "partial.parquet").write_bytes(b"partial")

        create_storage(temp_dir, strategy="flat")

        assert staging.is_dir()

    def test_corrupt_current_commit_falls_back_to_prior_valid_generation(self, temp_dir):
        storage = create_storage(temp_dir, strategy="flat")
        original = pl.DataFrame({"value": [1]})
        replacement = pl.DataFrame({"value": [2]})
        storage.write(original, "prices")
        storage.write(replacement, "prices")
        current = storage._current_commit("prices")
        commit_path = storage._key_path("prices") / "commits" / f"{current.commit_id}.json"
        commit_path.write_text("{invalid", encoding="utf-8")

        assert storage.read("prices").collect().equals(original)

    def test_generation_history_is_bounded(self, temp_dir):
        storage = create_storage(temp_dir, strategy="flat")
        for value in range(10):
            storage.write(pl.DataFrame({"value": [value]}), "prices")

        key_path = storage._key_path("prices")
        assert len(list((key_path / "commits").glob("*.json"))) == storage.GENERATION_RETENTION
        assert len(list((key_path / "generations").iterdir())) == storage.GENERATION_RETENTION
        assert storage.read("prices").collect()["value"].item() == 9

    def test_concurrent_flat_writes_publish_matching_data_and_metadata(self, temp_dir):
        storage = create_storage(temp_dir, strategy="flat")

        def write_generation(writer: int) -> None:
            data = pl.DataFrame(
                {
                    "timestamp": [datetime(2024, 1, 1)] * writer,
                    "writer": [writer] * writer,
                }
            )
            storage.write(data, "prices", metadata={"writer": writer})

        with ThreadPoolExecutor(max_workers=8) as executor:
            list(executor.map(write_generation, range(1, 25)))

        data = storage.read("prices").collect()
        metadata = storage.get_metadata("prices")
        assert metadata is not None
        assert data.height == metadata["row_count"]
        assert data["writer"].unique().to_list() == [metadata["custom"]["writer"]]

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
        assert config.lock_timeout == 30
        assert config.metadata_tracking
        assert config.partition_cols == ["year", "month"]

    def test_flat_config(self, temp_dir):
        """Test flat storage configuration."""
        config = StorageConfig(base_path=temp_dir, strategy="flat")
        assert config.strategy == "flat"
        assert config.partition_cols == []  # No partitions for flat

    def test_custom_config(self, temp_dir):
        """Test custom configuration."""
        config = StorageConfig(base_path=temp_dir, compression="lz4", lock_timeout=12)
        assert config.compression == "lz4"
        assert config.lock_timeout == 12
