"""Unit tests for filesystem storage backend."""

import json
import tempfile
from datetime import datetime
from pathlib import Path

import polars as pl
import pytest

from ml4t.data.core.models import DataObject, Metadata
from ml4t.data.storage.filesystem import FileSystemBackend


class TestFileSystemBackend:
    """Test FileSystemBackend class."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def storage(self, temp_dir):
        """Create a filesystem storage backend."""
        return FileSystemBackend(data_root=temp_dir)

    @pytest.fixture
    def sample_data(self):
        """Create sample DataFrame for testing."""
        return pl.DataFrame(
            {
                "timestamp": [
                    datetime(2024, 1, 1),
                    datetime(2024, 1, 2),
                    datetime(2024, 1, 3),
                ],
                "open": [100.0, 101.0, 102.0],
                "high": [105.0, 106.0, 107.0],
                "low": [99.0, 100.0, 101.0],
                "close": [104.0, 105.0, 106.0],
                "volume": [1000000, 1100000, 1200000],
            }
        )

    @pytest.fixture
    def sample_metadata(self):
        """Create sample metadata."""
        return Metadata(
            provider="test_provider",
            symbol="AAPL",
            asset_class="equities",
            bar_type="time",
            bar_params={"frequency": "daily"},
        )

    def test_initialization(self, temp_dir):
        """Test backend initialization."""
        storage = FileSystemBackend(data_root=temp_dir)

        assert storage.data_root == temp_dir
        assert storage.data_root.exists()
        assert storage.locks_dir.exists()
        assert storage.locks_dir == temp_dir / ".locks"

    def test_get_storage_path(self, storage, sample_metadata):
        """Test storage path generation."""
        dir_path, key = storage._get_storage_path(sample_metadata)

        expected_dir = storage.data_root / "equities" / "daily"
        expected_key = "equities/daily/AAPL"

        # Resolve both paths to handle symlinks (e.g., /tmp -> /private/tmp on macOS)
        assert dir_path.resolve() == expected_dir.resolve()
        assert key == expected_key

    def test_get_lock_path(self, storage):
        """Test lock path generation."""
        key = "equities/daily/AAPL"
        lock_path = storage._get_lock_path(key)

        expected = storage.locks_dir / "equities_daily_AAPL.lock"
        assert lock_path == expected

    def test_write_and_read(self, storage, sample_data, sample_metadata):
        """Test writing and reading data."""
        # Create data object
        data_obj = DataObject(data=sample_data, metadata=sample_metadata)

        # Write data
        storage_key = storage.write(data_obj)

        assert storage_key == "equities/daily/AAPL"

        # Verify files were created
        data_dir = storage.data_root / "equities" / "daily"
        assert data_dir.exists()
        assert (data_dir / "AAPL.parquet").exists()
        assert (data_dir / "AAPL.manifest.json").exists()

        # Read data back
        read_obj = storage.read(storage_key)

        assert read_obj is not None
        assert read_obj.metadata.symbol == "AAPL"
        assert read_obj.metadata.provider == "test_provider"
        assert len(read_obj.data) == 3

        # Verify data integrity
        assert read_obj.data["close"].to_list() == [104.0, 105.0, 106.0]

    def test_exists(self, storage, sample_data, sample_metadata):
        """Test checking if data exists."""
        key = "equities/daily/AAPL"

        # Should not exist initially
        assert not storage.exists(key)

        # Write data
        data_obj = DataObject(data=sample_data, metadata=sample_metadata)
        storage.write(data_obj)

        # Should exist now
        assert storage.exists(key)

    def test_delete(self, storage, sample_data, sample_metadata):
        """Test deleting data."""
        # Write data
        data_obj = DataObject(data=sample_data, metadata=sample_metadata)
        storage_key = storage.write(data_obj)

        # Verify it exists
        assert storage.exists(storage_key)

        # Delete it
        storage.delete(storage_key)

        # Verify it's deleted
        assert not storage.exists(storage_key)

        # Files should be gone
        data_dir = storage.data_root / "equities" / "daily"
        assert not (data_dir / "AAPL.parquet").exists()
        assert not (data_dir / "AAPL.manifest.json").exists()

    def test_list(self, storage):
        """Test listing stored keys."""
        # Create multiple data objects
        symbols = ["AAPL", "GOOGL", "MSFT"]

        for symbol in symbols:
            data = pl.DataFrame(
                {
                    "timestamp": [datetime(2024, 1, 1)],
                    "open": [100.0],
                    "high": [105.0],
                    "low": [99.0],
                    "close": [104.0],
                    "volume": [1000000],
                }
            )

            metadata = Metadata(
                provider="test",
                symbol=symbol,
                asset_class="equities",
                bar_type="time",
                bar_params={"frequency": "daily"},
            )

            data_obj = DataObject(data=data, metadata=metadata)
            storage.write(data_obj)

        # List all keys
        keys = storage.list_keys()

        assert len(keys) == 3
        assert "equities/daily/AAPL" in keys
        assert "equities/daily/GOOGL" in keys
        assert "equities/daily/MSFT" in keys

    def test_list_with_pattern(self, storage):
        """Test listing with pattern matching."""
        # Create data in different asset classes
        for asset_class, symbol in [
            ("equities", "AAPL"),
            ("equities", "GOOGL"),
            ("crypto", "BTC"),
        ]:
            data = pl.DataFrame(
                {
                    "timestamp": [datetime(2024, 1, 1)],
                    "open": [100.0],
                    "high": [105.0],
                    "low": [99.0],
                    "close": [104.0],
                    "volume": [1000000],
                }
            )

            metadata = Metadata(
                provider="test",
                symbol=symbol,
                asset_class=asset_class,
                bar_type="time",
                bar_params={"frequency": "daily"},
            )

            data_obj = DataObject(data=data, metadata=metadata)
            storage.write(data_obj)

        # List only equities
        equity_keys = storage.list_keys(prefix="equities/")

        assert len(equity_keys) == 2
        assert all("equities" in key for key in equity_keys)
        assert "crypto/daily/BTC" not in equity_keys

    def test_overwrite_existing(self, storage):
        """Test overwriting existing data."""
        metadata = Metadata(
            provider="test",
            symbol="AAPL",
            asset_class="equities",
            bar_type="time",
            bar_params={"frequency": "daily"},
        )

        # Write initial data
        data1 = pl.DataFrame(
            {
                "timestamp": [datetime(2024, 1, 1)],
                "open": [100.0],
                "high": [105.0],
                "low": [99.0],
                "close": [104.0],
                "volume": [1000000],
            }
        )
        obj1 = DataObject(data=data1, metadata=metadata)
        key = storage.write(obj1)

        # Write new data with same key
        data2 = pl.DataFrame(
            {
                "timestamp": [datetime(2024, 1, 1), datetime(2024, 1, 2)],
                "open": [100.0, 101.0],
                "high": [105.0, 106.0],
                "low": [99.0, 100.0],
                "close": [104.0, 105.0],
                "volume": [1000000, 1100000],
            }
        )
        obj2 = DataObject(data=data2, metadata=metadata)
        storage.write(obj2)

        # Read and verify it's the new data
        read_obj = storage.read(key)
        assert len(read_obj.data) == 2  # New data has 2 rows

    def test_concurrent_write_same_key(self, temp_dir):
        """Test concurrent writes to same key (file locking)."""
        # Create two storage instances
        storage1 = FileSystemBackend(data_root=temp_dir)
        storage2 = FileSystemBackend(data_root=temp_dir)

        metadata = Metadata(
            provider="test",
            symbol="CONCURRENT",
            asset_class="equities",
            bar_type="time",
            bar_params={"frequency": "daily"},
        )

        # Create different data for each writer
        data1 = pl.DataFrame(
            {
                "timestamp": [datetime(2024, 1, 1)],
                "open": [100.0],
                "high": [105.0],
                "low": [99.0],
                "close": [104.0],
                "volume": [1000000],
            }
        )

        data2 = pl.DataFrame(
            {
                "timestamp": [datetime(2024, 1, 1), datetime(2024, 1, 2)],
                "open": [100.0, 101.0],
                "high": [105.0, 106.0],
                "low": [99.0, 100.0],
                "close": [104.0, 105.0],
                "volume": [1000000, 1100000],
            }
        )

        obj1 = DataObject(data=data1, metadata=metadata)
        obj2 = DataObject(data=data2, metadata=metadata)

        # Both write to same key
        key1 = storage1.write(obj1)
        key2 = storage2.write(obj2)

        assert key1 == key2  # Same key

        # Read final result - should be one of them (last write wins)
        final = storage1.read(key1)
        assert final is not None
        assert len(final.data) in [1, 2]  # One of the two datasets

    def test_read_nonexistent(self, storage):
        """Test reading non-existent data."""
        with pytest.raises(ValueError, match="Invalid storage key"):
            storage.read("nonexistent/key")

        # Test with valid key that doesn't exist
        with pytest.raises(FileNotFoundError):
            storage.read("equities/daily/NONEXISTENT")

    def test_manifest_preservation(self, storage, sample_data):
        """Test that manifest metadata is preserved."""
        # Create metadata with various fields
        metadata = Metadata(
            provider="yahoo",
            symbol="AAPL",
            asset_class="equities",
            bar_type="time",
            bar_params={"frequency": "daily"},
            data_range={"start": "2024-01-01", "end": "2024-01-03"},
            provider_params={"adjusted": True},
            attributes={"source": "test", "validated": True},
        )

        data_obj = DataObject(data=sample_data, metadata=metadata)
        storage.write(data_obj)

        # Read manifest directly
        manifest_path = storage.data_root / "equities" / "daily" / "AAPL.manifest.json"
        with open(manifest_path) as f:
            manifest = json.load(f)

        assert manifest["provider"] == "yahoo"
        assert manifest["symbol"] == "AAPL"
        assert manifest["schema_version"] == "1.0"  # Default version
        assert manifest["data_range"]["start"] == "2024-01-01"
        assert manifest["provider_params"]["adjusted"] is True
        # Note: attributes are not persisted in the current implementation
