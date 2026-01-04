"""Tests for storage backends."""

import json
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import polars as pl
import pytest

from ml4t.data.core.models import DataObject, Metadata, SchemaVersion
from ml4t.data.storage.base import StorageBackend
from ml4t.data.storage.filesystem import FileSystemBackend


class TestStorageAbstraction:
    """Test storage abstraction."""

    def test_storage_interface(self) -> None:
        """Test that StorageBackend ABC cannot be instantiated."""
        with pytest.raises(TypeError):
            StorageBackend()  # type: ignore


class TestFileSystemBackend:
    """Test filesystem storage backend."""

    @pytest.fixture
    def temp_dir(self) -> Path:
        """Create a temporary directory for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def fs_backend(self, temp_dir: Path) -> FileSystemBackend:
        """Create a filesystem backend instance."""
        return FileSystemBackend(data_root=temp_dir)

    @pytest.fixture
    def sample_data(self) -> DataObject:
        """Create sample data for testing."""
        df = pl.DataFrame(
            {
                "timestamp": [
                    datetime(2024, 1, 1, 9, 30, tzinfo=UTC),
                    datetime(2024, 1, 2, 9, 30, tzinfo=UTC),
                ],
                "open": [100.0, 101.0],
                "high": [102.0, 103.0],
                "low": [99.0, 100.0],
                "close": [101.5, 102.5],
                "volume": [1000000.0, 1100000.0],
                "dividends": [0.0, 0.0],
                "splits": [1.0, 1.0],
            }
        )

        metadata = Metadata(
            provider="test",
            symbol="AAPL",
            asset_class="equities",
            bar_type="time",
            bar_params={"frequency": "daily"},
            schema_version=SchemaVersion.V1_0,
            data_range={"start": "2024-01-01T09:30:00Z", "end": "2024-01-02T09:30:00Z"},
        )

        return DataObject(data=df, metadata=metadata)

    def test_write_and_read(self, fs_backend: FileSystemBackend, sample_data: DataObject) -> None:
        """Test writing and reading data."""
        # Write data
        key = fs_backend.write(sample_data)
        assert key == "equities/daily/AAPL"

        # Check files exist
        parquet_path = fs_backend.data_root / "equities" / "daily" / "AAPL.parquet"
        manifest_path = fs_backend.data_root / "equities" / "daily" / "AAPL.manifest.json"
        assert parquet_path.exists()
        assert manifest_path.exists()

        # Read data back
        loaded_data = fs_backend.read(key)
        assert loaded_data.data.shape == sample_data.data.shape
        assert loaded_data.metadata.symbol == "AAPL"

        # Compare DataFrames
        assert loaded_data.data.equals(sample_data.data)

    def test_manifest_content(self, fs_backend: FileSystemBackend, sample_data: DataObject) -> None:
        """Test manifest file content."""
        _ = fs_backend.write(sample_data)

        manifest_path = fs_backend.data_root / "equities" / "daily" / "AAPL.manifest.json"
        with open(manifest_path) as f:
            manifest = json.load(f)

        assert manifest["provider"] == "test"
        assert manifest["symbol"] == "AAPL"
        assert manifest["asset_class"] == "equities"
        assert manifest["bar_type"] == "time"
        assert manifest["bar_params"]["frequency"] == "daily"
        assert manifest["schema_version"] == "1.0"
        assert "download_utc_timestamp" in manifest
        assert manifest["data_range"]["start"] == "2024-01-01T09:30:00Z"
        assert manifest["data_range"]["end"] == "2024-01-02T09:30:00Z"

    def test_exists(self, fs_backend: FileSystemBackend, sample_data: DataObject) -> None:
        """Test checking if data exists."""
        key = "equities/daily/AAPL"

        # Should not exist initially
        assert not fs_backend.exists(key)

        # Write data
        fs_backend.write(sample_data)

        # Should exist now
        assert fs_backend.exists(key)

    def test_delete(self, fs_backend: FileSystemBackend, sample_data: DataObject) -> None:
        """Test deleting data."""
        key = fs_backend.write(sample_data)
        assert fs_backend.exists(key)

        # Delete
        fs_backend.delete(key)
        assert not fs_backend.exists(key)

        # Files should be gone
        parquet_path = fs_backend.data_root / "equities" / "daily" / "AAPL.parquet"
        manifest_path = fs_backend.data_root / "equities" / "daily" / "AAPL.manifest.json"
        assert not parquet_path.exists()
        assert not manifest_path.exists()

    def test_list_keys(self, fs_backend: FileSystemBackend, sample_data: DataObject) -> None:
        """Test listing stored keys."""
        # Initially empty
        assert fs_backend.list_keys() == []

        # Write data
        fs_backend.write(sample_data)

        # Create another data object
        sample_data.metadata.symbol = "MSFT"
        fs_backend.write(sample_data)

        # List keys
        keys = fs_backend.list_keys()
        assert len(keys) == 2
        assert "equities/daily/AAPL" in keys
        assert "equities/daily/MSFT" in keys

    def test_read_nonexistent(self, fs_backend: FileSystemBackend) -> None:
        """Test reading non-existent data raises error."""
        with pytest.raises((FileNotFoundError, ValueError)):
            fs_backend.read("nonexistent/key")

    def test_directory_structure(
        self, fs_backend: FileSystemBackend, sample_data: DataObject
    ) -> None:
        """Test that proper directory structure is created."""
        fs_backend.write(sample_data)

        # Check directory structure
        assert (fs_backend.data_root / "equities").is_dir()
        assert (fs_backend.data_root / "equities" / "daily").is_dir()

        # Create minute data
        sample_data.metadata.bar_params = {"frequency": "minute"}
        fs_backend.write(sample_data)

        assert (fs_backend.data_root / "equities" / "minute").is_dir()
