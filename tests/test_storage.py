"""Tests for storage backend ABC and implementations."""

from pathlib import Path

import polars as pl
import pytest

from ml4t.data.storage.backend import StorageBackend, StorageConfig
from ml4t.data.storage.hive import HiveStorage


class TestStorageAbstraction:
    """Test storage abstraction."""

    def test_storage_interface(self) -> None:
        """Test that StorageBackend ABC cannot be instantiated."""
        with pytest.raises(TypeError):
            StorageBackend(StorageConfig(base_path=Path("/tmp/test")))  # type: ignore


class TestHiveStorageRoundTrip:
    """Test HiveStorage write/read/delete/list round-trip."""

    @pytest.fixture
    def storage(self, tmp_path: Path) -> HiveStorage:
        config = StorageConfig(base_path=tmp_path, strategy="hive")
        return HiveStorage(config)

    @pytest.fixture
    def sample_df(self) -> pl.LazyFrame:
        return pl.DataFrame(
            {
                "timestamp": pl.Series(
                    [
                        "2024-01-01T09:30:00",
                        "2024-01-02T09:30:00",
                    ]
                ).str.to_datetime(),
                "open": [100.0, 101.0],
                "high": [102.0, 103.0],
                "low": [99.0, 100.0],
                "close": [101.5, 102.5],
                "volume": [1_000_000.0, 1_100_000.0],
            }
        ).lazy()

    def test_write_and_read(self, storage: HiveStorage, sample_df: pl.LazyFrame) -> None:
        path = storage.write(sample_df, "AAPL")
        assert path.exists()

        result = storage.read("AAPL").collect()
        assert result.shape[0] == 2
        assert "close" in result.columns

    def test_exists(self, storage: HiveStorage, sample_df: pl.LazyFrame) -> None:
        assert not storage.exists("AAPL")
        storage.write(sample_df, "AAPL")
        assert storage.exists("AAPL")

    def test_delete(self, storage: HiveStorage, sample_df: pl.LazyFrame) -> None:
        storage.write(sample_df, "AAPL")
        assert storage.exists("AAPL")
        storage.delete("AAPL")
        assert not storage.exists("AAPL")

    def test_list_keys(self, storage: HiveStorage, sample_df: pl.LazyFrame) -> None:
        assert storage.list_keys() == []
        storage.write(sample_df, "AAPL")
        storage.write(sample_df, "MSFT")
        keys = storage.list_keys()
        assert len(keys) == 2
        assert "AAPL" in keys
        assert "MSFT" in keys

    def test_get_metadata(self, storage: HiveStorage, sample_df: pl.LazyFrame) -> None:
        storage.write(sample_df, "AAPL", metadata={"provider": "yahoo"})
        meta = storage.get_metadata("AAPL")
        assert meta is not None
