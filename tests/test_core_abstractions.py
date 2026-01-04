"""Tests for core abstractions."""

from datetime import UTC, datetime
from pathlib import Path

import polars as pl
import pytest
from pydantic import ValidationError

from ml4t.data.core.config import Config, StorageConfig
from ml4t.data.core.models import DataObject, Metadata, SchemaVersion
from ml4t.data.providers.base import Provider


class TestConfig:
    """Test configuration management."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = Config()
        assert config.data_root == Path.home() / ".qldm" / "data"
        assert config.storage.backend == "filesystem"
        assert config.log_level == "INFO"

    def test_config_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test configuration from environment variables."""
        monkeypatch.setenv("QLDM_DATA_ROOT", "/custom/path")
        monkeypatch.setenv("QLDM_LOG_LEVEL", "DEBUG")

        config = Config()
        assert config.data_root == Path("/custom/path")
        assert config.log_level == "DEBUG"

    def test_storage_config_validation(self) -> None:
        """Test storage configuration validation."""
        config = StorageConfig(backend="filesystem", compression="snappy")
        assert config.compression == "snappy"

        with pytest.raises(ValidationError):
            StorageConfig(backend="invalid", compression="snappy")


class TestDataModels:
    """Test data models."""

    def test_metadata_creation(self) -> None:
        """Test metadata model creation."""
        metadata = Metadata(
            provider="test",
            symbol="AAPL",
            asset_class="equities",
            frequency="daily",
            schema_version=SchemaVersion.V1_0,
        )
        assert metadata.provider == "test"
        assert metadata.symbol == "AAPL"
        assert metadata.download_utc_timestamp is not None

    def test_data_object_with_dataframe(self) -> None:
        """Test DataObject with Polars DataFrame."""
        df = pl.DataFrame(
            {
                "timestamp": [datetime.now(UTC)],
                "open": [100.0],
                "high": [101.0],
                "low": [99.0],
                "close": [100.5],
                "volume": [1000000.0],
            }
        )

        metadata = Metadata(
            provider="test",
            symbol="AAPL",
            asset_class="equities",
            frequency="daily",
            schema_version=SchemaVersion.V1_0,
        )

        data_obj = DataObject(data=df, metadata=metadata)
        assert data_obj.data.shape == (1, 6)
        assert data_obj.metadata.symbol == "AAPL"

    def test_data_object_validation(self) -> None:
        """Test DataObject validation."""
        # Invalid DataFrame schema
        df = pl.DataFrame({"wrong_column": [1, 2, 3]})

        metadata = Metadata(
            provider="test",
            symbol="AAPL",
            asset_class="equities",
            frequency="daily",
            schema_version=SchemaVersion.V1_0,
        )

        with pytest.raises(ValidationError):
            DataObject(data=df, metadata=metadata)


class TestProviderAbstraction:
    """Test provider abstraction."""

    def test_provider_interface(self) -> None:
        """Test that Provider ABC cannot be instantiated."""
        with pytest.raises(TypeError):
            Provider()  # type: ignore

    def test_mock_provider_implementation(self) -> None:
        """Test mock provider implementation."""
        from ml4t.data.providers.mock import MockProvider

        provider = MockProvider()
        df = provider.fetch_ohlcv(symbol="TEST", start="2024-01-01", end="2024-01-10")

        assert isinstance(df, pl.DataFrame)
        assert "timestamp" in df.columns
        assert "open" in df.columns
        assert "high" in df.columns
        assert "low" in df.columns
        assert "close" in df.columns
        assert "volume" in df.columns
        assert df.shape[0] > 0

    def test_provider_normalization(self) -> None:
        """Test that provider returns normalized schema."""
        from ml4t.data.providers.mock import MockProvider

        provider = MockProvider()
        df = provider.fetch_ohlcv(symbol="TEST", start="2024-01-01", end="2024-01-02")

        # Check data types
        assert df["timestamp"].dtype == pl.Datetime
        assert df["open"].dtype == pl.Float64
        assert df["high"].dtype == pl.Float64
        assert df["low"].dtype == pl.Float64
        assert df["close"].dtype == pl.Float64
        assert df["volume"].dtype == pl.Float64

        # Check OHLC invariants
        assert (df["high"] >= df["low"]).all()
        assert (df["high"] >= df["open"]).all()
        assert (df["high"] >= df["close"]).all()
        assert (df["low"] <= df["open"]).all()
        assert (df["low"] <= df["close"]).all()
