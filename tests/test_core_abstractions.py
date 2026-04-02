"""Tests for core abstractions."""

from datetime import UTC, datetime
from pathlib import Path

import polars as pl
import pytest
from ml4t.specs import (
    ArtifactStorage,
    FeedSpec,
    MarketDataSchema,
    MarketDataSemantics,
    MarketDataSpec,
    TimestampSemantics,
)
from pydantic import ValidationError

from ml4t.data.core.config import Config, StorageConfig
from ml4t.data.core.models import DataObject, Metadata, SchemaVersion
from ml4t.data.paths import default_ml4t_data_path
from ml4t.data.providers.base import Provider


class TestConfig:
    """Test configuration management."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = Config()
        assert config.data_root == default_ml4t_data_path()
        assert config.storage.backend == "filesystem"
        assert config.log_level == "INFO"

    def test_config_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test configuration from environment variables."""
        monkeypatch.setenv("ML4T_DATA_PATH", "/custom/path")
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


class TestArtifactContracts:
    """Test shared market-data artifact contracts."""

    def test_feed_spec_from_market_data_spec(self) -> None:
        spec = MarketDataSpec(
            artifact_id="prices",
            storage=ArtifactStorage(path="prices.parquet"),
            schema=MarketDataSchema(
                timestamp_col="ts",
                entity_col="ticker",
                price_col="last",
                open_col="o",
                high_col="h",
                low_col="l",
                close_col="c",
                volume_col="vol",
            ),
            semantics=MarketDataSemantics(
                data_frequency="1d",
                calendar="XNYS",
                timezone="UTC",
                timestamp_semantics=TimestampSemantics.BAR_CLOSE,
            ),
        )

        feed_spec = FeedSpec.from_any(spec)

        assert feed_spec.timestamp_col == "ts"
        assert feed_spec.entity_col == "ticker"
        assert feed_spec.price_col == "last"
        assert feed_spec.close_col == "c"
        assert feed_spec.volume_col == "vol"
        assert feed_spec.calendar == "XNYS"
        assert feed_spec.timezone == "UTC"
        assert feed_spec.data_frequency == "1d"
        assert feed_spec.timestamp_semantics is TimestampSemantics.BAR_CLOSE

    def test_feed_spec_from_market_data_spec_mapping(self) -> None:
        spec_dict = MarketDataSpec(
            artifact_id="bars",
            storage=ArtifactStorage(path="bars.parquet"),
            schema=MarketDataSchema(timestamp_col="date", entity_col="asset", close_col="close_px"),
            semantics=MarketDataSemantics(
                data_frequency="1h",
                timestamp_semantics=TimestampSemantics.SESSION_LABEL,
            ),
        ).to_dict()

        feed_spec = FeedSpec.from_any(spec_dict)

        assert feed_spec.timestamp_col == "date"
        assert feed_spec.entity_col == "asset"
        assert feed_spec.close_col == "close_px"
        assert feed_spec.price_col == "close"
        assert feed_spec.data_frequency == "1h"
        assert feed_spec.timestamp_semantics is TimestampSemantics.SESSION_LABEL

    def test_market_data_spec_to_feed_spec(self) -> None:
        spec = MarketDataSpec(
            artifact_id="intraday",
            storage=ArtifactStorage(path="intraday.parquet"),
            schema=MarketDataSchema(timestamp_col="timestamp", entity_col="symbol", price_col="mid"),
            semantics=MarketDataSemantics(calendar="24/7"),
        )

        feed_spec = spec.to_feed_spec()

        assert feed_spec.timestamp_col == "timestamp"
        assert feed_spec.entity_col == "symbol"
        assert feed_spec.price_col == "mid"
        assert feed_spec.calendar == "24/7"
