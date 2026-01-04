"""Tests for ETF downloader module."""

import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import polars as pl
import pytest

from ml4t.data.etfs.downloader import OHLCV_SCHEMA, ETFConfig, ETFDataManager


class TestETFConfig:
    """Tests for ETFConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = ETFConfig()

        assert config.provider == "yahoo"
        assert config.start == "2005-01-01"
        assert config.end == "2025-12-31"
        assert config.frequency == "daily"
        assert config.tickers == {}

    def test_storage_path_expanded(self):
        """Test that storage path is expanded."""
        config = ETFConfig(storage_path=Path("~/test-data/etfs"))

        assert "~" not in str(config.storage_path)
        assert config.storage_path.is_absolute()

    def test_get_all_symbols_empty(self):
        """Test get_all_symbols with empty tickers."""
        config = ETFConfig()
        assert config.get_all_symbols() == []

    def test_get_all_symbols_with_categories(self):
        """Test get_all_symbols with multiple categories."""
        config = ETFConfig(
            tickers={
                "equity": {"symbols": ["SPY", "QQQ"]},
                "bonds": {"symbols": ["TLT", "IEF"]},
            }
        )

        symbols = config.get_all_symbols()
        assert set(symbols) == {"SPY", "QQQ", "TLT", "IEF"}

    def test_get_all_symbols_deduplicates(self):
        """Test that get_all_symbols removes duplicates."""
        config = ETFConfig(
            tickers={
                "equity": {"symbols": ["SPY", "QQQ"]},
                "overlap": {"symbols": ["SPY", "TLT"]},  # SPY appears twice
            }
        )

        symbols = config.get_all_symbols()
        assert len(symbols) == 3
        assert symbols.count("SPY") == 1

    def test_get_categories(self):
        """Test get_categories returns organized dict."""
        config = ETFConfig(
            tickers={
                "equity": {"symbols": ["SPY", "QQQ"]},
                "bonds": {"symbols": ["TLT"]},
            }
        )

        categories = config.get_categories()
        assert categories["equity"] == ["SPY", "QQQ"]
        assert categories["bonds"] == ["TLT"]

    def test_get_categories_ignores_invalid(self):
        """Test that invalid category format is ignored."""
        config = ETFConfig(
            tickers={
                "equity": {"symbols": ["SPY"]},
                "invalid": "not a dict",  # Should be ignored
                "no_symbols": {"other_key": "value"},  # Missing symbols key
            }
        )

        categories = config.get_categories()
        assert "equity" in categories
        assert "invalid" not in categories
        assert "no_symbols" not in categories


class TestETFDataManager:
    """Tests for ETFDataManager class."""

    @pytest.fixture
    def temp_storage(self):
        """Create temporary storage directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def config(self, temp_storage):
        """Create test configuration."""
        return ETFConfig(
            provider="yahoo",
            start="2024-01-01",
            end="2024-12-31",
            frequency="daily",
            storage_path=temp_storage,
            tickers={
                "test": {"symbols": ["AAPL", "MSFT"]},
            },
        )

    @pytest.fixture
    def manager(self, config):
        """Create ETFDataManager instance."""
        return ETFDataManager(config)

    def test_init(self, manager, temp_storage):
        """Test initialization."""
        assert manager.config.storage_path == temp_storage
        assert manager._provider is None
        assert temp_storage.exists()

    def test_from_config_yaml(self, temp_storage):
        """Test creating manager from YAML config."""
        yaml_content = f"""
etfs:
  provider: yahoo
  start: "2020-01-01"
  end: "2024-12-31"
  frequency: daily
  storage_path: {temp_storage}
  tickers:
    equity:
      symbols: ["SPY", "QQQ"]
"""
        config_file = temp_storage / "test_config.yaml"
        config_file.write_text(yaml_content)

        manager = ETFDataManager.from_config(config_file)

        assert manager.config.provider == "yahoo"
        assert manager.config.start == "2020-01-01"
        assert "SPY" in manager.config.get_all_symbols()

    def test_provider_lazy_initialization(self, manager):
        """Test that provider is lazily initialized."""
        assert manager._provider is None

        with patch("ml4t.data.providers.yahoo.YahooFinanceProvider") as mock_provider:
            mock_provider.return_value = MagicMock()
            _ = manager.provider

            mock_provider.assert_called_once_with(enable_progress=True)

    def test_download_all_empty_result(self, manager):
        """Test download_all with empty result."""
        mock_provider = MagicMock()
        mock_provider.fetch_batch_ohlcv.return_value = pl.DataFrame()

        with patch.object(manager, "_provider", mock_provider):
            manager._provider = mock_provider
            stats = manager.download_all()

            assert stats == {}

    def test_download_all_with_data(self, manager, temp_storage):
        """Test download_all with mock data."""
        mock_data = pl.DataFrame(
            {
                "timestamp": [datetime(2024, 1, 1), datetime(2024, 1, 2)],
                "symbol": ["AAPL", "AAPL"],
                "open": [100.0, 101.0],
                "high": [102.0, 103.0],
                "low": [99.0, 100.0],
                "close": [101.0, 102.0],
                "volume": [1000.0, 1100.0],
            }
        )

        mock_provider = MagicMock()
        mock_provider.fetch_batch_ohlcv.return_value = mock_data

        with patch.object(manager, "_provider", mock_provider):
            manager._provider = mock_provider
            stats = manager.download_all()

            assert "AAPL" in stats
            assert stats["AAPL"] == 2

            # Check files were created
            assert (temp_storage / "ohlcv_1d" / "ticker=AAPL" / "data.parquet").exists()
            assert (temp_storage / "etf_universe.parquet").exists()
            assert (temp_storage / "etf_universe_metadata.json").exists()

    def test_load_ohlcv_no_data(self, manager):
        """Test load_ohlcv when no data exists."""
        df = manager.load_ohlcv("UNKNOWN")

        assert df.is_empty()
        assert set(df.columns) == set(OHLCV_SCHEMA.keys())

    def test_load_ohlcv_with_data(self, manager, temp_storage):
        """Test load_ohlcv with existing data."""
        # Create test data file
        ohlcv_dir = temp_storage / "ohlcv_1d" / "ticker=SPY"
        ohlcv_dir.mkdir(parents=True)

        test_data = pl.DataFrame(
            {
                "timestamp": [datetime(2024, 1, 1), datetime(2024, 1, 2)],
                "open": [100.0, 101.0],
                "high": [102.0, 103.0],
                "low": [99.0, 100.0],
                "close": [101.0, 102.0],
                "volume": [1000.0, 1100.0],
            }
        )
        test_data.write_parquet(ohlcv_dir / "data.parquet")

        # Load data
        df = manager.load_ohlcv("SPY")

        assert len(df) == 2
        assert "symbol" in df.columns
        assert df["symbol"][0] == "SPY"

    def test_load_ohlcv_handles_date_column(self, manager, temp_storage):
        """Test load_ohlcv renames date to timestamp."""
        ohlcv_dir = temp_storage / "ohlcv_1d" / "ticker=SPY"
        ohlcv_dir.mkdir(parents=True)

        test_data = pl.DataFrame(
            {
                "date": [datetime(2024, 1, 1)],  # Uses 'date' instead of 'timestamp'
                "open": [100.0],
                "high": [102.0],
                "low": [99.0],
                "close": [101.0],
                "volume": [1000.0],
            }
        )
        test_data.write_parquet(ohlcv_dir / "data.parquet")

        df = manager.load_ohlcv("SPY")
        assert "timestamp" in df.columns
        assert "date" not in df.columns

    def test_load_symbols_multiple(self, manager, temp_storage):
        """Test loading multiple symbols."""
        for symbol in ["SPY", "QQQ"]:
            ohlcv_dir = temp_storage / "ohlcv_1d" / f"ticker={symbol}"
            ohlcv_dir.mkdir(parents=True)
            pl.DataFrame(
                {
                    "timestamp": [datetime(2024, 1, 1)],
                    "open": [100.0],
                    "high": [102.0],
                    "low": [99.0],
                    "close": [101.0],
                    "volume": [1000.0],
                }
            ).write_parquet(ohlcv_dir / "data.parquet")

        df = manager.load_symbols(["SPY", "QQQ"])

        assert len(df) == 2
        assert set(df["symbol"].to_list()) == {"SPY", "QQQ"}

    def test_load_symbols_empty_list(self, manager):
        """Test loading empty symbol list."""
        df = manager.load_symbols([])
        assert df.is_empty()

    def test_load_all_from_combined_file(self, manager, temp_storage):
        """Test load_all uses combined file if available."""
        test_data = pl.DataFrame(
            {
                "timestamp": [datetime(2024, 1, 1)],
                "symbol": ["SPY"],
                "open": [100.0],
                "high": [102.0],
                "low": [99.0],
                "close": [101.0],
                "volume": [1000.0],
            }
        )
        test_data.write_parquet(temp_storage / "etf_universe.parquet")

        df = manager.load_all()
        assert len(df) == 1
        assert df["symbol"][0] == "SPY"

    def test_load_category_valid(self, manager, temp_storage):
        """Test load_category with valid category."""
        # Setup config with categories
        manager.config.tickers = {"equity": {"symbols": ["SPY"]}}

        # Create test data
        ohlcv_dir = temp_storage / "ohlcv_1d" / "ticker=SPY"
        ohlcv_dir.mkdir(parents=True)
        pl.DataFrame(
            {
                "timestamp": [datetime(2024, 1, 1)],
                "open": [100.0],
                "high": [102.0],
                "low": [99.0],
                "close": [101.0],
                "volume": [1000.0],
            }
        ).write_parquet(ohlcv_dir / "data.parquet")

        df = manager.load_category("equity")
        assert len(df) == 1

    def test_load_category_invalid(self, manager):
        """Test load_category with invalid category."""
        manager.config.tickers = {"equity": {"symbols": ["SPY"]}}

        with pytest.raises(ValueError, match="Unknown category"):
            manager.load_category("invalid")

    def test_get_available_symbols_empty(self, manager):
        """Test get_available_symbols when no data."""
        assert manager.get_available_symbols() == []

    def test_get_available_symbols_with_data(self, manager, temp_storage):
        """Test get_available_symbols with existing data."""
        for symbol in ["SPY", "QQQ"]:
            (temp_storage / "ohlcv_1d" / f"ticker={symbol}").mkdir(parents=True)

        symbols = manager.get_available_symbols()
        assert set(symbols) == {"SPY", "QQQ"}

    def test_get_data_summary_empty(self, manager):
        """Test get_data_summary when no data."""
        summary = manager.get_data_summary()
        assert summary.is_empty()

    def test_get_data_summary_with_data(self, manager, temp_storage):
        """Test get_data_summary with existing data."""
        ohlcv_dir = temp_storage / "ohlcv_1d" / "ticker=SPY"
        ohlcv_dir.mkdir(parents=True)
        pl.DataFrame(
            {
                "timestamp": [datetime(2024, 1, 1), datetime(2024, 1, 2)],
                "open": [100.0, 101.0],
                "high": [102.0, 103.0],
                "low": [99.0, 100.0],
                "close": [101.0, 102.0],
                "volume": [1000.0, 1100.0],
            }
        ).write_parquet(ohlcv_dir / "data.parquet")

        summary = manager.get_data_summary()
        assert len(summary) == 1
        assert summary["symbol"][0] == "SPY"
        assert summary["row_count"][0] == 2

    def test_update_no_existing_data(self, manager, temp_storage):
        """Test update when no existing data."""
        mock_provider = MagicMock()
        mock_provider.fetch_ohlcv.return_value = pl.DataFrame(
            {
                "timestamp": [datetime(2024, 1, 1)],
                "open": [100.0],
                "high": [102.0],
                "low": [99.0],
                "close": [101.0],
                "volume": [1000.0],
            }
        )

        with patch("ml4t.data.providers.yahoo.YahooFinanceProvider") as mock_yf:
            mock_yf.return_value = mock_provider
            stats = manager.update()

            assert "AAPL" in stats or "MSFT" in stats

    def test_save_metadata(self, manager, temp_storage):
        """Test _save_metadata creates JSON file."""
        manager._save_metadata()

        metadata_file = temp_storage / "etf_universe_metadata.json"
        assert metadata_file.exists()

        import json

        with open(metadata_file) as f:
            metadata = json.load(f)

        assert metadata["name"] == "ML4T 50-ETF Universe"
        assert "config" in metadata
        assert metadata["config"]["start"] == "2024-01-01"

    def test_save_by_ticker(self, manager, temp_storage):
        """Test _save_by_ticker partitions data correctly."""
        data = pl.DataFrame(
            {
                "timestamp": [datetime(2024, 1, 1), datetime(2024, 1, 1)],
                "symbol": ["SPY", "QQQ"],
                "open": [100.0, 200.0],
                "high": [102.0, 202.0],
                "low": [99.0, 199.0],
                "close": [101.0, 201.0],
                "volume": [1000.0, 2000.0],
            }
        )

        stats = manager._save_by_ticker(data)

        assert stats["SPY"] == 1
        assert stats["QQQ"] == 1
        assert (temp_storage / "ohlcv_1d" / "ticker=SPY" / "data.parquet").exists()
        assert (temp_storage / "ohlcv_1d" / "ticker=QQQ" / "data.parquet").exists()


class TestETFDataManagerIntegration:
    """Integration tests for ETFDataManager."""

    @pytest.fixture
    def temp_storage(self):
        """Create temporary storage directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def test_save_and_load_roundtrip(self, temp_storage):
        """Test complete save and load cycle."""
        config = ETFConfig(
            storage_path=temp_storage,
            tickers={"test": {"symbols": ["SPY"]}},
        )
        manager = ETFDataManager(config)

        # Create and save data
        data = pl.DataFrame(
            {
                "timestamp": [datetime(2024, 1, 1), datetime(2024, 1, 2)],
                "symbol": ["SPY", "SPY"],
                "open": [100.0, 101.0],
                "high": [102.0, 103.0],
                "low": [99.0, 100.0],
                "close": [101.0, 102.0],
                "volume": [1000.0, 1100.0],
            }
        )
        manager._save_by_ticker(data)
        manager._save_combined(data)

        # Load and verify
        loaded = manager.load_all()
        assert len(loaded) == 2

        spy_data = manager.load_ohlcv("SPY")
        assert len(spy_data) == 2
        assert spy_data["open"][0] == 100.0
