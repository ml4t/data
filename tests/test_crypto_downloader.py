"""Tests for Crypto downloader module."""

import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import polars as pl
import pytest

from ml4t.data.crypto.downloader import CryptoConfig, CryptoDataManager


class TestCryptoConfig:
    """Tests for CryptoConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = CryptoConfig()

        assert config.provider == "binance_public"
        assert config.market == "futures"
        assert config.start == "2021-01-01"
        assert config.end == "2025-12-31"
        assert config.interval == "8h"
        assert config.symbols == {}

    def test_storage_path_expanded(self):
        """Test that storage path is expanded."""
        config = CryptoConfig(storage_path=Path("~/test-data/crypto"))

        assert "~" not in str(config.storage_path)
        assert config.storage_path.is_absolute()

    def test_get_all_symbols_empty(self):
        """Test get_all_symbols with empty config."""
        config = CryptoConfig()
        assert config.get_all_symbols() == []

    def test_get_all_symbols_with_categories(self):
        """Test get_all_symbols with multiple categories."""
        config = CryptoConfig(
            symbols={
                "major": {"symbols": ["BTCUSDT", "ETHUSDT"]},
                "alt": {"symbols": ["SOLUSDT", "AVAXUSDT"]},
            }
        )

        symbols = config.get_all_symbols()
        assert set(symbols) == {"BTCUSDT", "ETHUSDT", "SOLUSDT", "AVAXUSDT"}

    def test_get_all_symbols_deduplicates(self):
        """Test that get_all_symbols removes duplicates."""
        config = CryptoConfig(
            symbols={
                "major": {"symbols": ["BTCUSDT", "ETHUSDT"]},
                "top_10": {"symbols": ["BTCUSDT", "SOLUSDT"]},  # BTCUSDT appears twice
            }
        )

        symbols = config.get_all_symbols()
        assert len(symbols) == 3
        assert symbols.count("BTCUSDT") == 1

    def test_get_categories(self):
        """Test get_categories returns organized dict."""
        config = CryptoConfig(
            symbols={
                "major": {"symbols": ["BTCUSDT"]},
                "alt": {"symbols": ["SOLUSDT"]},
            }
        )

        categories = config.get_categories()
        assert categories["major"] == ["BTCUSDT"]
        assert categories["alt"] == ["SOLUSDT"]

    def test_get_categories_ignores_invalid(self):
        """Test that invalid category format is ignored."""
        config = CryptoConfig(
            symbols={
                "major": {"symbols": ["BTCUSDT"]},
                "invalid": "not a dict",
                "no_symbols": {"other_key": "value"},
            }
        )

        categories = config.get_categories()
        assert "major" in categories
        assert "invalid" not in categories
        assert "no_symbols" not in categories


class TestCryptoDataManager:
    """Tests for CryptoDataManager class."""

    @pytest.fixture
    def temp_storage(self):
        """Create temporary storage directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def config(self, temp_storage):
        """Create test configuration."""
        return CryptoConfig(
            provider="binance_public",
            market="futures",
            start="2024-01-01",
            end="2024-12-31",
            interval="8h",
            storage_path=temp_storage,
            symbols={
                "test": {"symbols": ["BTCUSDT", "ETHUSDT"]},
            },
        )

    @pytest.fixture
    def manager(self, config):
        """Create CryptoDataManager instance."""
        return CryptoDataManager(config)

    def test_init(self, manager, temp_storage):
        """Test initialization."""
        assert manager.config.storage_path == temp_storage
        assert manager._provider is None
        assert temp_storage.exists()

    def test_from_config_yaml(self, temp_storage):
        """Test creating manager from YAML config."""
        yaml_content = f"""
crypto:
  provider: binance_public
  market: futures
  start: "2023-01-01"
  end: "2024-12-31"
  interval: 8h
  storage_path: {temp_storage}
  symbols:
    major:
      symbols: ["BTCUSDT", "ETHUSDT"]
"""
        config_file = temp_storage / "test_config.yaml"
        config_file.write_text(yaml_content)

        manager = CryptoDataManager.from_config(config_file)

        assert manager.config.provider == "binance_public"
        assert manager.config.start == "2023-01-01"
        assert "BTCUSDT" in manager.config.get_all_symbols()

    def test_provider_lazy_initialization(self, manager):
        """Test that provider is lazily initialized."""
        assert manager._provider is None

        with patch("ml4t.data.providers.binance_public.BinancePublicProvider") as mock_provider:
            mock_provider.return_value = MagicMock()
            _ = manager.provider

            mock_provider.assert_called_once_with(market="futures")

    def test_download_premium_index_empty_result(self, manager):
        """Test download_premium_index with empty result."""
        mock_provider = MagicMock()
        mock_provider.fetch_premium_index_multi.return_value = pl.DataFrame()

        with patch.object(manager, "_provider", mock_provider):
            manager._provider = mock_provider
            df = manager.download_premium_index()

            assert df.is_empty()

    def test_download_premium_index_with_data(self, manager, temp_storage):
        """Test download_premium_index with mock data."""
        mock_data = pl.DataFrame(
            {
                "timestamp": [datetime(2024, 1, 1), datetime(2024, 1, 1)],
                "symbol": ["BTCUSDT", "ETHUSDT"],
                "premium_index": [0.001, 0.002],
                "close": [42000.0, 2200.0],
                "volume": [1000.0, 500.0],
            }
        )

        mock_provider = MagicMock()
        mock_provider.fetch_premium_index_multi.return_value = mock_data

        with patch.object(manager, "_provider", mock_provider):
            manager._provider = mock_provider
            df = manager.download_premium_index()

            assert len(df) == 2

            # Check files were created
            assert (temp_storage / "premium_index.parquet").exists()
            assert (temp_storage / "premium_index" / "symbol=BTCUSDT" / "data.parquet").exists()
            assert (temp_storage / "premium_index" / "symbol=ETHUSDT" / "data.parquet").exists()

    def test_download_premium_index_default_symbols(self, manager):
        """Test that default symbols are used when config is empty."""
        manager.config.symbols = {}

        mock_provider = MagicMock()
        mock_provider.fetch_premium_index_multi.return_value = pl.DataFrame(
            {
                "timestamp": [datetime(2024, 1, 1)],
                "symbol": ["BTCUSDT"],
                "premium_index": [0.001],
                "close": [42000.0],
                "volume": [1000.0],
            }
        )

        with patch.object(manager, "_provider", mock_provider):
            manager._provider = mock_provider
            manager.download_premium_index()

            # Should be called with default symbols
            call_args = mock_provider.fetch_premium_index_multi.call_args
            assert "BTCUSDT" in call_args.kwargs.get("symbols", []) or "BTCUSDT" in call_args[
                1
            ].get("symbols", [])

    def test_save_premium_index(self, manager, temp_storage):
        """Test _save_premium_index creates combined file."""
        df = pl.DataFrame(
            {
                "timestamp": [datetime(2024, 1, 1)],
                "symbol": ["BTCUSDT"],
                "premium_index": [0.001],
            }
        )

        manager._save_premium_index(df)

        assert (temp_storage / "premium_index.parquet").exists()

    def test_save_by_symbol(self, manager, temp_storage):
        """Test _save_by_symbol partitions data correctly."""
        df = pl.DataFrame(
            {
                "timestamp": [datetime(2024, 1, 1), datetime(2024, 1, 1)],
                "symbol": ["BTCUSDT", "ETHUSDT"],
                "premium_index": [0.001, 0.002],
            }
        )

        manager._save_by_symbol(df)

        assert (temp_storage / "premium_index" / "symbol=BTCUSDT" / "data.parquet").exists()
        assert (temp_storage / "premium_index" / "symbol=ETHUSDT" / "data.parquet").exists()

    def test_load_premium_index_from_combined(self, manager, temp_storage):
        """Test load_premium_index uses combined file."""
        df = pl.DataFrame(
            {
                "timestamp": [datetime(2024, 1, 1), datetime(2024, 1, 2)],
                "symbol": ["BTCUSDT", "ETHUSDT"],
                "premium_index": [0.001, 0.002],
            }
        )
        df.write_parquet(temp_storage / "premium_index.parquet")

        loaded = manager.load_premium_index()

        assert len(loaded) == 2
        assert set(loaded["symbol"].to_list()) == {"BTCUSDT", "ETHUSDT"}

    def test_load_premium_index_filter_symbols(self, manager, temp_storage):
        """Test load_premium_index with symbol filter."""
        df = pl.DataFrame(
            {
                "timestamp": [datetime(2024, 1, 1), datetime(2024, 1, 1)],
                "symbol": ["BTCUSDT", "ETHUSDT"],
                "premium_index": [0.001, 0.002],
            }
        )
        df.write_parquet(temp_storage / "premium_index.parquet")

        loaded = manager.load_premium_index(symbols=["BTCUSDT"])

        assert len(loaded) == 1
        assert loaded["symbol"][0] == "BTCUSDT"

    def test_load_premium_index_from_partitions(self, manager, temp_storage):
        """Test load_premium_index falls back to partitioned files."""
        # Don't create combined file, only partitioned
        partition_dir = temp_storage / "premium_index"
        for symbol in ["BTCUSDT", "ETHUSDT"]:
            symbol_dir = partition_dir / f"symbol={symbol}"
            symbol_dir.mkdir(parents=True)
            pl.DataFrame(
                {
                    "timestamp": [datetime(2024, 1, 1)],
                    "symbol": [symbol],
                    "premium_index": [0.001],
                }
            ).write_parquet(symbol_dir / "data.parquet")

        loaded = manager.load_premium_index()

        assert len(loaded) == 2

    def test_load_premium_index_no_data(self, manager):
        """Test load_premium_index when no data exists."""
        df = manager.load_premium_index()
        assert df.is_empty()

    def test_load_symbol(self, manager, temp_storage):
        """Test load_symbol convenience method."""
        df = pl.DataFrame(
            {
                "timestamp": [datetime(2024, 1, 1), datetime(2024, 1, 1)],
                "symbol": ["BTCUSDT", "ETHUSDT"],
                "premium_index": [0.001, 0.002],
            }
        )
        df.write_parquet(temp_storage / "premium_index.parquet")

        loaded = manager.load_symbol("BTCUSDT")

        assert len(loaded) == 1
        assert loaded["symbol"][0] == "BTCUSDT"

    def test_get_available_symbols_empty(self, manager):
        """Test get_available_symbols when no data."""
        assert manager.get_available_symbols() == []

    def test_get_available_symbols_with_data(self, manager, temp_storage):
        """Test get_available_symbols with existing data."""
        partition_dir = temp_storage / "premium_index"
        for symbol in ["BTCUSDT", "ETHUSDT"]:
            (partition_dir / f"symbol={symbol}").mkdir(parents=True)

        symbols = manager.get_available_symbols()
        assert set(symbols) == {"BTCUSDT", "ETHUSDT"}

    def test_get_data_summary_empty(self, manager):
        """Test get_data_summary when no data."""
        summary = manager.get_data_summary()
        assert summary.is_empty()

    def test_get_data_summary_with_data(self, manager, temp_storage):
        """Test get_data_summary with existing data."""
        df = pl.DataFrame(
            {
                "timestamp": [datetime(2024, 1, 1), datetime(2024, 1, 2), datetime(2024, 1, 1)],
                "symbol": ["BTCUSDT", "BTCUSDT", "ETHUSDT"],
                "premium_index": [0.001, 0.002, 0.003],
            }
        )
        df.write_parquet(temp_storage / "premium_index.parquet")

        summary = manager.get_data_summary()

        assert len(summary) == 2
        assert "symbol" in summary.columns
        assert "start_date" in summary.columns
        assert "end_date" in summary.columns
        assert "row_count" in summary.columns


class TestCryptoDataManagerIntegration:
    """Integration tests for CryptoDataManager."""

    @pytest.fixture
    def temp_storage(self):
        """Create temporary storage directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def test_save_and_load_roundtrip(self, temp_storage):
        """Test complete save and load cycle."""
        config = CryptoConfig(
            storage_path=temp_storage,
            symbols={"test": {"symbols": ["BTCUSDT"]}},
        )
        manager = CryptoDataManager(config)

        # Create and save data
        data = pl.DataFrame(
            {
                "timestamp": [datetime(2024, 1, 1), datetime(2024, 1, 2)],
                "symbol": ["BTCUSDT", "BTCUSDT"],
                "premium_index": [0.001, 0.002],
                "close": [42000.0, 42500.0],
                "volume": [1000.0, 1100.0],
            }
        )
        manager._save_premium_index(data)
        manager._save_by_symbol(data)

        # Load and verify
        loaded = manager.load_premium_index()
        assert len(loaded) == 2

        btc_data = manager.load_symbol("BTCUSDT")
        assert len(btc_data) == 2

        # Verify summary
        summary = manager.get_data_summary()
        assert len(summary) == 1
        assert summary["row_count"][0] == 2

    def test_partitioned_filtering(self, temp_storage):
        """Test that loading from partitions respects symbol filter."""
        config = CryptoConfig(storage_path=temp_storage)
        manager = CryptoDataManager(config)

        # Create partitioned data (no combined file)
        partition_dir = temp_storage / "premium_index"
        for symbol in ["BTCUSDT", "ETHUSDT", "SOLUSDT"]:
            symbol_dir = partition_dir / f"symbol={symbol}"
            symbol_dir.mkdir(parents=True)
            pl.DataFrame(
                {
                    "timestamp": [datetime(2024, 1, 1)],
                    "symbol": [symbol],
                    "premium_index": [0.001],
                }
            ).write_parquet(symbol_dir / "data.parquet")

        # Load with filter
        loaded = manager.load_premium_index(symbols=["BTCUSDT", "ETHUSDT"])
        assert len(loaded) == 2
        assert set(loaded["symbol"].to_list()) == {"BTCUSDT", "ETHUSDT"}
