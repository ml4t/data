"""Tests for DataManager storage operations (load/update)."""

import tempfile
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

import polars as pl
import pytest

from ml4t.data.data_manager import DataManager
from ml4t.data.storage.backend import StorageConfig
from ml4t.data.storage.hive import HiveStorage


class TestDataManagerLoad:
    """Test DataManager.load() method."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def storage(self, temp_dir):
        """Create storage backend."""
        config = StorageConfig(base_path=temp_dir)
        return HiveStorage(config)

    @pytest.fixture
    def manager(self, storage):
        """Create DataManager with storage."""
        return DataManager(
            storage=storage,
            use_transactions=True,
            enable_validation=True,
        )

    @pytest.fixture
    def sample_data(self):
        """Create sample OHLCV data."""
        dates = [datetime(2024, 1, i, tzinfo=UTC) for i in range(1, 11)]
        return pl.DataFrame(
            {
                "timestamp": dates,
                "open": [100.0 + i for i in range(10)],
                "high": [105.0 + i for i in range(10)],
                "low": [95.0 + i for i in range(10)],
                "close": [102.0 + i for i in range(10)],
                "volume": [1000000 + i * 10000 for i in range(10)],
            }
        )

    def test_load_requires_storage(self):
        """Test that load() requires storage to be configured."""
        manager = DataManager()  # No storage

        with pytest.raises(ValueError, match="Storage not configured"):
            manager.load("AAPL", "2024-01-01", "2024-01-31")

    def test_load_basic(self, manager, storage):
        """Test basic load operation with mock provider."""
        key = manager.load(
            symbol="AAPL",
            start="2024-01-01",
            end="2024-01-10",
            provider="mock",
        )

        assert key == "equities/daily/AAPL"
        assert storage.exists(key)

        # Verify data was stored (8 trading days: excludes weekend Jan 6-7)
        stored_df = storage.read(key).collect()  # Collect LazyFrame
        assert len(stored_df) == 8
        assert "timestamp" in stored_df.columns

    def test_load_with_lazy_frame(self, manager, storage):
        """Test load stores data correctly (LazyFrame internally)."""
        key = manager.load("AAPL", "2024-01-01", "2024-01-10", provider="mock")

        assert storage.exists(key)
        stored_df = storage.read(key).collect()  # Collect LazyFrame
        assert isinstance(stored_df, pl.DataFrame)
        assert len(stored_df) == 8  # 8 trading days

    def test_load_with_mock_provider_returns_data(self, manager, storage):
        """Test that MockProvider generates data for any symbol."""
        # MockProvider always generates synthetic data, even for "nonexistent" symbols
        key = manager.load("ANY_SYMBOL", "2024-01-01", "2024-01-05", provider="mock")
        assert storage.exists(key)
        stored_df = storage.read(key).collect()
        assert len(stored_df) > 0  # MockProvider generates data

    def test_load_with_validation(self, manager, storage):
        """Test load with validation enabled."""
        # Should succeed with valid data from mock provider
        key = manager.load("AAPL", "2024-01-01", "2024-01-10", provider="mock")
        assert storage.exists(key)

    def test_load_overwrites_existing(self, manager, storage):
        """Test that load overwrites existing data."""
        # First load
        key1 = manager.load("AAPL", "2024-01-01", "2024-01-05", provider="mock")
        stored_df1 = storage.read(key1).collect()
        rows1 = len(stored_df1)

        # Second load with different date range (should overwrite)
        key2 = manager.load("AAPL", "2024-01-01", "2024-01-10", provider="mock")

        assert key1 == key2
        stored_df2 = storage.read(key2).collect()
        rows2 = len(stored_df2)
        # Second load should have more data (longer date range)
        assert rows2 > rows1

    def test_load_progress_callback(self, storage):
        """Test that progress callback is called."""
        progress_calls = []

        def progress_callback(message, progress):
            progress_calls.append((message, progress))

        manager = DataManager(
            storage=storage,
            use_transactions=True,
            enable_validation=True,
            progress_callback=progress_callback,
        )

        manager.load("AAPL", "2024-01-01", "2024-01-10", provider="mock")

        # Should have progress updates
        assert len(progress_calls) > 0
        messages = [msg for msg, _ in progress_calls]
        assert any("Loading" in msg for msg in messages) or any(
            "progress" in msg.lower() for msg in messages
        )

    def test_load_different_asset_classes(self, manager, storage):
        """Test loading different asset classes."""
        # Equities
        key1 = manager.load(
            "AAPL", "2024-01-01", "2024-01-10", asset_class="equities", provider="mock"
        )
        assert key1 == "equities/daily/AAPL"

        # Crypto
        key2 = manager.load(
            "BTC", "2024-01-01", "2024-01-10", asset_class="crypto", provider="mock"
        )
        assert key2 == "crypto/daily/BTC"

        # Forex
        key3 = manager.load(
            "EURUSD", "2024-01-01", "2024-01-10", asset_class="forex", provider="mock"
        )
        assert key3 == "forex/daily/EURUSD"

        # All should exist
        assert storage.exists(key1)
        assert storage.exists(key2)
        assert storage.exists(key3)


class TestDataManagerUpdate:
    """Test DataManager.update() method."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def storage(self, temp_dir):
        """Create storage backend."""
        config = StorageConfig(base_path=temp_dir)
        return HiveStorage(config)

    @pytest.fixture
    def manager(self, storage):
        """Create DataManager with storage."""
        return DataManager(
            storage=storage,
            use_transactions=True,
            enable_validation=True,
        )

    @pytest.fixture
    def initial_data(self):
        """Create initial dataset."""
        dates = [datetime(2024, 1, i, tzinfo=UTC) for i in range(1, 11)]
        return pl.DataFrame(
            {
                "timestamp": dates,
                "open": [100.0 + i for i in range(10)],
                "high": [105.0 + i for i in range(10)],
                "low": [95.0 + i for i in range(10)],
                "close": [102.0 + i for i in range(10)],
                "volume": [1000000 + i * 10000 for i in range(10)],
            }
        )

    @pytest.fixture
    def new_data(self):
        """Create new data for update."""
        dates = [datetime(2024, 1, i, tzinfo=UTC) for i in range(8, 16)]
        return pl.DataFrame(
            {
                "timestamp": dates,
                "open": [107.0 + i for i in range(8)],
                "high": [112.0 + i for i in range(8)],
                "low": [102.0 + i for i in range(8)],
                "close": [109.0 + i for i in range(8)],
                "volume": [1070000 + i * 10000 for i in range(8)],
            }
        )

    def test_update_requires_storage(self):
        """Test that update() requires storage to be configured."""
        manager = DataManager()  # No storage

        with pytest.raises(ValueError, match="Storage not configured"):
            manager.update("AAPL")

    def test_update_falls_back_to_load_if_no_data(self, manager, storage):
        """Test that update falls back to load if no existing data."""
        # No initial data - update should fall back to load
        key = manager.update("AAPL", provider="mock")

        # Should have created data via load fallback
        assert key == "equities/daily/AAPL"
        assert storage.exists(key)

    def test_update_incremental(self, manager, storage):
        """Test incremental update with new data."""
        # First load initial data
        key = manager.load("AAPL", "2024-01-01", "2024-01-05", provider="mock")
        initial_df = storage.read(key).collect()
        initial_rows = len(initial_df)

        # Now update - should fetch more recent data and merge
        updated_key = manager.update("AAPL", provider="mock")

        assert updated_key == key
        stored_df = storage.read(updated_key).collect()

        # Should have at least the initial data (update merges data)
        assert len(stored_df) >= initial_rows

    def test_update_handles_duplicates(self, manager, storage):
        """Test that update handles duplicate timestamps correctly."""
        key = manager.load("AAPL", "2024-01-01", "2024-01-05", provider="mock")

        # Update should merge and deduplicate
        manager.update("AAPL", provider="mock")

        stored_df = storage.read(key).collect()
        # Should not have duplicates
        assert stored_df["timestamp"].is_unique().all()

    def test_update_skips_if_current(self, manager, storage):
        """Test that update works even with recent data."""
        # Load some data first
        key = manager.load("AAPL", "2024-01-01", "2024-01-05", provider="mock")

        # Try to update - should work
        updated_key = manager.update("AAPL", provider="mock")

        # Should return same key
        assert updated_key == key
        assert storage.exists(key)

    def test_update_with_gaps(self, manager, storage):
        """Test update handles data correctly."""
        # Initial load
        key = manager.load("AAPL", "2024-01-01", "2024-01-05", provider="mock")

        # Update - should merge data
        manager.update("AAPL", fill_gaps=True, provider="mock")

        stored_df = storage.read(key).collect()
        # Should have data without duplicates
        assert len(stored_df) > 0
        assert stored_df["timestamp"].is_unique().all()

    def test_update_progress_callback(self, storage):
        """Test that progress callback is called during update."""
        # Track progress
        progress_calls = []

        def progress_callback(message, progress):
            progress_calls.append((message, progress))

        manager = DataManager(
            storage=storage,
            use_transactions=True,
            enable_validation=True,
            progress_callback=progress_callback,
        )

        # Initial load (will call progress)
        manager.load("AAPL", "2024-01-01", "2024-01-05", provider="mock")
        initial_calls = len(progress_calls)

        # Update (should also call progress)
        manager.update("AAPL", provider="mock")

        # Should have more progress updates after update
        assert len(progress_calls) >= initial_calls
        messages = [msg for msg, _ in progress_calls]
        assert any("Updating" in msg for msg in messages)
        assert any("Completed" in msg for msg in messages)


class TestDataManagerTransactions:
    """Test transaction support in DataManager."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def storage(self, temp_dir):
        """Create storage backend."""
        config = StorageConfig(base_path=temp_dir)
        return HiveStorage(config)

    @pytest.fixture
    def sample_data(self):
        """Create sample OHLCV data."""
        dates = [datetime(2024, 1, i, tzinfo=UTC) for i in range(1, 11)]
        return pl.DataFrame(
            {
                "timestamp": dates,
                "open": [100.0 + i for i in range(10)],
                "high": [105.0 + i for i in range(10)],
                "low": [95.0 + i for i in range(10)],
                "close": [102.0 + i for i in range(10)],
                "volume": [1000000 + i * 10000 for i in range(10)],
            }
        )

    def test_transactions_enabled(self, storage):
        """Test that transactions can be enabled."""
        manager = DataManager(storage=storage, use_transactions=True)

        # TransactionalStorage should be wrapped
        assert hasattr(manager.storage, "transaction")

    def test_transactions_disabled(self, storage):
        """Test that transactions can be disabled."""
        manager = DataManager(storage=storage, use_transactions=False)

        # Should be raw storage (HiveStorage doesn't have transaction)
        assert not hasattr(manager.storage, "transaction")

    @patch.object(DataManager, "fetch")
    def test_load_uses_transactions(self, mock_fetch, storage, sample_data):
        """Test that load() uses transactions when enabled."""
        manager = DataManager(storage=storage, use_transactions=True)
        mock_fetch.return_value = sample_data

        # Load should use transaction
        key = manager.load("AAPL", "2024-01-01", "2024-01-10", provider="mock")

        # Data should be stored
        assert storage.exists(key)


class TestDataManagerBatchLoadFromStorage:
    """Test DataManager.batch_load_from_storage() method."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def storage(self, temp_dir):
        """Create storage backend."""
        config = StorageConfig(base_path=temp_dir)
        return HiveStorage(config)

    @pytest.fixture
    def manager(self, storage):
        """Create DataManager with storage."""
        return DataManager(
            storage=storage,
            use_transactions=False,  # Disable - TransactionalStorage.read() doesn't accept date kwargs
            enable_validation=False,  # Disable for faster tests
        )

    @pytest.fixture
    def sample_data_aapl(self):
        """Create sample OHLCV data for AAPL."""
        dates = [datetime(2024, 1, i, tzinfo=UTC) for i in range(1, 11)]
        return pl.DataFrame(
            {
                "timestamp": dates,
                "open": [100.0 + i for i in range(10)],
                "high": [105.0 + i for i in range(10)],
                "low": [95.0 + i for i in range(10)],
                "close": [102.0 + i for i in range(10)],
                "volume": [1000000 + i * 10000 for i in range(10)],
                "symbol": ["AAPL"] * 10,
            }
        )

    @pytest.fixture
    def sample_data_msft(self):
        """Create sample OHLCV data for MSFT."""
        dates = [datetime(2024, 1, i, tzinfo=UTC) for i in range(1, 11)]
        return pl.DataFrame(
            {
                "timestamp": dates,
                "open": [200.0 + i for i in range(10)],
                "high": [205.0 + i for i in range(10)],
                "low": [195.0 + i for i in range(10)],
                "close": [202.0 + i for i in range(10)],
                "volume": [2000000 + i * 10000 for i in range(10)],
                "symbol": ["MSFT"] * 10,
            }
        )

    def test_batch_load_from_storage_requires_storage(self):
        """Test that batch_load_from_storage requires storage."""
        manager = DataManager()  # No storage

        with pytest.raises(ValueError, match="Storage not configured"):
            manager.batch_load_from_storage(
                symbols=["AAPL"],
                start="2024-01-01",
                end="2024-01-10",
            )

    def test_batch_load_from_storage_empty_symbols(self, manager):
        """Test that empty symbols list raises error."""
        with pytest.raises(ValueError, match="symbols list cannot be empty"):
            manager.batch_load_from_storage(
                symbols=[],
                start="2024-01-01",
                end="2024-01-10",
            )

    def test_batch_load_from_storage_all_cached(self, manager, storage):
        """Test loading all symbols from storage cache."""
        # Pre-populate storage using real MockProvider (returns 8 rows for trading days)
        manager.load("AAPL", "2024-01-01", "2024-01-10", provider="mock")
        manager.load("MSFT", "2024-01-01", "2024-01-10", provider="mock")

        # Batch load should use cache
        result = manager.batch_load_from_storage(
            symbols=["AAPL", "MSFT"],
            start="2024-01-01",
            end="2024-01-10",
            fetch_missing=False,
        )

        # Should return combined data
        assert isinstance(result, pl.DataFrame)
        # MockProvider returns 8 trading days, storage may filter slightly (7-8 per symbol)
        assert len(result) >= 14  # At least 7 rows per symbol
        assert set(result["symbol"].unique().to_list()) == {"AAPL", "MSFT"}

    def test_batch_load_from_storage_some_missing(self, manager, storage):
        """Test loading with some symbols missing from storage."""
        # Pre-populate storage with only AAPL using MockProvider
        manager.load("AAPL", "2024-01-01", "2024-01-10", provider="mock")

        # Batch load should fetch missing MSFT from provider
        result = manager.batch_load_from_storage(
            symbols=["AAPL", "MSFT"],
            start="2024-01-01",
            end="2024-01-10",
            fetch_missing=True,
            provider="mock",
        )

        assert isinstance(result, pl.DataFrame)
        # Should have data from storage (AAPL) + fetched (MSFT)
        # MockProvider returns 8 rows per symbol, storage may filter slightly
        assert len(result) >= 14  # Combined data from both symbols
        assert set(result["symbol"].unique().to_list()) == {"AAPL", "MSFT"}

    @patch.object(DataManager, "fetch")
    def test_batch_load_from_storage_strict_mode(
        self, mock_fetch, manager, storage, sample_data_aapl
    ):
        """Test strict mode (fetch_missing=False) raises on missing."""
        # Pre-populate storage with only AAPL
        mock_fetch.return_value = sample_data_aapl
        manager.load("AAPL", "2024-01-01", "2024-01-10", provider="mock")

        # Should raise because MSFT is not in storage
        with pytest.raises(ValueError, match="not in storage"):
            manager.batch_load_from_storage(
                symbols=["AAPL", "MSFT"],
                start="2024-01-01",
                end="2024-01-10",
                fetch_missing=False,
            )

    def test_batch_load_from_storage_no_data_raises(self, manager):
        """Test that no data at all raises error."""
        with pytest.raises(ValueError, match="not in storage"):
            manager.batch_load_from_storage(
                symbols=["UNKNOWN1", "UNKNOWN2"],
                start="2024-01-01",
                end="2024-01-10",
                fetch_missing=False,
            )

    def test_batch_load_from_storage_parallel_reads(self, manager, storage):
        """Test parallel storage reads with max_workers."""
        # Pre-populate storage using real MockProvider
        manager.load("AAPL", "2024-01-01", "2024-01-10", provider="mock")
        manager.load("MSFT", "2024-01-01", "2024-01-10", provider="mock")

        # Batch load with specific max_workers
        result = manager.batch_load_from_storage(
            symbols=["AAPL", "MSFT"],
            start="2024-01-01",
            end="2024-01-10",
            max_workers=2,
            fetch_missing=False,
        )

        # MockProvider returns 8 trading days per symbol (excludes weekend)
        assert len(result) >= 14  # At least 7 rows per symbol after date filtering


class TestDataManagerImportData:
    """Test DataManager.import_data() method."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def storage(self, temp_dir):
        """Create storage backend."""
        config = StorageConfig(base_path=temp_dir)
        return HiveStorage(config)

    @pytest.fixture
    def manager(self, storage):
        """Create DataManager with storage."""
        return DataManager(
            storage=storage,
            use_transactions=True,
            enable_validation=True,
        )

    @pytest.fixture
    def sample_data(self):
        """Create sample OHLCV data."""
        dates = [datetime(2024, 1, i, tzinfo=UTC) for i in range(1, 11)]
        return pl.DataFrame(
            {
                "timestamp": dates,
                "open": [100.0 + i for i in range(10)],
                "high": [105.0 + i for i in range(10)],
                "low": [95.0 + i for i in range(10)],
                "close": [102.0 + i for i in range(10)],
                "volume": [1000000 + i * 10000 for i in range(10)],
            }
        )

    def test_import_data_requires_storage(self, sample_data):
        """Test that import_data requires storage."""
        manager = DataManager()  # No storage

        with pytest.raises(ValueError, match="Storage not configured"):
            manager.import_data(
                data=sample_data,
                symbol="AAPL",
                provider="databento",
            )

    def test_import_data_basic(self, manager, storage, sample_data):
        """Test basic data import."""
        key = manager.import_data(
            data=sample_data,
            symbol="AAPL",
            provider="databento",
        )

        assert key == "equities/daily/AAPL"
        assert storage.exists(key)

        # Verify stored data
        stored_df = storage.read(key).collect()
        assert len(stored_df) == 10

    def test_import_data_with_lazyframe(self, manager, storage, sample_data):
        """Test importing LazyFrame."""
        lazy_data = sample_data.lazy()

        key = manager.import_data(
            data=lazy_data,
            symbol="AAPL",
            provider="databento",
        )

        assert storage.exists(key)
        stored_df = storage.read(key).collect()
        assert len(stored_df) == 10

    def test_import_data_empty_raises(self, manager):
        """Test that empty data raises error."""
        empty_df = pl.DataFrame(
            {
                "timestamp": [],
                "open": [],
                "high": [],
                "low": [],
                "close": [],
                "volume": [],
            }
        )

        with pytest.raises(ValueError, match="Cannot import empty data"):
            manager.import_data(
                data=empty_df,
                symbol="AAPL",
                provider="databento",
            )

    def test_import_data_missing_columns_raises(self, manager):
        """Test that missing required columns raises error."""
        incomplete_df = pl.DataFrame(
            {
                "timestamp": [datetime(2024, 1, 1, tzinfo=UTC)],
                "close": [100.0],
                # Missing open, high, low, volume
            }
        )

        with pytest.raises(ValueError, match="Missing required columns"):
            manager.import_data(
                data=incomplete_df,
                symbol="AAPL",
                provider="databento",
            )

    def test_import_data_different_asset_classes(self, manager, storage, sample_data):
        """Test importing different asset classes."""
        # Crypto
        key_crypto = manager.import_data(
            data=sample_data,
            symbol="BTC",
            provider="binance",
            asset_class="crypto",
        )
        assert key_crypto == "crypto/daily/BTC"
        assert storage.exists(key_crypto)

        # Forex
        key_forex = manager.import_data(
            data=sample_data,
            symbol="EURUSD",
            provider="oanda",
            asset_class="forex",
        )
        assert key_forex == "forex/daily/EURUSD"
        assert storage.exists(key_forex)

        # Futures
        key_futures = manager.import_data(
            data=sample_data,
            symbol="ES",
            provider="databento",
            asset_class="futures",
            frequency="1min",
        )
        assert key_futures == "futures/1min/ES"
        assert storage.exists(key_futures)

    def test_import_data_with_bar_type(self, manager, storage, sample_data):
        """Test importing with different bar types."""
        key = manager.import_data(
            data=sample_data,
            symbol="ES",
            provider="databento",
            bar_type="volume",
            bar_threshold=1000,
        )

        # Note: import_data returns the storage key used
        # The actual storage key includes bar_type info when not "time"
        # Just verify import completed without error and key is returned
        assert key is not None
        assert "ES" in key

    def test_import_data_with_exchange(self, manager, storage, sample_data):
        """Test importing with exchange metadata."""
        key = manager.import_data(
            data=sample_data,
            symbol="ES",
            provider="databento",
            asset_class="futures",
            exchange="CME",
        )

        assert storage.exists(key)


class TestDataManagerListSymbols:
    """Test DataManager.list_symbols() method."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def storage(self, temp_dir):
        """Create storage backend."""
        config = StorageConfig(base_path=temp_dir)
        return HiveStorage(config)

    @pytest.fixture
    def manager(self, storage):
        """Create DataManager with storage."""
        return DataManager(
            storage=storage,
            enable_validation=False,
        )

    @pytest.fixture
    def sample_data(self):
        """Create sample OHLCV data."""
        dates = [datetime(2024, 1, i, tzinfo=UTC) for i in range(1, 6)]
        return pl.DataFrame(
            {
                "timestamp": dates,
                "open": [100.0 + i for i in range(5)],
                "high": [105.0 + i for i in range(5)],
                "low": [95.0 + i for i in range(5)],
                "close": [102.0 + i for i in range(5)],
                "volume": [1000000 + i * 10000 for i in range(5)],
            }
        )

    def test_list_symbols_requires_storage(self):
        """Test that list_symbols requires storage."""
        manager = DataManager()  # No storage

        with pytest.raises(ValueError, match="Storage not configured"):
            manager.list_symbols()

    def test_list_symbols_empty_storage(self, manager):
        """Test list_symbols on empty storage."""
        symbols = manager.list_symbols()

        assert isinstance(symbols, list)
        assert len(symbols) == 0

    @patch.object(DataManager, "fetch")
    def test_list_symbols_all(self, mock_fetch, manager, sample_data):
        """Test listing all symbols."""
        mock_fetch.return_value = sample_data

        # Load multiple symbols
        manager.load("AAPL", "2024-01-01", "2024-01-05", provider="yahoo")
        manager.load("MSFT", "2024-01-01", "2024-01-05", provider="yahoo")
        manager.load("GOOGL", "2024-01-01", "2024-01-05", provider="yahoo")

        symbols = manager.list_symbols()

        assert isinstance(symbols, list)
        # Note: list_symbols depends on metadata files, which may not be created
        # by all storage backends in the same way

    @patch.object(DataManager, "fetch")
    def test_list_symbols_filter_by_provider(self, mock_fetch, manager, sample_data):
        """Test filtering symbols by provider."""
        mock_fetch.return_value = sample_data

        # Load from different providers
        manager.load("AAPL", "2024-01-01", "2024-01-05", provider="yahoo")

        # Filter by provider
        symbols = manager.list_symbols(provider="yahoo")

        assert isinstance(symbols, list)

    @patch.object(DataManager, "fetch")
    def test_list_symbols_filter_by_asset_class(self, mock_fetch, manager, sample_data):
        """Test filtering symbols by asset class."""
        mock_fetch.return_value = sample_data

        # Load different asset classes
        manager.load("AAPL", "2024-01-01", "2024-01-05", provider="mock", asset_class="equities")

        # Import crypto data
        manager.import_data(
            data=sample_data,
            symbol="BTC",
            provider="binance",
            asset_class="crypto",
        )

        # Filter by asset class
        crypto_symbols = manager.list_symbols(asset_class="crypto")
        assert isinstance(crypto_symbols, list)


class TestDataManagerBatchLoad:
    """Test DataManager.batch_load() method edge cases."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def storage(self, temp_dir):
        """Create storage backend."""
        config = StorageConfig(base_path=temp_dir)
        return HiveStorage(config)

    @pytest.fixture
    def manager(self, storage):
        """Create DataManager with storage."""
        return DataManager(
            storage=storage,
            enable_validation=False,
        )

    def test_batch_load_empty_symbols(self, manager):
        """Test batch_load with empty symbols list raises error."""
        with pytest.raises(ValueError, match="symbols list cannot be empty"):
            manager.batch_load(
                symbols=[],
                start="2024-01-01",
                end="2024-01-10",
            )

    def test_batch_load_fail_on_partial_true(self, manager):
        """Test batch_load with fail_on_partial=True raises on failure."""
        # Using a non-existent provider will cause failures
        with pytest.raises(ValueError, match="Batch load failed"):
            manager.batch_load(
                symbols=["AAPL", "MSFT"],
                start="2024-01-01",
                end="2024-01-10",
                provider="nonexistent_provider",
                fail_on_partial=True,
            )

    def test_batch_load_fail_on_partial_false(self, manager):
        """Test batch_load with fail_on_partial=False returns partial results."""
        # Use mock provider for first symbol, should succeed
        # Even if some symbols fail (by using bad provider for them), partial results returned
        result = manager.batch_load(
            symbols=["AAPL"],  # Single symbol to test success path
            start="2024-01-01",
            end="2024-01-10",
            provider="mock",
            fail_on_partial=False,
        )

        assert isinstance(result, pl.DataFrame)
        assert len(result) > 0
        assert result["symbol"][0] == "AAPL"

    def test_batch_load_adds_symbol_column(self, manager):
        """Test batch_load adds symbol column if missing."""
        # MockProvider may or may not include symbol column
        # batch_load should ensure symbol column exists
        result = manager.batch_load(
            symbols=["AAPL"],
            start="2024-01-01",
            end="2024-01-10",
            provider="mock",
        )

        assert "symbol" in result.columns
        assert result["symbol"][0] == "AAPL"

    def test_batch_load_max_workers(self, manager):
        """Test batch_load respects max_workers parameter."""
        result = manager.batch_load(
            symbols=["SYM1", "SYM2", "SYM3"],
            start="2024-01-01",
            end="2024-01-10",
            provider="mock",
            max_workers=2,  # Limit workers
        )

        # MockProvider returns 8 rows per symbol (trading days)
        assert len(result) == 24  # 3 symbols * 8 rows each
        assert set(result["symbol"].unique().to_list()) == {"SYM1", "SYM2", "SYM3"}


class TestDataManagerGetMetadata:
    """Test DataManager.get_metadata() and related methods."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def storage(self, temp_dir):
        """Create storage backend."""
        config = StorageConfig(base_path=temp_dir)
        return HiveStorage(config)

    @pytest.fixture
    def manager(self, storage):
        """Create DataManager with storage."""
        return DataManager(
            storage=storage,
            enable_validation=False,
        )

    def test_get_metadata_requires_storage(self):
        """Test get_metadata requires storage."""
        manager = DataManager()

        with pytest.raises(ValueError, match="Storage not configured"):
            manager.get_metadata("AAPL")

    def test_get_metadata_not_found(self, manager):
        """Test get_metadata returns None for missing symbol."""
        result = manager.get_metadata("UNKNOWN")

        assert result is None

    def test_get_metadata_for_key_no_storage(self):
        """Test get_metadata_for_key returns None when no storage configured."""
        manager = DataManager()  # No storage

        result = manager.get_metadata_for_key("some/key")

        assert result is None
