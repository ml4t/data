"""Tests for provider_updater module."""

from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock

import polars as pl

from ml4t.data.provider_updater import ProviderUpdater


class MockProviderUpdater(ProviderUpdater):
    """Concrete implementation for testing."""

    def __init__(self, storage, fetch_returns_data=True, transform_identity=True):
        super().__init__("mock_provider", storage)
        self.fetch_returns_data = fetch_returns_data
        self.transform_identity = transform_identity
        self.fetch_call_count = 0
        self.transform_call_count = 0

    def _fetch_data(self, symbol, start_time, end_time, **kwargs):
        """Mock data fetching."""
        self.fetch_call_count += 1
        if not self.fetch_returns_data:
            return pl.DataFrame()

        # Generate mock OHLCV data
        timestamps = []
        current = start_time
        while current < end_time:
            timestamps.append(current)
            current += timedelta(hours=1)

        if not timestamps:
            return pl.DataFrame()

        n = len(timestamps)
        return pl.DataFrame(
            {
                "timestamp": timestamps,
                "open": [100.0 + i for i in range(n)],
                "high": [101.0 + i for i in range(n)],
                "low": [99.0 + i for i in range(n)],
                "close": [100.5 + i for i in range(n)],
                "volume": [1000.0 + i * 100 for i in range(n)],
            }
        )

    def _transform_data(self, data, symbol, **kwargs):
        """Mock data transformation."""
        self.transform_call_count += 1
        if self.transform_identity:
            return data
        return pl.DataFrame()  # Empty after transform

    def _get_default_start_time(self, symbol):
        """Get default start time."""
        return datetime.now() - timedelta(days=30)


class MockStorage:
    """Mock storage backend for testing."""

    def __init__(self, latest_timestamp=None):
        self._latest_timestamp = latest_timestamp
        self.save_chunk_calls = []
        self.update_combined_calls = []
        self.update_metadata_calls = []

    def get_latest_timestamp(self, symbol, provider):
        return self._latest_timestamp

    def save_chunk(self, data, symbol, provider, start_time, end_time):
        self.save_chunk_calls.append(
            {
                "data": data,
                "symbol": symbol,
                "provider": provider,
                "start": start_time,
                "end": end_time,
            }
        )
        # Return mock Path object
        mock_path = MagicMock(spec=Path)
        mock_path.name = f"{symbol}_{start_time.isoformat()}_{end_time.isoformat()}.parquet"
        return mock_path

    def update_combined_file(self, data, symbol, provider):
        self.update_combined_calls.append({"data": data, "symbol": symbol, "provider": provider})
        return len(data)

    def update_metadata(self, symbol, provider, end_time, records_added, chunk_file):
        self.update_metadata_calls.append(
            {
                "symbol": symbol,
                "provider": provider,
                "end_time": end_time,
                "records_added": records_added,
                "chunk_file": chunk_file,
            }
        )


class TestProviderUpdater:
    """Tests for ProviderUpdater class."""

    def test_init(self):
        """Test initialization."""
        storage = MockStorage()
        updater = MockProviderUpdater(storage)

        assert updater.provider_name == "mock_provider"
        assert updater.storage is storage
        assert updater.safety_margin == timedelta(minutes=5)

    def test_init_custom_safety_margin(self):
        """Test initialization with custom safety margin."""
        storage = MockStorage()
        _updater = MockProviderUpdater(storage)  # noqa: F841
        # Override in parent __init__
        parent_updater = ProviderUpdater.__new__(MockProviderUpdater)
        ProviderUpdater.__init__(parent_updater, "test", storage, safety_margin_minutes=10)

        assert parent_updater.safety_margin == timedelta(minutes=10)

    def test_update_symbol_basic(self):
        """Test basic symbol update."""
        storage = MockStorage()
        updater = MockProviderUpdater(storage)

        start = datetime.now() - timedelta(days=1)
        end = datetime.now()

        result = updater.update_symbol("AAPL", start_time=start, end_time=end, incremental=False)

        assert result["symbol"] == "AAPL"
        assert result["success"] is True
        assert result["records_fetched"] > 0
        assert result["records_added"] > 0
        assert updater.fetch_call_count == 1
        assert updater.transform_call_count == 1

    def test_update_symbol_dry_run(self):
        """Test dry run mode doesn't save data."""
        storage = MockStorage()
        updater = MockProviderUpdater(storage)

        start = datetime.now() - timedelta(days=1)
        end = datetime.now()

        result = updater.update_symbol("AAPL", start_time=start, end_time=end, dry_run=True)

        assert result["success"] is True
        assert result["records_added"] > 0  # Shows what would be added
        assert len(storage.save_chunk_calls) == 0  # But nothing saved
        assert len(storage.update_combined_calls) == 0

    def test_update_symbol_incremental_no_existing_data(self):
        """Test incremental update with no existing data."""
        storage = MockStorage(latest_timestamp=None)
        updater = MockProviderUpdater(storage)

        result = updater.update_symbol("AAPL", incremental=True)

        assert result["success"] is True
        # Should use default start time
        assert updater.fetch_call_count == 1

    def test_update_symbol_incremental_with_existing_data(self):
        """Test incremental update with existing data."""
        latest = datetime.now() - timedelta(hours=12)
        storage = MockStorage(latest_timestamp=latest)
        updater = MockProviderUpdater(storage)

        result = updater.update_symbol("AAPL", incremental=True)

        assert result["success"] is True
        assert updater.fetch_call_count == 1

    def test_update_symbol_already_up_to_date(self):
        """Test update when data is already current."""
        # Latest timestamp is very recent
        latest = datetime.now() + timedelta(hours=1)  # In the future
        storage = MockStorage(latest_timestamp=latest)
        updater = MockProviderUpdater(storage)

        result = updater.update_symbol("AAPL", incremental=True)

        assert result["success"] is True
        assert result.get("skip_reason") == "already_up_to_date"
        assert result["records_fetched"] == 0
        assert result["records_added"] == 0

    def test_update_symbol_no_data_fetched(self):
        """Test handling when no data is fetched."""
        storage = MockStorage()
        updater = MockProviderUpdater(storage, fetch_returns_data=False)

        start = datetime.now() - timedelta(days=1)
        end = datetime.now()

        result = updater.update_symbol("AAPL", start_time=start, end_time=end)

        assert result["success"] is True
        assert result["records_fetched"] == 0
        assert result["records_added"] == 0

    def test_update_symbol_empty_after_transform(self):
        """Test handling when transform returns empty data."""
        storage = MockStorage()
        updater = MockProviderUpdater(storage, transform_identity=False)

        start = datetime.now() - timedelta(days=1)
        end = datetime.now()

        result = updater.update_symbol("AAPL", start_time=start, end_time=end)

        assert result["success"] is True
        assert result["records_fetched"] > 0
        assert result["records_added"] == 0

    def test_update_symbol_with_error(self):
        """Test error handling during update."""
        storage = MockStorage()
        updater = MockProviderUpdater(storage)

        # Make _fetch_data raise an exception
        _original_fetch = updater._fetch_data  # noqa: F841
        updater._fetch_data = MagicMock(side_effect=ValueError("API error"))

        start = datetime.now() - timedelta(days=1)
        end = datetime.now()

        result = updater.update_symbol("AAPL", start_time=start, end_time=end)

        assert result["success"] is False
        assert len(result["errors"]) > 0
        assert "API error" in result["errors"][0]

    def test_update_symbol_uppercase(self):
        """Test that symbol is uppercased."""
        storage = MockStorage()
        updater = MockProviderUpdater(storage)

        start = datetime.now() - timedelta(days=1)
        end = datetime.now()

        result = updater.update_symbol("aapl", start_time=start, end_time=end)

        assert result["symbol"] == "AAPL"

    def test_update_symbols_empty_list(self):
        """Test updating empty symbol list."""
        storage = MockStorage()
        updater = MockProviderUpdater(storage)

        result = updater.update_symbols([])

        assert result == {}

    def test_update_symbols_single(self):
        """Test updating single symbol."""
        storage = MockStorage()
        updater = MockProviderUpdater(storage)

        start = datetime.now() - timedelta(days=1)
        end = datetime.now()

        result = updater.update_symbols(["AAPL"], start_time=start, end_time=end)

        assert "AAPL" in result
        assert result["AAPL"]["success"] is True

    def test_update_symbols_multiple(self):
        """Test updating multiple symbols."""
        storage = MockStorage()
        updater = MockProviderUpdater(storage)

        start = datetime.now() - timedelta(days=1)
        end = datetime.now()

        result = updater.update_symbols(["AAPL", "MSFT", "GOOGL"], start_time=start, end_time=end)

        assert len(result) == 3
        assert all(r["success"] for r in result.values())

    def test_update_symbols_concurrent(self):
        """Test concurrent symbol updates."""
        storage = MockStorage()
        updater = MockProviderUpdater(storage)

        start = datetime.now() - timedelta(days=1)
        end = datetime.now()

        # Use max_workers > 1 to test concurrent path
        result = updater.update_symbols(
            ["AAPL", "MSFT", "GOOGL", "AMZN"], max_workers=2, start_time=start, end_time=end
        )

        assert len(result) == 4
        assert all(r["success"] for r in result.values())

    def test_update_symbols_with_failure(self):
        """Test handling failures in multi-symbol update."""
        storage = MockStorage()
        updater = MockProviderUpdater(storage)

        # Make MSFT fail
        original_fetch = updater._fetch_data

        def failing_fetch(symbol, start, end, **kwargs):
            if symbol == "MSFT":
                raise ValueError("MSFT fetch failed")
            return original_fetch(symbol, start, end, **kwargs)

        updater._fetch_data = failing_fetch

        start = datetime.now() - timedelta(days=1)
        end = datetime.now()

        result = updater.update_symbols(
            ["AAPL", "MSFT", "GOOGL"], start_time=start, end_time=end, max_workers=1
        )

        assert result["AAPL"]["success"] is True
        assert result["MSFT"]["success"] is False
        assert result["GOOGL"]["success"] is True

    def test_initialize_stats(self):
        """Test statistics initialization."""
        storage = MockStorage()
        updater = MockProviderUpdater(storage)

        stats = updater._initialize_stats("AAPL")

        assert stats["symbol"] == "AAPL"
        assert stats["provider"] == "mock_provider"
        assert stats["records_fetched"] == 0
        assert stats["records_added"] == 0
        assert stats["errors"] == []
        assert stats["success"] is False

    def test_get_time_range_explicit(self):
        """Test time range with explicit start/end."""
        storage = MockStorage()
        updater = MockProviderUpdater(storage)

        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 2)

        result_start, result_end = updater._get_time_range("AAPL", start, end, incremental=False)

        assert result_start == start
        assert result_end == end

    def test_get_time_range_incremental_with_latest(self):
        """Test time range determination for incremental update."""
        latest = datetime(2024, 1, 1, 12, 0)
        storage = MockStorage(latest_timestamp=latest)
        updater = MockProviderUpdater(storage)

        result_start, result_end = updater._get_time_range("AAPL", None, None, incremental=True)

        # Start should be after latest - safety_margin + 1 minute
        assert result_start is not None
        assert result_start > latest - updater.safety_margin

    def test_get_time_range_no_update_needed(self):
        """Test time range when already up to date."""
        storage = MockStorage()
        updater = MockProviderUpdater(storage)

        # Start is after end
        start = datetime(2024, 1, 2)
        end = datetime(2024, 1, 1)

        result_start, result_end = updater._get_time_range("AAPL", start, end, incremental=False)

        assert result_start is None
        assert result_end is None

    def test_save_data_empty(self):
        """Test saving empty data."""
        storage = MockStorage()
        updater = MockProviderUpdater(storage)

        result = updater._save_data(pl.DataFrame(), "AAPL", datetime.now(), datetime.now())

        assert result == 0
        assert len(storage.save_chunk_calls) == 0

    def test_save_data_with_records(self):
        """Test saving data with records."""
        storage = MockStorage()
        updater = MockProviderUpdater(storage)

        data = pl.DataFrame(
            {
                "timestamp": [datetime.now()],
                "close": [100.0],
            }
        )
        start = datetime.now() - timedelta(hours=1)
        end = datetime.now()

        result = updater._save_data(data, "AAPL", start, end)

        assert result == 1
        assert len(storage.save_chunk_calls) == 1
        assert len(storage.update_combined_calls) == 1
        assert len(storage.update_metadata_calls) == 1


class TestProviderUpdaterIntegration:
    """Integration tests for ProviderUpdater."""

    def test_full_update_workflow(self):
        """Test complete update workflow."""
        storage = MockStorage()
        updater = MockProviderUpdater(storage)

        # Full workflow: fetch -> transform -> save
        start = datetime.now() - timedelta(hours=24)
        end = datetime.now()

        result = updater.update_symbol("AAPL", start_time=start, end_time=end, incremental=False)

        assert result["success"] is True
        assert result["records_fetched"] > 0
        assert result["records_added"] > 0

        # Verify storage was called correctly
        assert len(storage.save_chunk_calls) == 1
        assert storage.save_chunk_calls[0]["symbol"] == "AAPL"
        assert storage.save_chunk_calls[0]["provider"] == "mock_provider"

        assert len(storage.update_combined_calls) == 1
        assert len(storage.update_metadata_calls) == 1

    def test_incremental_workflow(self):
        """Test incremental update workflow."""
        # First update
        storage1 = MockStorage(latest_timestamp=None)
        updater1 = MockProviderUpdater(storage1)

        result1 = updater1.update_symbol("AAPL", incremental=True)
        assert result1["success"] is True

        # Second incremental update (with existing data)
        latest = datetime.now() - timedelta(hours=6)
        storage2 = MockStorage(latest_timestamp=latest)
        updater2 = MockProviderUpdater(storage2)

        result2 = updater2.update_symbol("AAPL", incremental=True)
        assert result2["success"] is True

    def test_batch_update_workflow(self):
        """Test batch update of multiple symbols."""
        storage = MockStorage()
        updater = MockProviderUpdater(storage)

        symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]
        start = datetime.now() - timedelta(days=1)
        end = datetime.now()

        results = updater.update_symbols(symbols, max_workers=2, start_time=start, end_time=end)

        successful = sum(1 for r in results.values() if r["success"])
        assert successful == len(symbols)
