"""Integration tests for batch_load_from_storage with Universe.

Tests the integration between batch_load_from_storage and Universe loading.
"""

from datetime import UTC, datetime, timedelta

import polars as pl
import pytest

from ml4t.data.data_manager import DataManager
from ml4t.data.storage.backend import StorageConfig
from ml4t.data.storage.hive import HiveStorage


@pytest.fixture
def storage_with_sp10_data(tmp_path):
    """Create storage with 10 S&P 500 symbols for testing."""
    # Create storage
    config = StorageConfig(base_path=tmp_path / "test_storage")
    storage = HiveStorage(config)

    # Create DataManager with storage
    manager = DataManager(storage=storage, output_format="polars")

    # First 10 S&P 500 symbols (alphabetically)
    symbols = ["A", "AAL", "AAPL", "ABBV", "ABC", "ABMD", "ABT", "ACN", "ADBE", "ADI"]

    # Generate test data (1 year of daily data)
    start_date = datetime(2024, 1, 1, tzinfo=UTC)
    dates = [start_date + timedelta(days=i) for i in range(252)]  # Trading days

    for symbol in symbols:
        # Create OHLCV data
        df = pl.DataFrame(
            {
                "timestamp": dates,
                "open": [100.0 + i * 0.1 for i in range(252)],
                "high": [101.0 + i * 0.1 for i in range(252)],
                "low": [99.0 + i * 0.1 for i in range(252)],
                "close": [100.5 + i * 0.1 for i in range(252)],
                "volume": [1_000_000.0 for _ in range(252)],
            }
        )

        # Store data
        key = f"equities/daily/{symbol}"
        storage.write(df, key)

    return manager, symbols


def test_batch_load_from_storage_with_symbol_list(storage_with_sp10_data):
    """Test loading symbols from a list (like Universe would provide)."""
    manager, symbols = storage_with_sp10_data

    # Simulate what Universe.get() returns
    symbol_list = symbols[:5]  # First 5 symbols

    df = manager.batch_load_from_storage(
        symbols=symbol_list,
        start="2024-01-01",
        end="2024-12-31",
        frequency="daily",
        asset_class="equities",
        fetch_missing=False,
    )

    # Verify results
    assert not df.is_empty()
    assert len(df) == 5 * 252  # 5 symbols * 252 trading days
    assert df["symbol"].n_unique() == 5
    assert set(df["symbol"].unique().to_list()) == set(symbol_list)


def test_batch_load_storage_performance_realistic(storage_with_sp10_data):
    """Test realistic performance with partial date range."""
    import time

    manager, symbols = storage_with_sp10_data

    # Load just Q1 2024 data
    start_time = time.perf_counter()

    df = manager.batch_load_from_storage(
        symbols=symbols,  # All 10 symbols
        start="2024-01-01",
        end="2024-03-31",
        frequency="daily",
        asset_class="equities",
        fetch_missing=False,
    )

    elapsed = time.perf_counter() - start_time

    # Verify
    assert not df.is_empty()
    assert df["symbol"].n_unique() == 10

    # Performance check - should be very fast for small date range
    print(f"\n✓ Loaded 10 symbols (Q1 2024) in {elapsed:.3f}s")
    assert elapsed < 0.5, f"Should load quickly, took {elapsed:.3f}s"


def test_batch_load_empty_result_handling(tmp_path):
    """Test handling when date range returns no data."""
    # Create storage
    config = StorageConfig(base_path=tmp_path / "test_storage")
    storage = HiveStorage(config)
    manager = DataManager(storage=storage)

    # Create data for 2023
    dates_2023 = [datetime(2023, 1, 1, tzinfo=UTC) + timedelta(days=i) for i in range(100)]

    df = pl.DataFrame(
        {
            "timestamp": dates_2023,
            "open": [100.0 for _ in range(100)],
            "high": [101.0 for _ in range(100)],
            "low": [99.0 for _ in range(100)],
            "close": [100.5 for _ in range(100)],
            "volume": [1_000_000.0 for _ in range(100)],
        }
    )

    storage.write(df, "equities/daily/TEST")

    # Request data for 2024 (should fail with empty_data error)
    with pytest.raises(ValueError, match="No data loaded"):
        manager.batch_load_from_storage(
            symbols=["TEST"],
            start="2024-01-01",
            end="2024-12-31",
            fetch_missing=False,
        )
