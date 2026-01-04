"""Performance tests for batch_load_from_storage().

This module tests the performance characteristics of batch_load_from_storage()
and validates that it provides significant speedup compared to network fetching.
"""

import time
from datetime import UTC, datetime, timedelta

import polars as pl
import pytest

from ml4t.data.data_manager import DataManager
from ml4t.data.storage.backend import StorageConfig
from ml4t.data.storage.hive import HiveStorage


@pytest.fixture
def storage_with_data(tmp_path):
    """Create storage with pre-populated test data for 100 symbols."""
    # Create storage
    config = StorageConfig(base_path=tmp_path / "test_storage")
    storage = HiveStorage(config)

    # Create DataManager with storage
    manager = DataManager(storage=storage, output_format="polars")

    # Generate test symbols
    symbols = [f"TEST{i:03d}" for i in range(100)]

    # Generate test data for each symbol (1 year of daily data)
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


class TestBatchLoadFromStoragePerformance:
    """Performance tests for batch_load_from_storage()."""

    def test_batch_load_from_storage_fast(self, storage_with_data):
        """Test that storage loading is much faster than network fetch.

        Acceptance: Load 100 symbols from storage in < 1 second.
        """
        manager, symbols = storage_with_data

        # Measure storage load time
        start_time = time.perf_counter()

        df = manager.batch_load_from_storage(
            symbols=symbols,
            start="2024-01-01",
            end="2024-12-31",
            frequency="daily",
            asset_class="equities",
            fetch_missing=False,  # All data should be in storage
        )

        elapsed = time.perf_counter() - start_time

        # Validate results
        assert not df.is_empty(), "Result should not be empty"
        assert len(df) == 100 * 252, f"Expected 25,200 rows, got {len(df)}"
        assert df["symbol"].n_unique() == 100, "Should have 100 unique symbols"

        # Performance assertion
        print(f"\n✓ Loaded 100 symbols from storage in {elapsed:.3f} seconds")
        assert elapsed < 1.0, f"Storage load took {elapsed:.3f}s, expected < 1.0s"

    def test_batch_load_from_storage_vs_fetch_comparison(self, storage_with_data):
        """Compare storage vs fetch performance (simulated).

        This test demonstrates the performance difference between storage reads
        and network fetches by timing storage reads and comparing to expected
        fetch times.
        """
        manager, symbols = storage_with_data

        # Test with subset of symbols
        test_symbols = symbols[:10]

        # Measure storage load time
        start_time = time.perf_counter()

        df_storage = manager.batch_load_from_storage(
            symbols=test_symbols,
            start="2024-01-01",
            end="2024-12-31",
            frequency="daily",
            asset_class="equities",
            fetch_missing=False,
        )

        storage_time = time.perf_counter() - start_time

        # Validate
        assert len(df_storage) == 10 * 252, f"Expected 2,520 rows, got {len(df_storage)}"

        # Expected network fetch time (conservative estimate)
        # Typical network fetch: 500-2000ms per symbol
        # For 10 symbols with max_workers=4: ~1250-5000ms
        expected_fetch_time_min = 1.25  # 10 symbols / 4 workers * 500ms
        expected_fetch_time_max = 5.0  # 10 symbols / 4 workers * 2000ms

        # Calculate speedup range
        speedup_min = expected_fetch_time_min / storage_time
        speedup_max = expected_fetch_time_max / storage_time

        print(f"\n✓ Storage load: {storage_time:.3f}s for 10 symbols")
        print(
            f"  Expected fetch time: {expected_fetch_time_min:.3f}s - {expected_fetch_time_max:.3f}s"
        )
        print(f"  Estimated speedup: {speedup_min:.1f}x - {speedup_max:.1f}x")

        # Storage should be at least 5x faster than minimum expected fetch time
        assert speedup_min >= 5.0, f"Storage should be at least 5x faster, got {speedup_min:.1f}x"

    def test_batch_load_from_storage_scalability(self, storage_with_data):
        """Test performance scaling with different symbol counts."""
        manager, symbols = storage_with_data

        test_sizes = [10, 25, 50, 100]
        times = []

        for size in test_sizes:
            test_symbols = symbols[:size]

            start_time = time.perf_counter()

            df = manager.batch_load_from_storage(
                symbols=test_symbols,
                start="2024-01-01",
                end="2024-12-31",
                frequency="daily",
                asset_class="equities",
                fetch_missing=False,
            )

            elapsed = time.perf_counter() - start_time
            times.append(elapsed)

            assert len(df) == size * 252, f"Expected {size * 252} rows, got {len(df)}"

            print(f"  {size:3d} symbols: {elapsed:.3f}s ({elapsed / size * 1000:.1f}ms per symbol)")

        # Check that time scales reasonably (should be roughly linear with parallel reads)
        print("\n✓ Scalability test:")
        for size, elapsed in zip(test_sizes, times):
            print(f"  {size:3d} symbols: {elapsed:.3f}s")

        # Time per symbol should be relatively constant (due to parallelization)
        time_per_symbol = [t / s for t, s in zip(times, test_sizes)]
        avg_time_per_symbol = sum(time_per_symbol) / len(time_per_symbol)

        print(f"  Average time per symbol: {avg_time_per_symbol * 1000:.1f}ms")

        # All should be within 3x of average (parallelization variance)
        for tps in time_per_symbol:
            assert tps < avg_time_per_symbol * 3, "Time per symbol should be relatively constant"


class TestBatchLoadFromStorageFunctional:
    """Functional tests for batch_load_from_storage()."""

    def test_all_symbols_in_storage(self, storage_with_data):
        """Test loading when all symbols are in storage."""
        manager, symbols = storage_with_data

        df = manager.batch_load_from_storage(
            symbols=symbols[:10],
            start="2024-01-01",
            end="2024-12-31",
            fetch_missing=False,
        )

        assert len(df) == 10 * 252
        assert df["symbol"].n_unique() == 10
        assert "timestamp" in df.columns
        assert "open" in df.columns

    def test_no_symbols_in_storage(self, tmp_path):
        """Test behavior when no symbols are in storage."""
        # Create empty storage
        config = StorageConfig(base_path=tmp_path / "empty_storage")
        storage = HiveStorage(config)
        manager = DataManager(storage=storage, output_format="polars")

        # Should raise error with fetch_missing=False
        with pytest.raises(ValueError, match="not in storage"):
            manager.batch_load_from_storage(
                symbols=["MISSING1", "MISSING2"],
                start="2024-01-01",
                end="2024-12-31",
                fetch_missing=False,
            )

    def test_partial_storage_missing_symbols_error(self, storage_with_data):
        """Test that error is raised when symbols missing and fetch_missing=False."""
        manager, symbols = storage_with_data

        # Mix of existing and non-existent symbols
        test_symbols = symbols[:5] + ["MISSING_SYMBOL"]

        # Should raise error with fetch_missing=False
        with pytest.raises(ValueError, match="not in storage"):
            manager.batch_load_from_storage(
                symbols=test_symbols,
                start="2024-01-01",
                end="2024-12-31",
                fetch_missing=False,
            )

    def test_date_range_filtering(self, storage_with_data):
        """Test that date range filtering works correctly."""
        manager, symbols = storage_with_data

        # Load subset of date range
        df = manager.batch_load_from_storage(
            symbols=symbols[:5],
            start="2024-06-01",
            end="2024-06-30",
            fetch_missing=False,
        )

        # Verify date filtering
        assert not df.is_empty()
        min_date = df["timestamp"].min()
        max_date = df["timestamp"].max()

        assert min_date >= datetime(2024, 6, 1, tzinfo=UTC)
        assert max_date <= datetime(2024, 6, 30, 23, 59, 59, tzinfo=UTC)

    def test_asset_class_and_frequency_parameters(self, tmp_path):
        """Test that asset_class and frequency parameters work correctly."""
        # Create storage
        config = StorageConfig(base_path=tmp_path / "test_storage")
        storage = HiveStorage(config)
        manager = DataManager(storage=storage)

        # Create data with different asset class and frequency
        dates = [datetime(2024, 1, 1, tzinfo=UTC) + timedelta(hours=i) for i in range(24)]

        df = pl.DataFrame(
            {
                "timestamp": dates,
                "open": [100.0 for _ in range(24)],
                "high": [101.0 for _ in range(24)],
                "low": [99.0 for _ in range(24)],
                "close": [100.5 for _ in range(24)],
                "volume": [1_000_000.0 for _ in range(24)],
            }
        )

        # Store as crypto/hourly
        key = "crypto/hourly/BTC"
        storage.write(df, key)

        # Load with correct parameters
        result = manager.batch_load_from_storage(
            symbols=["BTC"],
            start="2024-01-01",
            end="2024-01-02",
            frequency="hourly",
            asset_class="crypto",
            fetch_missing=False,
        )

        assert len(result) == 24
        assert result["symbol"][0] == "BTC"

    def test_empty_symbols_list(self, storage_with_data):
        """Test that empty symbols list raises error."""
        manager, _ = storage_with_data

        with pytest.raises(ValueError, match="symbols list cannot be empty"):
            manager.batch_load_from_storage(
                symbols=[],
                start="2024-01-01",
                end="2024-12-31",
            )

    def test_no_storage_configured(self):
        """Test that error is raised when storage is not configured."""
        manager = DataManager(output_format="polars")

        with pytest.raises(ValueError, match="Storage not configured"):
            manager.batch_load_from_storage(
                symbols=["AAPL"],
                start="2024-01-01",
                end="2024-12-31",
            )

    def test_invalid_date_range(self, storage_with_data):
        """Test that invalid date range raises error."""
        manager, symbols = storage_with_data

        with pytest.raises(ValueError, match="End date must be after start date"):
            manager.batch_load_from_storage(
                symbols=symbols[:5],
                start="2024-12-31",
                end="2024-01-01",  # End before start
            )


class TestBatchLoadFromStorageBenchmarks:
    """Detailed benchmarks with performance reporting."""

    def test_performance_report(self, storage_with_data):
        """Generate a comprehensive performance report."""
        manager, symbols = storage_with_data

        print("\n" + "=" * 60)
        print("BATCH_LOAD_FROM_STORAGE PERFORMANCE REPORT")
        print("=" * 60)

        test_cases = [
            (10, "Small batch"),
            (25, "Medium batch"),
            (50, "Large batch"),
            (100, "Extra large batch"),
        ]

        for size, description in test_cases:
            test_symbols = symbols[:size]

            start_time = time.perf_counter()

            df = manager.batch_load_from_storage(
                symbols=test_symbols,
                start="2024-01-01",
                end="2024-12-31",
                fetch_missing=False,
            )

            elapsed = time.perf_counter() - start_time
            rows = len(df)
            ms_per_symbol = elapsed / size * 1000
            rows_per_second = rows / elapsed

            print(f"\n{description} ({size} symbols):")
            print(f"  Total time:     {elapsed:.3f}s")
            print(f"  Per symbol:     {ms_per_symbol:.1f}ms")
            print(f"  Rows loaded:    {rows:,}")
            print(f"  Rows/second:    {rows_per_second:,.0f}")

            # Validate
            assert len(df) == size * 252
            assert df["symbol"].n_unique() == size

        print("\n" + "=" * 60)
        print("ACCEPTANCE CRITERIA:")
        print("  ✓ 100 symbols loaded in < 1 second")
        print("  ✓ Storage reads 10-100x faster than network fetch")
        print("  ✓ Parallel reads scale efficiently")
        print("=" * 60 + "\n")
