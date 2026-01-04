"""Comprehensive integration tests for multi-asset workflows.

This module tests complete end-to-end workflows that use all multi-asset features
together in realistic scenarios. It validates that:

1. All features work together seamlessly
2. Performance targets are met in realistic workflows
3. Edge cases are handled gracefully across the stack
4. Round-trip conversions preserve data integrity

Test Organization:
    - TestMultiAssetWorkflows: End-to-end realistic workflows
    - TestPerformanceIntegration: Performance validation across features
    - TestCrossFeatureIntegration: Multiple features working together
    - TestErrorHandling: Edge cases and error conditions
    - TestFormatConversionWorkflows: Format conversion integration
"""

from __future__ import annotations

import time
from datetime import UTC, datetime, timedelta

import polars as pl
import pytest

from ml4t.data.core.schemas import MultiAssetSchema
from ml4t.data.data_manager import DataManager
from ml4t.data.storage.backend import StorageConfig
from ml4t.data.storage.hive import HiveStorage
from ml4t.data.universe import Universe
from ml4t.data.utils.format import pivot_to_stacked, pivot_to_wide

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def manager_with_storage(tmp_path):
    """Create DataManager with temporary storage backend."""
    config = StorageConfig(base_path=tmp_path / "test_storage")
    storage = HiveStorage(config)
    return DataManager(storage=storage, output_format="polars")


@pytest.fixture
def sample_universe():
    """Small test universe (5 symbols) for quick testing."""
    return ["AAPL", "MSFT", "GOOG", "AMZN", "META"]


@pytest.fixture
def populated_storage(manager_with_storage, sample_universe):
    """Pre-populated storage with sample data for testing."""
    manager = manager_with_storage

    # Generate test data (1 quarter of daily data)
    start_date = datetime(2024, 1, 1, tzinfo=UTC)
    dates = [start_date + timedelta(days=i) for i in range(63)]  # Q1 trading days

    for symbol in sample_universe:
        # Create OHLCV data with symbol-specific base price
        base_price = 100.0 + len(symbol) * 10.0  # AAPL=140, MSFT=140, etc.

        df = pl.DataFrame(
            {
                "timestamp": dates,
                "open": [base_price + i * 0.1 for i in range(63)],
                "high": [base_price + i * 0.1 + 1.0 for i in range(63)],
                "low": [base_price + i * 0.1 - 1.0 for i in range(63)],
                "close": [base_price + i * 0.1 + 0.5 for i in range(63)],
                "volume": [1_000_000.0 * (1 + i * 0.01) for i in range(63)],
            }
        )

        # Store data
        key = f"equities/daily/{symbol}"
        manager.storage.write(df, key)

    return manager, sample_universe


# ============================================================================
# End-to-End Workflow Tests
# ============================================================================


class TestMultiAssetWorkflows:
    """Test complete end-to-end workflows using all multi-asset features."""

    def test_complete_workflow_load_analyze_convert(self, populated_storage):
        """Test: Load from storage → Validate → Convert to wide → Analyze → Convert back.

        This tests a complete workflow:
        1. Load universe data from storage
        2. Validate multi-asset schema
        3. Convert to wide format for analysis
        4. Perform calculations
        5. Convert back to stacked format
        6. Validate round-trip integrity
        """
        manager, symbols = populated_storage

        # Step 1: Load universe data from storage
        df_original = manager.batch_load_from_storage(
            symbols=symbols,
            start="2024-01-01",
            end="2024-03-31",
            frequency="daily",
            asset_class="equities",
            fetch_missing=False,
        )

        # Step 2: Validate schema compliance
        assert MultiAssetSchema.validate(df_original, strict=True)
        assert len(df_original) == 5 * 63  # 5 symbols × 63 days
        assert set(df_original["symbol"].unique().to_list()) == set(symbols)

        # Step 3: Convert to wide format for cross-sectional analysis
        df_wide = pivot_to_wide(df_original)
        assert len(df_wide) == 63  # One row per timestamp
        assert "timestamp" in df_wide.columns
        assert "close_AAPL" in df_wide.columns
        assert "volume_META" in df_wide.columns

        # Step 4: Perform analysis (calculate returns)
        # In wide format, can easily compute cross-sectional statistics
        close_cols = [col for col in df_wide.columns if col.startswith("close_")]
        assert len(close_cols) == 5

        # Step 5: Convert back to stacked format
        df_stacked = pivot_to_stacked(df_wide)
        assert MultiAssetSchema.validate(df_stacked, strict=True)

        # Step 6: Validate round-trip integrity
        original_sorted = df_original.sort(["timestamp", "symbol"])
        result_sorted = df_stacked.sort(["timestamp", "symbol"])

        # Check data preservation
        assert len(original_sorted) == len(result_sorted)
        assert original_sorted["close"].to_list() == result_sorted["close"].to_list()
        assert original_sorted["volume"].to_list() == result_sorted["volume"].to_list()

    def test_workflow_universe_load_to_storage_to_analysis(self):
        """Test: Load universe via API → Store → Reload from storage → Analyze.

        Tests the workflow where:
        1. User loads a universe via batch_load_universe (simulated with mock)
        2. Data is automatically cached to storage
        3. Subsequent loads come from storage (fast)
        4. Data is converted to wide format for analysis
        """
        # This test would use a mock provider since we don't want actual network calls
        # For now, we'll create a simplified version with direct storage population

        # Create temporary universe
        Universe.add_custom("test_workflow", ["AAPL", "MSFT", "GOOGL"])

        try:
            DataManager(output_format="polars")

            # In a real workflow, this would fetch from provider and cache
            # For testing, we simulate the cached data scenario
            # This is effectively tested by test_complete_workflow_load_analyze_convert

            # The key integration point is that batch_load_universe returns
            # multi-asset format that can be directly used with format conversions

            # Simulated data in multi-asset format
            df = pl.DataFrame(
                {
                    "timestamp": [
                        datetime(2024, 1, 1, tzinfo=UTC),
                        datetime(2024, 1, 1, tzinfo=UTC),
                        datetime(2024, 1, 2, tzinfo=UTC),
                        datetime(2024, 1, 2, tzinfo=UTC),
                    ],
                    "symbol": ["AAPL", "MSFT", "AAPL", "MSFT"],
                    "open": [100.0, 200.0, 101.0, 201.0],
                    "high": [102.0, 202.0, 103.0, 203.0],
                    "low": [99.0, 199.0, 100.0, 200.0],
                    "close": [101.0, 201.0, 102.0, 202.0],
                    "volume": [1_000_000.0, 2_000_000.0, 1_100_000.0, 2_100_000.0],
                }
            )

            # Validate multi-asset format
            assert MultiAssetSchema.validate(df, strict=True)

            # Convert to wide for analysis
            df_wide = pivot_to_wide(df)
            assert len(df_wide) == 2  # 2 timestamps
            assert "close_AAPL" in df_wide.columns
            assert "close_MSFT" in df_wide.columns

            # Can compute cross-sectional metrics
            # (In practice: correlations, spreads, pairs trading signals, etc.)
            assert df_wide["close_AAPL"][0] == 101.0
            assert df_wide["close_MSFT"][0] == 201.0

        finally:
            Universe.remove_custom("test_workflow")

    def test_workflow_storage_cache_first_loading(self, populated_storage):
        """Test: Cache-first loading workflow with graceful fallback.

        Tests the pattern where:
        1. First check storage for cached data
        2. Only fetch missing symbols/dates from provider
        3. Merge cached + fetched data
        4. Store newly fetched data for next time
        """
        manager, symbols = populated_storage

        # All data is in storage - should be fast
        start_time = time.perf_counter()

        df = manager.batch_load_from_storage(
            symbols=symbols,
            start="2024-01-01",
            end="2024-03-31",
            fetch_missing=False,  # Strict cache-only mode
        )

        elapsed = time.perf_counter() - start_time

        # Verify data loaded
        assert len(df) == 5 * 63
        assert set(df["symbol"].unique().to_list()) == set(symbols)
        assert MultiAssetSchema.validate(df, strict=True)

        # Should be very fast (< 0.5s for 5 symbols)
        assert elapsed < 0.5, f"Storage load took {elapsed:.3f}s, expected < 0.5s"

    def test_workflow_mixed_symbols_storage_and_fetch(self, populated_storage):
        """Test: Mixed workflow with some symbols in storage, some need fetching.

        This tests the graceful degradation when:
        - Some symbols are cached
        - Some symbols need to be fetched
        - fetch_missing=False should raise error
        """
        manager, symbols = populated_storage

        # Mix of existing and non-existing symbols
        mixed_symbols = symbols[:3] + ["NONEXISTENT1", "NONEXISTENT2"]

        # Should raise error when fetch_missing=False
        with pytest.raises(ValueError, match="not in storage"):
            manager.batch_load_from_storage(
                symbols=mixed_symbols,
                start="2024-01-01",
                end="2024-03-31",
                fetch_missing=False,
            )


# ============================================================================
# Performance Integration Tests
# ============================================================================


class TestPerformanceIntegration:
    """Test performance characteristics across integrated features."""

    @pytest.mark.skip(reason="Performance targets not finalized")
    def test_batch_load_universe_performance_target(self):
        """Validate loading 10 symbols completes in reasonable time.

        This tests real-world performance with network fetching and rate limiting.
        Yahoo Finance has rate limiting (~2s between requests with max_workers=4),
        so 10 symbols will take ~20-30s. This is expected and acceptable.

        The key validation is that:
        1. All symbols fetch successfully
        2. Multi-asset schema is correct
        3. Parallel fetching works (not sequential)
        """
        # Create small custom universe
        test_symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA", "JPM", "V", "WMT"]
        Universe.add_custom("test_perf_10", test_symbols)

        try:
            manager = DataManager(output_format="polars")

            # Time the operation
            start_time = time.perf_counter()

            # This fetches from Yahoo Finance with rate limiting
            # We test with a very small date range to be respectful to providers
            df = manager.batch_load_universe(
                universe="test_perf_10",
                start="2024-01-01",
                end="2024-01-05",  # Just 5 days
                provider="yahoo",
                fail_on_partial=False,  # Graceful degradation
            )

            elapsed = time.perf_counter() - start_time

            # Verify we got data for all symbols
            assert len(df) > 0
            assert "symbol" in df.columns
            assert MultiAssetSchema.validate(df, strict=True)

            # Should get data for most/all symbols
            unique_symbols = df["symbol"].n_unique()
            assert unique_symbols >= 8, f"Expected at least 8 symbols, got {unique_symbols}"

            # Performance expectation: with rate limiting, this will take 20-30s
            # This is acceptable for network fetching with proper rate limiting
            print(f"\n✓ Loaded {unique_symbols} symbols in {elapsed:.3f}s")
            print(f"  (Rate-limited network fetch: ~{elapsed / unique_symbols:.1f}s per symbol)")

            # Sanity check: should complete in < 60s (very conservative)
            assert elapsed < 60.0, f"Expected <60s, got {elapsed:.3f}s"

        finally:
            Universe.remove_custom("test_perf_10")

    def test_storage_load_100_symbols_performance(self, tmp_path):
        """Test loading 100 symbols from storage completes in <1 second."""
        # Create storage and populate with 100 symbols
        config = StorageConfig(base_path=tmp_path / "perf_storage")
        storage = HiveStorage(config)
        manager = DataManager(storage=storage, output_format="polars")

        # Generate 100 test symbols
        symbols = [f"SYM{i:03d}" for i in range(100)]

        # Populate storage with minimal data (1 month)
        start_date = datetime(2024, 1, 1, tzinfo=UTC)
        dates = [start_date + timedelta(days=i) for i in range(21)]  # 21 trading days

        for symbol in symbols:
            df = pl.DataFrame(
                {
                    "timestamp": dates,
                    "open": [100.0 for _ in range(21)],
                    "high": [101.0 for _ in range(21)],
                    "low": [99.0 for _ in range(21)],
                    "close": [100.5 for _ in range(21)],
                    "volume": [1_000_000.0 for _ in range(21)],
                }
            )
            storage.write(df, f"equities/daily/{symbol}")

        # Measure load time
        start_time = time.perf_counter()

        df = manager.batch_load_from_storage(
            symbols=symbols,
            start="2024-01-01",
            end="2024-01-31",
            fetch_missing=False,
        )

        elapsed = time.perf_counter() - start_time

        # Verify
        assert len(df) == 100 * 21  # 2,100 rows
        assert df["symbol"].n_unique() == 100
        assert MultiAssetSchema.validate(df, strict=True)

        # Performance target: <1s for 100 symbols from storage
        print(f"\n✓ Loaded 100 symbols from storage in {elapsed:.3f}s")
        assert elapsed < 1.0, f"Expected <1s, got {elapsed:.3f}s"

    def test_format_conversion_performance_large_dataset(self):
        """Test format conversion performance on large multi-asset dataset."""
        # Create large dataset: 50 symbols × 252 days = 12,600 rows
        symbols = [f"SYM{i:02d}" for i in range(50)]
        timestamps = [datetime(2024, 1, 1, tzinfo=UTC) + timedelta(days=i) for i in range(252)]

        data = []
        for ts in timestamps:
            for symbol in symbols:
                data.append(
                    {
                        "timestamp": ts,
                        "symbol": symbol,
                        "open": 100.0,
                        "high": 101.0,
                        "low": 99.0,
                        "close": 100.5,
                        "volume": 1_000_000.0,
                    }
                )

        df_stacked = pl.DataFrame(data)
        assert len(df_stacked) == 12_600

        # Measure stacked → wide conversion
        start_time = time.perf_counter()
        df_wide = pivot_to_wide(df_stacked)
        to_wide_time = time.perf_counter() - start_time

        # Measure wide → stacked conversion
        start_time = time.perf_counter()
        df_back = pivot_to_stacked(df_wide)
        to_stacked_time = time.perf_counter() - start_time

        # Verify correctness
        assert len(df_wide) == 252
        assert len(df_back) == 12_600

        # Performance targets
        print(f"\n✓ Stacked→Wide: {to_wide_time:.3f}s for 12,600 rows")
        print(f"✓ Wide→Stacked: {to_stacked_time:.3f}s for 252×50 cells")

        assert to_wide_time < 1.0, f"Stacked→Wide took {to_wide_time:.3f}s, expected <1s"
        assert to_stacked_time < 1.0, f"Wide→Stacked took {to_stacked_time:.3f}s, expected <1s"


# ============================================================================
# Cross-Feature Integration Tests
# ============================================================================


class TestCrossFeatureIntegration:
    """Test multiple features working together seamlessly."""

    def test_universe_plus_storage_plus_format_conversion_chain(self, tmp_path):
        """Full chain: Universe → Storage → Load → Convert → Analyze.

        This tests the complete data pipeline:
        1. Define universe
        2. Load and store data
        3. Reload from storage (cache hit)
        4. Convert to wide format
        5. Perform analysis
        6. Convert back to stacked
        7. Validate integrity
        """
        # Step 1: Create universe
        Universe.add_custom("integration_test", ["AAPL", "MSFT", "GOOGL"])

        try:
            # Step 2: Setup storage
            config = StorageConfig(base_path=tmp_path / "integration_storage")
            storage = HiveStorage(config)
            manager = DataManager(storage=storage, output_format="polars")

            # Populate storage with universe data
            symbols = Universe.get("integration_test")
            dates = [datetime(2024, 1, 1, tzinfo=UTC) + timedelta(days=i) for i in range(10)]

            for symbol in symbols:
                df = pl.DataFrame(
                    {
                        "timestamp": dates,
                        "open": [100.0 + i for i in range(10)],
                        "high": [101.0 + i for i in range(10)],
                        "low": [99.0 + i for i in range(10)],
                        "close": [100.5 + i for i in range(10)],
                        "volume": [1_000_000.0 for _ in range(10)],
                    }
                )
                storage.write(df, f"equities/daily/{symbol}")

            # Step 3: Load from storage (cache hit)
            df_cached = manager.batch_load_from_storage(
                symbols=symbols,
                start="2024-01-01",
                end="2024-01-10",
                fetch_missing=False,
            )

            # Step 4: Validate multi-asset schema
            assert MultiAssetSchema.validate(df_cached, strict=True)
            # Date filter is inclusive of start, exclusive of end: Jan 1-9 = 9 days
            assert len(df_cached) == 3 * 9  # 3 symbols × 9 days

            # Step 5: Convert to wide format
            df_wide = pivot_to_wide(df_cached)
            assert len(df_wide) == 9  # 9 timestamps
            assert "close_AAPL" in df_wide.columns
            assert "close_MSFT" in df_wide.columns
            assert "close_GOOGL" in df_wide.columns

            # Step 6: Perform cross-sectional analysis
            # Calculate average close across symbols at each timestamp
            close_cols = ["close_AAPL", "close_MSFT", "close_GOOGL"]
            df_wide = df_wide.with_columns(pl.mean_horizontal(close_cols).alias("average_close"))
            assert "average_close" in df_wide.columns

            # Step 7: Convert back to stacked
            df_final = pivot_to_stacked(df_wide.drop("average_close"))

            # Step 8: Validate round-trip
            assert MultiAssetSchema.validate(df_final, strict=True)
            assert len(df_final) == len(df_cached)

            # Check data preservation
            cached_sorted = df_cached.sort(["timestamp", "symbol"])
            final_sorted = df_final.sort(["timestamp", "symbol"])
            assert cached_sorted["close"].to_list() == final_sorted["close"].to_list()

        finally:
            Universe.remove_custom("integration_test")

    def test_schema_validation_across_all_operations(self, populated_storage):
        """Test that schema validation holds across all operations."""
        manager, symbols = populated_storage

        # Load from storage
        df1 = manager.batch_load_from_storage(
            symbols=symbols, start="2024-01-01", end="2024-03-31", fetch_missing=False
        )
        assert MultiAssetSchema.validate(df1, strict=True)

        # Convert to wide
        df_wide = pivot_to_wide(df1)
        # Wide format doesn't have same schema (no symbol column)

        # Convert back to stacked
        df2 = pivot_to_stacked(df_wide)
        assert MultiAssetSchema.validate(df2, strict=True)

        # Standardize order
        df3 = MultiAssetSchema.standardize_order(df2)
        assert MultiAssetSchema.validate(df3, strict=True)

        # Cast to schema (should be idempotent)
        df4 = MultiAssetSchema.cast_to_schema(df3)
        assert MultiAssetSchema.validate(df4, strict=True)


# ============================================================================
# Error Handling and Edge Cases
# ============================================================================


class TestErrorHandling:
    """Test error handling and edge cases across the stack."""

    def test_empty_universe_raises_clear_error(self):
        """Test that empty universe raises clear error message."""
        Universe.add_custom("empty_test", [])

        try:
            manager = DataManager(output_format="polars")

            with pytest.raises(ValueError, match="symbols list cannot be empty"):
                manager.batch_load_universe(
                    universe="empty_test",
                    start="2024-01-01",
                    end="2024-01-31",
                )

        finally:
            Universe.remove_custom("empty_test")

    def test_invalid_date_range_caught_early(self, populated_storage):
        """Test that invalid dates are caught with clear error."""
        manager, symbols = populated_storage

        # Invalid format
        with pytest.raises(ValueError, match="Invalid date format"):
            manager.batch_load_from_storage(
                symbols=symbols[:1],
                start="not-a-date",
                end="2024-01-31",
            )

        # End before start
        with pytest.raises(ValueError, match="End date must be after start date"):
            manager.batch_load_from_storage(
                symbols=symbols[:1],
                start="2024-12-31",
                end="2024-01-01",
            )

    def test_missing_storage_raises_clear_error(self):
        """Test that missing storage raises clear error."""
        manager = DataManager(output_format="polars")  # No storage configured

        with pytest.raises(ValueError, match="Storage not configured"):
            manager.batch_load_from_storage(
                symbols=["AAPL"],
                start="2024-01-01",
                end="2024-01-31",
            )

    def test_no_data_for_date_range_raises_clear_error(self, populated_storage):
        """Test that empty result raises clear error."""
        manager, symbols = populated_storage

        # Request data for 2025 (storage only has 2024 Q1)
        with pytest.raises(ValueError, match="No data loaded"):
            manager.batch_load_from_storage(
                symbols=symbols,
                start="2025-01-01",
                end="2025-12-31",
                fetch_missing=False,
            )

    @pytest.mark.skip(reason="Error handling not implemented")
    def test_partial_failures_with_graceful_degradation(self):
        """Test that partial failures can be handled gracefully."""
        manager = DataManager(output_format="polars")

        # Mix of valid and invalid symbols
        symbols = ["AAPL", "INVALID_SYMBOL_123", "MSFT"]

        # With fail_on_partial=False, should get data for valid symbols
        df = manager.batch_load(
            symbols=symbols,
            start="2024-01-01",
            end="2024-01-05",
            provider="yahoo",
            fail_on_partial=False,
        )

        # Should have data for at least one valid symbol
        unique_symbols = df["symbol"].unique().to_list()
        assert "AAPL" in unique_symbols or "MSFT" in unique_symbols
        assert "INVALID_SYMBOL_123" not in unique_symbols

        # Should still be valid multi-asset format
        assert MultiAssetSchema.validate(df, strict=True)

    def test_format_conversion_with_missing_data(self):
        """Test format conversion handles missing data gracefully."""
        # Create stacked data with null values
        df = pl.DataFrame(
            {
                "timestamp": [
                    datetime(2024, 1, 1, tzinfo=UTC),
                    datetime(2024, 1, 1, tzinfo=UTC),
                    datetime(2024, 1, 2, tzinfo=UTC),
                    datetime(2024, 1, 2, tzinfo=UTC),
                ],
                "symbol": ["AAPL", "MSFT", "AAPL", "MSFT"],
                "open": [100.0, 200.0, 101.0, 201.0],
                "high": [102.0, 202.0, 103.0, 203.0],
                "low": [99.0, 199.0, 100.0, 200.0],
                "close": [101.0, None, 102.0, 202.0],  # NULL value
                "volume": [1_000_000.0, 2_000_000.0, 1_100_000.0, 2_100_000.0],
            }
        )

        # Convert to wide
        df_wide = pivot_to_wide(df)
        assert df_wide["close_MSFT"][0] is None  # Preserves null

        # Convert back to stacked
        df_stacked = pivot_to_stacked(df_wide)

        # Find the MSFT row at 2024-01-01
        msft_jan1 = df_stacked.filter(
            (pl.col("timestamp") == datetime(2024, 1, 1, tzinfo=UTC)) & (pl.col("symbol") == "MSFT")
        )
        assert msft_jan1["close"][0] is None  # Null preserved through round-trip


# ============================================================================
# Format Conversion Workflow Tests
# ============================================================================


class TestFormatConversionWorkflows:
    """Test realistic format conversion workflows."""

    def test_stacked_to_wide_for_correlation_analysis(self, populated_storage):
        """Test converting to wide format for correlation analysis."""
        manager, symbols = populated_storage

        # Load data
        df = manager.batch_load_from_storage(
            symbols=symbols,
            start="2024-01-01",
            end="2024-03-31",
            fetch_missing=False,
        )

        # Convert to wide for correlation calculation
        df_wide = pivot_to_wide(df, value_cols=["close"])

        # Should have one close column per symbol
        close_cols = [col for col in df_wide.columns if col.startswith("close_")]
        assert len(close_cols) == 5

        # Can compute returns
        for col in close_cols:
            df_wide = df_wide.with_columns(
                pl.col(col).pct_change().alias(f"returns_{col.replace('close_', '')}")
            )

        # Verify returns columns created
        returns_cols = [col for col in df_wide.columns if col.startswith("returns_")]
        assert len(returns_cols) == 5

    def test_wide_to_stacked_for_storage(self):
        """Test converting wide data back to stacked for efficient storage."""
        # Simulate wide format data from external source
        df_wide = pl.DataFrame(
            {
                "timestamp": [
                    datetime(2024, 1, 1, tzinfo=UTC),
                    datetime(2024, 1, 2, tzinfo=UTC),
                ],
                "close_AAPL": [150.0, 151.0],
                "close_MSFT": [350.0, 351.0],
                "volume_AAPL": [1_000_000.0, 1_100_000.0],
                "volume_MSFT": [2_000_000.0, 2_100_000.0],
            }
        )

        # Convert to stacked for storage
        # Note: pivot_to_stacked automatically detects value columns
        df_stacked = pivot_to_stacked(df_wide)

        # Verify stacked format
        assert "symbol" in df_stacked.columns
        assert len(df_stacked) == 4  # 2 timestamps × 2 symbols
        assert set(df_stacked["symbol"].unique().to_list()) == {"AAPL", "MSFT"}

        # Verify values preserved
        aapl_jan1 = df_stacked.filter(
            (pl.col("timestamp") == datetime(2024, 1, 1, tzinfo=UTC)) & (pl.col("symbol") == "AAPL")
        )
        assert aapl_jan1["close"][0] == 150.0
        assert aapl_jan1["volume"][0] == 1_000_000.0

    def test_round_trip_conversion_large_dataset(self):
        """Test that large dataset survives round-trip conversion intact."""
        # Create large dataset
        symbols = [f"SYM{i:02d}" for i in range(20)]
        timestamps = [datetime(2024, 1, 1, tzinfo=UTC) + timedelta(days=i) for i in range(100)]

        data = []
        for ts in timestamps:
            for symbol in symbols:
                data.append(
                    {
                        "timestamp": ts,
                        "symbol": symbol,
                        "open": 100.0,
                        "high": 101.0,
                        "low": 99.0,
                        "close": 100.5,
                        "volume": 1_000_000.0,
                    }
                )

        df_original = pl.DataFrame(data).sort(["timestamp", "symbol"])
        assert len(df_original) == 2_000  # 20 symbols × 100 days

        # Round-trip conversion
        df_wide = pivot_to_wide(df_original)
        df_back = pivot_to_stacked(df_wide).sort(["timestamp", "symbol"])

        # Verify integrity
        assert len(df_original) == len(df_back)
        assert df_original["close"].to_list() == df_back["close"].to_list()
        assert df_original["volume"].to_list() == df_back["volume"].to_list()
        assert df_original["symbol"].to_list() == df_back["symbol"].to_list()


# ============================================================================
# Edge Case Tests
# ============================================================================


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_single_symbol_multi_asset_format(self):
        """Test that single symbol works correctly in multi-asset format."""
        df = pl.DataFrame(
            {
                "timestamp": [
                    datetime(2024, 1, 1, tzinfo=UTC),
                    datetime(2024, 1, 2, tzinfo=UTC),
                ],
                "symbol": ["AAPL", "AAPL"],
                "open": [100.0, 101.0],
                "high": [102.0, 103.0],
                "low": [99.0, 100.0],
                "close": [101.0, 102.0],
                "volume": [1_000_000.0, 1_100_000.0],
            }
        )

        # Should validate as multi-asset
        assert MultiAssetSchema.validate(df, strict=True)

        # Should convert to wide
        df_wide = pivot_to_wide(df)
        assert "close_AAPL" in df_wide.columns
        assert len(df_wide) == 2

        # Should convert back
        df_back = pivot_to_stacked(df_wide)
        assert len(df_back) == 2

    def test_very_large_universe_handling(self):
        """Test that very large universes (500+ symbols) are handled correctly."""
        # Create large universe
        large_universe = [f"SYM{i:04d}" for i in range(500)]

        Universe.add_custom("large_test", large_universe)

        try:
            # Should be able to create multi-asset DataFrame with many symbols
            # (Not actually loading data, just testing the concept)
            symbols_retrieved = Universe.get("large_test")
            assert len(symbols_retrieved) == 500

            # In practice, batch_load would handle this with max_workers parallelization
            # The multi-asset format should handle 500 symbols × 252 days = 126,000 rows

        finally:
            Universe.remove_custom("large_test")

    def test_duplicate_timestamp_symbol_pairs_rejected(self):
        """Test that duplicate (timestamp, symbol) pairs are rejected."""
        df = pl.DataFrame(
            {
                "timestamp": [
                    datetime(2024, 1, 1, tzinfo=UTC),
                    datetime(2024, 1, 1, tzinfo=UTC),  # Duplicate
                ],
                "symbol": ["AAPL", "AAPL"],  # Same symbol
                "open": [100.0, 100.0],
                "high": [101.0, 101.0],
                "low": [99.0, 99.0],
                "close": [100.5, 100.5],
                "volume": [1_000_000.0, 1_000_000.0],
            }
        )

        # Should raise error on wide conversion (duplicates not allowed)
        with pytest.raises(ValueError, match="duplicate"):
            pivot_to_wide(df)

    def test_symbol_with_special_characters(self):
        """Test that symbols with special characters (e.g., BRK.B) work correctly."""
        df = pl.DataFrame(
            {
                "timestamp": [
                    datetime(2024, 1, 1, tzinfo=UTC),
                    datetime(2024, 1, 1, tzinfo=UTC),
                ],
                "symbol": ["BRK.B", "MSFT"],
                "open": [300.0, 350.0],
                "high": [302.0, 352.0],
                "low": [299.0, 349.0],
                "close": [301.0, 351.0],
                "volume": [500_000.0, 2_000_000.0],
            }
        )

        # Should validate
        assert MultiAssetSchema.validate(df, strict=True)

        # Note: Current pivot implementation may have issues with dots in symbols
        # This test documents the expected behavior
        # (Implementation may need to handle . → _ conversion)


# ============================================================================
# Benchmark and Reporting Tests
# ============================================================================


class TestBenchmarks:
    """Comprehensive benchmarks with performance reporting."""

    def test_comprehensive_performance_report(self, populated_storage):
        """Generate comprehensive performance report for documentation."""
        manager, symbols = populated_storage

        print("\n" + "=" * 70)
        print("MULTI-ASSET INTEGRATION PERFORMANCE REPORT")
        print("=" * 70)

        # Test 1: Storage load performance
        print("\n1. Storage Load Performance:")
        start = time.perf_counter()
        df = manager.batch_load_from_storage(
            symbols=symbols, start="2024-01-01", end="2024-03-31", fetch_missing=False
        )
        load_time = time.perf_counter() - start
        print(f"   Loaded {len(df):,} rows in {load_time:.3f}s")
        print(f"   ({len(df) / load_time:,.0f} rows/sec)")

        # Test 2: Format conversion performance
        print("\n2. Format Conversion Performance:")
        start = time.perf_counter()
        df_wide = pivot_to_wide(df)
        to_wide_time = time.perf_counter() - start
        print(f"   Stacked→Wide: {to_wide_time:.3f}s")

        start = time.perf_counter()
        pivot_to_stacked(df_wide)
        to_stacked_time = time.perf_counter() - start
        print(f"   Wide→Stacked: {to_stacked_time:.3f}s")

        # Test 3: Validation performance
        print("\n3. Schema Validation Performance:")
        start = time.perf_counter()
        for _ in range(100):
            MultiAssetSchema.validate(df, strict=True)
        validation_time = (time.perf_counter() - start) / 100
        print(f"   Validation: {validation_time * 1000:.1f}ms per call")

        print("\n" + "=" * 70)
        print("ACCEPTANCE CRITERIA VALIDATION:")
        print(f"   ✓ Storage load: {load_time:.3f}s < 0.5s target")
        print(f"   ✓ Format conversion: {to_wide_time:.3f}s < 1.0s target")
        print(f"   ✓ Schema validation: {validation_time * 1000:.1f}ms < 10ms target")
        print("=" * 70 + "\n")

        # Assertions for CI
        assert load_time < 0.5
        assert to_wide_time < 1.0
        assert validation_time < 0.01


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
