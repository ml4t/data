"""Tests for batch_load() multi-asset functionality."""

from datetime import UTC, datetime

import polars as pl
import pytest

from ml4t.data.core.schemas import MultiAssetSchema
from ml4t.data.data_manager import DataManager


class TestBatchLoad:
    """Test suite for DataManager.batch_load() method."""

    @pytest.fixture(autouse=True)
    def cleanup_yfinance(self):
        """Clean up yfinance global state before each test.

        yfinance uses global dictionaries (shared._DFS, _ERRORS, etc.) that persist
        across calls, causing cross-contamination between tests.
        See: https://github.com/ranaroussi/yfinance/issues/2557
        """
        try:
            from yfinance import shared

            shared._DFS.clear()
            shared._ERRORS.clear()
            shared._TRACEBACKS.clear()
        except (ImportError, AttributeError):
            pass  # yfinance not available or structure changed
        yield
        # Cleanup after test too
        try:
            from yfinance import shared

            shared._DFS.clear()
            shared._ERRORS.clear()
            shared._TRACEBACKS.clear()
        except (ImportError, AttributeError):
            pass

    @pytest.fixture
    def manager(self):
        """Create DataManager instance for testing."""
        return DataManager(output_format="polars")

    def test_batch_load_basic(self, manager):
        """Test basic batch loading of multiple symbols.

        Note: Uses max_workers=1 to avoid yfinance thread-safety bug
        (https://github.com/ranaroussi/yfinance/issues/2557)
        """
        symbols = ["AAPL", "MSFT", "GOOG"]
        df = manager.batch_load(
            symbols=symbols,
            start="2024-01-01",
            end="2024-01-31",
            frequency="daily",
            provider="yahoo",
            max_workers=1,  # Serial execution to avoid yfinance threading bugs
        )

        # Check schema compliance
        assert MultiAssetSchema.validate(df, strict=True)

        # Check all symbols present
        unique_symbols = df["symbol"].unique().sort().to_list()
        assert set(unique_symbols) == set(symbols)

        # Check sorting (timestamp, symbol)
        assert df["timestamp"].is_sorted()
        # Within each timestamp, symbols should be sorted
        for ts in df["timestamp"].unique():
            symbols_at_ts = df.filter(pl.col("timestamp") == ts)["symbol"].to_list()
            assert symbols_at_ts == sorted(symbols_at_ts)

        # Check column order
        expected_cols = ["timestamp", "symbol", "open", "high", "low", "close", "volume"]
        assert df.columns[:7] == expected_cols

    def test_batch_load_empty_symbols(self, manager):
        """Test that empty symbols list raises error."""
        with pytest.raises(ValueError, match="symbols list cannot be empty"):
            manager.batch_load(
                symbols=[],
                start="2024-01-01",
                end="2024-01-31",
            )

    def test_batch_load_partial_failure_graceful(self, manager):
        """Test graceful handling of partial failures.

        Note: Uses max_workers=1 to avoid yfinance thread-safety bug
        (https://github.com/ranaroussi/yfinance/issues/2557)
        """
        symbols = ["AAPL", "INVALID_SYMBOL_XYZ", "MSFT"]
        df = manager.batch_load(
            symbols=symbols,
            start="2024-01-01",
            end="2024-01-31",
            fail_on_partial=False,  # Graceful degradation
            provider="yahoo",
            max_workers=1,  # Serial execution to avoid yfinance threading bugs
        )

        # Should get data for valid symbols only
        unique_symbols = df["symbol"].unique().to_list()
        assert "AAPL" in unique_symbols or "MSFT" in unique_symbols
        assert "INVALID_SYMBOL_XYZ" not in unique_symbols

        # Should still be valid multi-asset format
        assert MultiAssetSchema.validate(df, strict=True)

    def test_batch_load_partial_failure_strict(self, manager):
        """Test strict error handling when fail_on_partial=True.

        Note: Uses max_workers=1 to avoid yfinance thread-safety bug
        (https://github.com/ranaroussi/yfinance/issues/2557)
        """
        symbols = ["AAPL", "INVALID_SYMBOL_XYZ", "MSFT"]

        with pytest.raises(ValueError, match="Batch load failed"):
            manager.batch_load(
                symbols=symbols,
                start="2024-01-01",
                end="2024-01-31",
                fail_on_partial=True,  # Strict mode
                provider="yahoo",
                max_workers=1,  # Serial execution to avoid yfinance threading bugs
            )

    def test_batch_load_all_failures(self, manager):
        """Test that all failures raises error even with fail_on_partial=False."""
        symbols = ["INVALID1", "INVALID2", "INVALID3"]

        with pytest.raises(ValueError, match="All .* symbols failed"):
            manager.batch_load(
                symbols=symbols,
                start="2024-01-01",
                end="2024-01-31",
                fail_on_partial=False,
                provider="yahoo",
            )

    def test_batch_load_performance(self, manager):
        """Test that batch loading is reasonably fast.

        Note: Uses max_workers=1 to avoid yfinance thread-safety bug
        (https://github.com/ranaroussi/yfinance/issues/2557)
        """
        import time

        symbols = ["AAPL", "MSFT", "GOOG", "AMZN", "META"]

        start_time = time.time()
        df = manager.batch_load(
            symbols=symbols,
            start="2024-01-01",
            end="2024-01-31",
            max_workers=1,  # Serial execution to avoid yfinance threading bugs
            provider="yahoo",
        )
        elapsed = time.time() - start_time

        # Should complete in reasonable time (< 15 seconds for 5 symbols serially)
        assert elapsed < 15.0

        # Should have data for all symbols
        assert len(df["symbol"].unique()) == len(symbols)

    def test_batch_load_date_validation(self, manager):
        """Test that invalid dates are rejected."""
        with pytest.raises(ValueError, match="Invalid date format"):
            manager.batch_load(
                symbols=["AAPL"],
                start="invalid-date",
                end="2024-01-31",
            )

        with pytest.raises(ValueError, match="End date must be after start date"):
            manager.batch_load(
                symbols=["AAPL"],
                start="2024-12-31",
                end="2024-01-01",  # Before start
            )

    def test_batch_load_data_quality(self, manager):
        """Test that returned data has expected quality."""
        symbols = ["AAPL", "MSFT"]
        df = manager.batch_load(
            symbols=symbols,
            start="2024-01-01",
            end="2024-01-31",
            provider="yahoo",
        )

        # No nulls in required columns
        for col in ["timestamp", "symbol", "open", "high", "low", "close", "volume"]:
            assert df[col].null_count() == 0

        # OHLC consistency (high >= low)
        assert (df["high"] >= df["low"]).all()

        # Prices are positive
        assert (df["close"] > 0).all()
        assert (df["open"] > 0).all()

        # Volume is non-negative
        assert (df["volume"] >= 0).all()


class TestMultiAssetSchema:
    """Test MultiAssetSchema utilities used by batch_load()."""

    def test_add_symbol_column(self):
        """Test adding symbol column to single-symbol data."""
        df = pl.DataFrame(
            {
                "timestamp": [
                    datetime(2024, 1, 1, 9, 30, tzinfo=UTC),
                    datetime(2024, 1, 2, 9, 30, tzinfo=UTC),
                ],
                "open": [100.0, 101.0],
                "high": [102.0, 103.0],
                "low": [99.0, 100.0],
                "close": [101.0, 102.0],
                "volume": [1000.0, 1100.0],
            }
        )

        df_with_symbol = MultiAssetSchema.add_symbol_column(df, "AAPL")

        assert "symbol" in df_with_symbol.columns
        assert df_with_symbol["symbol"].unique().to_list() == ["AAPL"]
        assert len(df_with_symbol) == len(df)

    def test_add_symbol_column_idempotent(self):
        """Test that adding same symbol twice is idempotent."""
        df = pl.DataFrame(
            {
                "timestamp": [datetime(2024, 1, 1, 9, 30, tzinfo=UTC)],
                "symbol": ["AAPL"],
                "open": [100.0],
                "high": [102.0],
                "low": [99.0],
                "close": [101.0],
                "volume": [1000.0],
            }
        )

        # Adding same symbol should be no-op
        df_again = MultiAssetSchema.add_symbol_column(df, "AAPL")
        assert df_again.equals(df)

    def test_add_symbol_column_conflict(self):
        """Test that adding different symbol raises error."""
        df = pl.DataFrame(
            {
                "timestamp": [datetime(2024, 1, 1, 9, 30, tzinfo=UTC)],
                "symbol": ["AAPL"],
                "open": [100.0],
                "high": [102.0],
                "low": [99.0],
                "close": [101.0],
                "volume": [1000.0],
            }
        )

        with pytest.raises(ValueError, match="already has 'symbol' column"):
            MultiAssetSchema.add_symbol_column(df, "MSFT")

    def test_standardize_order(self):
        """Test standardizing column order and sorting."""
        df = pl.DataFrame(
            {
                "volume": [1000.0, 2000.0, 1500.0],
                "close": [100.0, 101.0, 99.0],
                "symbol": ["AAPL", "MSFT", "AAPL"],
                "timestamp": [
                    datetime(2024, 1, 2, 9, 30, tzinfo=UTC),
                    datetime(2024, 1, 1, 9, 30, tzinfo=UTC),
                    datetime(2024, 1, 1, 9, 30, tzinfo=UTC),
                ],
                "open": [99.0, 100.0, 98.0],
                "high": [101.0, 102.0, 100.0],
                "low": [98.0, 99.0, 97.0],
            }
        )

        standardized = MultiAssetSchema.standardize_order(df)

        # Check column order
        assert standardized.columns[:7] == [
            "timestamp",
            "symbol",
            "open",
            "high",
            "low",
            "close",
            "volume",
        ]

        # Check sorting (timestamp first, then symbol)
        assert standardized["timestamp"].is_sorted()

        # At 2024-01-01, AAPL should come before MSFT
        jan1_data = standardized.filter(
            pl.col("timestamp") == datetime(2024, 1, 1, 9, 30, tzinfo=UTC)
        )
        assert jan1_data["symbol"].to_list() == ["AAPL", "MSFT"]

    def test_validate_multi_asset(self):
        """Test schema validation for multi-asset data."""
        valid_df = pl.DataFrame(
            {
                "timestamp": [
                    datetime(2024, 1, 1, 9, 30, tzinfo=UTC),
                    datetime(2024, 1, 1, 9, 30, tzinfo=UTC),
                ],
                "symbol": ["AAPL", "MSFT"],
                "open": [100.0, 200.0],
                "high": [102.0, 202.0],
                "low": [99.0, 199.0],
                "close": [101.0, 201.0],
                "volume": [1000.0, 2000.0],
            }
        )

        assert MultiAssetSchema.validate(valid_df, strict=True)

    def test_validate_missing_column(self):
        """Test validation fails with missing required column."""
        invalid_df = pl.DataFrame(
            {
                "timestamp": [datetime(2024, 1, 1, 9, 30, tzinfo=UTC)],
                "symbol": ["AAPL"],
                "open": [100.0],
                # Missing: high, low, close, volume
            }
        )

        with pytest.raises(ValueError, match="Missing required column"):
            MultiAssetSchema.validate(invalid_df, strict=True)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
