"""Integration tests for Wiki Prices provider (local Parquet file).

These tests verify the Wiki Prices provider works correctly with the local Parquet archive.

Requirements:
    - Local wiki_prices.parquet file (auto-detected or provided)
    - No API key needed (local file access)
    - Dataset: 1962-01-02 to 2018-03-27, 3,199 US companies

Test Coverage:
    - Provider initialization (auto-detect and explicit path)
    - Stock OHLCV data (AAPL, multiple date ranges)
    - Date range boundary handling (pre-1962, post-2018)
    - Symbol availability checks
    - Error handling (invalid symbols, frequencies)
    - Helper methods (list_symbols, get_date_range, get_stats)
    - Performance characteristics (lazy vs eager loading)
"""

import polars as pl
import pytest

from ml4t.data.core.exceptions import DataNotAvailableError
from ml4t.data.providers.wiki_prices import WikiPricesProvider

# Mark as integration test
pytestmark = pytest.mark.integration


@pytest.fixture
def provider():
    """Create Wiki Prices provider with auto-detection."""
    try:
        provider = WikiPricesProvider()
        yield provider
        provider.close()
    except FileNotFoundError:
        pytest.skip("Wiki Prices Parquet file not found (required for integration tests)")


@pytest.fixture
def provider_cached():
    """Create Wiki Prices provider with in-memory caching."""
    try:
        provider = WikiPricesProvider(cache_in_memory=True)
        yield provider
        provider.close()
    except FileNotFoundError:
        pytest.skip("Wiki Prices Parquet file not found (required for integration tests)")


class TestWikiPricesProvider:
    """Test Wiki Prices provider with local Parquet file."""

    def test_provider_initialization_autodetect(self):
        """Test provider can auto-detect Parquet file location."""
        try:
            provider = WikiPricesProvider()
            assert provider.name == "wiki_prices"
            assert provider.parquet_path.exists()
            assert provider.parquet_path.name == "wiki_prices.parquet"
            provider.close()
        except FileNotFoundError:
            pytest.skip("Wiki Prices Parquet not found - expected for CI/CD")

    def test_provider_initialization_explicit_path(self):
        """Test provider with explicit Parquet path."""
        explicit_path = "/home/stefan/ml4t/software/projects/daily_us_equities/wiki_prices.parquet"
        try:
            provider = WikiPricesProvider(parquet_path=explicit_path)
            assert provider.name == "wiki_prices"
            assert str(provider.parquet_path) == explicit_path
            provider.close()
        except FileNotFoundError:
            pytest.skip("Explicit path not found - expected for different environments")

    def test_provider_initialization_invalid_path(self):
        """Test provider fails gracefully with invalid path."""
        with pytest.raises(FileNotFoundError, match="not found at provided path"):
            WikiPricesProvider(parquet_path="/nonexistent/path/wiki_prices.parquet")

    def test_fetch_ohlcv_aapl_historical(self, provider):
        """Test fetching AAPL historical data (2010-2015)."""
        df = provider.fetch_ohlcv(
            symbol="AAPL",
            start="2010-01-01",
            end="2015-12-31",
            frequency="daily",
        )

        # Verify data structure
        assert isinstance(df, pl.DataFrame)
        assert not df.is_empty(), "Should fetch data for AAPL (2010-2015)"

        # Check required columns
        required_cols = ["timestamp", "open", "high", "low", "close", "volume"]
        assert all(col in df.columns for col in required_cols), f"Missing columns in {df.columns}"

        # Verify data types
        assert df["timestamp"].dtype == pl.Datetime
        assert df["open"].dtype == pl.Float64
        assert df["close"].dtype == pl.Float64
        assert df["volume"].dtype == pl.Float64

        # Verify OHLCV relationships (adjusted prices)
        assert (df["high"] >= df["low"]).all(), "High should be >= Low"
        assert (df["high"] >= df["open"]).all(), "High should be >= Open"
        assert (df["high"] >= df["close"]).all(), "High should be >= Close"
        assert (df["low"] <= df["open"]).all(), "Low should be <= Open"
        assert (df["low"] <= df["close"]).all(), "Low should be <= Close"

        # Check for reasonable prices (AAPL 2010-2015 range ~$10-$130 adjusted)
        assert (df["close"] > 0).all(), "Prices should be positive"
        assert (df["close"] < 200).all(), "AAPL adjusted close should be < $200 in 2010-2015"

        # Verify volume is positive
        assert (df["volume"] > 0).all(), "Volume should be positive"

        # Check row count (should be ~1,500 trading days for 6 years)
        assert 1200 < len(df) < 1800, f"Expected ~1,500 days, got {len(df)}"

    def test_fetch_ohlcv_recent_range_near_cutoff(self, provider):
        """Test fetching data near dataset end (2017-2018)."""
        df = provider.fetch_ohlcv(
            symbol="AAPL",
            start="2017-01-01",
            end="2018-03-27",  # Exact dataset end
            frequency="daily",
        )

        assert not df.is_empty(), "Should fetch data up to 2018-03-27"
        assert df["timestamp"].max().strftime("%Y-%m-%d") == "2018-03-27"

    def test_fetch_ohlcv_multiple_symbols(self, provider):
        """Test fetching data for multiple blue-chip stocks."""
        symbols = ["AAPL", "MSFT", "GOOGL", "JPM", "XOM"]
        start = "2015-01-01"
        end = "2016-12-31"

        results = {}
        for symbol in symbols:
            try:
                df = provider.fetch_ohlcv(symbol, start, end)
                results[symbol] = len(df)
            except DataNotAvailableError:
                # GOOGL might not be in dataset (check)
                pass

        # Should successfully fetch at least 4 out of 5
        assert len(results) >= 4, f"Should fetch most symbols, got {list(results.keys())}"

        # Each should have similar row counts (~500 trading days for 2 years)
        for symbol, row_count in results.items():
            assert 400 < row_count < 600, f"{symbol}: expected ~500 days, got {row_count}"

    def test_fetch_ohlcv_long_history(self, provider):
        """Test fetching very long historical range (30 years)."""
        df = provider.fetch_ohlcv(
            symbol="AAPL",
            start="1985-01-01",
            end="2015-12-31",
            frequency="daily",
        )

        assert not df.is_empty()
        # 30 years ~ 7,500 trading days
        assert len(df) > 7000, f"Expected > 7,000 days for 30 years, got {len(df)}"

    def test_frequency_validation(self, provider):
        """Test that only daily frequency is supported."""
        with pytest.raises(ValueError, match="only supports daily frequency"):
            provider.fetch_ohlcv("AAPL", "2015-01-01", "2015-12-31", frequency="hourly")

        with pytest.raises(ValueError, match="only supports daily frequency"):
            provider.fetch_ohlcv("AAPL", "2015-01-01", "2015-12-31", frequency="minute")

    def test_invalid_symbol(self, provider):
        """Test error handling for non-existent symbol."""
        with pytest.raises(DataNotAvailableError, match="No data available"):
            provider.fetch_ohlcv(
                symbol="INVALID_XYZ",
                start="2015-01-01",
                end="2015-12-31",
                frequency="daily",
            )

    def test_date_range_completely_before_dataset(self, provider):
        """Test error handling for date range before dataset starts."""
        with pytest.raises(DataNotAvailableError, match="No data available"):
            provider.fetch_ohlcv(
                symbol="AAPL",
                start="1950-01-01",
                end="1960-12-31",
                frequency="daily",
            )

    def test_date_range_completely_after_dataset(self, provider):
        """Test error handling for date range after dataset ends."""
        with pytest.raises(DataNotAvailableError, match="No data available"):
            provider.fetch_ohlcv(
                symbol="AAPL",
                start="2019-01-01",
                end="2024-12-31",
                frequency="daily",
            )

    def test_date_range_adjustment_before_start(self, provider):
        """Test automatic adjustment for dates before dataset start."""
        df = provider.fetch_ohlcv(
            symbol="AAPL",
            start="1950-01-01",  # Before 1962 start
            end="2015-12-31",
            frequency="daily",
        )

        # Should adjust to dataset start (1962 or AAPL IPO, whichever is later)
        assert not df.is_empty()
        first_date = df["timestamp"].min().strftime("%Y-%m-%d")
        # AAPL IPO was 1980-12-12, so first date should be around then
        assert first_date >= "1962-01-02", f"Date should be >= 1962, got {first_date}"

    def test_date_range_adjustment_after_end(self, provider):
        """Test automatic adjustment for dates after dataset end."""
        df = provider.fetch_ohlcv(
            symbol="AAPL",
            start="2017-01-01",
            end="2024-12-31",  # After 2018-03-27 end
            frequency="daily",
        )

        # Should adjust to dataset end
        assert not df.is_empty()
        last_date = df["timestamp"].max().strftime("%Y-%m-%d")
        assert last_date == "2018-03-27", f"Should adjust to 2018-03-27, got {last_date}"

    def test_list_available_symbols(self, provider):
        """Test listing all available symbols."""
        symbols = provider.list_available_symbols()

        # Check properties
        assert isinstance(symbols, list)
        assert len(symbols) == 3199, f"Expected 3,199 symbols, got {len(symbols)}"
        assert "AAPL" in symbols, "AAPL should be in dataset"
        assert "MSFT" in symbols, "MSFT should be in dataset"
        assert symbols == sorted(symbols), "Symbols should be sorted"

    def test_get_date_range_for_symbol(self, provider):
        """Test getting date range for specific symbol."""
        start, end = provider.get_date_range("AAPL")

        # AAPL IPO was 1980-12-12
        assert start >= "1980-12-12", f"AAPL should start >= 1980-12-12, got {start}"
        assert end == "2018-03-27", f"Dataset ends 2018-03-27, got {end}"

        # Check format
        assert len(start) == 10, "Date should be YYYY-MM-DD"
        assert len(end) == 10, "Date should be YYYY-MM-DD"

    def test_get_date_range_invalid_symbol(self, provider):
        """Test error handling for invalid symbol in date range query."""
        with pytest.raises(DataNotAvailableError, match="No data available"):
            provider.get_date_range("INVALID_XYZ")

    def test_get_dataset_stats(self, provider):
        """Test getting dataset statistics."""
        stats = provider.get_dataset_stats()

        # Check required fields
        assert "total_rows" in stats
        assert "total_symbols" in stats
        assert "date_range" in stats
        assert "file_size_mb" in stats

        # Verify values
        assert stats["total_rows"] == 15_389_314, (
            f"Expected 15.4M rows, got {stats['total_rows']:,}"
        )
        assert stats["total_symbols"] == 3199, (
            f"Expected 3,199 symbols, got {stats['total_symbols']}"
        )
        assert stats["date_range"][0] == "1962-01-02", "Should start 1962-01-02"
        assert stats["date_range"][1] == "2018-03-27", "Should end 2018-03-27"
        assert stats["file_size_mb"] > 600, f"File should be ~632MB, got {stats['file_size_mb']}MB"
        assert stats["cached_in_memory"] is False, "Should be lazy loading by default"

    def test_provider_lazy_loading(self, provider):
        """Test that default provider uses lazy loading (minimal memory)."""
        # Check that _data is None (not loaded into memory)
        assert provider._data is None, "Should use lazy loading by default"
        assert provider.cache_in_memory is False

        # Query should still work (lazy scan)
        df = provider.fetch_ohlcv("AAPL", "2015-01-01", "2015-12-31")
        assert not df.is_empty()

    def test_provider_eager_loading(self, provider_cached):
        """Test that cached provider loads data into memory."""
        # Check that _data is loaded
        assert provider_cached._data is not None, "Should load data into memory"
        assert provider_cached.cache_in_memory is True

        # Verify loaded data properties
        assert len(provider_cached._data) == 15_389_314
        assert provider_cached._data["ticker"].n_unique() == 3199

        # Query should work from memory
        df = provider_cached.fetch_ohlcv("AAPL", "2015-01-01", "2015-12-31")
        assert not df.is_empty()

    def test_performance_lazy_vs_eager(self, provider, provider_cached):
        """Test performance characteristics of lazy vs eager loading."""
        import time

        symbol = "AAPL"
        start = "2015-01-01"
        end = "2015-12-31"

        # Lazy loading (first query)
        start_time = time.time()
        df_lazy = provider.fetch_ohlcv(symbol, start, end)
        lazy_time = time.time() - start_time

        # Eager loading (from memory)
        start_time = time.time()
        df_cached = provider_cached.fetch_ohlcv(symbol, start, end)
        cached_time = time.time() - start_time

        # Both should return same data
        assert len(df_lazy) == len(df_cached)

        # Cached should be faster (or similar for small queries)
        # Note: For single-symbol queries, difference may be negligible
        print(f"\nLazy: {lazy_time:.3f}s, Cached: {cached_time:.3f}s")

    def test_context_manager(self):
        """Test provider works with context manager."""
        try:
            with WikiPricesProvider() as provider:
                assert provider.name == "wiki_prices"
                df = provider.fetch_ohlcv("AAPL", "2015-01-01", "2015-12-31")
                assert not df.is_empty()
        except FileNotFoundError:
            pytest.skip("Wiki Prices Parquet not found")

    def test_schema_correctness(self, provider):
        """Test that output schema matches ml4t-data standard."""
        df = provider.fetch_ohlcv("AAPL", "2015-01-01", "2015-12-31")

        # Verify column names
        expected_cols = ["timestamp", "open", "high", "low", "close", "volume"]
        assert list(df.columns) == expected_cols, f"Schema mismatch: {df.columns}"

        # Verify column order (should match standard schema)
        for i, col in enumerate(expected_cols):
            assert df.columns[i] == col, f"Column {i} should be '{col}', got '{df.columns[i]}'"

    def test_adjusted_prices_used(self, provider):
        """Test that provider uses adjusted prices (not raw prices)."""
        # Fetch AAPL data around a known stock split
        # AAPL 7-for-1 split on 2014-06-09
        df = provider.fetch_ohlcv("AAPL", "2014-06-01", "2014-06-30")

        # Adjusted prices should be continuous (no 7x jump on split date)
        # Calculate max daily return
        returns = df["close"].pct_change().abs()
        max_return = returns.max()

        # Max return should be < 10% (not ~700% from unadjusted split)
        assert max_return < 0.10, (
            f"Max daily return {max_return:.2%} too high - provider may be using unadjusted prices"
        )

    def test_data_quality_no_nulls(self, provider):
        """Test that fetched data has no null values."""
        df = provider.fetch_ohlcv("AAPL", "2015-01-01", "2015-12-31")

        # Check for nulls in all columns
        for col in df.columns:
            null_count = df[col].null_count()
            assert null_count == 0, f"Column '{col}' has {null_count} null values"

    def test_data_quality_chronological_order(self, provider):
        """Test that data is returned in chronological order."""
        df = provider.fetch_ohlcv("AAPL", "2015-01-01", "2015-12-31")

        # Check timestamps are sorted
        timestamps = df["timestamp"].to_list()
        assert timestamps == sorted(timestamps), "Timestamps should be in chronological order"

    def test_data_quality_no_duplicates(self, provider):
        """Test that data has no duplicate timestamps."""
        df = provider.fetch_ohlcv("AAPL", "2015-01-01", "2015-12-31")

        # Check for duplicate dates
        unique_dates = df["timestamp"].n_unique()
        total_dates = len(df)
        assert unique_dates == total_dates, f"Found duplicates: {total_dates - unique_dates}"
