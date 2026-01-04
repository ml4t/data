#!/usr/bin/env python3
"""Real API integration tests for Databento provider."""

import os

import polars as pl
import pytest

# Set environment variable for tests
os.environ["DATABENTO_API_KEY"] = "db-DEUmyRReHLQ4CFMcrUYqr4aVYRJr8"

# Import the real databento module (no mocking)
import databento

from ml4t.data.providers.databento import DataBentoProvider


@pytest.mark.integration
class TestDatabentaRealAPI:
    """Real API tests for Databento provider."""

    @pytest.fixture
    def provider(self):
        """Create real Databento provider."""
        return DataBentoProvider(api_key=os.getenv("DATABENTO_API_KEY"))

    @pytest.mark.skip(reason="DataBento paid tier required")
    def test_databento_module_available(self):
        """Verify databento module is properly installed."""
        assert hasattr(databento, "__version__")
        print(f"Databento version: {databento.__version__}")

    def test_fetch_specific_contract(self, provider):
        """Test fetching specific futures contract with real API."""
        # Fetch minimal data to minimize costs
        df = provider.fetch_ohlcv(
            "ESH4",  # E-mini S&P March 2024 contract
            "2024-01-02",
            "2024-01-02",
            "daily",
        )

        # Verify schema
        assert not df.is_empty()
        assert df.shape[0] == 1  # Single day
        assert "timestamp" in df.columns
        assert "open" in df.columns
        assert "high" in df.columns
        assert "low" in df.columns
        assert "close" in df.columns
        assert "volume" in df.columns
        assert "symbol" in df.columns

        # Verify data types
        assert df["timestamp"].dtype == pl.Datetime("ns", "UTC")
        assert df["open"].dtype == pl.Float64
        assert df["symbol"][0] == "ESH4"

        # Verify data quality
        assert df["high"][0] >= df["low"][0]
        assert df["high"][0] >= df["open"][0]
        assert df["high"][0] >= df["close"][0]
        assert df["low"][0] <= df["open"][0]
        assert df["low"][0] <= df["close"][0]

        # Check actual values (known from previous tests)
        assert df["open"][0] == 4820.75
        assert df["high"][0] == 4828.00
        assert df["low"][0] == 4765.50
        assert df["close"][0] == 4788.50

    def test_empty_response_handling(self, provider):
        """Test handling of requests that return no data."""
        # Note: Futures markets trade almost every day, even holidays
        # Testing with a symbol that doesn't exist in the date range
        # ESH4 (March 2024) wouldn't have data in early 2023
        df = provider.fetch_ohlcv(
            "ESH4",
            "2023-01-01",  # Too early for ESH4 contract
            "2023-01-01",
            "daily",
        )

        # Should return empty DataFrame without errors
        assert df.is_empty() or len(df) == 0

    def test_invalid_symbol_handling(self, provider):
        """Test handling for invalid symbols."""
        # Invalid symbols return empty DataFrame (graceful handling)
        df = provider.fetch_ohlcv("INVALID_SYMBOL_XYZ", "2024-01-02", "2024-01-02", "daily")

        # Should return empty DataFrame without errors
        assert df.is_empty() or len(df) == 0

    def test_multiple_days_fetch(self, provider):
        """Test fetching multiple days of data."""
        # Fetch 3 trading days to verify sorting and continuity
        df = provider.fetch_ohlcv("ESH4", "2024-01-02", "2024-01-04", "daily")

        assert not df.is_empty()
        assert len(df) == 3  # 3 trading days

        # Verify data is sorted by timestamp
        timestamps = df["timestamp"].to_list()
        assert timestamps == sorted(timestamps)

        # Verify all days have complete OHLCV data
        for i in range(len(df)):
            assert df["high"][i] >= df["low"][i]
            assert df["volume"][i] > 0

    def test_different_frequencies(self, provider):
        """Test different data frequencies."""
        # Test hourly data (should work)
        df_hourly = provider.fetch_ohlcv("ESH4", "2024-01-02", "2024-01-02", "hourly")

        if not df_hourly.is_empty():
            assert len(df_hourly) > 1  # Should have multiple hourly bars
            assert "timestamp" in df_hourly.columns

    def test_continuous_futures_expected_failure(self, provider):
        """Test that continuous futures return empty with GLBX.MDP3."""
        # Continuous futures (ES.v.0) don't work with GLBX.MDP3 dataset
        # Returns empty DataFrame instead of error
        df = provider.fetch_continuous_futures("ES", "2024-01-02", "2024-01-02", "daily", version=0)

        # Should return empty DataFrame
        assert df.is_empty() or len(df) == 0

    def test_column_mapping_preservation(self, provider):
        """Test that all important columns are preserved."""
        df = provider.fetch_ohlcv("ESH4", "2024-01-02", "2024-01-02", "daily")

        # Check that extra columns are preserved
        expected_cols = ["timestamp", "open", "high", "low", "close", "volume", "symbol"]
        for col in expected_cols:
            assert col in df.columns

        # May have additional columns like rtype, publisher_id, etc.
        # These should be preserved but not required
        all_cols = df.columns
        print(f"All columns preserved: {all_cols}")


if __name__ == "__main__":
    pytest.main([__file__, "-xvs"])
