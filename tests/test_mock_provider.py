"""Tests for mock provider implementation."""

import polars as pl
import pytest

from ml4t.data.providers.mock import DataPattern, MockProvider


class TestMockProvider:
    """Test suite for mock provider."""

    def test_initialization(self):
        """Test provider initialization with default parameters."""
        provider = MockProvider()
        assert provider.name == "mock"
        assert provider.seed == 42
        assert provider.pattern == DataPattern.RANDOM
        assert provider.base_price == 100.0
        assert provider.volatility == 0.01

    def test_initialization_with_custom_params(self):
        """Test provider initialization with custom parameters."""
        provider = MockProvider(
            seed=123,
            pattern=DataPattern.TREND,
            base_price=50.0,
            volatility=0.02,
            trend_rate=0.001,
        )
        assert provider.seed == 123
        assert provider.pattern == DataPattern.TREND
        assert provider.base_price == 50.0
        assert provider.volatility == 0.02
        assert provider.trend_rate == 0.001

    def test_fetch_daily_data(self):
        """Test fetching daily OHLCV data."""
        provider = MockProvider(seed=42)
        df = provider.fetch_ohlcv("MOCK", "2024-01-01", "2024-01-05", "daily")

        # Should have 5 business days (Jan 1-5, 2024)
        assert len(df) == 5

        # Verify schema
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
        assert df["close"].dtype == pl.Float64
        assert df["volume"].dtype == pl.Float64
        assert df["symbol"].dtype == pl.Utf8

        # Verify OHLC invariants
        assert (df["high"] >= df["low"]).all()
        assert (df["high"] >= df["open"]).all()
        assert (df["high"] >= df["close"]).all()
        assert (df["low"] <= df["open"]).all()
        assert (df["low"] <= df["close"]).all()

        # Verify all prices are positive
        assert (df["open"] > 0).all()
        assert (df["close"] > 0).all()
        assert (df["volume"] >= 0).all()

        # Verify symbol is set correctly
        assert (df["symbol"] == "MOCK").all()

    def test_fetch_minute_data(self):
        """Test fetching minute-level data."""
        provider = MockProvider(seed=42)
        df = provider.fetch_ohlcv("MOCK", "2024-01-02", "2024-01-02", "minute")

        # Should have 390 minutes for a single trading day
        assert len(df) == 390

        # Verify timestamps are sequential
        timestamps = df["timestamp"].to_list()
        for i in range(1, len(timestamps)):
            time_diff = (timestamps[i] - timestamps[i - 1]).total_seconds()
            assert time_diff == 60  # 60 seconds between minutes

    def test_empty_date_range(self):
        """Test handling of empty date ranges."""
        provider = MockProvider()
        # Weekend dates - should return empty
        df = provider.fetch_ohlcv("MOCK", "2024-01-06", "2024-01-07", "daily")
        assert df.is_empty() or len(df) == 0

    def test_deterministic_generation(self):
        """Test that data generation is deterministic with same seed."""
        provider1 = MockProvider(seed=42)
        provider2 = MockProvider(seed=42)

        df1 = provider1.fetch_ohlcv("MOCK", "2024-01-01", "2024-01-05", "daily")
        df2 = provider2.fetch_ohlcv("MOCK", "2024-01-01", "2024-01-05", "daily")

        # Should produce identical data
        assert df1["open"].to_list() == df2["open"].to_list()
        assert df1["close"].to_list() == df2["close"].to_list()

    def test_different_seeds_produce_different_data(self):
        """Test that different seeds produce different data."""
        provider1 = MockProvider(seed=42)
        provider2 = MockProvider(seed=123)

        df1 = provider1.fetch_ohlcv("MOCK", "2024-01-01", "2024-01-05", "daily")
        df2 = provider2.fetch_ohlcv("MOCK", "2024-01-01", "2024-01-05", "daily")

        # Should produce different data
        assert df1["open"].to_list() != df2["open"].to_list()

    def test_trend_pattern(self):
        """Test trend pattern generation."""
        provider = MockProvider(
            seed=42,
            pattern=DataPattern.TREND,
            trend_rate=0.01,  # 1% daily trend
            base_price=100.0,
        )
        df = provider.fetch_ohlcv("MOCK", "2024-01-01", "2024-01-31", "daily")

        # Calculate average daily returns
        returns = (df["close"][1:] / df["close"][:-1] - 1).mean()

        # With 1% daily trend, average returns should be positive
        assert returns > 0

    def test_volatile_pattern(self):
        """Test volatile pattern generation."""
        provider_low_vol = MockProvider(
            seed=42,
            pattern=DataPattern.RANDOM,
            volatility=0.01,
        )
        provider_high_vol = MockProvider(
            seed=42,
            pattern=DataPattern.VOLATILE,
            volatility=0.01,
        )

        df_low = provider_low_vol.fetch_ohlcv("MOCK", "2024-01-01", "2024-01-31", "daily")
        df_high = provider_high_vol.fetch_ohlcv("MOCK", "2024-01-01", "2024-01-31", "daily")

        # Calculate standard deviation of returns
        returns_low = (df_low["close"][1:] / df_low["close"][:-1] - 1).std()
        returns_high = (df_high["close"][1:] / df_high["close"][:-1] - 1).std()

        # Volatile pattern should have higher standard deviation
        assert returns_high > returns_low

    def test_flat_pattern(self):
        """Test flat/range-bound pattern."""
        provider = MockProvider(
            seed=42,
            pattern=DataPattern.FLAT,
            base_price=100.0,
            range_bound=0.05,  # 5% range
        )
        df = provider.fetch_ohlcv("MOCK", "2024-01-01", "2024-01-31", "daily")

        # All prices should be within 5% of base price
        assert df["close"].min() > 100.0 * 0.95
        assert df["close"].max() < 100.0 * 1.05

    def test_gaps_pattern(self):
        """Test gaps pattern generation."""
        provider = MockProvider(
            seed=42,
            pattern=DataPattern.GAPS,
            gap_probability=0.2,  # 20% chance of gaps
        )
        df = provider.fetch_ohlcv("MOCK", "2024-01-01", "2024-01-31", "daily")

        # Calculate gaps (open significantly different from previous close)
        gaps = []
        for i in range(1, len(df)):
            gap = abs(df["open"][i] - df["close"][i - 1]) / df["close"][i - 1]
            gaps.append(gap)

        # Should have some meaningful gaps
        significant_gaps = [g for g in gaps if g > 0.01]  # > 1% gap
        assert len(significant_gaps) > 0

    def test_template_method_integration(self):
        """Test that mock provider works with base provider's template method."""
        provider = MockProvider(seed=42)

        # This should use the template method from BaseProvider
        df = provider.fetch_ohlcv("MOCK", "2024-01-01", "2024-01-05", "daily")

        # Data should be validated by BaseProvider
        assert not df.is_empty()
        assert len(df) == 5

        # Data should be sorted by timestamp (done in _validate_response)
        timestamps = df["timestamp"].to_list()
        assert timestamps == sorted(timestamps)

    def test_invalid_date_format(self):
        """Test handling of invalid date format."""
        provider = MockProvider()

        with pytest.raises(ValueError, match="Invalid date format"):
            provider.fetch_ohlcv("MOCK", "2024/01/01", "2024/01/05", "daily")

    def test_start_after_end(self):
        """Test handling of start date after end date."""
        provider = MockProvider()

        with pytest.raises(ValueError, match="Start date must be before"):
            provider.fetch_ohlcv("MOCK", "2024-01-05", "2024-01-01", "daily")

    def test_empty_symbol(self):
        """Test handling of empty symbol."""
        provider = MockProvider()

        with pytest.raises(ValueError, match="Symbol cannot be empty"):
            provider.fetch_ohlcv("", "2024-01-01", "2024-01-05", "daily")

    def test_data_validation(self):
        """Test that provider's validate_data method works correctly."""
        provider = MockProvider()
        df = provider.fetch_ohlcv("MOCK", "2024-01-01", "2024-01-05", "daily")

        result = provider.validate_data(df)
        assert result["valid"] is True
        assert result["total_records"] == 5
        assert len(result["issues"]) == 0

    def test_error_simulation(self):
        """Test that mock provider can be configured to simulate errors."""
        # This is a placeholder for future error simulation functionality
        # Mock provider could be enhanced to simulate network errors, etc.
