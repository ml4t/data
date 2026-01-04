"""Tests for SyntheticProvider.

These tests verify the synthetic data generator creates realistic OHLCV data
with proper relationships and statistical properties.
"""

import numpy as np
import polars as pl
import pytest

from ml4t.data.providers.synthetic import SyntheticProvider


class TestSyntheticProviderBasics:
    """Basic functionality tests."""

    @pytest.fixture
    def provider(self):
        """Create provider with fixed seed."""
        return SyntheticProvider(seed=42)

    def test_provider_name(self, provider):
        """Test provider name is correct."""
        assert provider.name == "synthetic"

    def test_default_parameters(self):
        """Test default parameter values."""
        provider = SyntheticProvider()
        assert provider.model == "gbm"
        assert provider.annual_return == 0.08
        assert provider.annual_volatility == 0.20
        assert provider.base_price == 100.0
        assert provider.base_volume == 1_000_000

    def test_custom_parameters(self):
        """Test custom parameter values."""
        provider = SyntheticProvider(
            model="gbm_jump",
            annual_return=0.12,
            annual_volatility=0.30,
            base_price=50.0,
            base_volume=500_000,
        )
        assert provider.model == "gbm_jump"
        assert provider.annual_return == 0.12
        assert provider.annual_volatility == 0.30
        assert provider.base_price == 50.0
        assert provider.base_volume == 500_000

    def test_invalid_model(self, provider):
        """Test invalid model raises error."""
        provider.model = "invalid"
        with pytest.raises(ValueError, match="Unknown model"):
            provider.fetch_ohlcv("SYNTH", "2024-01-01", "2024-01-31", "daily")


class TestSyntheticProviderDataGeneration:
    """Data generation tests."""

    @pytest.fixture
    def provider(self):
        """Create provider with fixed seed."""
        return SyntheticProvider(seed=42)

    def test_daily_data_generation(self, provider):
        """Test daily data generation."""
        df = provider.fetch_ohlcv("SYNTH", "2024-01-01", "2024-01-31", "daily")

        # Should have data
        assert len(df) > 0

        # Check schema
        assert "timestamp" in df.columns
        assert "open" in df.columns
        assert "high" in df.columns
        assert "low" in df.columns
        assert "close" in df.columns
        assert "volume" in df.columns

        # Check data types
        assert df["timestamp"].dtype == pl.Datetime("us", "UTC")
        assert df["open"].dtype == pl.Float64
        assert df["volume"].dtype == pl.Float64

    def test_hourly_data_generation(self, provider):
        """Test hourly data generation."""
        df = provider.fetch_ohlcv("SYNTH", "2024-01-02", "2024-01-03", "hourly")

        # Should have multiple bars per day
        assert len(df) > 10

    def test_minute_data_generation(self, provider):
        """Test minute data generation."""
        df = provider.fetch_ohlcv("SYNTH", "2024-01-02", "2024-01-02", "minute")

        # Should have ~390 bars per day
        assert len(df) >= 300

    def test_reproducibility_with_seed(self):
        """Test that same seed produces same data."""
        provider1 = SyntheticProvider(seed=42)
        provider2 = SyntheticProvider(seed=42)

        df1 = provider1.fetch_ohlcv("SYNTH", "2024-01-01", "2024-01-10", "daily")
        df2 = provider2.fetch_ohlcv("SYNTH", "2024-01-01", "2024-01-10", "daily")

        # Should be identical
        assert df1["close"].to_list() == df2["close"].to_list()

    def test_different_symbols_different_data(self):
        """Test that different symbols produce different data."""
        provider = SyntheticProvider(seed=42)

        df1 = provider.fetch_ohlcv("SYNTH_A", "2024-01-01", "2024-01-10", "daily")

        # Reset and fetch different symbol
        provider.reset_seed(42)
        df2 = provider.fetch_ohlcv("SYNTH_B", "2024-01-01", "2024-01-10", "daily")

        # Should be different (different symbol hash)
        assert df1["close"].to_list() != df2["close"].to_list()


class TestOHLCInvariants:
    """Test OHLC data invariants."""

    @pytest.fixture
    def provider(self):
        """Create provider with fixed seed."""
        return SyntheticProvider(seed=42)

    def test_high_is_highest(self, provider):
        """Test that High >= all other prices."""
        df = provider.fetch_ohlcv("SYNTH", "2024-01-01", "2024-12-31", "daily")

        assert (df["high"] >= df["open"]).all()
        assert (df["high"] >= df["close"]).all()
        assert (df["high"] >= df["low"]).all()

    def test_low_is_lowest(self, provider):
        """Test that Low <= all other prices."""
        df = provider.fetch_ohlcv("SYNTH", "2024-01-01", "2024-12-31", "daily")

        assert (df["low"] <= df["open"]).all()
        assert (df["low"] <= df["close"]).all()
        assert (df["low"] <= df["high"]).all()

    def test_prices_positive(self, provider):
        """Test that all prices are positive."""
        df = provider.fetch_ohlcv("SYNTH", "2024-01-01", "2024-12-31", "daily")

        assert (df["open"] > 0).all()
        assert (df["high"] > 0).all()
        assert (df["low"] > 0).all()
        assert (df["close"] > 0).all()

    def test_volume_positive(self, provider):
        """Test that volume is positive."""
        df = provider.fetch_ohlcv("SYNTH", "2024-01-01", "2024-12-31", "daily")

        assert (df["volume"] > 0).all()

    def test_timestamps_sorted(self, provider):
        """Test that timestamps are sorted."""
        df = provider.fetch_ohlcv("SYNTH", "2024-01-01", "2024-06-30", "daily")

        timestamps = df["timestamp"].to_list()
        assert timestamps == sorted(timestamps)

    def test_no_duplicate_timestamps(self, provider):
        """Test that there are no duplicate timestamps."""
        df = provider.fetch_ohlcv("SYNTH", "2024-01-01", "2024-06-30", "daily")

        assert len(df) == df["timestamp"].n_unique()


class TestModels:
    """Test different price models."""

    def test_gbm_model(self):
        """Test GBM model generates reasonable data."""
        provider = SyntheticProvider(model="gbm", seed=42)
        df = provider.fetch_ohlcv("SYNTH", "2024-01-01", "2024-12-31", "daily")

        # Check we have data
        assert len(df) > 200

        # Check price stays reasonable
        assert df["close"].min() > 10  # Doesn't crash to near zero
        assert df["close"].max() < 1000  # Doesn't explode

    def test_gbm_jump_model(self):
        """Test GBM with jumps generates data with occasional large moves."""
        provider = SyntheticProvider(model="gbm_jump", seed=42)
        df = provider.fetch_ohlcv("SYNTH", "2024-01-01", "2024-12-31", "daily")

        # Calculate returns
        returns = df["close"].pct_change().drop_nulls()

        # GBM with jumps should have some larger returns
        assert len(df) > 200
        assert returns.std() > 0.005  # Some volatility

    def test_mean_revert_model(self):
        """Test mean-reverting model."""
        provider = SyntheticProvider(model="mean_revert", seed=42)
        df = provider.fetch_ohlcv("SYNTH", "2024-01-01", "2024-12-31", "daily")

        # Mean reversion should keep prices bounded
        assert len(df) > 200
        # Prices should not drift too far from starting point
        assert df["close"].mean() < 200  # Reasonable bound


class TestStatisticalProperties:
    """Test statistical properties of generated data."""

    @pytest.fixture
    def long_series(self):
        """Generate a longer time series for statistical tests."""
        provider = SyntheticProvider(seed=42)
        return provider.fetch_ohlcv("SYNTH", "2020-01-01", "2024-12-31", "daily")

    def test_returns_have_expected_volatility(self, long_series):
        """Test that returns have approximately expected volatility."""
        returns = long_series["close"].pct_change().drop_nulls()

        # Annualized volatility should be roughly 20% (what we set)
        daily_vol = returns.std()
        annual_vol = daily_vol * np.sqrt(252)

        # Allow significant tolerance since it's random
        assert 0.10 < annual_vol < 0.40

    def test_returns_approximately_normal(self, long_series):
        """Test that returns are approximately normally distributed."""
        returns = long_series["close"].pct_change().drop_nulls().to_numpy()

        # Check skewness is reasonable (using numpy - no scipy dependency)
        n = len(returns)
        mean = np.mean(returns)
        std = np.std(returns)
        skew = np.sum(((returns - mean) / std) ** 3) / n

        assert -1.0 < skew < 1.0

    def test_volume_correlated_with_volatility(self, long_series):
        """Test that volume increases on volatile days."""
        # Calculate absolute returns as proxy for volatility
        abs_returns = long_series["close"].pct_change().abs().drop_nulls()
        volumes = long_series["volume"][1:]  # Align with returns

        # Calculate correlation
        corr = np.corrcoef(abs_returns.to_numpy(), volumes.to_numpy())[0, 1]

        # Should be positive correlation
        assert corr > 0


class TestEdgeCases:
    """Test edge cases."""

    def test_empty_date_range(self):
        """Test empty date range returns empty DataFrame."""
        provider = SyntheticProvider(seed=42)
        # Weekend dates only
        df = provider.fetch_ohlcv("SYNTH", "2024-01-06", "2024-01-07", "daily")

        # Should be empty or very few rows (weekend)
        assert len(df) == 0

    def test_single_day(self):
        """Test single day generates one bar."""
        provider = SyntheticProvider(seed=42)
        df = provider.fetch_ohlcv("SYNTH", "2024-01-02", "2024-01-02", "daily")

        # Should have exactly one bar
        assert len(df) == 1

    def test_get_available_symbols(self):
        """Test available symbols returns list."""
        provider = SyntheticProvider()
        symbols = provider.get_available_symbols()

        assert len(symbols) > 0
        assert "SYNTH" in symbols

    def test_reset_seed(self):
        """Test seed reset produces reproducible results."""
        provider = SyntheticProvider(seed=42)

        df1 = provider.fetch_ohlcv("SYNTH", "2024-01-01", "2024-01-10", "daily")

        # Reset and regenerate
        provider.reset_seed(42)
        df2 = provider.fetch_ohlcv("SYNTH", "2024-01-01", "2024-01-10", "daily")

        assert df1["close"].to_list() == df2["close"].to_list()

    def test_context_manager(self):
        """Test provider works as context manager."""
        with SyntheticProvider(seed=42) as provider:
            df = provider.fetch_ohlcv("SYNTH", "2024-01-01", "2024-01-10", "daily")
            assert len(df) > 0
