"""Tests for enhanced mock provider features."""

import polars as pl

from ml4t.data.providers.mock import DataPattern, MockProvider


class TestEnhancedMockProvider:
    """Test enhanced mock provider functionality."""

    def test_trend_pattern(self) -> None:
        """Test generating data with trend pattern."""
        provider = MockProvider(
            seed=42,
            pattern=DataPattern.TREND,
            trend_rate=0.01,  # 1% daily trend
        )

        df = provider.fetch_ohlcv(
            symbol="TEST",
            start="2024-01-01",
            end="2024-01-10",
        )

        # Check that prices generally trend upward
        first_close = df["close"][0]
        last_close = df["close"][-1]
        assert last_close > first_close

    def test_volatile_pattern(self) -> None:
        """Test generating volatile data."""
        provider = MockProvider(
            seed=42,
            pattern=DataPattern.VOLATILE,
            volatility=0.05,  # 5% volatility
        )

        df = provider.fetch_ohlcv(
            symbol="TEST",
            start="2024-01-01",
            end="2024-01-10",
        )

        # Check that high-low spread is larger with higher volatility
        spreads = (df["high"] - df["low"]) / df["open"]
        avg_spread = spreads.mean()
        assert avg_spread > 0.02  # Should have meaningful spread

    def test_gaps_pattern(self) -> None:
        """Test generating data with gaps."""
        provider = MockProvider(
            seed=42,
            pattern=DataPattern.GAPS,
            gap_probability=0.3,  # 30% chance of gaps
            volatility=0.01,
        )

        df = provider.fetch_ohlcv(
            symbol="TEST",
            start="2024-01-01",
            end="2024-01-31",
        )

        # Check that the pattern generates more volatile opens
        # With gaps pattern, some opens should deviate significantly from previous close
        opens = df["open"].to_list()
        closes = df["close"].to_list()

        max_gap = 0
        for i in range(1, len(opens)):
            gap = abs(opens[i] - closes[i - 1]) / closes[i - 1]
            max_gap = max(max_gap, gap)

        # With gap pattern, we should see larger overnight moves
        assert max_gap > 0.005  # At least 0.5% gap somewhere

    def test_flat_pattern(self) -> None:
        """Test generating flat/ranging data."""
        provider = MockProvider(
            seed=42,
            pattern=DataPattern.FLAT,
            range_bound=0.02,  # 2% range
        )

        df = provider.fetch_ohlcv(
            symbol="TEST",
            start="2024-01-01",
            end="2024-01-10",
        )

        # Check that prices stay within a range
        all_prices = pl.concat(
            [
                df["open"],
                df["high"],
                df["low"],
                df["close"],
            ]
        )

        price_range = (all_prices.max() - all_prices.min()) / all_prices.mean()
        assert price_range < 0.05  # Should stay within 5% range

    def test_custom_base_price(self) -> None:
        """Test setting custom base price."""
        provider = MockProvider(
            seed=42,
            base_price=500.0,
        )

        df = provider.fetch_ohlcv(
            symbol="TEST",
            start="2024-01-01",
            end="2024-01-02",
        )

        # Prices should be around the base price
        avg_price = df["close"].mean()
        assert 450 < avg_price < 550

    def test_minute_data_generation(self) -> None:
        """Test generating minute-level data."""
        provider = MockProvider(seed=42)

        df = provider.fetch_ohlcv(
            symbol="TEST",
            start="2024-01-01",
            end="2024-01-01",
            frequency="minute",
        )

        # Should generate data for market hours (9:30 AM - 4:00 PM ET)
        # That's 6.5 hours = 390 minutes
        assert len(df) == 390

        # Check timestamps are properly spaced
        timestamps = df["timestamp"].to_list()
        for i in range(1, len(timestamps)):
            diff = (timestamps[i] - timestamps[i - 1]).total_seconds()
            assert diff == 60  # 60 seconds between minutes

    def test_reproducible_with_seed(self) -> None:
        """Test that same seed produces same data."""
        provider1 = MockProvider(seed=123)
        provider2 = MockProvider(seed=123)

        df1 = provider1.fetch_ohlcv("TEST", "2024-01-01", "2024-01-10")
        df2 = provider2.fetch_ohlcv("TEST", "2024-01-01", "2024-01-10")

        assert df1.equals(df2)

    def test_different_seeds_produce_different_data(self) -> None:
        """Test that different seeds produce different data."""
        provider1 = MockProvider(seed=123)
        provider2 = MockProvider(seed=456)

        df1 = provider1.fetch_ohlcv("TEST", "2024-01-01", "2024-01-10")
        df2 = provider2.fetch_ohlcv("TEST", "2024-01-01", "2024-01-10")

        assert not df1.equals(df2)
