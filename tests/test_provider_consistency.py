"""Provider API consistency tests.

These tests ensure all providers follow the same API contracts:
- Standard column ordering
- Frequency parameter handling
- Empty DataFrame schemas
- Required column presence
"""

import polars as pl

from ml4t.data.providers.mock import MockProvider
from ml4t.data.providers.yahoo import YahooFinanceProvider


class TestStandardColumnOrder:
    """Test that all providers return columns in standard order."""

    STANDARD_COLUMNS = ["timestamp", "symbol", "open", "high", "low", "close", "volume"]

    def test_mock_provider_returns_standard_columns(self):
        """Mock provider returns columns in standard order."""
        provider = MockProvider()
        df = provider.fetch_ohlcv("AAPL", "2024-01-01", "2024-01-05", "daily")

        # Check column order
        assert list(df.columns[:7]) == self.STANDARD_COLUMNS

    def test_yahoo_provider_empty_dataframe_has_standard_columns(self):
        """Yahoo provider empty DataFrame has correct schema."""
        provider = YahooFinanceProvider()
        df = provider._create_empty_dataframe()

        # Empty DataFrame should have same columns
        assert list(df.columns) == self.STANDARD_COLUMNS


class TestDataTypes:
    """Test that returned DataFrames have correct data types."""

    def test_timestamp_is_datetime(self):
        """Timestamp column is datetime type."""
        provider = MockProvider()
        df = provider.fetch_ohlcv("AAPL", "2024-01-01", "2024-01-05", "daily")

        if not df.is_empty():
            assert df["timestamp"].dtype in [
                pl.Datetime,
                pl.Datetime("ns", "UTC"),
                pl.Datetime("ms", "UTC"),
                pl.Datetime("us", "UTC"),
            ]

    def test_ohlcv_are_float64(self):
        """OHLCV columns are Float64."""
        provider = MockProvider()
        df = provider.fetch_ohlcv("AAPL", "2024-01-01", "2024-01-05", "daily")

        if not df.is_empty():
            for col in ["open", "high", "low", "close", "volume"]:
                assert df[col].dtype == pl.Float64, f"{col} should be Float64"

    def test_symbol_is_string(self):
        """Symbol column is string type."""
        provider = MockProvider()
        df = provider.fetch_ohlcv("AAPL", "2024-01-01", "2024-01-05", "daily")

        if not df.is_empty():
            assert df["symbol"].dtype in [pl.String, pl.Utf8]


class TestOHLCInvariants:
    """Test OHLC invariants are maintained."""

    def test_high_greater_than_or_equal_to_low(self):
        """High is always >= Low."""
        provider = MockProvider()
        df = provider.fetch_ohlcv("AAPL", "2024-01-01", "2024-01-05", "daily")

        if not df.is_empty():
            assert (df["high"] >= df["low"]).all()

    def test_high_greater_than_or_equal_to_open(self):
        """High is always >= Open."""
        provider = MockProvider()
        df = provider.fetch_ohlcv("AAPL", "2024-01-01", "2024-01-05", "daily")

        if not df.is_empty():
            assert (df["high"] >= df["open"]).all()

    def test_high_greater_than_or_equal_to_close(self):
        """High is always >= Close."""
        provider = MockProvider()
        df = provider.fetch_ohlcv("AAPL", "2024-01-01", "2024-01-05", "daily")

        if not df.is_empty():
            assert (df["high"] >= df["close"]).all()

    def test_low_less_than_or_equal_to_open(self):
        """Low is always <= Open."""
        provider = MockProvider()
        df = provider.fetch_ohlcv("AAPL", "2024-01-01", "2024-01-05", "daily")

        if not df.is_empty():
            assert (df["low"] <= df["open"]).all()

    def test_low_less_than_or_equal_to_close(self):
        """Low is always <= Close."""
        provider = MockProvider()
        df = provider.fetch_ohlcv("AAPL", "2024-01-01", "2024-01-05", "daily")

        if not df.is_empty():
            assert (df["low"] <= df["close"]).all()


class TestTimestampBehavior:
    """Test timestamp sorting and uniqueness."""

    def test_data_sorted_by_timestamp(self):
        """Data is sorted by timestamp ascending."""
        provider = MockProvider()
        df = provider.fetch_ohlcv("AAPL", "2024-01-01", "2024-01-10", "daily")

        if len(df) > 1:
            # Check timestamps are monotonically increasing
            timestamps = df["timestamp"].to_list()
            assert timestamps == sorted(timestamps)

    def test_no_duplicate_timestamps(self):
        """No duplicate timestamps in results."""
        provider = MockProvider()
        df = provider.fetch_ohlcv("AAPL", "2024-01-01", "2024-01-10", "daily")

        if not df.is_empty():
            assert df["timestamp"].n_unique() == len(df)


class TestSymbolHandling:
    """Test symbol column behavior."""

    def test_symbol_uppercase(self):
        """Symbol column values are uppercase."""
        provider = MockProvider()
        df = provider.fetch_ohlcv("aapl", "2024-01-01", "2024-01-05", "daily")

        if not df.is_empty():
            # All symbols should be uppercase
            symbols = df["symbol"].unique().to_list()
            for symbol in symbols:
                assert symbol == symbol.upper(), f"Symbol {symbol} should be uppercase"


class TestFrequencyParameter:
    """Test frequency parameter handling."""

    def test_daily_frequency_accepted(self):
        """Daily frequency is accepted by all providers."""
        provider = MockProvider()
        # Should not raise
        df = provider.fetch_ohlcv("AAPL", "2024-01-01", "2024-01-05", "daily")
        assert df is not None


class TestEmptyDataFrameSchema:
    """Test empty DataFrame has correct schema."""

    def test_empty_dataframe_has_all_columns(self):
        """Empty DataFrame has all required columns."""
        provider = YahooFinanceProvider()
        df = provider._create_empty_dataframe()

        required_columns = ["timestamp", "symbol", "open", "high", "low", "close", "volume"]
        for col in required_columns:
            assert col in df.columns, f"Missing column: {col}"

    def test_empty_dataframe_is_empty(self):
        """Empty DataFrame has zero rows."""
        provider = YahooFinanceProvider()
        df = provider._create_empty_dataframe()
        assert len(df) == 0
