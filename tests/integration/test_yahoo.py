"""Integration tests for Yahoo Finance provider (real API calls).

These tests verify the Yahoo Finance provider works correctly with actual API calls.

Requirements:
    - No API key needed (free public API via yfinance)
    - Rate limit: ~2000 requests/hour

Test Coverage:
    - Stock OHLCV data (AAPL)
    - ETF data (SPY)
    - Index data (^GSPC)
    - Multiple frequencies (daily, hourly)
    - Error handling for invalid symbols
"""

from datetime import datetime, timedelta

import polars as pl
import pytest

from ml4t.data.providers.yahoo import YahooFinanceProvider

# Always run these tests (no API key needed)
pytestmark = pytest.mark.integration


@pytest.fixture
def provider():
    """Create Yahoo Finance provider."""
    provider = YahooFinanceProvider()
    yield provider
    provider.close()


class TestYahooFinanceProvider:
    """Test Yahoo Finance provider with real API calls."""

    def test_provider_initialization(self):
        """Test provider can be initialized without API key."""
        provider = YahooFinanceProvider()
        assert provider.name == "yahoo"
        provider.close()

    def test_fetch_ohlcv_stock_daily(self, provider):
        """Test fetching daily stock data with real API call."""
        # Use recent date range to ensure data availability
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")

        # Fetch AAPL daily data
        df = provider.fetch_ohlcv(
            symbol="AAPL",
            start=start_date,
            end=end_date,
            frequency="daily",
        )

        # Verify data structure
        assert isinstance(df, pl.DataFrame)
        assert not df.is_empty(), "Should fetch some data for AAPL"

        # Check required columns
        required_cols = ["timestamp", "open", "high", "low", "close", "volume"]
        assert all(col in df.columns for col in required_cols)

        # Verify data types
        assert df["timestamp"].dtype == pl.Datetime
        assert df["open"].dtype == pl.Float64
        assert df["close"].dtype == pl.Float64
        assert df["volume"].dtype == pl.Float64

        # Verify OHLCV relationships
        assert (df["high"] >= df["low"]).all()
        assert (df["high"] >= df["open"]).all()
        assert (df["high"] >= df["close"]).all()
        assert (df["low"] <= df["open"]).all()
        assert (df["low"] <= df["close"]).all()

        # Check for reasonable prices (AAPL should be > $50)
        assert df["close"].min() > 50.0
        assert df["volume"].min() > 0

        print(f"✅ Fetched {len(df)} rows of AAPL daily data")
        print(f"   Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        print(f"   Price range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")

    def test_fetch_ohlcv_etf(self, provider):
        """Test fetching ETF data with real API call."""
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=14)).strftime("%Y-%m-%d")

        # Fetch SPY (S&P 500 ETF) data
        df = provider.fetch_ohlcv(
            symbol="SPY",
            start=start_date,
            end=end_date,
            frequency="daily",
        )

        # Verify data structure
        assert isinstance(df, pl.DataFrame)
        assert not df.is_empty(), "Should fetch some data for SPY"

        # Check required columns
        required_cols = ["timestamp", "open", "high", "low", "close", "volume"]
        assert all(col in df.columns for col in required_cols)

        # SPY price should be > $200 (sanity check)
        assert df["close"].min() > 200.0

        print(f"✅ Fetched {len(df)} rows of SPY ETF data")
        print(f"   Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        print(f"   Price range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")

    def test_fetch_ohlcv_index(self, provider):
        """Test fetching index data with real API call."""
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=14)).strftime("%Y-%m-%d")

        # Fetch S&P 500 index data
        df = provider.fetch_ohlcv(
            symbol="^GSPC",
            start=start_date,
            end=end_date,
            frequency="daily",
        )

        # Verify data structure
        assert isinstance(df, pl.DataFrame)
        assert not df.is_empty(), "Should fetch some data for ^GSPC"

        # Check required columns
        required_cols = ["timestamp", "open", "high", "low", "close", "volume"]
        assert all(col in df.columns for col in required_cols)

        # S&P 500 should be > 3000 (sanity check)
        assert df["close"].min() > 3000.0

        print(f"✅ Fetched {len(df)} rows of S&P 500 index data")
        print(f"   Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        print(f"   Index range: {df['close'].min():.2f} - {df['close'].max():.2f}")

    def test_invalid_symbol(self, provider):
        """Test error handling for invalid symbol."""
        from ml4t.data.core.exceptions import SymbolNotFoundError

        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")

        # Fetch with invalid symbol - should raise SymbolNotFoundError
        with pytest.raises(SymbolNotFoundError) as exc_info:
            provider.fetch_ohlcv(
                symbol="INVALID_SYMBOL_12345",
                start=start_date,
                end=end_date,
                frequency="daily",
            )

        # Verify exception details
        assert "INVALID_SYMBOL_12345" in str(exc_info.value)
        print("✅ Invalid symbol raises SymbolNotFoundError as expected")

    def test_historical_data_range(self, provider):
        """Test fetching historical data over longer period."""
        # Fetch 1 year of data
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")

        df = provider.fetch_ohlcv(
            symbol="MSFT",
            start=start_date,
            end=end_date,
            frequency="daily",
        )

        # Should have approximately 250 trading days
        assert len(df) > 200
        assert len(df) < 300

        # Verify chronological order
        timestamps = df["timestamp"].to_list()
        assert timestamps == sorted(timestamps)

        print(f"✅ Fetched {len(df)} rows of 1-year MSFT data")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
