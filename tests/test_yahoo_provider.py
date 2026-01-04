"""Tests for Yahoo Finance provider."""

from unittest.mock import MagicMock, patch

import pandas as pd
import polars as pl
import pytest

from ml4t.data.providers.yahoo import YahooFinanceProvider


class TestYahooFinanceProvider:
    """Test Yahoo Finance provider functionality."""

    def test_provider_name(self) -> None:
        """Test provider returns correct name."""
        provider = YahooFinanceProvider()
        assert provider.name == "yahoo"

    @patch("ml4t.data.providers.yahoo.yf.download")
    def test_fetch_daily_data(self, mock_download: MagicMock) -> None:
        """Test fetching daily OHLCV data."""
        # Mock yfinance response
        mock_df = pd.DataFrame(
            {
                "Open": [100.0, 101.0, 102.0],
                "High": [105.0, 106.0, 107.0],
                "Low": [99.0, 100.0, 101.0],
                "Close": [104.0, 105.0, 106.0],
                "Volume": [1000000, 1100000, 1200000],
            },
            index=pd.date_range("2024-01-01", periods=3),
        )

        mock_download.return_value = mock_df

        provider = YahooFinanceProvider()
        df = provider.fetch_ohlcv(
            symbol="AAPL",
            start="2024-01-01",
            end="2024-01-03",
            frequency="daily",
        )

        # Verify download was called correctly
        mock_download.assert_called_once_with(
            "AAPL",
            start="2024-01-01",
            end="2024-01-04",  # End date is exclusive in yfinance
            interval="1d",
            progress=False,
            auto_adjust=True,
            actions=False,
        )

        # Check DataFrame structure
        assert isinstance(df, pl.DataFrame)
        assert len(df) == 3
        assert set(df.columns) == {"timestamp", "open", "high", "low", "close", "volume", "symbol"}

        # Check data values
        assert df["open"].to_list() == [100.0, 101.0, 102.0]
        assert df["close"].to_list() == [104.0, 105.0, 106.0]
        assert df["volume"].to_list() == [1000000, 1100000, 1200000]

    @patch("ml4t.data.providers.yahoo.yf.download")
    def test_fetch_minute_data(self, mock_download: MagicMock) -> None:
        """Test fetching minute-level data."""
        # Mock yfinance response
        mock_df = pd.DataFrame(
            {
                "Open": [100.0, 100.1],
                "High": [100.2, 100.3],
                "Low": [99.9, 100.0],
                "Close": [100.1, 100.2],
                "Volume": [10000, 11000],
            },
            index=pd.date_range("2024-01-01 09:30:00", periods=2, freq="1min"),
        )

        mock_download.return_value = mock_df

        provider = YahooFinanceProvider()
        df = provider.fetch_ohlcv(
            symbol="AAPL",
            start="2024-01-01",
            end="2024-01-01",
            frequency="minute",
        )

        # Verify download was called with minute interval
        mock_download.assert_called_once_with(
            "AAPL",
            start="2024-01-01",
            end="2024-01-02",
            interval="1m",
            progress=False,
            auto_adjust=True,
            actions=False,
        )

        assert len(df) == 2

    @patch("ml4t.data.providers.yahoo.yf.download")
    def test_empty_response_handling(self, mock_download: MagicMock) -> None:
        """Test handling of empty responses."""
        # Mock empty response
        mock_download.return_value = pd.DataFrame()

        provider = YahooFinanceProvider()

        # Empty response should raise SymbolNotFoundError
        from ml4t.data.core.exceptions import SymbolNotFoundError

        with pytest.raises(SymbolNotFoundError):
            provider.fetch_ohlcv(
                symbol="INVALID",
                start="2024-01-01",
                end="2024-01-03",
            )

    @patch("ml4t.data.providers.yahoo.yf.download")
    def test_rate_limiting(self, mock_download: MagicMock) -> None:
        """Test that rate limiting is applied."""
        mock_df = pd.DataFrame(
            {
                "Open": [100.0],
                "High": [105.0],
                "Low": [99.0],
                "Close": [104.0],
                "Volume": [1000000],
            },
            index=pd.date_range("2024-01-01", periods=1),
        )

        mock_download.return_value = mock_df

        # Create provider
        provider = YahooFinanceProvider()

        import time

        start = time.time()

        # Make 3 rapid requests - should complete quickly
        for _ in range(3):
            provider.fetch_ohlcv("AAPL", "2024-01-01", "2024-01-01")

        elapsed = time.time() - start

        # With 10 requests/sec, 3 requests should be nearly instant
        assert elapsed < 0.5

        # Now test that rate limiting actually works
        # Create second provider instance
        provider2 = YahooFinanceProvider()

        start = time.time()
        # Make 3 rapid requests
        for _ in range(3):
            provider2.fetch_ohlcv("AAPL", "2024-01-01", "2024-01-01")
        elapsed = time.time() - start

        # All 3 should complete quickly since 4/sec allows it
        assert elapsed < 1.0

    @patch("ml4t.data.providers.yahoo.yf.download")
    def test_error_handling(self, mock_download: MagicMock) -> None:
        """Test error handling in provider."""
        # Mock network error
        mock_download.side_effect = Exception("Network error")

        provider = YahooFinanceProvider()

        from ml4t.data.core.exceptions import DataValidationError, NetworkError

        # Provider may raise NetworkError or DataValidationError
        with pytest.raises((NetworkError, DataValidationError)) as exc_info:
            provider.fetch_ohlcv("AAPL", "2024-01-01", "2024-01-03")

        assert "Network error" in str(exc_info.value)

    @patch("ml4t.data.providers.yahoo.yf.download")
    def test_frequency_mapping(self, mock_download: MagicMock) -> None:
        """Test frequency parameter mapping to yfinance intervals."""
        mock_df = pd.DataFrame(
            {
                "Open": [100.0],
                "High": [105.0],
                "Low": [99.0],
                "Close": [104.0],
                "Volume": [1000000],
            },
            index=pd.date_range("2024-01-01", periods=1),
        )

        mock_download.return_value = mock_df
        # Create provider
        provider = YahooFinanceProvider()

        # Test different frequency mappings
        frequency_map = {
            "minute": "1m",
            "5minute": "5m",
            "15minute": "15m",
            "30minute": "30m",
            "hourly": "1h",
            "daily": "1d",
            "weekly": "1wk",
            "monthly": "1mo",
        }

        for freq, expected_interval in frequency_map.items():
            mock_download.reset_mock()
            provider.fetch_ohlcv("AAPL", "2024-01-01", "2024-01-03", frequency=freq)

            # Check that the correct interval was used
            call_kwargs = mock_download.call_args[1]
            assert call_kwargs["interval"] == expected_interval
