"""Tests for CryptoCompare provider."""

from datetime import datetime
from unittest.mock import MagicMock, patch

import httpx
import pytest

from ml4t.data.providers.cryptocompare import CryptoCompareProvider


class TestCryptoCompareProvider:
    """Test CryptoCompare provider functionality."""

    def test_provider_name(self) -> None:
        """Test provider name."""
        provider = CryptoCompareProvider()
        assert provider.name == "cryptocompare"

    def test_initialization(self) -> None:
        """Test provider initialization."""
        # Without API key
        provider1 = CryptoCompareProvider()
        assert provider1.api_key is None
        assert provider1.exchange == "CCCAGG"

        # With API key
        provider2 = CryptoCompareProvider(api_key="test_key", exchange="Binance")
        assert provider2.api_key == "test_key"
        assert provider2.exchange == "Binance"

    def test_symbol_normalization(self) -> None:
        """Test symbol normalization."""
        provider = CryptoCompareProvider()

        # Different formats
        assert provider._normalize_symbol("BTC") == ("BTC", "USD")
        assert provider._normalize_symbol("BTC-USD") == ("BTC", "USD")
        assert provider._normalize_symbol("BTC/USD") == ("BTC", "USD")
        assert provider._normalize_symbol("BTCUSD") == ("BTC", "USD")
        assert provider._normalize_symbol("BTCUSDT") == ("BTC", "USDT")
        assert provider._normalize_symbol("ETH") == ("ETH", "USD")
        assert provider._normalize_symbol("eth-eur") == ("ETH", "EUR")

    @patch("ml4t.data.providers.cryptocompare.httpx.Client")
    def test_fetch_daily_data(self, mock_client_class: MagicMock) -> None:
        """Test fetching daily OHLCV data."""
        # Setup mock response
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "Response": "Success",
            "Data": {
                "Data": [
                    {
                        "time": 1704067200,  # 2024-01-01
                        "open": 42000.0,
                        "high": 42500.0,
                        "low": 41500.0,
                        "close": 42200.0,
                        "volumefrom": 1234.56,
                    },
                    {
                        "time": 1704153600,  # 2024-01-02
                        "open": 42200.0,
                        "high": 43000.0,
                        "low": 42000.0,
                        "close": 42800.0,
                        "volumefrom": 2345.67,
                    },
                ]
            },
        }
        mock_response.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.get.return_value = mock_response
        mock_client_class.return_value = mock_client

        # Fetch data
        provider = CryptoCompareProvider()
        df = provider.fetch_ohlcv(
            symbol="BTC-USD",
            start="2024-01-01",
            end="2024-01-02",
            frequency="daily",
        )

        # Verify request
        mock_client.get.assert_called_once()
        call_args = mock_client.get.call_args
        assert call_args[0][0] == "https://min-api.cryptocompare.com/data/v2/histoday"

        params = call_args[1]["params"]
        assert params["fsym"] == "BTC"
        assert params["tsym"] == "USD"
        assert params["e"] == "CCCAGG"

        # Verify DataFrame
        assert len(df) == 2
        # DataFrame now includes symbol column
        required_columns = ["timestamp", "open", "high", "low", "close", "volume"]
        assert all(col in df.columns for col in required_columns)

        # Check data values
        assert df["open"][0] == 42000.0
        assert df["close"][1] == 42800.0
        assert df["volume"][0] == 1234.56

        # Check timestamp conversion
        ts = df["timestamp"][0]
        assert isinstance(ts, datetime)
        assert ts.date() == datetime(2024, 1, 1).date()

    @patch("ml4t.data.providers.cryptocompare.httpx.Client")
    def test_fetch_minute_data(self, mock_client_class: MagicMock) -> None:
        """Test fetching minute-level data."""
        # Setup mock response
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "Response": "Success",
            "Data": {
                "Data": [
                    {
                        "time": 1704067200,  # 2024-01-01 00:00
                        "open": 42000.0,
                        "high": 42010.0,
                        "low": 41990.0,
                        "close": 42005.0,
                        "volumefrom": 12.34,
                    },
                    {
                        "time": 1704067260,  # 2024-01-01 00:01
                        "open": 42005.0,
                        "high": 42015.0,
                        "low": 42000.0,
                        "close": 42010.0,
                        "volumefrom": 23.45,
                    },
                ]
            },
        }
        mock_response.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.get.return_value = mock_response
        mock_client_class.return_value = mock_client

        # Fetch data
        provider = CryptoCompareProvider()
        df = provider.fetch_ohlcv(
            symbol="ETH/USD",
            start="2024-01-01",
            end="2024-01-01",
            frequency="minute",
        )

        # Verify request
        mock_client.get.assert_called_once()
        call_args = mock_client.get.call_args
        assert call_args[0][0] == "https://min-api.cryptocompare.com/data/v2/histominute"

        params = call_args[1]["params"]
        assert params["fsym"] == "ETH"
        assert params["tsym"] == "USD"

        # Verify DataFrame
        assert len(df) == 2
        assert df["close"][0] == 42005.0

    @patch("ml4t.data.providers.cryptocompare.httpx.Client")
    def test_api_error_handling(self, mock_client_class: MagicMock) -> None:
        """Test API error handling."""
        # Setup mock error response
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "Response": "Error",
            "Message": "Invalid API key",
        }
        mock_response.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.get.return_value = mock_response
        mock_client_class.return_value = mock_client

        # Should raise ValueError
        provider = CryptoCompareProvider()
        with pytest.raises(ValueError, match="API error: Invalid API key"):
            provider.fetch_ohlcv(
                symbol="BTC",
                start="2024-01-01",
                end="2024-01-02",
                frequency="daily",
            )

    @patch("ml4t.data.providers.cryptocompare.httpx.Client")
    def test_rate_limit_handling(self, mock_client_class: MagicMock) -> None:
        """Test rate limit handling."""
        # Setup mock rate limit response
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Rate limited",
            request=MagicMock(),
            response=MagicMock(status_code=429),
        )

        # Second call succeeds
        mock_success_response = MagicMock()
        mock_success_response.json.return_value = {"Response": "Success", "Data": {"Data": []}}
        mock_success_response.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.get.side_effect = [mock_response, mock_success_response]
        mock_client_class.return_value = mock_client

        # Should retry after rate limit
        provider = CryptoCompareProvider()

        with patch("time.sleep") as mock_sleep:
            provider.fetch_ohlcv(
                symbol="BTC",
                start="2024-01-01",
                end="2024-01-02",
                frequency="daily",
            )

            # Should have slept for 60 seconds
            mock_sleep.assert_called_with(60)

        # Should have made 2 requests
        assert mock_client.get.call_count == 2

    @patch("ml4t.data.providers.cryptocompare.httpx.Client")
    def test_empty_data_handling(self, mock_client_class: MagicMock) -> None:
        """Test handling of empty data response."""
        # Setup mock empty response
        mock_response = MagicMock()
        mock_response.json.return_value = {"Response": "Success", "Data": {"Data": []}}
        mock_response.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.get.return_value = mock_response
        mock_client_class.return_value = mock_client

        # Should return empty DataFrame
        provider = CryptoCompareProvider()
        df = provider.fetch_ohlcv(
            symbol="UNKNOWN",
            start="2024-01-01",
            end="2024-01-02",
            frequency="daily",
        )

        assert df.is_empty()

    def test_frequency_mapping(self) -> None:
        """Test frequency mapping."""
        provider = CryptoCompareProvider()

        # Valid frequencies
        assert provider.FREQUENCY_MAP["minute"] == "histominute"
        assert provider.FREQUENCY_MAP["hourly"] == "histohour"
        assert provider.FREQUENCY_MAP["daily"] == "histoday"

        # Invalid frequency should raise
        with pytest.raises(ValueError, match="Unsupported frequency"):
            provider.fetch_ohlcv(
                symbol="BTC",
                start="2024-01-01",
                end="2024-01-02",
                frequency="weekly",  # Not supported
            )

    def test_with_api_key(self) -> None:
        """Test provider with API key sets authorization header."""
        # Create provider with API key
        provider = CryptoCompareProvider(api_key="test_api_key")

        # Check headers were set on the session
        assert "authorization" in provider.session.headers
        assert provider.session.headers["authorization"] == "Apikey test_api_key"

        provider.close()
