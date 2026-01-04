"""Tests for Binance provider."""

from datetime import datetime
from unittest.mock import MagicMock, patch

import httpx
import pytest

from ml4t.data.providers.binance import BinanceProvider


class TestBinanceProvider:
    """Test Binance provider functionality."""

    def test_provider_name(self) -> None:
        """Test provider name."""
        provider_spot = BinanceProvider(market="spot")
        assert provider_spot.name == "binance"

        provider_futures = BinanceProvider(market="futures")
        assert provider_futures.name == "binance"

    def test_initialization(self) -> None:
        """Test provider initialization."""
        # Spot market
        provider1 = BinanceProvider()
        assert provider1.market == "spot"
        assert provider1.base_url == "https://api.binance.com/api/v3"

        # Futures market
        provider2 = BinanceProvider(market="futures")
        assert provider2.market == "futures"
        assert provider2.base_url == "https://fapi.binance.com/fapi/v1"

        # Invalid market
        with pytest.raises(ValueError, match="Invalid market"):
            BinanceProvider(market="options")

    def test_symbol_normalization(self) -> None:
        """Test symbol normalization to Binance format."""
        provider = BinanceProvider()

        # Different input formats
        assert provider._normalize_symbol("BTC") == "BTCUSDT"
        assert provider._normalize_symbol("BTC-USD") == "BTCUSDT"
        assert provider._normalize_symbol("BTC/USD") == "BTCUSDT"
        assert provider._normalize_symbol("BTCUSD") == "BTCUSDT"
        assert provider._normalize_symbol("BTCUSDT") == "BTCUSDT"
        assert provider._normalize_symbol("ETH") == "ETHUSDT"
        assert provider._normalize_symbol("eth") == "ETHUSDT"
        assert provider._normalize_symbol("ADA") == "ADAUSDT"
        assert provider._normalize_symbol("ADABTC") == "ADABTC"

    @patch("time.sleep")
    @patch("ml4t.data.providers.binance.httpx.Client")
    def test_fetch_daily_data(self, mock_client_class: MagicMock, mock_sleep: MagicMock) -> None:
        """Test fetching daily OHLCV data."""
        # Setup mock response - Binance kline format
        mock_response = MagicMock()
        mock_response.json.return_value = [
            [
                1704067200000,  # Open time (2024-01-01)
                "42000.00",  # Open
                "42500.00",  # High
                "41500.00",  # Low
                "42200.00",  # Close
                "1234.56",  # Volume
                1704153599999,  # Close time
                "51849600.00",  # Quote volume
                10000,  # Trade count
                "600.00",  # Taker buy base volume
                "25200000.00",  # Taker buy quote volume
                "0",  # Ignore
            ],
            [
                1704153600000,  # Open time (2024-01-02)
                "42200.00",
                "43000.00",
                "42000.00",
                "42800.00",
                "2345.67",
                1704239999999,
                "100000000.00",
                15000,
                "1200.00",
                "51600000.00",
                "0",
            ],
        ]
        mock_response.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.get.return_value = mock_response
        mock_client_class.return_value = mock_client

        # Fetch data
        provider = BinanceProvider()
        df = provider.fetch_ohlcv(
            symbol="BTC-USD",
            start="2024-01-01",
            end="2024-01-02",
            frequency="daily",
        )

        # Verify request
        mock_client.get.assert_called_once()
        call_args = mock_client.get.call_args
        assert call_args[0][0] == "https://api.binance.com/api/v3/klines"

        params = call_args[1]["params"]
        assert params["symbol"] == "BTCUSDT"
        assert params["interval"] == "1d"
        assert params["limit"] == 1000

        # Verify DataFrame
        assert len(df) == 2
        assert list(df.columns) == ["timestamp", "symbol", "open", "high", "low", "close", "volume"]

        # Check data values
        assert df["open"][0] == 42000.0
        assert df["close"][1] == 42800.0
        assert df["volume"][0] == 1234.56

        # Check timestamp is datetime
        assert isinstance(df["timestamp"][0], datetime)

    @patch("time.sleep")
    @patch("ml4t.data.providers.binance.httpx.Client")
    def test_fetch_minute_data(self, mock_client_class: MagicMock, mock_sleep: MagicMock) -> None:
        """Test fetching minute-level data."""
        # Setup mock response
        mock_response = MagicMock()
        mock_response.json.return_value = [
            [
                1704067200000,  # 2024-01-01 00:00
                "42000.00",
                "42010.00",
                "41990.00",
                "42005.00",
                "12.34",
                1704067259999,
                "518496.00",
                100,
                "6.00",
                "252000.00",
                "0",
            ],
            [
                1704153599000,  # 2024-01-01 23:59:59 - last minute of the day
                "42005.00",
                "42015.00",
                "42000.00",
                "42010.00",
                "23.45",
                1704153599999,
                "985000.00",
                150,
                "12.00",
                "504000.00",
                "0",
            ],
        ]
        mock_response.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.get.return_value = mock_response
        mock_client_class.return_value = mock_client

        # Fetch data
        provider = BinanceProvider()
        df = provider.fetch_ohlcv(
            symbol="ETH/USD",
            start="2024-01-01",
            end="2024-01-01",
            frequency="minute",
        )

        # Verify request
        mock_client.get.assert_called_once()
        call_args = mock_client.get.call_args
        params = call_args[1]["params"]
        assert params["symbol"] == "ETHUSDT"
        assert params["interval"] == "1m"

        # Verify DataFrame
        assert len(df) == 2
        assert df["close"][0] == 42005.0

    @patch("ml4t.data.providers.binance.httpx.Client")
    def test_rate_limit_handling(self, mock_client_class: MagicMock) -> None:
        """Test rate limit handling."""
        # First call returns rate limit error
        mock_response_429 = MagicMock()
        mock_response_429.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Rate limited",
            request=MagicMock(),
            response=MagicMock(status_code=429),
        )

        # Second call succeeds
        mock_response_success = MagicMock()
        mock_response_success.json.return_value = []
        mock_response_success.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.get.side_effect = [mock_response_429, mock_response_success]
        mock_client_class.return_value = mock_client

        # The provider now raises RateLimitError instead of sleeping
        provider = BinanceProvider()
        from ml4t.data.core.exceptions import RateLimitError

        # First call raises rate limit error (caught by retry decorator)
        with pytest.raises(RateLimitError):
            provider._fetch_and_transform_data("BTC", "2024-01-01", "2024-01-02", "daily")

        # Verify rate limit was detected
        assert mock_client.get.call_count == 1

    @patch("time.sleep")
    @patch("ml4t.data.providers.binance.httpx.Client")
    def test_invalid_symbol_handling(
        self, mock_client_class: MagicMock, mock_sleep: MagicMock
    ) -> None:
        """Test handling of invalid symbol."""
        # Setup mock 400 response
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Bad request",
            request=MagicMock(),
            response=MagicMock(status_code=400),
        )

        mock_client = MagicMock()
        mock_client.get.return_value = mock_response
        mock_client_class.return_value = mock_client

        # Should raise SymbolNotFoundError
        provider = BinanceProvider()
        from ml4t.data.core.exceptions import SymbolNotFoundError

        with pytest.raises(SymbolNotFoundError):
            provider.fetch_ohlcv(
                symbol="INVALID",
                start="2024-01-01",
                end="2024-01-02",
                frequency="daily",
            )

    def test_frequency_mapping(self) -> None:
        """Test frequency to interval mapping."""
        provider = BinanceProvider()

        # Valid frequencies
        assert provider.INTERVAL_MAP["minute"] == "1m"
        assert provider.INTERVAL_MAP["5minute"] == "5m"
        assert provider.INTERVAL_MAP["hourly"] == "1h"
        assert provider.INTERVAL_MAP["daily"] == "1d"
        assert provider.INTERVAL_MAP["weekly"] == "1w"
        assert provider.INTERVAL_MAP["monthly"] == "1M"

        # Invalid frequency should raise
        with pytest.raises(ValueError, match="Unsupported frequency"):
            provider.fetch_ohlcv(
                symbol="BTC",
                start="2024-01-01",
                end="2024-01-02",
                frequency="2minute",  # Not in map
            )

    @patch("time.sleep")
    @patch("ml4t.data.providers.binance.httpx.Client")
    def test_futures_market(self, mock_client_class: MagicMock, mock_sleep: MagicMock) -> None:
        """Test futures market endpoint."""
        mock_response = MagicMock()
        mock_response.json.return_value = []
        mock_response.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.get.return_value = mock_response
        mock_client_class.return_value = mock_client

        # Create futures provider
        provider = BinanceProvider(market="futures")
        provider.fetch_ohlcv(
            symbol="BTC",
            start="2024-01-01",
            end="2024-01-02",
            frequency="daily",
        )

        # Verify futures URL was used
        mock_client.get.assert_called_once()
        call_args = mock_client.get.call_args
        assert call_args[0][0] == "https://fapi.binance.com/fapi/v1/klines"

    # NOTE: get_available_symbols() is not part of the base provider interface
    # and is not required by TASK-008. Commenting out for now.
    # TODO: Implement get_available_symbols() as optional enhancement
    # @patch("ml4t.data.providers.binance.httpx.Client")
    # def test_get_available_symbols(self, mock_client_class: MagicMock) -> None:
    #     """Test getting available symbols."""
    #     mock_response = MagicMock()
    #     mock_response.json.return_value = {
    #         "symbols": [
    #             {"symbol": "BTCUSDT", "status": "TRADING"},
    #             {"symbol": "ETHUSDT", "status": "TRADING"},
    #             {"symbol": "BNBUSDT", "status": "TRADING"},
    #             {"symbol": "XYZUSDT", "status": "BREAK"},  # Not trading
    #         ]
    #     }
    #     mock_response.raise_for_status = MagicMock()

    #     mock_client = MagicMock()
    #     mock_client.get.return_value = mock_response
    #     mock_client_class.return_value = mock_client

    #     provider = BinanceProvider()
    #     symbols = provider.get_available_symbols()

    #     # Should only return trading symbols
    #     assert len(symbols) == 3
    #     assert "BTCUSDT" in symbols
    #     assert "ETHUSDT" in symbols
    #     assert "BNBUSDT" in symbols
    #     assert "XYZUSDT" not in symbols  # Status is BREAK
