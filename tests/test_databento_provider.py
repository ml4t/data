"""Tests for Databento provider implementation."""

import os
from unittest.mock import Mock, patch

import pandas as pd
import polars as pl
import pytest

# Import real databento exceptions (databento package is installed)
from databento.common.error import BentoClientError, BentoServerError

from ml4t.data.core.exceptions import (
    AuthenticationError,
    NetworkError,
    RateLimitError,
)
from ml4t.data.providers.databento import DataBentoProvider


class TestDataBentoProvider:
    """Test suite for Databento provider."""

    @pytest.fixture
    def mock_client(self):
        """Create mock Databento client."""
        client = Mock()
        client.metadata.list_datasets.return_value = ["GLBX.MDP3", "XNAS.ITCH"]
        client.metadata.list_schemas.return_value = ["ohlcv-1m", "trades", "tbbo"]
        return client

    @pytest.fixture
    def provider(self, mock_client):
        """Create provider with mocked client."""
        with patch("ml4t.data.providers.databento.Historical") as mock_historical:
            mock_historical.return_value = mock_client
            provider = DataBentoProvider(api_key="test_key")
            provider.client = mock_client
            return provider

    def test_initialization_with_api_key(self):
        """Test provider initialization with API key."""
        with patch("ml4t.data.providers.databento.Historical") as mock_historical:
            provider = DataBentoProvider(api_key="test_key")
            assert provider.api_key == "test_key"
            assert provider.dataset == "GLBX.MDP3"
            assert provider.default_schema == "ohlcv-1m"
            mock_historical.assert_called_once_with("test_key")

    def test_initialization_from_environment(self, monkeypatch):
        """Test provider initialization from environment variable."""
        monkeypatch.setenv("DATABENTO_API_KEY", "env_key")
        with patch("ml4t.data.providers.databento.Historical") as mock_historical:
            provider = DataBentoProvider()
            assert provider.api_key == "env_key"
            mock_historical.assert_called_once_with("env_key")

    def test_initialization_without_api_key(self):
        """Test that initialization fails without API key."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(AuthenticationError, match="API key not provided"):
                DataBentoProvider()

    def test_name(self, provider):
        """Test provider name."""
        assert provider.name == "databento"

    def test_fetch_raw_data_success(self, provider, mock_client):
        """Test successful raw data fetch."""
        # Mock response
        mock_response = Mock()
        mock_response.to_df = Mock(
            return_value=pd.DataFrame(
                {
                    "ts_event": [1704067200000000000],  # 2024-01-01 in nanoseconds
                    "open": [4700.0],
                    "high": [4710.0],
                    "low": [4695.0],
                    "close": [4705.0],
                    "volume": [1000.0],
                }
            )
        )

        mock_client.timeseries.get_range.return_value = mock_response

        result = provider._fetch_raw_data("ES.v.0", "2024-01-01", "2024-01-31", "daily")

        assert result == mock_response
        mock_client.timeseries.get_range.assert_called_once()

        # Check call arguments
        call_args = mock_client.timeseries.get_range.call_args
        assert call_args.kwargs["symbols"] == ["ES.v.0"]
        assert call_args.kwargs["schema"] == "ohlcv-1d"

    def test_fetch_raw_data_with_session_date_adjustment(self, mock_client):
        """Test that session date adjustment works when enabled."""
        with patch("ml4t.data.providers.databento.Historical") as mock_historical:
            mock_historical.return_value = mock_client
            # Enable session date adjustment
            provider = DataBentoProvider(
                api_key="test_key",
                adjust_session_dates=True,
                session_start_hour_utc=22,  # 10pm UTC
            )
            provider.client = mock_client

            mock_response = Mock()
            mock_client.timeseries.get_range.return_value = mock_response

            provider._fetch_raw_data("ES.v.0", "2024-01-01", "2024-01-01", "daily")

            # Check that start date was adjusted back one day
            call_args = mock_client.timeseries.get_range.call_args
            start_dt = call_args.kwargs["start"]
            # Should be Dec 31, 10pm UTC
            assert start_dt.day == 31
            assert start_dt.month == 12
            assert start_dt.year == 2023
            assert start_dt.hour == 22

    def test_fetch_raw_data_without_session_adjustment(self, provider, mock_client):
        """Test that dates are not adjusted when session adjustment is disabled."""
        # Default provider has adjust_session_dates=False
        mock_response = Mock()
        mock_client.timeseries.get_range.return_value = mock_response

        provider._fetch_raw_data("AAPL", "2024-01-01", "2024-01-01", "daily")

        # Check that start date was NOT adjusted
        call_args = mock_client.timeseries.get_range.call_args
        start_dt = call_args.kwargs["start"]
        # Should still be Jan 1
        assert start_dt.day == 1
        assert start_dt.month == 1
        assert start_dt.year == 2024
        assert start_dt.hour == 0

    def test_transform_data_ohlcv(self, provider):
        """Test transforming OHLCV data to standard schema."""
        # Create mock response with to_df method
        mock_response = Mock()
        mock_response.to_df.return_value = pd.DataFrame(
            {
                "ts_event": [1704067200000000000, 1704067260000000000],
                "open": [4700.0, 4705.0],
                "high": [4710.0, 4715.0],
                "low": [4695.0, 4700.0],
                "close": [4705.0, 4710.0],
                "volume": [1000.0, 1100.0],
            }
        )

        df = provider._transform_data(mock_response, "ES.v.0")

        # Check schema
        assert "timestamp" in df.columns
        assert "open" in df.columns
        assert "high" in df.columns
        assert "low" in df.columns
        assert "close" in df.columns
        assert "volume" in df.columns
        assert "symbol" in df.columns

        # Check data types
        assert df["timestamp"].dtype == pl.Datetime("ns")
        assert df["open"].dtype == pl.Float64
        assert df["symbol"][0] == "ES.v.0"

        # Check sorting
        timestamps = df["timestamp"].to_list()
        assert timestamps == sorted(timestamps)

    def test_transform_data_trades(self, provider):
        """Test transforming trade data."""
        mock_response = Mock()
        mock_response.to_df.return_value = pd.DataFrame(
            {
                "ts_event": [1704067200000000000],
                "price": [4700.0],
                "size": [10],
                "side": ["B"],
            }
        )

        df = provider._transform_data(mock_response, "AAPL")

        assert "timestamp" in df.columns
        assert "price" in df.columns
        assert "size" in df.columns
        assert "symbol" in df.columns
        assert df["symbol"][0] == "AAPL"

    def test_map_frequency_to_schema(self, provider):
        """Test frequency to schema mapping."""
        assert provider._map_frequency_to_schema("daily") == "ohlcv-1d"
        assert provider._map_frequency_to_schema("day") == "ohlcv-1d"
        assert provider._map_frequency_to_schema("1d") == "ohlcv-1d"

        assert provider._map_frequency_to_schema("minute") == "ohlcv-1m"
        assert provider._map_frequency_to_schema("min") == "ohlcv-1m"
        assert provider._map_frequency_to_schema("1m") == "ohlcv-1m"

        assert provider._map_frequency_to_schema("hour") == "ohlcv-1h"
        assert provider._map_frequency_to_schema("hourly") == "ohlcv-1h"

        assert provider._map_frequency_to_schema("trades") == "trades"
        assert provider._map_frequency_to_schema("quotes") == "tbbo"
        assert provider._map_frequency_to_schema("mbo") == "mbo"

    def test_session_date_configuration(self):
        """Test session date adjustment configuration."""
        with patch("ml4t.data.providers.databento.Historical"):
            # Test with adjustment enabled
            provider1 = DataBentoProvider(
                api_key="test_key",
                adjust_session_dates=True,
                session_start_hour_utc=17,  # 5pm UTC
            )
            assert provider1.adjust_session_dates is True
            assert provider1.session_start_hour_utc == 17

            # Test with adjustment disabled (default)
            provider2 = DataBentoProvider(api_key="test_key")
            assert provider2.adjust_session_dates is False

    def test_fetch_continuous_futures(self, provider, mock_client):
        """Test fetching continuous futures contracts."""
        mock_response = Mock()
        mock_response.to_df.return_value = pd.DataFrame(
            {
                "ts_event": [1704067200000000000],
                "open": [4700.0],
                "high": [4710.0],
                "low": [4695.0],
                "close": [4705.0],
                "volume": [1000.0],
            }
        )
        mock_client.timeseries.get_range.return_value = mock_response

        df = provider.fetch_continuous_futures("ES", "2024-01-01", "2024-01-31", "daily", version=0)

        # Check that it called with correct continuous symbol
        call_args = mock_client.timeseries.get_range.call_args
        assert call_args.kwargs["symbols"] == ["ES.v.0"]

        assert not df.is_empty()
        assert "timestamp" in df.columns

    def test_fetch_multiple_schemas(self, provider, mock_client):
        """Test fetching multiple schemas."""
        # Mock different responses for different schemas
        ohlcv_response = Mock()
        ohlcv_response.to_df.return_value = pd.DataFrame(
            {
                "ts_event": [1704067200000000000],
                "open": [100.0],
                "high": [101.0],
                "low": [99.0],
                "close": [100.5],
                "volume": [1000000.0],
            }
        )

        trades_response = Mock()
        trades_response.to_df.return_value = pd.DataFrame(
            {
                "ts_event": [1704067200000000000],
                "price": [100.0],
                "size": [100],
            }
        )

        mock_client.timeseries.get_range.side_effect = [ohlcv_response, trades_response]

        results = provider.fetch_multiple_schemas(
            "AAPL", "2024-01-01", "2024-01-01", ["ohlcv-1m", "trades"]
        )

        assert "ohlcv-1m" in results
        assert "trades" in results
        assert results["ohlcv-1m"] is not None
        assert results["trades"] is not None

    def test_get_available_datasets(self, provider, mock_client):
        """Test getting available datasets."""
        datasets = provider.get_available_datasets()
        assert datasets == ["GLBX.MDP3", "XNAS.ITCH"]
        mock_client.metadata.list_datasets.assert_called_once()

    def test_get_available_schemas(self, provider, mock_client):
        """Test getting available schemas."""
        schemas = provider.get_available_schemas()
        assert schemas == ["ohlcv-1m", "trades", "tbbo"]
        mock_client.metadata.list_schemas.assert_called_once_with(dataset="GLBX.MDP3")

    def test_error_handling_authentication(self, provider, mock_client):
        """Test authentication error handling."""
        # Use module-level BentoClientError (imported from real databento package)
        mock_client.timeseries.get_range.side_effect = BentoClientError(
            "Unauthorized: Invalid API key"
        )

        with pytest.raises(AuthenticationError, match="authentication failed"):
            provider._fetch_raw_data("ES.v.0", "2024-01-01", "2024-01-31", "daily")

    def test_error_handling_rate_limit(self, provider, mock_client):
        """Test rate limit error handling."""
        # Use module-level BentoClientError (imported from real databento package)
        mock_client.timeseries.get_range.side_effect = BentoClientError("Rate limit exceeded")

        with pytest.raises(RateLimitError, match="(?i)rate limit exceeded"):
            provider._fetch_raw_data("ES.v.0", "2024-01-01", "2024-01-31", "daily")

    def test_error_handling_server_error(self, provider, mock_client):
        """Test server error handling."""
        # Use module-level BentoServerError (imported from real databento package)
        mock_client.timeseries.get_range.side_effect = BentoServerError("Internal server error")

        with pytest.raises(NetworkError, match="server error"):
            provider._fetch_raw_data("ES.v.0", "2024-01-01", "2024-01-31", "daily")

    def test_integration_with_base_provider(self, provider, mock_client):
        """Test integration with base provider's template method."""
        mock_response = Mock()
        mock_response.to_df.return_value = pd.DataFrame(
            {
                "ts_event": [1704067200000000000],
                "open": [4700.0],
                "high": [4710.0],
                "low": [4695.0],
                "close": [4705.0],
                "volume": [1000.0],
            }
        )
        mock_client.timeseries.get_range.return_value = mock_response

        # Call the main fetch_ohlcv method (from base class)
        df = provider.fetch_ohlcv("ES.v.0", "2024-01-01", "2024-01-31", "daily")

        # Verify the full pipeline worked
        assert not df.is_empty()
        assert "timestamp" in df.columns
        assert "open" in df.columns
        assert df["open"][0] == 4700.0


@pytest.mark.integration
class TestDataBentoProviderIntegration:
    """Integration tests for Databento provider (requires API key)."""

    @pytest.mark.skipif(not os.getenv("DATABENTO_API_KEY"), reason="DATABENTO_API_KEY not set")
    def test_real_api_fetch(self):
        """Test with real Databento API (if key available)."""
        provider = DataBentoProvider()

        # Fetch a small amount of data to minimize costs
        # Note: ES.v.0 continuous futures don't work with GLBX.MDP3 dataset
        # Using specific contract instead
        df = provider.fetch_ohlcv(
            "ESH4",  # E-mini S&P March 2024 contract
            "2024-01-02",  # Single day
            "2024-01-02",
            "daily",
        )

        assert not df.is_empty()
        assert "timestamp" in df.columns
        assert "open" in df.columns
        assert "close" in df.columns

        # Verify data quality
        assert df["high"].max() >= df["low"].min()
        assert df["high"].max() >= df["open"].max()
        assert df["high"].max() >= df["close"].max()

    @pytest.mark.skipif(not os.getenv("DATABENTO_API_KEY"), reason="DATABENTO_API_KEY not set")
    @pytest.mark.skip(reason="DataBento paid tier required")
    def test_continuous_contract_real(self):
        """Test continuous contract with real API."""
        provider = DataBentoProvider()

        # Note: Continuous futures (symbol.v.0) may not work with GLBX.MDP3
        # This test is expected to handle the error gracefully
        try:
            df = provider.fetch_continuous_futures(
                "CL",  # Crude oil
                "2024-01-02",
                "2024-01-02",
                "daily",
                version=0,  # Front month
            )

            # If it works (unlikely with GLBX.MDP3):
            assert not df.is_empty()
            assert "symbol" in df.columns
            assert "CL.v.0" in df["symbol"][0]
        except Exception as e:
            # Expected to fail with GLBX.MDP3 dataset
            # Continuous futures require specific dataset configuration
            assert "No data available" in str(e) or "not resolve" in str(e)
            # This is expected behavior
