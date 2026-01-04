"""Tests for DataManager unified interface."""

from __future__ import annotations

import os
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import polars as pl
import pytest

from ml4t.data.data_manager import DataManager
from ml4t.data.managers.provider_manager import ProviderRouter


class TestDataManagerInitialization:
    """Test DataManager initialization and configuration."""

    def test_init_with_defaults(self):
        """Test DataManager initialization with default settings."""
        dm = DataManager()
        assert dm is not None
        assert dm.config is not None
        assert dm.providers == {}  # No providers loaded by default

    @pytest.mark.skip(reason="Config initialization not implemented in PRE-RELEASE")
    def test_init_with_env_vars(self):
        """Test DataManager initialization from environment variables."""
        with patch.dict(
            os.environ,
            {
                "CRYPTOCOMPARE_API_KEY": "test_key",
                "DATABENTO_API_KEY": "db-test",
                "OANDA_API_KEY": "oanda_test",
            },
        ):
            dm = DataManager()
            assert "CRYPTOCOMPARE_API_KEY" in os.environ
            # Provider instances are created lazily
            assert dm._available_providers == ["cryptocompare", "databento", "oanda"]

    def test_init_with_yaml_config(self):
        """Test DataManager initialization from YAML config file."""
        config_content = r"""
providers:
  cryptocompare:
    api_key: ${CRYPTOCOMPARE_API_KEY}
    rate_limit: 10
  databento:
    api_key: ${DATABENTO_API_KEY}
    dataset: GLBX.MDP3
  oanda:
    api_key: ${OANDA_API_KEY}
    account_id: test_account

routing:
  patterns:
    - pattern: '^[A-Z]{6}$'
      provider: oanda
    - pattern: '^(BTC|ETH|SOL)'
      provider: cryptocompare
    - pattern: '\.(v|V)\.[0-9]+$'
      provider: databento

defaults:
  output_format: polars
  timezone: UTC
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(config_content)
            config_path = f.name

        try:
            dm = DataManager(config_path=config_path)
            assert dm.config is not None
            assert dm.config.get("providers") is not None
            assert dm.config["defaults"]["output_format"] == "polars"
        finally:
            Path(config_path).unlink()

    def test_config_hierarchy(self):
        """Test configuration hierarchy: defaults < YAML < env < params."""
        config_content = """
defaults:
  output_format: pandas
providers:
  cryptocompare:
    rate_limit: 5
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(config_content)
            config_path = f.name

        try:
            # Parameters override everything
            dm = DataManager(
                config_path=config_path,
                output_format="lazy",
                providers={"cryptocompare": {"rate_limit": 20}},
            )
            assert dm.output_format == "lazy"  # Parameter overrides YAML
            assert dm.config["providers"]["cryptocompare"]["rate_limit"] == 20
        finally:
            Path(config_path).unlink()


class TestProviderRouting:
    """Test provider routing and symbol pattern matching."""

    def test_forex_routing(self):
        """Test that forex symbols route to OANDA."""
        router = ProviderRouter()
        router.add_pattern(r"^[A-Z]{6}$", "oanda")  # EURUSD format
        router.add_pattern(r"^[A-Z]{3}_[A-Z]{3}$", "oanda")  # EUR_USD format

        assert router.get_provider("EURUSD") == "oanda"
        assert router.get_provider("EUR_USD") == "oanda"
        assert router.get_provider("GBPJPY") == "oanda"

    def test_crypto_routing(self):
        """Test that crypto symbols route to CryptoCompare."""
        router = ProviderRouter()
        router.add_pattern(r"^(BTC|ETH|SOL|ADA|DOT)", "cryptocompare")
        router.add_pattern(r"^[A-Z]{3,5}(-USD)?$", "cryptocompare")  # Generic crypto

        assert router.get_provider("BTC") == "cryptocompare"
        assert router.get_provider("BTC-USD") == "cryptocompare"
        assert router.get_provider("ETH") == "cryptocompare"
        assert router.get_provider("SOL-USD") == "cryptocompare"

    def test_futures_routing(self):
        """Test that futures symbols route to Databento."""
        router = ProviderRouter()
        router.add_pattern(r"^[A-Z]+\.(v|V)\.[0-9]+$", "databento")  # Continuous futures
        router.add_pattern(
            r"^[A-Z]{2,3}[FGHJKMNQUVXZ][0-9]$", "databento"
        )  # Contract codes (2-3 letter root)

        assert router.get_provider("ES.v.0") == "databento"
        assert router.get_provider("CL.V.1") == "databento"
        assert router.get_provider("ESH4") == "databento"
        assert router.get_provider("CLZ5") == "databento"

    def test_override_routing(self):
        """Test manual provider override."""
        router = ProviderRouter()
        router.add_pattern(r"^BTC", "cryptocompare")

        # Normal routing
        assert router.get_provider("BTC") == "cryptocompare"

        # Override routing
        assert router.get_provider("BTC", override="databento") == "databento"

    def test_no_match_returns_none(self):
        """Test that unmatched symbols return None."""
        router = ProviderRouter()
        router.add_pattern(r"^BTC", "cryptocompare")

        assert router.get_provider("UNKNOWN") is None
        assert router.get_provider("") is None


class TestDataManagerFetch:
    """Test DataManager fetch functionality."""

    def test_fetch_basic(self):
        """Test basic fetch operation."""
        # Create mock provider
        mock_provider = MagicMock()
        mock_provider.fetch_ohlcv.return_value = pl.DataFrame(
            {
                "timestamp": [datetime(2024, 1, 1)],
                "open": [100.0],
                "high": [105.0],
                "low": [95.0],
                "close": [102.0],
                "volume": [1000.0],
                "symbol": ["BTC"],
            }
        )

        dm = DataManager()
        # Clear default patterns and inject mock provider
        dm.router.patterns.clear()
        dm._provider_manager._provider_classes["mock_crypto"] = lambda **kwargs: mock_provider
        dm._provider_manager._available_providers.append("mock_crypto")
        dm.router.add_pattern(r"^BTC", "mock_crypto")

        df = dm.fetch("BTC", "2024-01-01", "2024-01-01")

        assert isinstance(df, pl.DataFrame)
        assert len(df) == 1
        assert df["symbol"][0] == "BTC"
        mock_provider.fetch_ohlcv.assert_called_once_with(
            "BTC", "2024-01-01", "2024-01-01", "daily"
        )

    def test_fetch_with_frequency(self):
        """Test fetch with custom frequency."""
        mock_provider = MagicMock()
        mock_provider.fetch_ohlcv.return_value = pl.DataFrame()

        dm = DataManager()
        dm.router.patterns.clear()
        dm._provider_manager._provider_classes["mock_crypto"] = lambda **kwargs: mock_provider
        dm._provider_manager._available_providers.append("mock_crypto")
        dm.router.add_pattern(r"^ETH", "mock_crypto")

        dm.fetch("ETH", "2024-01-01", "2024-01-02", frequency="hourly")

        mock_provider.fetch_ohlcv.assert_called_once_with(
            "ETH", "2024-01-01", "2024-01-02", "hourly"
        )

    def test_fetch_no_provider_found(self):
        """Test fetch when no provider matches symbol."""
        dm = DataManager()

        with pytest.raises(ValueError, match="No provider found for symbol"):
            dm.fetch("UNKNOWN_SYMBOL", "2024-01-01", "2024-01-01")

    def test_fetch_provider_not_available(self):
        """Test fetch when matched provider is not configured."""
        dm = DataManager()
        # Add a pattern for a non-existent provider
        dm.router.add_pattern(r"^NONEXISTENT", "nonexistent_provider")

        with pytest.raises(ValueError, match="Provider.*not available"):
            dm.fetch("NONEXISTENT", "2024-01-01", "2024-01-01")

    def test_fetch_with_provider_override(self):
        """Test fetch with explicit provider override."""
        mock_provider = MagicMock()
        mock_provider.fetch_ohlcv.return_value = pl.DataFrame(
            {
                "timestamp": [],
                "open": [],
                "high": [],
                "low": [],
                "close": [],
                "volume": [],
                "symbol": [],
            }
        )

        dm = DataManager()
        dm._provider_manager._provider_classes["special"] = lambda **kwargs: mock_provider
        dm._provider_manager._available_providers.append("special")
        dm.router.add_pattern(r"^BTC", "cryptocompare")  # Would normally use cryptocompare

        dm.fetch("BTC", "2024-01-01", "2024-01-01", provider="special")

        # Should use override provider, not routed one
        mock_provider.fetch_ohlcv.assert_called_once()


class TestOutputFormats:
    """Test different output format conversions."""

    def test_output_format_polars(self):
        """Test Polars DataFrame output (default)."""
        mock_provider = MagicMock()
        test_df = pl.DataFrame(
            {
                "timestamp": [datetime(2024, 1, 1)],
                "open": [100.0],
                "high": [105.0],
                "low": [95.0],
                "close": [102.0],
                "volume": [1000.0],
                "symbol": ["BTC"],
            }
        )
        mock_provider.fetch_ohlcv.return_value = test_df

        dm = DataManager(output_format="polars")
        dm.router.patterns.clear()
        dm._provider_manager._provider_classes["mock_crypto"] = lambda **kwargs: mock_provider
        dm._provider_manager._available_providers.append("mock_crypto")
        dm.router.add_pattern(r"^BTC", "mock_crypto")

        result = dm.fetch("BTC", "2024-01-01", "2024-01-01")

        assert isinstance(result, pl.DataFrame)
        assert result.equals(test_df)

    def test_output_format_pandas(self):
        """Test pandas DataFrame output."""
        mock_provider = MagicMock()
        test_df = pl.DataFrame(
            {
                "timestamp": [datetime(2024, 1, 1)],
                "open": [100.0],
                "high": [105.0],
                "low": [95.0],
                "close": [102.0],
                "volume": [1000.0],
                "symbol": ["BTC"],
            }
        )
        mock_provider.fetch_ohlcv.return_value = test_df

        dm = DataManager(output_format="pandas")
        dm._provider_manager._provider_classes["mock_crypto"] = lambda **kwargs: mock_provider
        dm._provider_manager._available_providers.append("mock_crypto")
        dm.router.add_pattern(r"^BTC", "mock_crypto")

        result = dm.fetch("BTC", "2024-01-01", "2024-01-01")

        # Check if it's a pandas DataFrame
        assert hasattr(result, "iloc")  # pandas-specific attribute
        assert len(result) == 1
        assert result["symbol"].iloc[0] == "BTC"

    def test_output_format_lazy(self):
        """Test lazy DataFrame output."""
        mock_provider = MagicMock()
        test_df = pl.DataFrame(
            {
                "timestamp": [datetime(2024, 1, 1)],
                "open": [100.0],
                "high": [105.0],
                "low": [95.0],
                "close": [102.0],
                "volume": [1000.0],
                "symbol": ["BTC"],
            }
        )
        mock_provider.fetch_ohlcv.return_value = test_df

        dm = DataManager(output_format="lazy")
        dm._provider_manager._provider_classes["mock_crypto"] = lambda **kwargs: mock_provider
        dm._provider_manager._available_providers.append("mock_crypto")
        dm.router.add_pattern(r"^BTC", "mock_crypto")

        result = dm.fetch("BTC", "2024-01-01", "2024-01-01")

        assert isinstance(result, pl.LazyFrame)
        # Collect to verify data
        collected = result.collect()
        assert len(collected) == 1
        assert collected["symbol"][0] == "BTC"


class TestConnectionPooling:
    """Test connection pooling and session management."""

    def test_provider_reuse(self):
        """Test that providers are reused across calls."""
        mock_provider = MagicMock()
        mock_provider.fetch_ohlcv.return_value = pl.DataFrame(
            {
                "timestamp": [],
                "open": [],
                "high": [],
                "low": [],
                "close": [],
                "volume": [],
                "symbol": [],
            }
        )

        call_count = [0]

        def mock_provider_factory(**kwargs):
            call_count[0] += 1
            return mock_provider

        dm = DataManager()
        dm.router.patterns.clear()
        dm._provider_manager._provider_classes["mock_crypto"] = mock_provider_factory
        dm._provider_manager._available_providers.append("mock_crypto")
        dm.router.add_pattern(r"^BTC", "mock_crypto")

        # Multiple fetches
        dm.fetch("BTC", "2024-01-01", "2024-01-01")
        dm.fetch("BTC", "2024-01-02", "2024-01-02")

        # Provider should be instantiated only once
        assert call_count[0] == 1
        # But fetch should be called twice
        assert mock_provider.fetch_ohlcv.call_count == 2

    def test_multiple_providers(self):
        """Test managing multiple provider instances."""
        mock_crypto = MagicMock()
        mock_crypto.fetch_ohlcv.return_value = pl.DataFrame(
            {
                "timestamp": [],
                "open": [],
                "high": [],
                "low": [],
                "close": [],
                "volume": [],
                "symbol": [],
            }
        )

        mock_oanda = MagicMock()
        mock_oanda.fetch_ohlcv.return_value = pl.DataFrame(
            {
                "timestamp": [],
                "open": [],
                "high": [],
                "low": [],
                "close": [],
                "volume": [],
                "symbol": [],
            }
        )

        crypto_count = [0]
        oanda_count = [0]

        def mock_crypto_factory(**kwargs):
            crypto_count[0] += 1
            return mock_crypto

        def mock_oanda_factory(**kwargs):
            oanda_count[0] += 1
            return mock_oanda

        dm = DataManager()
        dm.router.patterns.clear()
        dm._provider_manager._provider_classes["mock_crypto"] = mock_crypto_factory
        dm._provider_manager._provider_classes["mock_oanda"] = mock_oanda_factory
        dm._provider_manager._available_providers.extend(["mock_crypto", "mock_oanda"])
        dm.router.add_pattern(r"^BTC", "mock_crypto")
        dm.router.add_pattern(r"^EUR", "mock_oanda")

        # Fetch from different providers
        dm.fetch("BTC", "2024-01-01", "2024-01-01")
        dm.fetch("EURUSD", "2024-01-01", "2024-01-01")

        # Both providers should be instantiated
        assert crypto_count[0] == 1
        assert oanda_count[0] == 1


class TestErrorHandling:
    """Test error handling and user-friendly messages."""

    def test_invalid_date_format(self):
        """Test handling of invalid date formats."""
        mock_provider = MagicMock()
        dm = DataManager()
        dm._provider_manager._provider_classes["mock_crypto"] = lambda **kwargs: mock_provider
        dm._provider_manager._available_providers.append("mock_crypto")
        dm.router.add_pattern(r"^BTC", "mock_crypto")

        with pytest.raises(ValueError, match="Invalid date format"):
            dm.fetch("BTC", "01-01-2024", "2024-01-01")  # Wrong format

    def test_end_before_start(self):
        """Test handling when end date is before start date."""
        mock_provider = MagicMock()
        dm = DataManager()
        dm._provider_manager._provider_classes["mock_crypto"] = lambda **kwargs: mock_provider
        dm._provider_manager._available_providers.append("mock_crypto")
        dm.router.add_pattern(r"^BTC", "mock_crypto")

        with pytest.raises(ValueError, match="End date must be after start date"):
            dm.fetch("BTC", "2024-01-02", "2024-01-01")

    def test_provider_error_handling(self):
        """Test handling of provider errors."""
        mock_provider = MagicMock()
        mock_provider.fetch_ohlcv.side_effect = Exception("API Error")

        dm = DataManager()
        dm.router.patterns.clear()
        dm._provider_manager._provider_classes["mock_crypto"] = lambda **kwargs: mock_provider
        dm._provider_manager._available_providers.append("mock_crypto")
        dm.router.add_pattern(r"^BTC", "mock_crypto")

        with pytest.raises(Exception, match="Failed to fetch data.*API Error"):
            dm.fetch("BTC", "2024-01-01", "2024-01-01")

    def test_invalid_output_format(self):
        """Test handling of invalid output format."""
        with pytest.raises(ValueError, match="Invalid output format"):
            DataManager(output_format="invalid")


class TestIntegration:
    """Integration tests with real provider classes."""

    def test_full_configuration_flow(self):
        """Test complete configuration and fetch flow."""
        config_content = """
        providers:
          cryptocompare:
            api_key: test_key

        routing:
          patterns:
            - pattern: "^BTC"
              provider: cryptocompare

        defaults:
          output_format: polars
          frequency: daily
        """

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(config_content)
            config_path = f.name

        try:
            dm = DataManager(config_path=config_path)

            # Verify configuration loaded
            assert dm.config is not None
            assert "providers" in dm.config
            assert dm.output_format == "polars"

            # Verify routing configured
            provider = dm.router.get_provider("BTC")
            assert provider == "cryptocompare"
        finally:
            Path(config_path).unlink()

    def test_batch_fetch(self):
        """Test fetching multiple symbols efficiently."""
        dm = DataManager()
        dm.router.add_pattern(r"^(BTC|ETH)", "cryptocompare")
        dm.router.add_pattern(r"^EUR", "oanda")

        symbols = ["BTC", "ETH", "EURUSD"]
        results = dm.fetch_batch(symbols, "2024-01-01", "2024-01-01")

        assert isinstance(results, dict)
        assert set(results.keys()) == set(symbols)
        for _symbol, df in results.items():
            assert df is None or isinstance(df, pl.DataFrame | pl.LazyFrame)
