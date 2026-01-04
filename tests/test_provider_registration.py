"""Test provider registration in DataManager."""

from ml4t.data.data_manager import DataManager
from ml4t.data.providers.binance import BinanceProvider
from ml4t.data.providers.binance_public import BinancePublicProvider
from ml4t.data.providers.cryptocompare import CryptoCompareProvider
from ml4t.data.providers.databento import DataBentoProvider
from ml4t.data.providers.mock import MockProvider
from ml4t.data.providers.oanda import OandaProvider
from ml4t.data.providers.okx import OKXProvider
from ml4t.data.providers.synthetic import SyntheticProvider
from ml4t.data.providers.yahoo import YahooFinanceProvider


def test_all_providers_registered():
    """Test that all 9 OHLCV providers are registered in DataManager.

    Note: Specialized providers (factor, prediction market) are not in DataManager.
    - Factor providers: aqr, fama_french (standalone, different API)
    - Prediction markets: kalshi, polymarket (standalone, different API)
    - Historical: wiki_prices (standalone, local file only)
    """
    expected_providers = {
        "binance": BinanceProvider,
        "binance_public": BinancePublicProvider,
        "cryptocompare": CryptoCompareProvider,
        "databento": DataBentoProvider,
        "mock": MockProvider,
        "oanda": OandaProvider,
        "okx": OKXProvider,
        "synthetic": SyntheticProvider,
        "yahoo": YahooFinanceProvider,
    }

    assert expected_providers == DataManager.PROVIDER_CLASSES
    assert len(DataManager.PROVIDER_CLASSES) == 9


def test_provider_imports_work():
    """Test that all provider imports are functional."""
    # This test ensures the imports at the top of data_manager.py work
    providers = [
        BinanceProvider,
        BinancePublicProvider,
        CryptoCompareProvider,
        DataBentoProvider,
        MockProvider,
        OandaProvider,
        SyntheticProvider,
        YahooFinanceProvider,
    ]

    for provider_class in providers:
        assert provider_class is not None
        assert hasattr(provider_class, "__init__")
        assert hasattr(provider_class, "fetch_ohlcv") or hasattr(provider_class, "_fetch_raw_data")


def test_free_providers_detected():
    """Test that free providers are detected without API keys."""
    dm = DataManager()

    # These providers should be available even without API keys
    free_providers = ["yahoo", "binance", "binance_public", "mock", "cryptocompare", "synthetic"]

    for provider in free_providers:
        assert provider in dm._available_providers, (
            f"{provider} should be available without API key"
        )


def test_provider_instantiation():
    """Test that each registered provider can be instantiated."""
    dm = DataManager()

    for provider_name, _provider_class in DataManager.PROVIDER_CLASSES.items():
        # Try to get provider instance
        try:
            if provider_name in ["databento", "oanda"]:
                # These require API keys, so we expect None or skip
                continue
            provider = dm._get_provider(provider_name)
            assert provider is not None or provider_name in ["databento", "oanda"]
        except Exception as e:
            # Only databento and oanda should fail without API keys
            assert provider_name in ["databento", "oanda"], (
                f"Unexpected error for {provider_name}: {e}"
            )


def test_provider_count():
    """Test that we have exactly 9 OHLCV providers registered in DataManager.

    Provider categories:
    - In DataManager (9): General OHLCV providers with unified fetch_ohlcv() interface
    - Standalone (5): Specialized providers with unique APIs
      - Factor data: aqr, fama_french
      - Prediction markets: kalshi, polymarket
      - Historical: wiki_prices
    """
    assert len(DataManager.PROVIDER_CLASSES) == 9, (
        f"Expected 9 providers, got {len(DataManager.PROVIDER_CLASSES)}"
    )

    # List all provider names for clarity
    provider_names = list(DataManager.PROVIDER_CLASSES.keys())
    assert "binance" in provider_names
    assert "binance_public" in provider_names
    assert "cryptocompare" in provider_names
    assert "databento" in provider_names
    assert "mock" in provider_names
    assert "oanda" in provider_names
    assert "okx" in provider_names
    assert "synthetic" in provider_names
    assert "yahoo" in provider_names


if __name__ == "__main__":
    # Run tests
    test_all_providers_registered()
    test_provider_imports_work()
    test_free_providers_detected()
    test_provider_instantiation()
    test_provider_count()
    print("✅ All provider registration tests passed!")
