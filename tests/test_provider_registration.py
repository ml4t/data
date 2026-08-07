"""Test provider registration in DataManager."""

import pytest

pytest.importorskip("databento")

from ml4t.data.data_manager import DataManager
from ml4t.data.providers.registry import PROVIDER_REGISTRY


def test_all_providers_registered():
    """Every manager-compatible registry entry resolves through DataManager."""
    expected_names = {spec.name for spec in PROVIDER_REGISTRY.values() if spec.manager_compatible}

    assert set(DataManager.PROVIDER_CLASSES) == expected_names
    for provider_name, provider_class in DataManager.PROVIDER_CLASSES.items():
        assert provider_class is PROVIDER_REGISTRY[provider_name].load_class()


def test_provider_imports_work():
    """Every manager-compatible provider class can be imported."""
    for spec in PROVIDER_REGISTRY.values():
        if not spec.manager_compatible:
            continue
        provider_class = spec.load_class()
        assert provider_class is not None
        assert hasattr(provider_class, "__init__")
        assert hasattr(provider_class, "fetch_ohlcv") or hasattr(provider_class, "_fetch_raw_data")


def test_free_providers_detected():
    """Test that free providers are detected without API keys."""
    dm = DataManager()

    # These providers should be available even without API keys
    free_providers = [
        "yahoo",
        "binance",
        "binance_public",
        "coingecko",
        "mock",
        "cryptocompare",
        "synthetic",
    ]

    for provider in free_providers:
        assert provider in dm._available_providers, (
            f"{provider} should be available without API key"
        )


def test_provider_instantiation():
    """Every provider reported as available can be instantiated."""
    dm = DataManager()

    for provider_name in dm.list_providers():
        provider = dm._get_provider(provider_name)
        assert provider is not None


def test_provider_count():
    """Provider counts and names come from the canonical registry."""
    expected = [spec.name for spec in PROVIDER_REGISTRY.values() if spec.manager_compatible]

    assert list(DataManager.PROVIDER_CLASSES) == expected


if __name__ == "__main__":
    # Run tests
    test_all_providers_registered()
    test_provider_imports_work()
    test_free_providers_detected()
    test_provider_instantiation()
    test_provider_count()
    print("✅ All provider registration tests passed!")
