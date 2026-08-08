"""Contract tests for provider discovery and manager routing."""

from __future__ import annotations

import pytest
from click.testing import CliRunner

from ml4t.data import providers as provider_namespace
from ml4t.data.cli_interface import cli
from ml4t.data.config.models import ProviderType
from ml4t.data.data_manager import DataManager
from ml4t.data.providers.oanda import OandaProvider
from ml4t.data.providers.registry import PROVIDER_REGISTRY, advertised_provider_specs


def test_typed_config_and_registry_have_the_same_provider_names():
    assert {provider.value for provider in ProviderType} == set(PROVIDER_REGISTRY)


def test_cli_provider_discovery_is_derived_from_registry():
    result = CliRunner().invoke(cli, ["providers"])

    assert result.exit_code == 0, result.output
    for spec in advertised_provider_specs():
        assert spec.name in result.output
    assert "binance_futures" not in result.output


def test_advertised_provider_classes_are_exported_from_public_namespace():
    for spec in advertised_provider_specs():
        exported_class = getattr(provider_namespace, spec.class_name)
        try:
            registered_class = spec.load_class()
        except ModuleNotFoundError as error:
            assert spec.extra is not None
            assert error.name is not None
            assert exported_class is None
        else:
            assert exported_class is registered_class


def test_manager_constructs_registered_ohlcv_provider_with_configuration():
    manager = DataManager(providers={"fred": {"api_key": "dummy"}})

    provider = manager._get_provider("fred")

    assert provider.name == "fred"


def test_fetch_rejects_non_ohlcv_capability_before_provider_construction(monkeypatch):
    manager = DataManager()
    get_provider = manager._provider_manager.get_provider
    calls: list[str] = []

    def track_get_provider(provider_name: str, *, required_capability: str | None = None):
        calls.append(provider_name)
        return get_provider(provider_name, required_capability=required_capability)

    monkeypatch.setattr(manager._provider_manager, "get_provider", track_get_provider)

    try:
        manager.fetch("QMJ", "2024-01-01", "2024-01-31", provider="aqr")
    except ValueError as error:
        assert "does not support 'ohlcv'" in str(error)
    else:
        raise AssertionError("Expected unsupported-capability failure")
    assert calls == ["aqr"]


def test_provider_info_reports_capabilities_before_credentials_are_configured(monkeypatch):
    monkeypatch.delenv("TIINGO_API_KEY", raising=False)
    manager = DataManager()

    info = manager.get_provider_info("tiingo")

    assert info["available"] is False
    assert info["capabilities"] == ["ohlcv"]
    assert info["credential_environment"] == ["TIINGO_API_KEY"]


def test_oanda_environment_configuration_only_passes_constructor_arguments(monkeypatch):
    monkeypatch.setenv("OANDA_API_KEY", "dummy-key")
    monkeypatch.setenv("OANDA_ACCOUNT_ID", "unused-account")
    monkeypatch.setattr(OandaProvider, "__init__", lambda self, api_key=None: None)
    manager = DataManager()

    manager._get_provider("oanda")


def test_provider_configuration_rejects_credentials_in_nested_extra():
    with pytest.raises(ValueError, match="extra contains reserved fields: api_key"):
        DataManager(
            providers={
                "tiingo": {
                    "api_key": "outer-secret",
                    "extra": {"api_key": "nested-secret", "exchange": "US"},
                }
            }
        )


def test_unresolved_environment_reference_does_not_enable_provider(monkeypatch):
    monkeypatch.delenv("DATABENTO_API_KEY", raising=False)
    manager = DataManager(providers={"databento": {"api_key": "${DATABENTO_API_KEY}"}})

    assert "databento" not in manager.list_providers()


def test_custom_registration_is_scoped_to_one_manager_instance():
    first = DataManager()
    second = DataManager()

    class CustomProvider:
        pass

    first._provider_manager.register_provider("custom", CustomProvider)

    assert "custom" in first._provider_manager._provider_classes
    assert "custom" not in second._provider_manager._provider_classes
    assert "custom" not in DataManager.PROVIDER_CLASSES
