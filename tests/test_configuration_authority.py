"""End-to-end contracts for the canonical configuration boundary."""

from pathlib import Path

import pytest

from ml4t.data import DataManager
from ml4t.data.config import load_config
from ml4t.data.providers.registry import PROVIDER_REGISTRY

EXAMPLE_CONFIGS = Path(__file__).parents[1] / "examples" / "configs"


def test_canonical_example_loads_through_typed_and_manager_entry_points():
    path = EXAMPLE_CONFIGS / "basic.yaml"

    config = load_config(path)
    manager = DataManager(config_path=str(path))

    assert config.get_provider("yahoo_main") is not None
    assert "yahoo_main" in manager.list_providers()
    assert manager._get_provider("yahoo_main").name == "yahoo"


def test_beta_mapping_example_has_one_explicit_migration_path(tmp_path):
    path = tmp_path / "beta.yaml"
    path.write_text(
        "storage:\n"
        "  path: ./data\n"
        "providers:\n"
        "  yahoo: {}\n"
        "datasets:\n"
        "  demo_stocks:\n"
        "    provider: yahoo\n"
        "    symbols: [AAPL]\n",
        encoding="utf-8",
    )

    config = load_config(path)
    manager = DataManager(config_path=str(path))

    assert config.get_dataset("demo_stocks") is not None
    assert "yahoo" in manager.list_providers()


def test_beta_mapping_datasets_receive_defaults_before_validation(tmp_path):
    path = tmp_path / "beta-defaults.yaml"
    path.write_text(
        "defaults:\n"
        "  frequency: weekly\n"
        "providers:\n"
        "  yahoo: {}\n"
        "datasets:\n"
        "  demo_stocks:\n"
        "    provider: yahoo\n"
        "    symbols: [AAPL]\n",
        encoding="utf-8",
    )

    config = load_config(path)

    assert config.get_dataset("demo_stocks").frequency.value == "weekly"


def test_relative_storage_path_is_resolved_from_config_location(tmp_path):
    config_path = tmp_path / "config" / "ml4t-data.yaml"
    config_path.parent.mkdir()
    config_path.write_text("storage:\n  base_path: ../market-data\n", encoding="utf-8")

    config = load_config(config_path)

    assert config.storage.base_path == (tmp_path / "market-data").resolve()


@pytest.mark.parametrize(
    "path",
    sorted(EXAMPLE_CONFIGS.rglob("*.yaml")),
    ids=lambda path: str(path.relative_to(EXAMPLE_CONFIGS)),
)
def test_every_shipped_configuration_uses_the_canonical_loader(path, monkeypatch):
    for spec in PROVIDER_REGISTRY.values():
        for environment in spec.credential_environment + spec.optional_credential_environment:
            monkeypatch.delenv(environment, raising=False)

    config = load_config(path)
    manager = DataManager(config_path=str(path))

    for provider in config.providers:
        info = manager.get_provider_info(provider.name)
        if provider.enabled and info["available"]:
            assert manager._get_provider(provider.name) is not None


def test_invalid_configuration_diagnostic_does_not_disclose_values(tmp_path, monkeypatch, capsys):
    credential = "credential-canary-that-must-not-appear"
    monkeypatch.setenv("CONFIG_SECRET_CANARY", credential)
    path = tmp_path / "invalid.yaml"
    path.write_text("unknown_setting: ${CONFIG_SECRET_CANARY}\n", encoding="utf-8")

    with pytest.raises(ValueError) as caught:
        load_config(path)

    assert "unknown_setting" in str(caught.value)
    assert credential not in str(caught.value)
    assert credential not in capsys.readouterr().out


def test_runtime_provider_overrides_use_the_same_validation_boundary():
    with pytest.raises(ValueError, match="does not accept api_key"):
        DataManager(providers={"yahoo": {"api_key": "unused-key"}})
