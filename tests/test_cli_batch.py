"""Tests for batch CLI commands."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from click.testing import CliRunner

from ml4t.data.cli import cli


def test_update_all_passes_storage_and_dataset_config() -> None:
    """update-all should honor YAML storage and dataset settings."""
    runner = CliRunner()
    config_text = """
storage:
  strategy: hive
  path: ./marketdata
  compression: zstd
  partition_granularity: year
  atomic_writes: true
  enable_locking: false
  metadata_tracking: true
datasets:
  taa:
    symbols: [SPY]
    provider: yahoo
    frequency: daily
    asset_class: equity
    lookback_days: 14
    fill_gaps: false
    start: 2010-01-01
    end: 2024-12-31
"""

    with (
        runner.isolated_filesystem(),
        patch("ml4t.data.cli.batch.create_storage") as mock_create_storage,
        patch("ml4t.data.cli.batch.DataManager") as mock_manager_class,
    ):
        storage = MagicMock()
        mock_create_storage.return_value = storage
        manager = MagicMock()
        manager.update.return_value = "equity/daily/SPY"
        mock_manager_class.return_value = manager

        with open("ml4t-data.yaml", "w") as f:
            f.write(config_text)

        result = runner.invoke(cli, ["update-all", "-c", "ml4t-data.yaml"])

    assert result.exit_code == 0, result.output
    mock_create_storage.assert_called_once()
    _, kwargs = mock_create_storage.call_args
    assert kwargs == {
        "strategy": "hive",
        "compression": "zstd",
        "partition_granularity": "year",
        "atomic_writes": True,
        "enable_locking": False,
        "metadata_tracking": True,
    }
    manager.update.assert_called_once_with(
        "SPY",
        frequency="daily",
        asset_class="equity",
        provider="yahoo",
        lookback_days=14,
        fill_gaps=False,
        initial_start="2010-01-01",
        initial_end="2024-12-31",
        initial_load_days=365,
    )


def test_update_all_uses_initial_load_days_when_configured() -> None:
    """update-all should pass configured first-load history length."""
    runner = CliRunner()
    config_text = """
storage:
  path: ./marketdata
datasets:
  demo:
    symbols: [AAPL]
    provider: mock
    initial_load_days: 3650
"""

    with (
        runner.isolated_filesystem(),
        patch("ml4t.data.cli.batch.create_storage") as mock_create_storage,
        patch("ml4t.data.cli.batch.DataManager") as mock_manager_class,
    ):
        mock_create_storage.return_value = MagicMock()
        manager = MagicMock()
        manager.update.return_value = "equities/daily/AAPL"
        mock_manager_class.return_value = manager

        with open("ml4t-data.yaml", "w") as f:
            f.write(config_text)

        result = runner.invoke(cli, ["update-all", "-c", "ml4t-data.yaml"])

    assert result.exit_code == 0, result.output
    manager.update.assert_called_once_with(
        "AAPL",
        frequency="daily",
        asset_class="equities",
        provider="mock",
        lookback_days=7,
        fill_gaps=True,
        initial_start=None,
        initial_end=None,
        initial_load_days=3650,
    )
