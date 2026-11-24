"""Configuration controls validation through the real update-all workflow."""

from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from ml4t.data.cli_interface import cli
from ml4t.data.storage import create_storage


@pytest.mark.parametrize("mode,rows", [("warn", 2), ("drop", 1), ("strict", None), ("loose", None)])
def test_configured_ohlc_mode_controls_stored_adjusted_bars(tmp_path, mode, rows):
    config = tmp_path / "data.yaml"
    root = tmp_path / "stored"
    config.write_text(
        f"storage:\n  base_path: {root}\n"
        "providers:\n  - name: alpaca\n    type: alpaca\n"
        "    api_key: test-key\n    api_secret: test-secret\n    extra:\n"
        f"      adjustment: all\n      ohlc_mode: {mode}\n"
        "datasets:\n  - name: adjusted\n    provider: alpaca\n"
        "    symbols: [SPY]\n    asset_class: equities\n    frequency: daily\n"
        "    start_date: '2026-11-19'\n    end_date: '2026-11-20'\n"
    )
    response = MagicMock(status_code=200)
    response.json.return_value = {
        "bars": [
            {"t": "2026-11-19T05:00:00Z", "o": 100, "h": 101, "l": 99, "c": 100, "v": 1000},
            {"t": "2026-11-20T05:00:00Z", "o": 100, "h": 101, "l": 99, "c": 101.01, "v": 1000},
        ],
        "next_page_token": None,
    }
    with patch("httpx.Client.get", return_value=response) as get:
        result = CliRunner().invoke(cli, ["update-all", "--config", str(config)])
    storage = create_storage(root)
    key = "equities/daily/SPY"
    if rows is None:
        assert result.exit_code != 0, result.output
        assert "OK Update complete!" not in result.output
        assert "FAIL" in result.output
        assert not storage.exists(key)
        if mode == "loose":
            get.assert_not_called()
            assert "ohlc_mode must be strict, drop or warn" in result.output
    else:
        assert result.exit_code == 0, result.output
        assert "FAIL" not in result.output
        stored = storage.read(key).collect()
        assert stored.height == rows
        assert stored["close"].to_list() == ([100.0, 101.01] if mode == "warn" else [100.0])
        assert get.call_args.kwargs["params"]["adjustment"] == "all"
