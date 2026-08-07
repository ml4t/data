"""Regression tests for release workflow policy."""

from __future__ import annotations

import re
from pathlib import Path

import yaml

WORKFLOW_DIR = Path(__file__).parents[1] / ".github" / "workflows"
PINNED_ACTION = re.compile(r"^[^./][^@]*@[0-9a-f]{40}$")


def _load(name: str) -> dict:
    return yaml.load((WORKFLOW_DIR / name).read_text(encoding="utf-8"), Loader=yaml.BaseLoader)


def _steps(workflow: dict):
    for job in workflow["jobs"].values():
        yield from job.get("steps", [])


def test_external_actions_are_pinned_and_checkouts_do_not_persist_credentials() -> None:
    for path in WORKFLOW_DIR.glob("*.yml"):
        workflow = _load(path.name)
        assert workflow["permissions"] == {"contents": "read"}
        for step in _steps(workflow):
            action = step.get("uses")
            if action and not action.startswith("./"):
                assert PINNED_ACTION.fullmatch(action), (
                    f"mutable action reference in {path}: {action}"
                )
            if action and action.startswith("actions/checkout@"):
                assert step.get("with", {}).get("persist-credentials") == "false"


def test_compatibility_matrix_covers_release_policy() -> None:
    matrix = _load("compatibility.yml")["jobs"]["compatibility"]["strategy"]["matrix"]

    assert matrix["os"] == ["ubuntu-latest", "macos-latest", "windows-latest"]
    assert matrix["python-version"] == ["3.12", "3.13", "3.14", "3.15"]


def test_publish_uses_only_the_validated_package_directory() -> None:
    release = _load("release.yml")
    build = release["jobs"]["build"]
    publish = release["jobs"]["publish"]

    assert build["needs"] == "compatibility"
    assert publish["needs"] == "build"
    assert publish["environment"] == "pypi"
    assert publish["permissions"] == {"contents": "read", "id-token": "write"}

    publish_action = next(
        step for step in publish["steps"] if step.get("name") == "Publish to PyPI"
    )
    assert publish_action["with"]["packages-dir"] == "dist/packages/"

    github_release = release["jobs"]["github-release"]
    create_step = next(
        step
        for step in github_release["steps"]
        if step.get("name") == "Create GitHub release with validated distributions"
    )
    assert create_step["env"]["GH_REPO"] == "${{ github.repository }}"


def test_compatibility_checkout_fetches_release_tags() -> None:
    compatibility = _load("compatibility.yml")["jobs"]["compatibility"]
    checkout = next(
        step
        for step in compatibility["steps"]
        if step.get("uses", "").startswith("actions/checkout@")
    )
    assert checkout["with"]["fetch-depth"] == "0"


def test_ci_does_not_run_an_empty_optional_dependency_lane() -> None:
    jobs = _load("ci.yml")["jobs"]
    assert "optional-dependency" not in jobs


def test_provider_contract_jobs_isolate_credentials() -> None:
    jobs = _load("provider-contracts.yml")["jobs"]
    credentials = {
        "cryptocompare": "CRYPTOCOMPARE_API_KEY",
        "databento": "DATABENTO_API_KEY",
        "oanda": "OANDA_API_KEY",
    }

    for provider, expected_credential in credentials.items():
        serialized = yaml.safe_dump(jobs[provider])
        assert expected_credential in serialized
        for other_credential in set(credentials.values()) - {expected_credential}:
            assert other_credential not in serialized


def test_public_provider_integrations_are_manually_reachable() -> None:
    workflow = _load("provider-contracts.yml")
    options = workflow["on"]["workflow_dispatch"]["inputs"]["provider"]["options"]
    public_job = workflow["jobs"]["public"]
    command = next(
        step["run"]
        for step in public_job["steps"]
        if step.get("name") == "Run public live integrations"
    )

    assert "public" in options
    for contract in (
        "test_binance_public.py::TestBinancePublicProvider::test_fetch_daily_spot_btc",
        "test_coingecko.py::TestCoinGeckoProvider::test_fetch_ohlcv_btc",
        "test_kalshi.py::TestKalshiProvider::test_list_markets",
        "test_polymarket.py::TestPolymarketProvider::test_get_market_by_slug",
        "test_yahoo.py::TestYahooFinanceProvider::test_fetch_ohlcv_stock_daily",
        "test_release_provider_contracts.py",
    ):
        assert contract in command
