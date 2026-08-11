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
    compatibility = _load("compatibility.yml")["jobs"]["compatibility"]
    matrix = compatibility["strategy"]["matrix"]

    assert matrix["os"] == ["ubuntu-latest", "macos-latest", "windows-latest"]
    assert matrix["python-version"] == ["3.12", "3.13", "3.14"]
    assert "continue-on-error" not in compatibility


def test_publish_uses_only_the_validated_package_directory() -> None:
    release = _load("release.yml")
    build = release["jobs"]["build"]
    publish = release["jobs"]["publish"]

    assert build["needs"] == [
        "ecosystem-qualification",
        "compatibility",
        "provider-contracts",
        "quality",
    ]
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
        "alpaca": {"ALPACA_API_KEY", "ALPACA_API_SECRET"},
        "cryptocompare": {"CRYPTOCOMPARE_API_KEY"},
        "databento": {"DATABENTO_API_KEY"},
        "finnhub": {"FINNHUB_API_KEY"},
        "fred": {"FRED_API_KEY"},
        "massive": {"MASSIVE_API_KEY"},
        "oanda": {"OANDA_API_KEY"},
        "tiingo": {"TIINGO_API_KEY"},
    }
    all_credentials = set().union(*credentials.values())

    for provider, expected_credentials in credentials.items():
        serialized = yaml.safe_dump(jobs[provider])
        for expected_credential in expected_credentials:
            assert expected_credential in serialized
        for other_credential in all_credentials - expected_credentials:
            assert other_credential not in serialized


def test_provider_contract_guards_prevent_skipped_green_runs() -> None:
    workflow = _load("provider-contracts.yml")
    jobs = workflow["jobs"]
    guarded_credentials = {
        "alpaca": {"PROVIDER_API_KEY", "PROVIDER_API_SECRET"},
        "cryptocompare": {"PROVIDER_API_KEY"},
        "databento": {"PROVIDER_API_KEY"},
        "finnhub": {"PROVIDER_API_KEY"},
        "fred": {"PROVIDER_API_KEY"},
        "massive": {"PROVIDER_API_KEY"},
        "oanda": {"PROVIDER_API_KEY"},
        "tiingo": {"PROVIDER_API_KEY"},
    }
    for provider, expected_env in guarded_credentials.items():
        guard = next(
            step for step in jobs[provider]["steps"] if step.get("name", "").startswith("Require")
        )
        assert set(guard["env"]) >= expected_env
        for variable in expected_env:
            assert f"os.environ.get('{variable}')" in guard["run"]
        assert "ALLOW_PAID_REQUESTS" not in guard["env"]


def test_provider_contract_dispatch_selects_exactly_one_job() -> None:
    workflow = _load("provider-contracts.yml")
    jobs = workflow["jobs"]
    inputs = workflow["on"]["workflow_dispatch"]["inputs"]
    options = inputs["provider"]["options"]

    assert workflow["run-name"] == "Provider Contract - ${{ inputs.provider }}"
    assert set(options) == set(jobs)
    for option in options:
        assert jobs[option]["if"] == f"inputs.provider == '{option}'"

    assert "allow-paid-requests" not in inputs


def test_release_requires_provider_contracts_for_the_tagged_commit() -> None:
    jobs = _load("release.yml")["jobs"]
    verification = jobs["provider-contracts"]
    step = next(
        step
        for step in verification["steps"]
        if step.get("name") == "Require successful live contracts for this commit"
    )

    assert verification["permissions"] == {"actions": "read", "contents": "read"}
    assert step["env"]["RELEASE_COMMIT"] == "${{ github.sha }}"
    assert "verify_provider_contract_runs.py" in step["run"]


def test_release_rechecks_quality_docs_and_dependencies() -> None:
    quality = _load("release.yml")["jobs"]["quality"]
    commands = "\n".join(step.get("run", "") for step in quality["steps"])

    assert "ruff check" in commands
    assert "ruff format --check" in commands
    assert "ty check" in commands
    assert "mkdocs build --strict" in commands
    assert "pip-audit --requirement" in commands


def test_public_provider_integrations_are_manually_reachable() -> None:
    workflow = _load("provider-contracts.yml")
    public_job = workflow["jobs"]["public"]
    command = next(
        step["run"]
        for step in public_job["steps"]
        if step.get("name") == "Run public live integrations"
    )

    for contract in (
        "test_binance_public.py::TestBinancePublicProvider::test_fetch_daily_spot_btc",
        "test_coingecko.py::TestCoinGeckoProvider::test_fetch_ohlcv_btc",
        "test_kalshi.py::TestKalshiProvider::test_list_markets",
        "test_polymarket.py::TestPolymarketProvider::test_get_market_by_slug",
        "test_yahoo.py::TestYahooFinanceProvider::test_fetch_ohlcv_stock_daily",
        "test_release_provider_contracts.py",
    ):
        assert contract in command


def test_credentialed_contract_jobs_run_one_bounded_test() -> None:
    jobs = _load("provider-contracts.yml")["jobs"]
    expected_tests = {
        "cryptocompare": (
            "test_cryptocompare.py::TestCryptoCompareProvider::test_fetch_ohlcv_btc_daily"
        ),
        "oanda": "test_oanda.py::TestOandaProvider::test_fetch_ohlcv_eurusd_daily",
        "finnhub": "test_finnhub.py::TestFinnhubProvider::test_fetch_quote",
        "databento": "test_databento_acceptance.py::test_databento_metadata_contract",
    }

    for provider, expected_test in expected_tests.items():
        command = next(
            step["run"]
            for step in jobs[provider]["steps"]
            if step.get("name") == "Run the live contract"
        )
        assert expected_test in command
