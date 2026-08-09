"""Tests for release provider-contract evidence verification."""

from __future__ import annotations

import pytest

from scripts.verify_provider_contract_runs import (
    REQUIRED_CONTRACTS,
    UNVERIFIED_CONTRACTS,
    validate_contract_runs,
)


def _run(provider: str, commit: str, conclusion: str = "success") -> dict[str, str]:
    return {
        "conclusion": conclusion,
        "displayTitle": f"Provider Contract - {provider}",
        "headSha": commit,
        "url": f"https://github.com/ml4t/data/actions/runs/{provider}",
    }


def test_all_required_contracts_must_pass_for_the_release_commit() -> None:
    commit = "a" * 40
    runs = [_run(provider, commit) for provider in REQUIRED_CONTRACTS]

    successful = validate_contract_runs(runs, commit)

    assert tuple(successful) == REQUIRED_CONTRACTS


def test_unverified_cryptocompare_is_not_release_qualified() -> None:
    assert set(UNVERIFIED_CONTRACTS) == {"cryptocompare"}
    assert set(REQUIRED_CONTRACTS).isdisjoint(UNVERIFIED_CONTRACTS)


def test_contracts_from_another_commit_do_not_satisfy_release_policy() -> None:
    commit = "a" * 40
    runs = [_run(provider, "b" * 40) for provider in REQUIRED_CONTRACTS]

    with pytest.raises(ValueError, match="missing successful provider contracts"):
        validate_contract_runs(runs, commit)


def test_failed_or_missing_contracts_are_reported() -> None:
    commit = "a" * 40
    runs = [
        _run(provider, commit)
        for provider in REQUIRED_CONTRACTS
        if provider not in {"databento", "oanda"}
    ]
    runs.append(_run("databento", commit, conclusion="failure"))

    with pytest.raises(ValueError, match="databento, oanda"):
        validate_contract_runs(runs, commit)
