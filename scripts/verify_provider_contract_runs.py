"""Require successful provider contract runs for one release commit."""

from __future__ import annotations

import argparse
import json
import subprocess
from collections.abc import Iterable, Mapping
from typing import Any

REQUIRED_CONTRACTS = (
    "public",
    "alpaca",
    "fred",
    "tiingo",
    "massive",
    "finnhub",
    "databento",
    "oanda",
)
UNVERIFIED_CONTRACTS = {
    "cryptocompare": "account registration unavailable during 0.1.0 release validation",
}
RUN_TITLE_PREFIX = "Provider Contract - "


def validate_contract_runs(runs: Iterable[Mapping[str, Any]], commit: str) -> dict[str, str]:
    """Return successful run URLs or raise when a required contract is missing."""
    successful: dict[str, str] = {}
    for run in runs:
        title = run.get("displayTitle")
        if (
            run.get("headSha") != commit
            or run.get("conclusion") != "success"
            or not isinstance(title, str)
            or not title.startswith(RUN_TITLE_PREFIX)
        ):
            continue
        provider = title.removeprefix(RUN_TITLE_PREFIX)
        if provider in REQUIRED_CONTRACTS:
            url = run.get("url")
            successful[provider] = url if isinstance(url, str) else ""

    missing = sorted(set(REQUIRED_CONTRACTS) - successful.keys())
    if missing:
        names = ", ".join(missing)
        raise ValueError(f"missing successful provider contracts for commit {commit}: {names}")
    return successful


def load_contract_runs(repository: str, commit: str) -> list[dict[str, Any]]:
    """Read completed workflow runs through the authenticated GitHub CLI."""
    result = subprocess.run(
        [
            "gh",
            "run",
            "list",
            "--repo",
            repository,
            "--workflow",
            "provider-contracts.yml",
            "--commit",
            commit,
            "--event",
            "workflow_dispatch",
            "--status",
            "success",
            "--limit",
            "100",
            "--json",
            "conclusion,displayTitle,headSha,url",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    runs = json.loads(result.stdout)
    if not isinstance(runs, list):
        raise TypeError("GitHub CLI returned an invalid workflow-run response")
    return runs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", required=True)
    parser.add_argument("--commit", required=True)
    args = parser.parse_args()

    successful = validate_contract_runs(
        load_contract_runs(args.repository, args.commit), args.commit
    )
    for provider in REQUIRED_CONTRACTS:
        print(f"{provider}: {successful[provider]}")
    for provider, reason in UNVERIFIED_CONTRACTS.items():
        print(f"{provider}: unverified - {reason}")


if __name__ == "__main__":
    main()
