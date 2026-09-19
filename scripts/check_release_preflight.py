"""Reject an ambiguous or already-published ml4t-data release candidate."""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass

from packaging.version import InvalidVersion, Version

COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}")


@dataclass(frozen=True)
class PublicationState:
    """Existing public records for one proposed version."""

    git_tag: bool
    github_release: bool
    pypi_version: bool


def version_failure(value: str) -> str | None:
    """Return why a version cannot identify a new release."""
    try:
        version = Version(value)
    except InvalidVersion:
        return "release version is not PEP 440 compliant"
    if (
        str(version) != value
        or len(version.release) != 3
        or version.is_devrelease
        or version.is_postrelease
        or version.local is not None
    ):
        return "release version must be normalized X.Y.Z or X.Y.ZaN/X.Y.ZbN/X.Y.ZrcN"
    return None


def preflight_failures(
    *,
    version: str,
    candidate_sha: str,
    checked_out_sha: str,
    workflow_sha: str,
    main_sha: str,
    publication: PublicationState,
) -> list[str]:
    """Return every condition that makes publication unsafe."""
    failures = []
    if failure := version_failure(version):
        failures.append(failure)
    if not COMMIT_PATTERN.fullmatch(candidate_sha):
        failures.append("candidate commit is not a full lowercase SHA")
    if checked_out_sha != candidate_sha:
        failures.append("checked-out commit differs from the requested candidate")
    if workflow_sha != candidate_sha:
        failures.append("workflow revision differs from the requested candidate")
    if main_sha != candidate_sha:
        failures.append("candidate commit is not the current origin/main revision")
    if publication.git_tag:
        failures.append(f"Git tag v{version} already exists")
    if publication.github_release:
        failures.append(f"GitHub release v{version} already exists")
    if publication.pypi_version:
        failures.append(f"PyPI version {version} already exists")
    return failures


def _git_output(*arguments: str) -> str:
    result = subprocess.run(["git", *arguments], check=True, capture_output=True, text=True)
    return result.stdout.strip()


def _resource_exists(url: str, *, token: str | None = None) -> bool:
    headers = {"Accept": "application/vnd.github+json", "User-Agent": "ml4t-data-release"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
        headers["X-GitHub-Api-Version"] = "2022-11-28"
    request = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=30):  # noqa: S310
            return True
    except urllib.error.HTTPError as error:
        if error.code == 404:
            return False
        raise


def publication_state(repository: str, version: str) -> PublicationState:
    """Read tag, release, and PyPI existence without changing public state."""
    encoded_tag = urllib.parse.quote(f"tags/v{version}", safe="")
    encoded_version = urllib.parse.quote(version, safe="")
    return PublicationState(
        git_tag=_resource_exists(
            f"https://api.github.com/repos/{repository}/git/ref/{encoded_tag}",
            token=os.environ.get("GITHUB_TOKEN"),
        ),
        github_release=_resource_exists(
            f"https://api.github.com/repos/{repository}/releases/tags/v{encoded_version}",
            token=os.environ.get("GITHUB_TOKEN"),
        ),
        pypi_version=_resource_exists(f"https://pypi.org/pypi/ml4t-data/{encoded_version}/json"),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", required=True)
    parser.add_argument("--candidate-sha", required=True)
    parser.add_argument("--repository", default="ml4t/data")
    parser.add_argument("--workflow-sha", required=True)
    args = parser.parse_args()
    failures = preflight_failures(
        version=args.version,
        candidate_sha=args.candidate_sha,
        checked_out_sha=_git_output("rev-parse", "HEAD"),
        workflow_sha=args.workflow_sha,
        main_sha=_git_output("rev-parse", "refs/remotes/origin/main"),
        publication=publication_state(args.repository, args.version),
    )
    print(f"release preflight: {'PASS' if not failures else 'FAIL'}")
    for failure in failures:
        print(f"- {failure}")
    return int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())
