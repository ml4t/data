"""Verify PyPI and GitHub records against the immutable release manifest."""

from __future__ import annotations

import argparse
import json
import os
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

from packaging.specifiers import SpecifierSet

try:
    from scripts.release_candidate import (
        EXPECTED_AUTHOR,
        EXPECTED_CLASSIFIERS,
        EXPECTED_DESCRIPTION,
        EXPECTED_KEYWORDS,
        EXPECTED_MAINTAINER,
        EXPECTED_URLS,
    )
except ModuleNotFoundError:  # Direct execution adds scripts/, not the repository root, to sys.path.
    from release_candidate import (
        EXPECTED_AUTHOR,
        EXPECTED_CLASSIFIERS,
        EXPECTED_DESCRIPTION,
        EXPECTED_KEYWORDS,
        EXPECTED_MAINTAINER,
        EXPECTED_URLS,
    )

HASH_PATTERN = re.compile(r"[0-9a-f]{64}")


def _artifact_digests(manifest: dict[str, Any]) -> dict[str, str]:
    records = manifest.get("artifacts", [])
    if not isinstance(records, list):
        return {}
    return {
        str(record.get("filename", "")): str(record.get("sha256", ""))
        for record in records
        if isinstance(record, dict)
    }


def published_release_failures(
    manifest: dict[str, Any], pypi: dict[str, Any], release: dict[str, Any]
) -> list[str]:
    """Return mismatches across the manifest, PyPI JSON, and GitHub release."""
    failures: list[str] = []
    version = str(manifest.get("version", ""))
    commit = str(manifest.get("commit_sha", ""))
    artifacts = _artifact_digests(manifest)
    if (
        manifest.get("schema_version") != 1
        or manifest.get("name") != "ml4t-data"
        or not version
        or len(commit) != 40
        or not artifacts
        or any(not HASH_PATTERN.fullmatch(digest) for digest in artifacts.values())
    ):
        return ["release manifest is incomplete"]

    info = pypi.get("info", {})
    keywords = {
        keyword.strip() for keyword in str(info.get("keywords", "")).split(",") if keyword.strip()
    }
    pypi_files = {
        str(record.get("filename", "")): str(record.get("digests", {}).get("sha256", ""))
        for record in pypi.get("urls", [])
        if isinstance(record, dict)
    }
    metadata_matches = (
        info.get("name") == "ml4t-data"
        and info.get("version") == version
        and info.get("summary") == EXPECTED_DESCRIPTION
        and info.get("author_email") == EXPECTED_AUTHOR
        and info.get("maintainer_email") == EXPECTED_MAINTAINER
        and info.get("license_expression") == "MIT"
        and SpecifierSet(str(info.get("requires_python", ""))) == SpecifierSet(">=3.12,<3.15")
        and info.get("project_urls") == EXPECTED_URLS
        and keywords >= EXPECTED_KEYWORDS
        and set(info.get("classifiers", [])) >= EXPECTED_CLASSIFIERS
    )
    if not metadata_matches:
        failures.append("PyPI metadata differs from the qualified source")
    if pypi_files != artifacts:
        failures.append("PyPI artifact names or digests differ from the release manifest")

    release_assets = {
        str(asset.get("name", "")): str(asset.get("digest", "")).removeprefix("sha256:")
        for asset in release.get("assets", [])
        if isinstance(asset, dict) and str(asset.get("name", "")) in artifacts
    }
    if release.get("tag_name") != f"v{version}" or release.get("target_commitish") != commit:
        failures.append("GitHub release tag or target differs from the release manifest")
    if release_assets != artifacts:
        failures.append("GitHub release artifact names or digests differ from the release manifest")
    return failures


def _read_json(url: str, *, token: str | None = None) -> dict[str, Any]:
    headers = {"Accept": "application/vnd.github+json", "User-Agent": "ml4t-data-release"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
        headers["X-GitHub-Api-Version"] = "2022-11-28"
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=30) as response:  # noqa: S310
        payload = json.load(response)
    if not isinstance(payload, dict):
        raise ValueError(f"{url} did not return a JSON object")
    return payload


def remote_release_failures(
    manifest: dict[str, Any], repository: str, *, attempts: int = 12, wait_seconds: float = 10.0
) -> list[str]:
    """Poll public records until publication propagation completes."""
    version = urllib.parse.quote(str(manifest.get("version", "")), safe="")
    tag = urllib.parse.quote(f"v{manifest.get('version', '')}", safe="")
    failures: list[str] = []
    for attempt in range(attempts):
        try:
            failures = published_release_failures(
                manifest,
                _read_json(f"https://pypi.org/pypi/ml4t-data/{version}/json"),
                _read_json(
                    f"https://api.github.com/repos/{repository}/releases/tags/{tag}",
                    token=os.environ.get("GITHUB_TOKEN"),
                ),
            )
        except Exception as error:  # noqa: BLE001
            failures = [f"public release records are unavailable: {error}"]
        if not failures:
            return []
        if attempt + 1 < attempts:
            time.sleep(wait_seconds)
    return failures


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--repository", default="ml4t/data")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    failures = remote_release_failures(manifest, args.repository)
    print(f"published release identity: {'PASS' if not failures else 'FAIL'}")
    for failure in failures:
        print(f"- {failure}")
    return int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())
