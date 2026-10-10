"""Behavioral contracts for the commit-bound release helpers."""

from __future__ import annotations

import io
import re
import tarfile
import zipfile
from copy import deepcopy
from email.message import Message
from pathlib import Path

import pytest

from scripts.check_release_preflight import PublicationState, preflight_failures, version_failure
from scripts.release_candidate import (
    EXPECTED_AUTHOR,
    EXPECTED_CLASSIFIERS,
    EXPECTED_DESCRIPTION,
    EXPECTED_KEYWORDS,
    EXPECTED_MAINTAINER,
    EXPECTED_URLS,
    create,
    metadata_failures,
)
from scripts.run_readme_quickstart import extract_quick_start
from scripts.verify_documentation_identity import (
    identity_failures,
    page_reference_urls,
    site_link_failures,
)
from scripts.verify_published_release import published_release_failures
from scripts.verify_readme_links import extract_markdown_targets, readme_link_failures

COMMIT = "a" * 40
TREE = "b" * 40
VERSION = "1.2.3"
REPOSITORY_ROOT = Path(__file__).parents[1]


@pytest.mark.parametrize("version", ["1.2", "v1.2.3", "1.2.3.dev1", "1.2.3.post1", "1.2.3+local"])
def test_release_preflight_rejects_ambiguous_versions(version: str) -> None:
    assert version_failure(version) is not None


def test_release_preflight_requires_exact_unpublished_main_candidate() -> None:
    assert (
        preflight_failures(
            version=VERSION,
            candidate_sha=COMMIT,
            checked_out_sha=COMMIT,
            workflow_sha=COMMIT,
            main_sha=COMMIT,
            publication=PublicationState(False, False, False),
        )
        == []
    )
    failures = preflight_failures(
        version=VERSION,
        candidate_sha=COMMIT,
        checked_out_sha="c" * 40,
        workflow_sha="d" * 40,
        main_sha="e" * 40,
        publication=PublicationState(True, True, True),
    )
    assert len(failures) == 6


def canonical_metadata() -> Message:
    metadata = Message()
    for name, value in {
        "Name": "ml4t-data",
        "Version": VERSION,
        "Summary": EXPECTED_DESCRIPTION,
        "Author-email": EXPECTED_AUTHOR,
        "Maintainer-email": EXPECTED_MAINTAINER,
        "License-Expression": "MIT",
        "Requires-Python": ">=3.12,<3.15",
        "Keywords": ",".join(sorted(EXPECTED_KEYWORDS)),
        "License-File": "LICENSE",
    }.items():
        metadata[name] = value
    for classifier in sorted(EXPECTED_CLASSIFIERS):
        metadata["Classifier"] = classifier
    for label, url in EXPECTED_URLS.items():
        metadata["Project-URL"] = f"{label}, {url}"
    return metadata


def test_candidate_manifest_records_exact_artifact_bytes(tmp_path: Path) -> None:
    candidate = tmp_path / "candidate"
    dist = candidate / "dist"
    dist.mkdir(parents=True)
    metadata = canonical_metadata().as_bytes()
    wheel = dist / "ml4t_data-1.2.3-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("ml4t_data-1.2.3.dist-info/METADATA", metadata)
    sdist = dist / "ml4t_data-1.2.3.tar.gz"
    with tarfile.open(sdist, "w:gz") as archive:
        info = tarfile.TarInfo("ml4t_data-1.2.3/PKG-INFO")
        info.size = len(metadata)
        archive.addfile(info, io.BytesIO(metadata))

    create(candidate, COMMIT, TREE)

    assert (candidate / "candidate.json").is_file()


def test_metadata_validation_detects_identity_drift() -> None:
    metadata = canonical_metadata()
    assert metadata_failures(metadata, expected_version=VERSION) == []
    metadata.replace_header("Summary", "Different package")
    assert any(
        "Summary" in failure for failure in metadata_failures(metadata, expected_version=VERSION)
    )


def published_fixture() -> tuple[dict, dict, dict]:
    artifacts = [
        {"filename": "ml4t_data-1.2.3-py3-none-any.whl", "sha256": "b" * 64},
        {"filename": "ml4t_data-1.2.3.tar.gz", "sha256": "c" * 64},
    ]
    manifest = {
        "schema_version": 1,
        "name": "ml4t-data",
        "version": VERSION,
        "commit_sha": COMMIT,
        "artifacts": deepcopy(artifacts),
    }
    pypi = {
        "info": {
            "name": "ml4t-data",
            "version": VERSION,
            "summary": EXPECTED_DESCRIPTION,
            "author_email": EXPECTED_AUTHOR,
            "maintainer_email": EXPECTED_MAINTAINER,
            "license_expression": "MIT",
            "requires_python": ">=3.12,<3.15",
            "project_urls": EXPECTED_URLS,
            "keywords": ",".join(sorted(EXPECTED_KEYWORDS)),
            "classifiers": sorted(EXPECTED_CLASSIFIERS),
        },
        "urls": [
            {"filename": record["filename"], "digests": {"sha256": record["sha256"]}}
            for record in artifacts
        ],
    }
    release = {
        "tag_name": f"v{VERSION}",
        "target_commitish": COMMIT,
        "assets": [
            {"name": record["filename"], "digest": f"sha256:{record['sha256']}"}
            for record in artifacts
        ],
    }
    return manifest, pypi, release


def test_published_release_must_match_manifest() -> None:
    manifest, pypi, release = published_fixture()
    assert published_release_failures(manifest, pypi, release) == []
    release["assets"][0]["digest"] = "sha256:" + "d" * 64
    assert "GitHub release artifact names or digests differ from the release manifest" in (
        published_release_failures(manifest, pypi, release)
    )


def test_documentation_identity_and_quickstart_are_executable_contracts() -> None:
    html = (
        '<meta name="ml4t-library" content="data">'
        f'<meta name="ml4t-version" content="{VERSION}">'
        f'<meta name="ml4t-commit" content="{COMMIT}">'
    )
    assert (
        identity_failures(
            html,
            expected_library="data",
            expected_version=VERSION,
            expected_commit=COMMIT,
            source="index.html",
        )
        == []
    )
    assert extract_quick_start("## Quick start\n\n```python\nvalue = 1\n```\n") == "value = 1\n"


def test_rendered_documentation_rejects_broken_links_assets_and_fragments(tmp_path: Path) -> None:
    site = tmp_path / "site"
    (site / "guide").mkdir(parents=True)
    (site / "assets").mkdir()
    (site / "assets/app.js").write_text("", encoding="utf-8")
    (site / "guide/index.html").write_text('<h1 id="working">Guide</h1>', encoding="utf-8")
    (site / "index.html").write_text(
        '<a href="guide/#missing">Guide</a><script src="assets/app.js"></script>'
        '<img src="assets/missing.png">',
        encoding="utf-8",
    )

    failures = site_link_failures(site)

    assert any("no matching anchor" in failure for failure in failures)
    assert any("does not resolve" in failure for failure in failures)


def test_deployed_documentation_discovers_only_same_route_references() -> None:
    html = (
        '<a href="guide/">Guide</a><img src="/docs/data/assets/logo.svg">'
        '<a href="/docs/backtest/">Other library</a><a href="https://example.com/">External</a>'
    )

    assert page_reference_urls(
        html,
        page_url="https://www.ml4trading.io/docs/data/",
        site_root="https://www.ml4trading.io/docs/data/",
    ) == [
        "https://www.ml4trading.io/docs/data/guide/",
        "https://www.ml4trading.io/docs/data/assets/logo.svg",
    ]


def test_readme_link_check_rejects_missing_local_and_unavailable_remote_targets(
    tmp_path: Path,
) -> None:
    readme = tmp_path / "README.md"
    (tmp_path / "LICENSE").write_text("MIT\n", encoding="utf-8")
    readme.write_text(
        "[license](LICENSE) [missing](missing.md) [remote](https://example.invalid/)\n",
        encoding="utf-8",
    )

    failures = readme_link_failures(
        readme,
        probe_url=lambda url: (_ for _ in ()).throw(ValueError(f"unavailable: {url}")),
    )

    assert extract_markdown_targets(readme.read_text(encoding="utf-8")) == [
        "LICENSE",
        "missing.md",
        "https://example.invalid/",
    ]
    assert any("local target does not exist" in failure for failure in failures)
    assert any("unavailable" in failure for failure in failures)


def test_external_workflow_actions_use_full_commit_pins() -> None:
    action = re.compile(r"^\s*uses:\s*(?!\./)([^\s#]+)@([^\s#]+)", re.MULTILINE)
    failures = []
    for workflow in sorted((REPOSITORY_ROOT / ".github/workflows").glob("*.yml")):
        for name, revision in action.findall(workflow.read_text(encoding="utf-8")):
            if not re.fullmatch(r"[0-9a-f]{40}", revision):
                failures.append(f"{workflow.name}: {name}@{revision}")
    assert failures == []
