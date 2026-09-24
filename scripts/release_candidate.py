"""Create and verify the immutable ml4t-data release-candidate manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
import zipfile
from email import policy
from email.message import Message
from email.parser import BytesParser
from pathlib import Path
from typing import Any

from packaging.specifiers import SpecifierSet

SCHEMA_VERSION = 1
EXPECTED_DESCRIPTION = (
    "Market data acquisition, storage, and update workflows for machine learning for trading."
)
EXPECTED_AUTHOR = "Stefan Jansen <stefan@applied-ai.com>"
EXPECTED_MAINTAINER = "Stefan Jansen <pm@ml4trading.io>"
EXPECTED_URLS = {
    "Homepage": "https://www.ml4trading.io/",
    "Documentation": "https://www.ml4trading.io/docs/data/",
    "Repository": "https://github.com/ml4t/data",
    "Issues": "https://github.com/ml4t/data/issues",
    "Changelog": "https://github.com/ml4t/data/releases",
}
EXPECTED_KEYWORDS = {"finance", "quantitative-finance", "algorithmic-trading", "market-data"}
EXPECTED_CLASSIFIERS = {
    "Development Status :: 5 - Production/Stable",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3.12",
    "Programming Language :: Python :: 3.13",
    "Programming Language :: Python :: 3.14",
    "Typing :: Typed",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _distribution_files(candidate_dir: Path) -> tuple[Path, Path]:
    dist_dir = candidate_dir / "dist"
    wheels = tuple(dist_dir.glob("*.whl"))
    sdists = tuple(dist_dir.glob("*.tar.gz"))
    if len(wheels) != 1 or len(sdists) != 1:
        raise ValueError(
            f"candidate must contain one wheel and one sdist, found {len(wheels)} and {len(sdists)}"
        )
    return wheels[0], sdists[0]


def _wheel_metadata(wheel: Path) -> Message:
    with zipfile.ZipFile(wheel) as archive:
        metadata_names = [
            name for name in archive.namelist() if name.endswith(".dist-info/METADATA")
        ]
        if len(metadata_names) != 1:
            raise ValueError(f"wheel must contain one METADATA file: {wheel}")
        return BytesParser(policy=policy.default).parsebytes(archive.read(metadata_names[0]))


def _sdist_metadata(sdist: Path) -> Message:
    with tarfile.open(sdist, "r:gz") as archive:
        metadata_members = [
            member
            for member in archive.getmembers()
            if member.name.count("/") == 1 and member.name.endswith("/PKG-INFO")
        ]
        if len(metadata_members) != 1:
            raise ValueError(f"sdist must contain one top-level PKG-INFO file: {sdist}")
        stream = archive.extractfile(metadata_members[0])
        if stream is None:
            raise ValueError(f"cannot read sdist metadata: {sdist}")
        return BytesParser(policy=policy.default).parsebytes(stream.read())


def metadata_failures(metadata: Message, *, expected_version: str) -> list[str]:
    """Return canonical public metadata mismatches."""
    failures: list[str] = []
    expected_scalars = {
        "Name": "ml4t-data",
        "Version": expected_version,
        "Summary": EXPECTED_DESCRIPTION,
        "Author-email": EXPECTED_AUTHOR,
        "Maintainer-email": EXPECTED_MAINTAINER,
        "License-Expression": "MIT",
    }
    for field, expected in expected_scalars.items():
        if metadata.get(field) != expected:
            failures.append(f"{field} is {metadata.get(field)!r}, expected {expected!r}")
    if SpecifierSet(str(metadata.get("Requires-Python", ""))) != SpecifierSet(">=3.12,<3.15"):
        failures.append("Requires-Python differs from >=3.12,<3.15")
    project_urls = {}
    for value in metadata.get_all("Project-URL", []):
        label, separator, url = value.partition(", ")
        if separator:
            project_urls[label] = url
    if project_urls != EXPECTED_URLS:
        failures.append("Project-URL metadata differs from the canonical URLs")
    keywords = {
        value.strip() for value in str(metadata.get("Keywords", "")).split(",") if value.strip()
    }
    if not keywords >= EXPECTED_KEYWORDS:
        failures.append("Keywords omit a required ecosystem or library keyword")
    if not set(metadata.get_all("Classifier", [])) >= EXPECTED_CLASSIFIERS:
        failures.append("Classifiers omit required support metadata")
    if "LICENSE" not in metadata.get_all("License-File", []):
        failures.append("License-File metadata does not identify LICENSE")
    return failures


def _identity(metadata: Message, path: Path) -> tuple[str, str]:
    name = metadata.get("Name")
    version = metadata.get("Version")
    if not name or not version:
        raise ValueError(f"distribution metadata is missing Name or Version: {path}")
    return name, version


def _public_metadata(metadata: Message) -> dict[str, Any]:
    project_urls = {}
    for value in metadata.get_all("Project-URL", []):
        label, url = value.split(",", maxsplit=1)
        project_urls[label.strip()] = url.strip()
    return {
        "author_email": metadata.get("Author-email"),
        "classifiers": sorted(metadata.get_all("Classifier", [])),
        "description": metadata.get("Summary"),
        "keywords": sorted(
            keyword.strip() for keyword in (metadata.get("Keywords") or "").split(",") if keyword
        ),
        "license": metadata.get("License-Expression") or metadata.get("License"),
        "maintainer_email": metadata.get("Maintainer-email"),
        "project_urls": project_urls,
        "requires_python": metadata.get("Requires-Python"),
    }


def _artifact_record(path: Path) -> dict[str, str | int]:
    return {"filename": path.name, "sha256": _sha256(path), "size": path.stat().st_size}


def create(candidate_dir: Path, commit_sha: str, git_tree: str) -> None:
    """Write a manifest after validating the candidate metadata and artifact identity."""
    wheel, sdist = _distribution_files(candidate_dir)
    wheel_metadata = _wheel_metadata(wheel)
    sdist_metadata = _sdist_metadata(sdist)
    wheel_identity = _identity(wheel_metadata, wheel)
    if wheel_identity != _identity(sdist_metadata, sdist):
        raise ValueError("wheel and sdist identities differ")
    name, version = wheel_identity
    failures = metadata_failures(wheel_metadata, expected_version=version)
    failures.extend(metadata_failures(sdist_metadata, expected_version=version))
    if failures:
        raise ValueError(f"candidate metadata failed: {failures}")
    public_metadata = _public_metadata(wheel_metadata)
    if public_metadata != _public_metadata(sdist_metadata):
        raise ValueError("wheel and sdist public metadata differ")
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "commit_sha": commit_sha,
        "git_tree": git_tree,
        "name": name,
        "version": version,
        "metadata": public_metadata,
        "artifacts": [_artifact_record(wheel), _artifact_record(sdist)],
    }
    (candidate_dir / "candidate.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _load_manifest(candidate_dir: Path) -> dict[str, Any]:
    value = json.loads((candidate_dir / "candidate.json").read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("candidate manifest must be a JSON object")
    return value


def verify(
    candidate_dir: Path,
    *,
    expected_commit: str | None = None,
    expected_tree: str | None = None,
    expected_tag: str | None = None,
) -> None:
    """Verify manifest identity, artifact bytes, and canonical metadata."""
    manifest = _load_manifest(candidate_dir)
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"unsupported candidate schema: {manifest.get('schema_version')!r}")
    if expected_commit is not None and manifest.get("commit_sha") != expected_commit:
        raise ValueError("candidate commit differs from the requested release commit")
    if expected_tree is not None and manifest.get("git_tree") != expected_tree:
        raise ValueError("candidate tree differs from the requested release tree")
    if expected_tag is not None and expected_tag != f"v{manifest.get('version')}":
        raise ValueError("release tag differs from the candidate version")

    wheel, sdist = _distribution_files(candidate_dir)
    actual_files = {wheel.name: wheel, sdist.name: sdist}
    records = manifest.get("artifacts")
    if not isinstance(records, list) or len(records) != 2:
        raise ValueError("candidate manifest must describe exactly two artifacts")
    recorded_files = {record.get("filename"): record for record in records}
    if set(recorded_files) != set(actual_files):
        raise ValueError("candidate manifest artifact inventory differs from the dist directory")
    for filename, path in actual_files.items():
        record = recorded_files[filename]
        if record.get("sha256") != _sha256(path) or record.get("size") != path.stat().st_size:
            raise ValueError(f"candidate artifact integrity check failed: {filename}")

    wheel_metadata = _wheel_metadata(wheel)
    sdist_metadata = _sdist_metadata(sdist)
    identity = _identity(wheel_metadata, wheel)
    if identity != _identity(sdist_metadata, sdist):
        raise ValueError("candidate wheel and sdist identities differ")
    if identity != (manifest.get("name"), manifest.get("version")):
        raise ValueError("candidate metadata differs from the manifest")
    failures = metadata_failures(wheel_metadata, expected_version=str(manifest.get("version")))
    failures.extend(
        metadata_failures(sdist_metadata, expected_version=str(manifest.get("version")))
    )
    if failures:
        raise ValueError(f"candidate metadata failed: {failures}")
    if _public_metadata(wheel_metadata) != manifest.get("metadata"):
        raise ValueError("candidate public metadata differs from the manifest")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    create_parser = subparsers.add_parser("create")
    create_parser.add_argument("candidate_dir", type=Path)
    create_parser.add_argument("--commit-sha", required=True)
    create_parser.add_argument("--git-tree", required=True)
    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("candidate_dir", type=Path)
    verify_parser.add_argument("--expected-commit")
    verify_parser.add_argument("--expected-tree")
    verify_parser.add_argument("--expected-tag")
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command == "create":
        create(args.candidate_dir, args.commit_sha, args.git_tree)
    else:
        verify(
            args.candidate_dir,
            expected_commit=args.expected_commit,
            expected_tree=args.expected_tree,
            expected_tag=args.expected_tag,
        )


if __name__ == "__main__":
    main()
