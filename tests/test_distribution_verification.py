"""Tests for release distribution validation."""

from __future__ import annotations

import io
import tarfile
import zipfile
from pathlib import Path

import pytest

from scripts.verify_distribution import validate_distributions, write_checksums


def _write_distributions(
    dist_dir: Path,
    version: str = "0.1.0",
    license_expression: str | None = "MIT",
    license_file: str | None = "LICENSE",
) -> None:
    metadata_fields = ["Metadata-Version: 2.4", "Name: ml4t-data", f"Version: {version}"]
    if license_expression is not None:
        metadata_fields.append(f"License-Expression: {license_expression}")
    if license_file is not None:
        metadata_fields.append(f"License-File: {license_file}")
    metadata_text = "\n".join(metadata_fields) + "\n"

    wheel = dist_dir / f"ml4t_data-{version}-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr(
            f"ml4t_data-{version}.dist-info/METADATA",
            metadata_text,
        )

    sdist = dist_dir / f"ml4t_data-{version}.tar.gz"
    metadata = metadata_text.encode()
    member = tarfile.TarInfo(f"ml4t_data-{version}/PKG-INFO")
    member.size = len(metadata)
    with tarfile.open(sdist, "w:gz") as archive:
        archive.addfile(member, io.BytesIO(metadata))


def test_distribution_metadata_matches_release_tag(tmp_path: Path) -> None:
    _write_distributions(tmp_path)

    archives, version = validate_distributions(tmp_path, "v0.1.0")

    assert version == "0.1.0"
    assert {path.suffix for path in archives} == {".whl", ".gz"}


def test_distribution_metadata_rejects_tag_mismatch(tmp_path: Path) -> None:
    _write_distributions(tmp_path)

    with pytest.raises(ValueError, match="does not match"):
        validate_distributions(tmp_path, "v0.1.1")


@pytest.mark.parametrize(
    ("license_expression", "license_file"),
    [(None, "LICENSE"), ("Apache-2.0", "LICENSE"), ("MIT", None)],
)
def test_distribution_metadata_rejects_incorrect_license_metadata(
    tmp_path: Path, license_expression: str | None, license_file: str | None
) -> None:
    _write_distributions(
        tmp_path,
        license_expression=license_expression,
        license_file=license_file,
    )

    with pytest.raises(ValueError, match="license metadata"):
        validate_distributions(tmp_path)


@pytest.mark.parametrize("tag", ["0.1.0", "v0.1", "release-0.1.0", "v0.1.0.dev1"])
def test_distribution_metadata_rejects_invalid_release_tag(tmp_path: Path, tag: str) -> None:
    _write_distributions(tmp_path)

    with pytest.raises(ValueError, match="release tag must match"):
        validate_distributions(tmp_path, tag)


def test_checksum_manifest_covers_both_archives(tmp_path: Path) -> None:
    _write_distributions(tmp_path)
    archives, _ = validate_distributions(tmp_path)

    manifest = tmp_path / "SHA256SUMS"
    write_checksums(archives, manifest)

    lines = manifest.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert all(len(line.split()[0]) == 64 for line in lines)
    assert {line.split()[-1] for line in lines} == {archive.name for archive in archives}
