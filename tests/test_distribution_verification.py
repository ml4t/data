"""Tests for release distribution validation."""

from __future__ import annotations

import io
import tarfile
import zipfile
from pathlib import Path

import pytest

from scripts.verify_distribution import validate_distributions, write_checksums


def _write_distributions(dist_dir: Path, version: str = "0.1.0") -> None:
    wheel = dist_dir / f"ml4t_data-{version}-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr(
            f"ml4t_data-{version}.dist-info/METADATA",
            f"Metadata-Version: 2.4\nName: ml4t-data\nVersion: {version}\n",
        )

    sdist = dist_dir / f"ml4t_data-{version}.tar.gz"
    metadata = f"Metadata-Version: 2.4\nName: ml4t-data\nVersion: {version}\n".encode()
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
