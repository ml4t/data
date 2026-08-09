"""Validate and install the release distributions in isolated environments."""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import subprocess
import sys
import tarfile
import tempfile
import zipfile
from email.message import Message
from email.parser import Parser
from pathlib import Path

PROJECT_NAME = "ml4t-data"
TAG_PATTERN = re.compile(r"^v(?P<version>\d+\.\d+\.\d+(?:(?:a|b|rc)\d+)?)$")


def _metadata_from_wheel(path: Path) -> Message:
    with zipfile.ZipFile(path) as archive:
        metadata_files = [
            name for name in archive.namelist() if name.endswith(".dist-info/METADATA")
        ]
        if len(metadata_files) != 1:
            raise ValueError(f"{path.name} must contain exactly one METADATA file")
        content = archive.read(metadata_files[0]).decode("utf-8")
    return Parser().parsestr(content)


def _metadata_from_sdist(path: Path) -> Message:
    with tarfile.open(path, mode="r:gz") as archive:
        metadata_files = [
            member
            for member in archive.getmembers()
            if member.name.count("/") == 1 and member.name.endswith("/PKG-INFO")
        ]
        if len(metadata_files) != 1:
            raise ValueError(f"{path.name} must contain exactly one top-level PKG-INFO file")
        extracted = archive.extractfile(metadata_files[0])
        if extracted is None:
            raise ValueError(f"could not read PKG-INFO from {path.name}")
        content = extracted.read().decode("utf-8")
    return Parser().parsestr(content)


def validate_distributions(
    dist_dir: Path, expected_tag: str | None = None
) -> tuple[list[Path], str]:
    wheels = sorted(dist_dir.glob("*.whl"))
    sdists = sorted(dist_dir.glob("*.tar.gz"))
    if len(wheels) != 1 or len(sdists) != 1:
        raise ValueError(
            f"expected one wheel and one sdist in {dist_dir}, found {len(wheels)} wheel(s) "
            f"and {len(sdists)} sdist(s)"
        )

    expected_version = None
    if expected_tag is not None:
        match = TAG_PATTERN.fullmatch(expected_tag)
        if match is None:
            raise ValueError(
                f"release tag must match vX.Y.Z, vX.Y.ZaN, vX.Y.ZbN, or vX.Y.ZrcN: {expected_tag}"
            )
        expected_version = match.group("version")

    archives = [*wheels, *sdists]
    metadata = [_metadata_from_wheel(wheels[0]), _metadata_from_sdist(sdists[0])]
    names = [item["Name"] for item in metadata]
    if any(name != PROJECT_NAME for name in names):
        raise ValueError(f"distribution name must be {PROJECT_NAME!r}: {names}")

    for item in metadata:
        license_expression = item["License-Expression"]
        license_files = item.get_all("License-File", [])
        if license_expression != "MIT" or "LICENSE" not in license_files:
            raise ValueError(
                "distribution license metadata must declare "
                f"License-Expression: MIT and License-File: LICENSE: "
                f"expression={license_expression!r}, files={license_files!r}"
            )

    versions = {item["Version"] for item in metadata}
    if len(versions) != 1:
        raise ValueError(f"wheel and sdist versions differ: {sorted(versions)}")
    version = versions.pop()
    if expected_version is not None and version != expected_version:
        raise ValueError(
            f"release tag {expected_tag!r} does not match distribution version {version!r}"
        )
    return archives, version


def write_checksums(archives: list[Path], output: Path) -> None:
    lines = []
    for archive in sorted(archives, key=lambda path: path.name):
        digest = hashlib.sha256(archive.read_bytes()).hexdigest()
        lines.append(f"{digest}  {archive.name}")
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _venv_python(venv: Path) -> Path:
    return venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def _venv_cli(venv: Path) -> Path:
    return venv / ("Scripts/ml4t-data.exe" if os.name == "nt" else "bin/ml4t-data")


def verify_install(archive: Path, version: str) -> None:
    with tempfile.TemporaryDirectory(prefix="ml4t-data-package-") as temporary:
        venv = Path(temporary) / "venv"
        subprocess.run(["uv", "venv", "--python", sys.executable, str(venv)], check=True)
        python = _venv_python(venv)
        subprocess.run(["uv", "pip", "install", "--python", str(python), str(archive)], check=True)
        subprocess.run(["uv", "pip", "check", "--python", str(python)], check=True)
        completed = subprocess.run(
            [
                str(python),
                "-c",
                (
                    "import ml4t.data as package; "
                    f"assert package.__version__ == {version!r}, package.__version__"
                ),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        if completed.stdout or completed.stderr:
            raise ValueError(
                f"{archive.name} import must be silent: "
                f"stdout={completed.stdout!r}, stderr={completed.stderr!r}"
            )
        subprocess.run([str(_venv_cli(venv)), "--help"], check=True, stdout=subprocess.DEVNULL)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dist-dir", type=Path, default=Path("dist"))
    parser.add_argument("--expected-tag")
    parser.add_argument("--skip-install", action="store_true")
    parser.add_argument("--write-checksums", action="store_true")
    parser.add_argument("--checksum-file", type=Path)
    args = parser.parse_args()

    archives, version = validate_distributions(args.dist_dir, args.expected_tag)
    if not args.skip_install:
        for archive in archives:
            verify_install(archive.resolve(), version)
    if args.write_checksums:
        checksum_file = args.checksum_file or args.dist_dir / "SHA256SUMS"
        write_checksums(archives, checksum_file)


if __name__ == "__main__":
    main()
