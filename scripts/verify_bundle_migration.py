"""Exercise the optional bundle route from an installed immutable Data distribution."""

from __future__ import annotations

import hashlib
import json
import subprocess
from importlib.metadata import distribution, version
from pathlib import Path

import pytest
import zipline  # noqa: F401

import ml4t.data

ZIPLINE_REVISION = "cf649ac06fddce76f6caa2403cd8494ac84f72ad"


def main() -> int:
    """Reject source-shadowed installs before testing real standard writers and readers."""
    root = Path(__file__).resolve().parents[1]
    package = Path(ml4t.data.__file__).resolve().parent
    installed_data = distribution("ml4t-data")
    direct_data = json.loads(installed_data.read_text("direct_url.json") or "{}")
    if (
        package != Path(installed_data.locate_file("ml4t/data")).resolve()
        or package.is_relative_to(root / "src")
        or direct_data.get("dir_info", {}).get("editable", False)
    ):
        raise ValueError("Bundle qualification must use installed Data, not an editable checkout")
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    if "g" + revision[:9] not in version("ml4t-data"):
        raise ValueError("Installed Data does not identify the checked-out revision")
    names = subprocess.check_output(
        ["git", "ls-tree", "-r", "--name-only", revision, "src/ml4t/data"],
        cwd=root,
        text=True,
    ).splitlines()
    for name in names:
        if name.endswith("/_version.py"):
            continue
        original = subprocess.check_output(["git", "show", f"{revision}:{name}"], cwd=root)
        installed = package / name.removeprefix("src/ml4t/data/")
        if hashlib.sha256(installed.read_bytes()).digest() != hashlib.sha256(original).digest():
            raise ValueError(f"Installed Data source differs: {name}")
    direct = json.loads(distribution("zipline-reloaded").read_text("direct_url.json") or "{}")
    if direct.get("vcs_info", {}).get("commit_id") != ZIPLINE_REVISION:
        raise ValueError("Bundle qualification requires the declared immutable Zipline revision")
    return pytest.main(
        [
            str(root / "tests/test_zipline_bundle_exchange.py"),
            "-q",
            "-ra",
            "-W",
            "error::ResourceWarning",
            "-o",
            "addopts=",
        ]
    )


if __name__ == "__main__":
    raise SystemExit(main())
