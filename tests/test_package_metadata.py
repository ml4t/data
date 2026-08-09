"""Release metadata contract tests."""

import subprocess
import sys
import textwrap
import tomllib
from pathlib import Path

PROJECT_ROOT = Path(__file__).parents[1]


def load_config() -> dict:
    """Load the authoritative project configuration."""
    with (PROJECT_ROOT / "pyproject.toml").open("rb") as config_file:
        return tomllib.load(config_file)


def load_project() -> dict:
    """Load the authoritative package metadata."""
    return load_config()["project"]


def test_supported_python_and_platform_metadata() -> None:
    """Distribution metadata matches the stable support policy."""
    project = load_project()
    classifiers = set(project["classifiers"])

    assert project["requires-python"] == ">=3.12"
    assert "Development Status :: 5 - Production/Stable" in classifiers
    assert "Operating System :: OS Independent" not in classifiers
    assert {
        "Operating System :: MacOS",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX :: Linux",
    } <= classifiers
    assert {
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Programming Language :: Python :: 3.14",
    } <= classifiers
    assert "Programming Language :: Python :: 3.11" not in classifiers
    assert "Programming Language :: Python :: 3.15" not in classifiers


def test_license_uses_spdx_metadata() -> None:
    """Built distributions declare the MIT license using PEP 639 metadata."""
    project = load_project()

    assert project["license"] == "MIT"
    assert project["license-files"] == ["LICENSE"]


def test_source_distribution_contains_release_verifiers() -> None:
    """Tests shipped in the source archive can import their release helpers."""
    included = set(load_config()["tool"]["hatch"]["build"]["targets"]["sdist"]["include"])

    assert {
        "/scripts/verify_distribution.py",
        "/scripts/verify_provider_contract_runs.py",
    } <= included


def test_oanda_extra_declares_undeclared_client_dependency() -> None:
    """The standalone OANDA extra supplies oandapyV20's requests import."""
    dependencies = load_project()["optional-dependencies"]["oanda"]

    assert any(dependency.startswith("oandapyV20") for dependency in dependencies)
    assert any(dependency.startswith("requests") for dependency in dependencies)


def test_core_futures_import_does_not_require_databento_extra() -> None:
    """Core futures algorithms remain importable without the optional SDK."""
    script = textwrap.dedent(
        """
        import builtins

        original_import = builtins.__import__

        def block_databento(name, *args, **kwargs):
            if name == "databento" or name.startswith("databento."):
                raise ModuleNotFoundError("blocked optional dependency", name="databento")
            return original_import(name, *args, **kwargs)

        builtins.__import__ = block_databento

        import ml4t.data.futures as futures

        assert hasattr(futures, "ContinuousContractBuilder")
        try:
            futures.FuturesDownloader
        except AttributeError as error:
            assert "ml4t-data[databento]" in str(error)
        else:
            raise AssertionError("Databento-backed symbol was unexpectedly available")
        assert not hasattr(futures, "FuturesDownloader")
        assert "FuturesDownloader" not in futures.__all__
        try:
            futures.require_databento("FuturesDownloader")
        except ImportError as error:
            assert "ml4t-data[databento]" in str(error)
        else:
            raise AssertionError("Databento requirement check unexpectedly passed")
        """
    )

    subprocess.run([sys.executable, "-c", script], cwd=PROJECT_ROOT, check=True)


def test_core_import_is_silent_without_optional_provider_dependencies() -> None:
    """Missing optional provider clients do not write during a core import."""
    script = textwrap.dedent(
        """
        import builtins

        original_import = builtins.__import__

        def block_optional_clients(name, *args, **kwargs):
            if (
                name == "databento"
                or name.startswith("databento.")
                or name == "oandapyV20"
                or name.startswith("oandapyV20.")
            ):
                raise ModuleNotFoundError("blocked optional dependency", name=name)
            return original_import(name, *args, **kwargs)

        builtins.__import__ = block_optional_clients

        import ml4t.data
        """
    )

    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert completed.stdout == ""
    assert completed.stderr == ""
