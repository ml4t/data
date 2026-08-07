"""Release metadata contract tests."""

import subprocess
import sys
import textwrap
import tomllib
from pathlib import Path

PROJECT_ROOT = Path(__file__).parents[1]


def load_project() -> dict:
    """Load the authoritative package metadata."""
    with (PROJECT_ROOT / "pyproject.toml").open("rb") as config_file:
        return tomllib.load(config_file)["project"]


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
        except ImportError as error:
            assert "ml4t-data[databento]" in str(error)
        else:
            raise AssertionError("Databento-backed symbol was unexpectedly available")
        """
    )

    subprocess.run([sys.executable, "-c", script], cwd=PROJECT_ROOT, check=True)
