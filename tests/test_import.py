"""Test basic imports work."""

import re

import ml4t.data
from ml4t.data import cli_interface


def test_ml4t_data_import():
    """Test that ml4t.data can be imported."""
    assert re.match(r"^\d+\.\d+(?:\.\d+)?(?:[a-zA-Z0-9.+-]*)?$", ml4t.data.__version__)


def test_ml4t_data_metadata():
    """Test ml4t.data metadata is correct."""
    assert ml4t.data.__author__ == "ML4T Team"
    assert ml4t.data.__email__ == "info@ml4trading.io"


def test_cli_interface_exports_entrypoints():
    """CLI module should expose callable entrypoints used by package scripts."""
    assert callable(cli_interface.cli)
    assert callable(cli_interface.main)
