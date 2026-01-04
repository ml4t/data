"""Test basic imports work."""

import ml4t.data


def test_ml4t_data_import():
    """Test that ml4t.data can be imported."""
    assert ml4t.data.__version__.startswith("0.1.0")


def test_ml4t_data_metadata():
    """Test ml4t.data metadata is correct."""
    assert ml4t.data.__author__ == "QuantLab Team"
    assert ml4t.data.__email__ == "info@quantlab.io"
