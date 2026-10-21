"""Legacy acquisition failures warn while preserving the public provider error."""

from unittest.mock import patch

import pytest

from ml4t.data.compat.empyrical import get_symbol_returns_from_yahoo
from ml4t.data.core.exceptions import NetworkError


def test_yahoo_compatibility_transport_failure_warns_and_reraises_without_a_partial_return():
    with (
        patch(
            "ml4t.data.providers.yahoo.yf.download", side_effect=OSError("offline capture failure")
        ),
        pytest.warns(UserWarning, match="Yahoo Finance read failed"),
        pytest.raises(NetworkError, match="offline capture failure"),
    ):
        get_symbol_returns_from_yahoo("SPY", "2024-01-02", "2024-01-05")
