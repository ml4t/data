"""Exercise the actual CSV producer with Windows' writable-flush requirement."""

import os
import warnings
from unittest.mock import patch

import pandas as pd
import pytest

from ml4t.data.compat.empyrical import get_returns_cached


@pytest.mark.parametrize("existing", [False, True])
def test_csv_cache_publication_flushes_a_writable_handle_before_atomic_replacement(
    tmp_path, existing
):
    path = tmp_path / "returns.csv"
    frame = pd.DataFrame(
        {"SPY": [0.01, -0.02]}, index=pd.date_range("2024-01-02", periods=2, tz="UTC")
    )
    if existing:
        get_returns_cached(path, lambda: frame * 2, "2020-01-01")
    sync = os.fsync

    def windows_flush(descriptor):
        # A zero-byte OS write checks the actual producing descriptor's access mode
        # without changing its bytes. Windows rejects fsync on read-only handles.
        os.write(descriptor, b"")
        sync(descriptor)

    with (
        warnings.catch_warnings(),
        patch("ml4t.data.utils.atomic.os.fsync", side_effect=windows_flush),
    ):
        warnings.simplefilter("error", UserWarning)
        result = get_returns_cached(path, lambda: frame, "2100-01-01")
    assert path.is_file()
    restored = pd.read_csv(path, index_col=0, parse_dates=True)
    pd.testing.assert_frame_equal(result, frame, check_freq=False)
    pd.testing.assert_frame_equal(restored, frame, check_freq=False)
    assert not list(tmp_path.glob("*.tmp"))
