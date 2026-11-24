"""Public storage loads retain the entire inclusive end date."""

from datetime import UTC, datetime

import polars as pl
import pytest

from ml4t.data import DataManager
from ml4t.data.storage import create_storage


@pytest.mark.parametrize("strategy", ["hive", "flat"])
@pytest.mark.parametrize("start,end", [("2026-11-16", "2026-11-20"), ("2026-11-20", "2026-11-20")])
def test_batch_storage_load_keeps_end_date_bars_and_excludes_next_day(
    tmp_path, strategy, start, end
):
    storage = create_storage(tmp_path, strategy=strategy)
    stamps = [
        datetime(2026, 11, 15, 23, 59, tzinfo=UTC),
        datetime(2026, 11, 16, 5, tzinfo=UTC),
        datetime(2026, 11, 19, 5, tzinfo=UTC),
        datetime(2026, 11, 20, tzinfo=UTC),
        datetime(2026, 11, 20, 5, tzinfo=UTC),
        datetime(2026, 11, 20, 23, 59, 59, 999999, tzinfo=UTC),
        datetime(2026, 11, 21, tzinfo=UTC),
    ]
    bars = pl.DataFrame(
        {
            "timestamp": stamps,
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.0,
            "volume": 1000.0,
        }
    )
    storage.write(bars, "equities/daily/SPY")
    with DataManager(storage=storage) as manager:
        result = manager.batch_load_from_storage(["SPY"], start, end, fetch_missing=False)
    expected = [stamp for stamp in stamps if start <= stamp.date().isoformat() <= end]
    assert result["timestamp"].to_list() == expected
    assert result["symbol"].to_list() == ["SPY"] * len(expected)

    # The lower-level storage timestamp contract remains half-open.
    exclusive = storage.read(
        "equities/daily/SPY", end_date=datetime(2026, 11, 20, tzinfo=UTC)
    ).collect()
    assert exclusive["timestamp"].to_list() == stamps[:3]
