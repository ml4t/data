"""Offline price-table migration: source CSV -> storage -> pandas panel -> native table."""

from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd
import polars as pl
from polars.testing import assert_frame_equal

from ml4t.data.storage import FlatStorage, StorageConfig
from ml4t.data.utils.conversion import from_price_panel, to_price_panel
from ml4t.specs import FeedSpec


def migrate_prices(directory: Path) -> tuple[pl.DataFrame, pd.DataFrame]:
    source = pl.read_csv(
        StringIO(
            "timestamp,asset,close\n"
            "2025-07-02T15:00:00Z,A,100\n2025-07-02T15:00:00Z,B,50\n"
            "2025-07-03T15:00:00Z,A,101\n2025-07-03T15:00:00Z,B,52\n"
            "2025-07-07T15:00:00Z,A,98\n2025-07-07T15:00:00Z,B,\n"
            "2025-07-08T15:00:00Z,A,103\n2025-07-08T15:00:00Z,B,53\n"
        ),
        schema_overrides={"close": pl.Float64},
    ).with_columns(pl.col("timestamp").str.to_datetime(time_unit="ns", time_zone="UTC"))
    feed = FeedSpec(
        time_contract_version=2,
        calendar="NYSE",
        timestamp_semantics="event_time",
        entity_col="asset",
        data_frequency="daily",
    )
    storage = FlatStorage(StorageConfig(base_path=directory))
    storage.write(source, "captured/prices", feed.to_dict())
    native = storage.read("captured/prices").collect()
    panel = to_price_panel(native, feed_spec=feed, metadata={"price_basis": "raw"})
    assert panel["A"].tolist() == [100, 101, 98, 103]
    assert pd.isna(panel["B"].iloc[2])
    assert_frame_equal(from_price_panel(panel), source)
    return native, panel


if __name__ == "__main__":
    with TemporaryDirectory(prefix="ml4t-price-handoff-") as temporary:
        native, panel = migrate_prices(Path(temporary))
        print(panel)
        print(f"Retained {native.height} observations and the original missing quote")
