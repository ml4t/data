"""End-to-end contracts for the canonical storage configuration."""

from datetime import datetime

import polars as pl
import pytest
from pydantic import ValidationError

from ml4t.data.config import StorageConfig as ConfigStorageConfig
from ml4t.data.core.config import StorageConfig as CoreStorageConfig
from ml4t.data.storage import StorageConfig, create_storage


def test_every_public_import_exposes_the_same_storage_config():
    assert ConfigStorageConfig is StorageConfig
    assert CoreStorageConfig is StorageConfig


@pytest.mark.parametrize("strategy", ["hive", "flat"])
def test_every_accepted_strategy_constructs_and_round_trips(tmp_path, strategy):
    data = pl.DataFrame({"timestamp": [datetime(2026, 1, 2)], "symbol": ["AAPL"], "close": [100.0]})
    config = StorageConfig(base_path=tmp_path / strategy, strategy=strategy)

    storage = create_storage(config)
    storage.write(data, "equity/daily/AAPL")

    assert storage.config is config
    assert storage.read("equity/daily/AAPL").collect().equals(data)


@pytest.mark.parametrize("backend", ["s3", "memory"])
def test_unimplemented_beta_backends_are_rejected(backend):
    with pytest.raises(ValidationError, match="Unsupported storage backend"):
        StorageConfig(backend=backend)


def test_filesystem_beta_backend_migrates_to_hive(tmp_path):
    config = StorageConfig(base_path=tmp_path, backend="filesystem")

    assert config.strategy == "hive"


def test_non_atomic_write_option_is_rejected():
    with pytest.raises(ValidationError, match="always atomic"):
        StorageConfig(atomic_writes=False)


def test_profile_generation_option_is_rejected():
    with pytest.raises(ValidationError, match="generate_profile"):
        StorageConfig(generate_profile=True)


def test_lock_timeout_reaches_backend_lock(tmp_path):
    storage = create_storage(StorageConfig(base_path=tmp_path, lock_timeout=17))

    assert storage._key_lock("AAPL").timeout == 17


@pytest.mark.parametrize("compression", ["zstd", "lz4", "snappy", "gzip", None, "none"])
def test_every_accepted_compression_writes_readable_parquet(tmp_path, compression):
    data = pl.DataFrame({"timestamp": [datetime(2026, 1, 2)], "close": [100.0]})
    storage = create_storage(
        StorageConfig(base_path=tmp_path / str(compression), compression=compression)
    )

    storage.write(data, "prices")

    assert storage.read("prices").collect().equals(data)


@pytest.mark.parametrize(
    ("granularity", "parts"),
    [
        ("year", ["year=2026"]),
        ("month", ["year=2026", "month=1"]),
        ("day", ["year=2026", "month=1", "day=2"]),
        ("hour", ["year=2026", "month=1", "day=2", "hour=0"]),
    ],
)
def test_every_partition_granularity_changes_the_hive_layout(tmp_path, granularity, parts):
    data = pl.DataFrame({"timestamp": [datetime(2026, 1, 2)], "close": [100.0]})
    storage = create_storage(
        StorageConfig(base_path=tmp_path / granularity, partition_granularity=granularity)
    )

    generation = storage.write(data, "prices")

    assert generation.joinpath(*parts, "data.parquet").is_file()


def test_disabling_metadata_tracking_removes_domain_metadata(tmp_path):
    data = pl.DataFrame({"timestamp": [datetime(2026, 1, 2)], "close": [100.0]})
    storage = create_storage(StorageConfig(base_path=tmp_path, metadata_tracking=False))

    storage.write(data, "prices", metadata={"provider": "test"})

    assert storage.get_metadata("prices") is None
    assert storage.read("prices").collect().equals(data)
