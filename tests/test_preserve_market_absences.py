"""Production update paths must never invent market observations or prices."""

from datetime import UTC, date, datetime

import polars as pl
import pytest
from click.testing import CliRunner
from polars.testing import assert_frame_equal

from ml4t.data import DataManager
from ml4t.data.cli import cli
from ml4t.data.config.models import DatasetConfig
from ml4t.data.providers import SyntheticProvider
from ml4t.data.sessions import SessionCompleter
from ml4t.data.storage.config import StorageConfig
from ml4t.data.storage.flat import FlatStorage
from ml4t.data.storage.hive import HiveStorage
from ml4t.data.utils.gaps import GapDetector


@pytest.mark.parametrize("backend", [FlatStorage, HiveStorage])
@pytest.mark.parametrize("check", ["absent_timestamp", "observed_null"])
def test_default_real_provider_update_preserves_historical_absence_and_observed_null(
    tmp_path, monkeypatch, backend, check
):
    class Clock(datetime):
        @classmethod
        def now(cls, tz=UTC):
            return cls(2025, 7, 15, 23, tzinfo=tz)

    monkeypatch.setattr("ml4t.data.managers.storage_manager.datetime", Clock)
    with SyntheticProvider(seed=42) as provider:
        frame = provider.fetch_ohlcv("SPY", "2025-07-07", "2025-07-14", "daily")
    observed = frame.filter(pl.col("timestamp").dt.date() != date(2025, 7, 9)).with_columns(
        pl.when(pl.col("timestamp").dt.date() == date(2025, 7, 8))
        .then(None)
        .otherwise(pl.col("close"))
        .alias("close")
    )
    storage = backend(StorageConfig(base_path=tmp_path))
    with DataManager(
        storage=storage, providers={"synthetic": {"extra": {"seed": 42}}}, enable_validation=False
    ) as manager:
        key = manager.import_data(observed, "SPY", "synthetic")
        manager.update("SPY", lookback_days=0)
        actual = storage.read(key).collect()
    if check == "absent_timestamp":
        assert date(2025, 7, 9) not in actual["timestamp"].dt.date().to_list()
    else:
        assert (
            actual.filter(pl.col("timestamp").dt.date() == date(2025, 7, 8))["close"].null_count()
            == 1
        )
    retained = actual.filter(pl.col("timestamp").dt.date() < date(2025, 7, 14))
    assert_frame_equal(retained, observed.filter(pl.col("timestamp").dt.date() < date(2025, 7, 14)))
    assert (
        actual.filter(pl.col("timestamp").dt.date() == date(2025, 7, 8))["close"].null_count() == 1
    )


@pytest.mark.parametrize("backend", [FlatStorage, HiveStorage])
def test_explicit_legacy_imputation_is_rejected_before_initial_dataset_creation(tmp_path, backend):
    storage = backend(StorageConfig(base_path=tmp_path))
    with DataManager(storage=storage, providers={"synthetic": {}}) as manager:
        with pytest.raises(ValueError, match="[Ss]ynthetic|imputation"):
            manager.update(
                "SPY",
                provider="synthetic",
                fill_gaps=True,
                initial_start="2025-07-07",
                initial_end="2025-07-14",
            )
    assert storage.list_keys() == []


def test_explicit_grid_expansion_keeps_missing_market_values_null():
    with SyntheticProvider(calendar_mode="continuous") as provider:
        frame = provider.fetch_ohlcv("SPY", "2025-07-07", "2025-07-07", "minute")
    frame = frame.filter(pl.col("timestamp").dt.strftime("%H:%M").is_in(["13:30", "13:32"]))
    result = SessionCompleter("NYSE").complete_sessions(frame)
    missing = result.filter(pl.col("is_imputed"))
    assert missing.height == 1
    for name in ["open", "high", "low", "close", "volume"]:
        assert missing[name].null_count() == 1
    retained = result.filter(~pl.col("is_imputed")).select(frame.columns).cast(frame.schema)
    assert_frame_equal(retained, frame)


def test_legacy_gap_imputation_cannot_create_market_values():
    with SyntheticProvider(seed=42) as provider:
        frame = provider.fetch_ohlcv("SPY", "2025-07-07", "2025-07-14", "daily")
    observed = frame.filter(pl.col("timestamp").dt.date() != date(2025, 7, 9))
    detector = GapDetector()
    with pytest.raises(ValueError, match="[Ss]ynthetic|imputation"):
        detector.fill_gaps(observed, detector.detect_gaps(observed))


def test_configuration_never_enables_synthetic_filling_by_default():
    config = DatasetConfig(name="market", symbols=["SPY"], provider="synthetic")
    assert config.fill_gaps is not True


def test_configuration_rejects_explicit_synthetic_imputation():
    with pytest.raises(ValueError, match="imputation"):
        DatasetConfig(name="market", symbols=["SPY"], provider="synthetic", fill_gaps=True)


def test_real_cli_rejects_imputation_before_creating_a_dataset(tmp_path):
    result = CliRunner().invoke(
        cli,
        [
            "update",
            "--symbol",
            "SPY",
            "--provider",
            "synthetic",
            "--fill-gaps",
            "--initial-start",
            "2025-07-07",
            "--initial-end",
            "2025-07-14",
            "--storage-path",
            str(tmp_path),
        ],
    )
    assert result.exit_code != 0
    assert "imputation" in result.output
    assert HiveStorage(StorageConfig(base_path=tmp_path)).list_keys() == []


def test_real_cli_default_update_preserves_a_historical_missing_observation(tmp_path, monkeypatch):
    class Clock(datetime):
        @classmethod
        def now(cls, tz=UTC):
            return cls(2025, 7, 15, 23, tzinfo=tz)

    monkeypatch.setattr("ml4t.data.managers.storage_manager.datetime", Clock)
    with SyntheticProvider(seed=42) as provider:
        complete = provider.fetch_ohlcv("SPY", "2025-07-07", "2025-07-14", "daily")
    observed = complete.filter(pl.col("timestamp").dt.date() != date(2025, 7, 9))
    storage = HiveStorage(StorageConfig(base_path=tmp_path))
    with DataManager(storage=storage, enable_validation=False) as manager:
        key = manager.import_data(observed, "SPY", "synthetic")
    result = CliRunner().invoke(
        cli,
        [
            "update",
            "--symbol",
            "SPY",
            "--provider",
            "synthetic",
            "--lookback-days",
            "0",
            "--storage-path",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 0, result.output
    actual = storage.read(key).collect()
    assert date(2025, 7, 9) not in actual["timestamp"].dt.date().to_list()
    assert storage.get_metadata(key)["custom"]["attributes"]["gaps_filled"] is False
