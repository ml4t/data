"""A recorded provider must control acquisition, not only rewritten metadata."""

import pytest
from polars.testing import assert_frame_equal

from ml4t.data import DataManager
from ml4t.data.providers.synthetic import SyntheticProvider
from ml4t.data.storage.config import StorageConfig
from ml4t.data.storage.flat import FlatStorage
from ml4t.data.storage.hive import HiveStorage


@pytest.fixture(params=[FlatStorage, HiveStorage])
def stored_provider(request, tmp_path):
    storage = request.param(StorageConfig(base_path=tmp_path))
    with SyntheticProvider() as provider:
        frame = provider.fetch_ohlcv("SPY", "2025-07-07", "2025-07-07", "daily")
    storage.write(frame, "equities/daily/SPY", {"provider": "synthetic"})
    return storage, frame


def test_real_incremental_fetch_reuses_the_recorded_adapter_without_a_route(stored_provider):
    storage, frame = stored_provider
    with DataManager(storage=storage) as manager:
        key = manager.update("SPY", fill_gaps=False)
        assert key == "equities/daily/SPY"
        assert manager.get_metadata_for_key(key)["provider"] == "synthetic"
    assert storage.read(key).collect().height > frame.height


def test_explicit_provider_still_overrides_the_recorded_adapter(stored_provider):
    storage, frame = stored_provider
    storage.write(frame, "equities/daily/SPY", {"provider": "no-such-recorded-provider"})
    with DataManager(storage=storage) as manager:
        key = manager.update("SPY", fill_gaps=False, provider="synthetic")
        assert manager.get_metadata_for_key(key)["provider"] == "synthetic"


def test_invalid_explicit_adapter_does_not_fall_back_to_the_recorded_one(stored_provider):
    storage, frame = stored_provider
    before = storage.get_metadata("equities/daily/SPY")
    with (
        DataManager(storage=storage) as manager,
        pytest.raises(ValueError, match="not.*available|not found|Unknown|unknown"),
    ):
        manager.update("SPY", fill_gaps=False, provider="not-a-provider")
    assert storage.get_metadata("equities/daily/SPY") == before
    assert_frame_equal(storage.read("equities/daily/SPY").collect(), frame)
