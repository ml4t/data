"""Updates must not splice a provider's restated history onto stored rows (ml4t/data#163).

A provider serving back-adjusted prices restates every earlier price after a dividend or split.
The update used to replace only the refetched overlap rows, so the store kept the rows before
the overlap on the old basis and gained a spurious return where the two bases met.

The fake provider below serves a deterministic daily series back-adjusted for the dividends it
currently knows about. Storage is seeded through the storage API with the series as it was
served before the ex-date, and the update runs through ``DataManager.update``, the path the
``ml4t-data update`` command uses.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from typing import Any, ClassVar

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from ml4t.data import DataManager
from ml4t.data.provider_updater import ProviderUpdater
from ml4t.data.providers.base import BaseProvider
from ml4t.data.storage.config import StorageConfig
from ml4t.data.storage.hive import HiveStorage
from ml4t.data.update_manager import IncrementalUpdater, UpdateStrategy

KEY = "equities/daily/SPY"
FIRST_SESSION = date(2025, 1, 2)
LAST_STORED = date(2025, 6, 18)
EX_DATE = date(2025, 6, 20)
# SPY's 2025-06-20 dividend: 1 - 1.7610 / close(2025-06-18), as measured in the issue.
FACTOR = 0.997052
# Repeated Yahoo retrievals of the same adjusted close differ by up to about 1.6e-6 relative.
RETRIEVAL_NOISE = 2e-6


def _sessions(start: date, end: date) -> list[date]:
    days = pl.date_range(start, end, "1d", eager=True)
    return [day for day in days.to_list() if day.weekday() < 5]


def adjusted_series(
    start: date,
    end: date,
    dividends: list[tuple[date, float]],
    *,
    noise: float = 0.0,
    first_available: date = FIRST_SESSION,
) -> pl.DataFrame:
    """Daily bars back-adjusted for ``dividends``, as an adjusted-price provider serves them."""
    rows = []
    for day in _sessions(max(start, first_available), end):
        index = (day - FIRST_SESSION).days
        close = 500.0 * 1.0004**index
        for ex_date, factor in dividends:
            if day < ex_date:
                close *= factor
        close *= 1.0 + noise
        rows.append(
            {
                "timestamp": datetime(day.year, day.month, day.day, tzinfo=UTC),
                "symbol": "SPY",
                "open": close,
                "high": close * 1.01,
                "low": close * 0.99,
                "close": close,
                "volume": 1_000_000.0,
            }
        )
    frame = pl.DataFrame(rows)
    return frame.with_columns(pl.col("timestamp").cast(pl.Datetime("us", "UTC")))


class AdjustedProvider(BaseProvider):
    """Serves ``adjusted_series`` for the dividends declared on the class at call time."""

    dividends: ClassVar[list[tuple[date, float]]] = []
    noise: ClassVar[float] = 0.0
    first_available: ClassVar[date] = FIRST_SESSION
    calls: ClassVar[list[tuple[str, str]]] = []

    @property
    def name(self) -> str:
        return "adjusted_stub"

    def _fetch_and_transform_data(
        self, symbol: str, start: str, end: str, frequency: str
    ) -> pl.DataFrame:
        type(self).calls.append((start, end))
        return adjusted_series(
            date.fromisoformat(start),
            date.fromisoformat(end),
            type(self).dividends,
            noise=type(self).noise,
            first_available=type(self).first_available,
        )


@pytest.fixture
def provider_class():
    class Provider(AdjustedProvider):
        dividends: ClassVar[list[tuple[date, float]]] = []
        noise: ClassVar[float] = 0.0
        first_available: ClassVar[date] = FIRST_SESSION
        calls: ClassVar[list[tuple[str, str]]] = []

    return Provider


@pytest.fixture
def storage(tmp_path):
    """A store holding SPY as the provider served it before the ex-date."""
    storage = HiveStorage(StorageConfig(base_path=tmp_path))
    seeded = adjusted_series(FIRST_SESSION, LAST_STORED, [])
    storage.write(seeded, KEY, {"provider": "adjusted_stub", "symbol": "SPY"})
    return storage


def _update(storage: HiveStorage, provider_class: type[AdjustedProvider]) -> None:
    with DataManager(storage=storage) as manager:
        manager._provider_manager.register_provider("adjusted_stub", provider_class)
        manager.update("SPY", asset_class="equities", frequency="daily")


def _today() -> date:
    return datetime.now(UTC).date()


def _returns(frame: pl.DataFrame) -> pl.Series:
    return frame.sort("timestamp")["close"].pct_change().drop_nulls()


class TestUpdateAfterACorporateAction:
    def test_stored_history_is_rewritten_on_the_providers_current_basis(
        self, storage, provider_class
    ):
        """Prevents: rows before the overlap staying on the pre-dividend basis.

        On main the store ends up 0.2948% too high on every row before the lookback window.
        """
        provider_class.dividends = [(EX_DATE, FACTOR)]

        _update(storage, provider_class)

        stored = storage.read(KEY).collect().sort("timestamp")
        current = adjusted_series(FIRST_SESSION, _today(), provider_class.dividends)
        assert stored["timestamp"].to_list() == current["timestamp"].to_list()
        assert_frame_equal(
            stored.select("timestamp", "open", "high", "low", "close"),
            current.select("timestamp", "open", "high", "low", "close"),
            rtol=1e-12,
        )

    def test_no_spurious_return_where_the_refetch_meets_stored_rows(self, storage, provider_class):
        """Prevents: a dividend-sized return on the first overlap row instead of the ex-date."""
        provider_class.dividends = [(EX_DATE, FACTOR)]

        _update(storage, provider_class)

        stored = storage.read(KEY).collect()
        current = adjusted_series(FIRST_SESSION, _today(), provider_class.dividends)
        gap = (_returns(stored) - _returns(current)).abs().max()
        assert gap is not None and gap < 1e-9

    def test_the_rewrite_is_a_full_refetch_from_the_first_stored_session(
        self, storage, provider_class
    ):
        """Prevents: rebasing the old rows by the overlap ratio instead of refetching them.

        A ratio is correct only for one multiplicative event in the gap; the provider's own
        series is the only reference for anything else.
        """
        provider_class.dividends = [(EX_DATE, FACTOR)]

        _update(storage, provider_class)

        assert len(provider_class.calls) == 2
        assert provider_class.calls[1][0] == FIRST_SESSION.isoformat()
        metadata = storage.get_metadata(KEY)["custom"]
        assert metadata["attributes"]["update_type"] == "full_refresh"

    def test_history_the_provider_no_longer_serves_is_left_untouched(self, storage, provider_class):
        """Prevents: writing a mixed-basis store when the full range cannot be refetched."""
        provider_class.dividends = [(EX_DATE, FACTOR)]
        provider_class.first_available = date(2025, 3, 3)
        before = storage.read(KEY).collect()

        with pytest.raises(ValueError, match="restated"):
            _update(storage, provider_class)

        assert_frame_equal(storage.read(KEY).collect(), before)


class TestUpdateWithoutACorporateAction:
    def test_retrieval_noise_keeps_the_update_incremental(self, storage, provider_class):
        """Prevents: a full refetch on every update because of float noise in the overlap.

        The overlap differs from the stored rows by Yahoo-sized retrieval noise only, so the
        update must fetch once and leave the stored rows before the lookback window as they are.
        """
        provider_class.noise = RETRIEVAL_NOISE
        before = storage.read(KEY).collect()

        _update(storage, provider_class)

        assert len(provider_class.calls) == 1
        stored = storage.read(KEY).collect().sort("timestamp")
        assert stored["timestamp"].max().date() > LAST_STORED
        untouched = stored.filter(pl.col("timestamp") < before["timestamp"].max() - timedelta(7))
        assert_frame_equal(
            untouched, before.filter(pl.col("timestamp").is_in(untouched["timestamp"]))
        )
        metadata = storage.get_metadata(KEY)["custom"]
        assert metadata["attributes"]["update_type"] == "incremental"

    def test_a_bar_still_forming_when_stored_keeps_the_update_incremental(self, tmp_path):
        """Prevents: a full refetch after every update run during the trading day.

        The last stored bar was retrieved before the session closed, so its close differs
        from the final one. Earlier overlap rows match, so nothing was restated.
        """
        storage = HiveStorage(StorageConfig(base_path=tmp_path))
        seeded = adjusted_series(FIRST_SESSION, LAST_STORED, [])
        seeded = seeded.with_columns(
            pl.when(pl.col("timestamp") == pl.col("timestamp").max())
            .then(pl.col(column) * 0.995)
            .otherwise(pl.col(column))
            .alias(column)
            for column in ("open", "high", "low", "close")
        )
        storage.write(seeded, KEY, {"provider": "adjusted_stub", "symbol": "SPY"})

        class Provider(AdjustedProvider):
            calls: ClassVar[list[tuple[str, str]]] = []

        _update(storage, Provider)

        assert len(Provider.calls) == 1
        stored = storage.read(KEY).collect().sort("timestamp")
        final = adjusted_series(FIRST_SESSION, _today(), [])
        assert_frame_equal(
            stored.select("timestamp", "close"), final.select("timestamp", "close"), rtol=1e-12
        )


class TestIncrementalUpdaterRefusesToSplice:
    """``IncrementalUpdater`` receives data, not a provider, so it cannot refetch."""

    def _refetch(self, start: date, dividends: list[tuple[date, float]]) -> pl.DataFrame:
        return adjusted_series(start, date(2025, 7, 3), dividends)

    @pytest.mark.parametrize("strategy", [UpdateStrategy.INCREMENTAL, UpdateStrategy.APPEND_ONLY])
    def test_a_restated_overlap_fails_and_leaves_the_store_unchanged(self, storage, strategy):
        """Prevents: splicing restated rows onto stored history through the library API."""
        before = storage.read(KEY).collect()
        refetched = self._refetch(date(2025, 6, 11), [(EX_DATE, FACTOR)])

        result = IncrementalUpdater().update_incremental(
            storage, None, KEY, refetched, "adjusted_stub", strategy
        )

        assert not result.success
        assert result.errors and "FULL_REFRESH" in result.errors[0]
        assert_frame_equal(storage.read(KEY).collect(), before)

    def test_a_refetch_covering_the_whole_store_replaces_it(self, storage):
        """Prevents the refusal overreaching: a refetch of the full range is not a splice."""
        refetched = self._refetch(FIRST_SESSION, [(EX_DATE, FACTOR)])

        result = IncrementalUpdater().update_incremental(
            storage, None, KEY, refetched, "adjusted_stub", UpdateStrategy.INCREMENTAL
        )

        assert result.success
        stored = storage.read(KEY).collect().sort("timestamp")
        assert_frame_equal(stored.select(refetched.columns), refetched, check_dtypes=False)


class _AdjustedUpdater(ProviderUpdater):
    """A ``ProviderUpdater`` over an adjusted-price source, as the class docstring shows."""

    def __init__(self, storage: HiveStorage, dividends: list[tuple[date, float]]) -> None:
        super().__init__("adjusted_stub", storage)
        self.dividends = dividends
        self.calls: list[tuple[datetime, datetime]] = []

    def _fetch_data(
        self, symbol: str, start_time: datetime, end_time: datetime, **kwargs: Any
    ) -> pl.DataFrame:
        self.calls.append((start_time, end_time))
        return adjusted_series(start_time.date(), end_time.date(), self.dividends)

    def _transform_data(self, data: pl.DataFrame, symbol: str, **kwargs: Any) -> pl.DataFrame:
        return data

    def _get_default_start_time(self, symbol: str) -> datetime:
        return datetime(2025, 1, 2, tzinfo=UTC)


class TestProviderUpdaterAfterACorporateAction:
    def test_combined_file_is_rewritten_on_the_current_basis(self, tmp_path):
        """Prevents: ``ProviderUpdater`` splicing a restated last row onto stored history."""
        storage = HiveStorage(StorageConfig(base_path=tmp_path))
        seed = _AdjustedUpdater(storage, [])
        seed.update_symbol(
            "SPY",
            start_time=datetime(2025, 1, 2, tzinfo=UTC),
            end_time=datetime(2025, 6, 18, tzinfo=UTC),
        )
        updater = _AdjustedUpdater(storage, [(EX_DATE, FACTOR)])

        stats = updater.update_symbol("SPY", end_time=datetime(2025, 7, 3, tzinfo=UTC))

        assert stats["success"], stats["errors"]
        stored = storage.read_data("SPY", "adjusted_stub").sort("timestamp")
        current = adjusted_series(FIRST_SESSION, date(2025, 7, 3), [(EX_DATE, FACTOR)])
        assert_frame_equal(stored.select(current.columns), current, check_dtypes=False)
