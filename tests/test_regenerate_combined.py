"""Regression tests for combined-file regeneration."""

from datetime import datetime

import polars as pl

from ml4t.data.etfs.downloader import ETFConfig, ETFDataManager


def _bars(symbol: str, timestamps: list[datetime]) -> pl.DataFrame:
    """Minimal OHLCV frame for one symbol."""
    n = len(timestamps)
    return pl.DataFrame(
        {
            "timestamp": timestamps,
            "symbol": [symbol] * n,
            "open": [1.0] * n,
            "high": [1.0] * n,
            "low": [1.0] * n,
            "close": [1.0] * n,
            "volume": [100.0] * n,
        },
        schema_overrides={"timestamp": pl.Datetime},
    )


class TestRegenerateCombined:
    """Tests for ETFDataManager._regenerate_combined."""

    def test_rebuilds_from_partitions_when_aggregate_is_stale(self, tmp_path):
        """A stale combined file must not survive regeneration.

        ``update()`` writes fresh bars to the per-ticker partitions and then
        calls ``_regenerate_combined()``. If regeneration reads the existing
        aggregate instead of the partitions, the stale file is rewritten
        unchanged and every downstream reader silently sees old data.
        """
        config = ETFConfig(
            storage_path=tmp_path,
            tickers={"equity": {"symbols": ["SPY"]}},
        )
        manager = ETFDataManager(config)

        # Partition holds bars through the newer date.
        fresh = _bars("SPY", [datetime(2025, 12, 31), datetime(2026, 8, 20)])
        manager._save_ticker(fresh, "SPY")

        # Aggregate is stale: it stops at the older date.
        stale = _bars("SPY", [datetime(2025, 12, 31)])
        manager._save_combined(stale)

        manager._regenerate_combined()

        combined = pl.read_parquet(tmp_path / "etf_universe.parquet")
        assert combined["timestamp"].max() == datetime(2026, 8, 20), (
            "combined file was not rebuilt from the ticker partitions"
        )
        assert combined.height == 2

    def test_picks_up_a_symbol_absent_from_the_aggregate(self, tmp_path):
        """A partition missing from the aggregate must appear after regeneration."""
        config = ETFConfig(
            storage_path=tmp_path,
            tickers={"equity": {"symbols": ["SPY", "QQQ"]}},
        )
        manager = ETFDataManager(config)

        ts = [datetime(2026, 8, 20)]
        manager._save_ticker(_bars("SPY", ts), "SPY")
        manager._save_ticker(_bars("QQQ", ts), "QQQ")

        # Aggregate predates QQQ being added to the universe.
        manager._save_combined(_bars("SPY", ts))

        manager._regenerate_combined()

        combined = pl.read_parquet(tmp_path / "etf_universe.parquet")
        assert set(combined["symbol"].unique()) == {"SPY", "QQQ"}
