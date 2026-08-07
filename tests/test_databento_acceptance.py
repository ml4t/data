"""Opt-in live contract test for the Databento provider.

The test process does not load ``.env`` files. Export both required variables explicitly.
"""

from __future__ import annotations

import os

import polars as pl
import pytest

pytest.importorskip("databento")

from ml4t.data.providers.databento import DataBentoProvider


@pytest.fixture
def provider() -> DataBentoProvider:
    """Create a provider only after an explicit live-test opt in."""
    if os.getenv("ML4T_RUN_DATABENTO_LIVE") != "1":
        pytest.skip("export ML4T_RUN_DATABENTO_LIVE=1 to authorize the live contract request")

    api_key = os.getenv("DATABENTO_API_KEY")
    if not api_key:
        pytest.skip("export DATABENTO_API_KEY for the Databento live contract test")
    return DataBentoProvider(api_key=api_key)


@pytest.mark.integration
def test_databento_daily_futures_contract(provider: DataBentoProvider) -> None:
    """Fetch one historical bar and verify the public output contract."""
    data = provider.fetch_ohlcv("ESH4", "2024-01-02", "2024-01-02", "daily")

    assert data.height == 1
    canonical_schema = {
        "timestamp": pl.Datetime("us", "UTC"),
        "symbol": pl.String,
        "open": pl.Float64,
        "high": pl.Float64,
        "low": pl.Float64,
        "close": pl.Float64,
        "volume": pl.Float64,
    }
    assert data.columns[:7] == list(canonical_schema)
    assert {column: data.schema[column] for column in canonical_schema} == canonical_schema
    assert data["symbol"].item() == "ESH4"
    assert data["high"].item() >= max(data["open"].item(), data["close"].item())
    assert data["low"].item() <= min(data["open"].item(), data["close"].item())
    assert data["volume"].item() >= 0
