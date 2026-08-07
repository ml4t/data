"""Opt-in live contract test for the Databento provider."""

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
        pytest.skip("set ML4T_RUN_DATABENTO_LIVE=1 to authorize the live contract request")

    api_key = os.getenv("DATABENTO_API_KEY")
    if not api_key:
        pytest.skip("DATABENTO_API_KEY is required for the Databento live contract test")
    return DataBentoProvider(api_key=api_key)


@pytest.mark.integration
def test_databento_daily_futures_contract(provider: DataBentoProvider) -> None:
    """Fetch one historical bar and verify the public output contract."""
    data = provider.fetch_ohlcv("ESH4", "2024-01-02", "2024-01-02", "daily")

    assert data.height == 1
    assert data.columns == ["timestamp", "open", "high", "low", "close", "volume", "symbol"]
    assert data.schema == {
        "timestamp": pl.Datetime("ns", "UTC"),
        "open": pl.Float64,
        "high": pl.Float64,
        "low": pl.Float64,
        "close": pl.Float64,
        "volume": pl.Float64,
        "symbol": pl.String,
    }
    assert data["symbol"].item() == "ESH4"
    assert data["high"].item() >= max(data["open"].item(), data["close"].item())
    assert data["low"].item() <= min(data["open"].item(), data["close"].item())
    assert data["volume"].item() >= 0
