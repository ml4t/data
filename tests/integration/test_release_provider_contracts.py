"""Bounded live contracts for public providers used by the release workflow."""

from datetime import UTC, datetime, timedelta

import httpx
import polars as pl
import pytest

from ml4t.data.providers import (
    AQRFactorProvider,
    BinanceProvider,
    EODHDProvider,
    FamaFrenchProvider,
    FXMacroDataProvider,
    ITCHSampleProvider,
    OKXProvider,
    TwelveDataProvider,
)

pytestmark = pytest.mark.integration


def _recent_dates() -> tuple[str, str]:
    end = datetime.now(UTC).date()
    return (end - timedelta(days=3)).isoformat(), end.isoformat()


def _assert_ohlcv(frame: pl.DataFrame, symbol: str, start: str, end: str) -> None:
    assert frame.height > 0
    assert frame.columns == ["timestamp", "symbol", "open", "high", "low", "close", "volume"]
    assert frame.get_column("symbol").unique().to_list() == [symbol]
    assert frame.get_column("timestamp").is_sorted()
    assert frame.schema["timestamp"] == pl.Datetime("us", "UTC")
    assert all(
        frame.schema[column] == pl.Float64 for column in ("open", "high", "low", "close", "volume")
    )
    start_bound = datetime.fromisoformat(start).replace(tzinfo=UTC) - timedelta(hours=12)
    end_bound = datetime.fromisoformat(end).replace(tzinfo=UTC) + timedelta(days=1)
    assert frame.get_column("timestamp").min() >= start_bound
    assert frame.get_column("timestamp").max() < end_bound
    assert frame.height <= (datetime.fromisoformat(end) - datetime.fromisoformat(start)).days + 2
    assert (
        frame.select("timestamp", "open", "high", "low", "close", "volume")
        .null_count()
        .sum_horizontal()
        .item()
        == 0
    )


def test_fama_french_current_ff3_contract(tmp_path) -> None:
    with FamaFrenchProvider(cache_path=tmp_path) as provider:
        frame = provider.fetch("ff3", frequency="monthly")

    assert frame.height > 1_000
    assert frame.columns == ["timestamp", "Mkt-RF", "SMB", "HML", "RF"]
    assert frame.get_column("timestamp").is_sorted()


def test_aqr_current_qmj_contract(tmp_path) -> None:
    data_path = AQRFactorProvider.download(output_path=tmp_path, datasets=["qmj_factors"])
    assert (data_path / "qmj_factors.parquet").exists()
    with AQRFactorProvider(data_path) as provider:
        frame = provider.fetch("qmj_factors", region="USA")

    assert frame.height > 500
    assert frame.columns == ["timestamp", "USA"]
    assert frame.get_column("timestamp").is_sorted()


def test_binance_market_data_contract() -> None:
    start, end = _recent_dates()
    with BinanceProvider() as provider:
        frame = provider.fetch_ohlcv("BTCUSDT", start, end, "daily")

    _assert_ohlcv(frame, "BTCUSDT", start, end)


def test_okx_market_data_contract() -> None:
    start, end = _recent_dates()
    with OKXProvider() as provider:
        frame = provider.fetch_ohlcv("BTC-USDT-SWAP", start, end, "daily")

    _assert_ohlcv(frame, "BTC-USDT-SWAP", start, end)


def test_fxmacrodata_public_catalogue_contract() -> None:
    with FXMacroDataProvider(api_key=None) as provider:
        health = provider.health()
        frame = provider.fetch_catalogue("usd")

    assert health.get("status") == "ok"
    assert frame.height > 0
    assert "indicator" in frame.columns


def test_eodhd_demo_contract() -> None:
    with EODHDProvider(api_key="demo") as provider:
        frame = provider.fetch_ohlcv("AAPL", "2025-01-02", "2025-01-10", "daily")

    _assert_ohlcv(frame, "AAPL", "2025-01-02", "2025-01-10")


def test_twelve_data_demo_contract() -> None:
    with TwelveDataProvider(api_key="demo") as provider:
        frame = provider.fetch_ohlcv("AAPL", "2025-01-02", "2025-01-10", "daily")

    _assert_ohlcv(frame, "AAPL", "2025-01-02", "2025-01-10")


def test_nasdaq_itch_sample_catalogue_contract(tmp_path) -> None:
    provider = ITCHSampleProvider(download_path=tmp_path / "raw", parsed_path=tmp_path / "parsed")

    with httpx.Client(timeout=30.0, follow_redirects=True) as client:
        for sample in provider.list_available_files():
            response = client.head(sample["url"])
            response.raise_for_status()
            content_length = response.headers.get("content-length")
            assert content_length is not None, sample["name"]
            assert int(content_length) == provider.KNOWN_FILES[sample["name"]], sample["name"]
