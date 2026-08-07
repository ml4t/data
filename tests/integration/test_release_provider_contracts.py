"""Bounded live contracts for public providers used by the release workflow."""

from datetime import UTC, datetime, timedelta

import polars as pl

from ml4t.data.providers import AQRFactorProvider, BinanceProvider, FamaFrenchProvider, OKXProvider


def _recent_dates() -> tuple[str, str]:
    end = datetime.now(UTC).date()
    return (end - timedelta(days=3)).isoformat(), end.isoformat()


def _assert_ohlcv(frame: pl.DataFrame, symbol: str) -> None:
    assert frame.height > 0
    assert frame.columns == ["timestamp", "symbol", "open", "high", "low", "close", "volume"]
    assert frame.get_column("symbol").unique().to_list() == [symbol]
    assert frame.get_column("timestamp").is_sorted()
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
    with AQRFactorProvider(data_path) as provider:
        frame = provider.fetch("qmj_factors", region="USA")

    assert frame.height > 500
    assert frame.columns == ["timestamp", "USA"]
    assert frame.get_column("timestamp").is_sorted()


def test_binance_market_data_contract() -> None:
    start, end = _recent_dates()
    with BinanceProvider() as provider:
        frame = provider.fetch_ohlcv("BTCUSDT", start, end, "daily")

    _assert_ohlcv(frame, "BTCUSDT")


def test_okx_market_data_contract() -> None:
    start, end = _recent_dates()
    with OKXProvider() as provider:
        frame = provider.fetch_ohlcv("BTC-USDT-SWAP", start, end, "daily")

    _assert_ohlcv(frame, "BTC-USDT-SWAP")
