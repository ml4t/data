"""Exercise incremental updates with provider vintages on disk."""

from unittest.mock import patch

import pandas as pd
import pytest
from click.testing import CliRunner

from ml4t.data.cli_interface import cli
from ml4t.data.storage import create_storage


@pytest.mark.parametrize(
    "factor, incomplete", [(0.997052, False), (1.0000016, False), (0.997052, True)]
)
def test_adjusted_update_preserves_returns_and_avoids_refetch_for_noise(
    tmp_path, factor, incomplete
):
    dates = pd.bdate_range("2025-06-02", "2025-06-24", name="Date")
    dates = dates[dates != pd.Timestamp("2025-06-19")]
    prices = pd.Series(range(100, 100 + len(dates)), index=dates, dtype=float)
    old = pd.DataFrame(
        {"Open": prices, "High": prices + 1, "Low": prices - 1, "Close": prices, "Volume": 1000}
    )
    fresh = old.copy()
    price_columns = ["Open", "High", "Low", "Close"]
    fresh.loc[:"2025-06-18", price_columns] *= factor
    calls = []

    def download(symbol, *, start, end, **kwargs):
        calls.append(start)
        source = old if len(calls) == 1 else fresh
        if incomplete and len(calls) == 3:
            source = source.iloc[1:]
        return source[(source.index >= start) & (source.index < end)]

    runner = CliRunner()
    args = [
        "update",
        "--symbol",
        "SPY",
        "--provider",
        "yahoo",
        "--frequency",
        "daily",
        "--asset-class",
        "equities",
        "--storage-path",
        str(tmp_path),
    ]
    with patch("yfinance.download", side_effect=download):
        load = runner.invoke(
            cli, [*args, "--initial-start", "2025-06-02", "--initial-end", "2025-06-18"]
        )
        assert load.exit_code == 0, load.output
        update = runner.invoke(cli, args)
    bars = create_storage(tmp_path).read("equities/daily/SPY").collect().sort("timestamp")
    if incomplete:
        assert update.exit_code != 0
        assert "omitted previously observed timestamps" in update.output
        assert bars["close"].to_list() == old.loc[:"2025-06-18", "Close"].to_list()
        return
    assert update.exit_code == 0, update.output
    close = pd.Series(bars["close"].to_list(), index=dates)
    assert close.to_list() == pytest.approx(fresh["Close"].to_list(), rel=1e-5)
    assert close.pct_change().dropna().to_list() == pytest.approx(
        fresh["Close"].pct_change().dropna().to_list(), abs=1e-5
    )
    assert calls == ["2025-06-02", "2025-06-11"] + (["2025-06-02"] if factor < 1 else [])
