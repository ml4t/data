"""Offline acquisition-import migration using retained public provider response captures."""

import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from unittest.mock import patch

import httpx
import pandas as pd

from ml4t.data.compat.empyrical import (
    default_returns_func,
    get_fama_french,
    get_returns_cached,
    get_symbol_returns_from_yahoo,
    get_treasury_yield,
)
from ml4t.data.providers import FamaFrenchProvider, FREDProvider, YahooFinanceProvider


def migrate_acquisition(directory: Path, captures: Path) -> dict[str, Any]:
    """Run actual native providers and import-compatible helpers; only transport is replayed."""
    directory.mkdir(parents=True, exist_ok=True)
    capture = json.loads((captures / "yahoo-dst-provider.json").read_text())
    request = capture["request"]
    start, end = (pd.Timestamp(request[name]) for name in ("start", "end"))
    record = capture["records"]["True"]
    adjusted = pd.DataFrame(
        record["data"],
        columns=pd.MultiIndex.from_tuples(record["columns"]),
        index=pd.DatetimeIndex(record["index"])
        .astype(record["index_dtype"])
        .rename(record["index_name"]),
    )

    def http_get(_client: httpx.Client, url: str, **kwargs: Any) -> httpx.Response:
        if str(url).startswith(FamaFrenchProvider.BASE_URL + "/"):
            payload = (captures / str(url).rsplit("/", 1)[-1]).read_bytes()
        elif url == "https://fred.stlouisfed.org/graph/fredgraph.csv":
            assert kwargs["params"] == {"id": "DGS3MO", "cosd": "2024-01-02", "coed": "2024-01-04"}
            payload = (captures / "fred-dgs3mo.csv").read_bytes()
        else:
            raise AssertionError(f"Unexpected offline source request: {url}")
        return httpx.Response(200, content=payload, request=httpx.Request("GET", url))

    def download(symbol: str, **kwargs: Any) -> pd.DataFrame:
        assert symbol == "SPY" and kwargs["start"] == start and kwargs["end"] == end
        assert kwargs["auto_adjust"] is True and kwargs["actions"] is False
        return adjusted.copy(deep=True)

    from ml4t.data.providers import yahoo

    with (
        patch.object(httpx.Client, "get", http_get),
        patch.object(yahoo.yf, "download", download),
        patch.dict(os.environ, {"EMPYRICAL_CACHE_DIR": str(directory), "FRED_API_KEY": ""}),
    ):
        with FamaFrenchProvider(cache_path=directory / "native-factors") as provider:
            native_factors = provider.fetch("ff3_mom", "daily", "2024-01-02", "2024-01-05")
            cached = provider.fetch("ff3_mom", "daily", "2024-01-02", "2024-01-05", offline=True)
            assert native_factors.equals(cached)
        factors = get_fama_french("2024-01-02", "2024-01-05")
        assert factors.columns.to_list() == ["Mkt-RF", "SMB", "HML", "RF", "Mom"]
        assert abs(factors.iloc[0]["Mkt-RF"] + 0.007) < 4e-18
        assert factors.iloc[0]["RF"] == 0.0002
        with FREDProvider(anonymous=True) as provider:
            native_yields = provider.fetch_series("DGS3MO", "2024-01-02", "2024-01-04")
        yields = get_treasury_yield("2024-01-02", "2024-01-04")
        assert yields["DGS3MO"].to_list() == [5.46, 5.48, 5.48]
        assert native_yields["available_at"].null_count() == 3
        with YahooFinanceProvider() as provider:
            prices = provider.fetch_adjusted_close("SPY", start=start, end=end)
        get_returns_cached(
            directory / "spy.csv",
            get_symbol_returns_from_yahoo,
            end,
            symbol="SPY",
            start=start,
            end=end,
        )
        benchmark = default_returns_func("SPY", start=start, end=end)
        assert benchmark.name == "SPY" and len(benchmark) == 1
        assert benchmark.index[0] == pd.Timestamp("2025-03-10", tz="UTC")
        expected = 549.7793579101562 / 564.8236694335938 - 1
        assert abs(benchmark.iloc[0] - expected) < 1e-15
    return {
        "native_factors": native_factors,
        "factors": factors,
        "native_yields": native_yields,
        "yields": yields,
        "adjusted_prices": prices,
        "benchmark": benchmark,
    }


if __name__ == "__main__":
    captures = Path(__file__).resolve().parents[1] / "tests/fixtures/acquisition"
    with TemporaryDirectory(prefix="ml4t-acquisition-migration-") as temporary:
        result = migrate_acquisition(Path(temporary), captures)
        print(result["factors"])
        print(result["yields"])
        print(result["benchmark"])
