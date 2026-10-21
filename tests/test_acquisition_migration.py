"""Exercise captured acquisition, disk cache and scalar-series contracts without live services."""

import json
import os
from datetime import date, datetime, timedelta
from pathlib import Path

import httpx
import numpy as np
import pandas as pd
import polars as pl
import pytest
from pandas.testing import assert_frame_equal, assert_series_equal

from ml4t.data.compat import empyrical as compat
from ml4t.data.core.exceptions import DataNotAvailableError, DataValidationError, NetworkError
from ml4t.data.providers.fama_french import FamaFrenchProvider
from ml4t.data.providers.fred import FREDProvider

CAPTURES = Path(__file__).parent / "fixtures" / "acquisition"


@pytest.fixture
def french_http(monkeypatch):
    calls = []

    def get(client, url, **kwargs):
        calls.append(str(url))
        content = (CAPTURES / str(url).rsplit("/", 1)[-1]).read_bytes()
        return httpx.Response(200, content=content, request=httpx.Request("GET", url))

    monkeypatch.setattr(httpx.Client, "get", get)
    return calls


def test_official_zip_parser_and_real_cache_preserve_decimal_rf_and_complete_dates(
    tmp_path, french_http, monkeypatch
):
    with FamaFrenchProvider(cache_path=tmp_path) as provider:
        native = provider.fetch("ff3_mom", frequency="daily", start="2024-01-02", end="2024-01-05")
        assert native["timestamp"].to_list() == [datetime(2024, 1, day) for day in (2, 3, 4, 5)]
        assert native.drop("timestamp").row(0, named=True) == pytest.approx(
            {
                "Mkt-RF": -0.007,
                "SMB": -0.0055,
                "HML": 0.0079,
                "RF": 0.0002,
                "MOM": -0.0211,
            },
            rel=0,
            abs=4e-18,
        )
        assert len(french_http) == 2
        assert sorted(path.name for path in tmp_path.glob("*.parquet")) == [
            "ff3_daily.parquet",
            "mom_daily.parquet",
        ]

        def no_network(*args, **kwargs):
            pytest.fail("An offline cache hit must never try a provider request")

        monkeypatch.setattr(httpx.Client, "get", no_network)
        hit = provider.fetch(
            "ff3_mom", frequency="daily", start="2024-01-02", end="2024-01-05", offline=True
        )
        assert hit.equals(native)


def test_declared_cache_freshness_rejects_offline_stale_unless_explicitly_accepted(
    tmp_path, french_http
):
    with FamaFrenchProvider(cache_path=tmp_path) as provider:
        expected = provider.fetch("ff3", "daily", "2024-01-02", "2024-01-05")
        cache = tmp_path / "ff3_daily.parquet"
        os.utime(cache, (946684800, 946684800))
        with pytest.raises(DataNotAvailableError, match="stale"):
            provider.fetch("ff3", "daily", offline=True, max_age=timedelta(days=1))
        assert len(french_http) == 1
        accepted = provider.fetch(
            "ff3",
            "daily",
            "2024-01-02",
            "2024-01-05",
            offline=True,
            max_age=timedelta(days=1),
            allow_stale=True,
        )
        assert accepted.equals(expected)
        assert provider.last_fetch_metadata["stale_accepted"] is True
        assert provider.last_fetch_metadata["cache_hit"] is True
        provider.fetch("ff3", "daily", max_age=timedelta(days=1))
        assert len(french_http) == 2
        assert provider.last_fetch_metadata["stale_accepted"] is False


def test_explicit_refresh_uses_source_and_failed_download_preserves_cache(
    tmp_path, french_http, monkeypatch
):
    with FamaFrenchProvider(cache_path=tmp_path) as provider:
        provider.fetch("ff3", "daily")
        provider.fetch("ff3", "daily", refresh=True)
        assert len(french_http) == 2
        cache = tmp_path / "ff3_daily.parquet"
        before = cache.read_bytes()

        def fail(*args, **kwargs):
            raise httpx.ConnectError("captured refresh failure")

        monkeypatch.setattr(httpx.Client, "get", fail)
        with pytest.raises(NetworkError):
            provider.fetch("ff3", "daily", refresh=True)
        assert cache.read_bytes() == before
        assert pl.read_parquet(cache).height > 0


def test_interrupted_parquet_refresh_does_not_replace_last_valid_cache(
    tmp_path, french_http, monkeypatch
):
    with FamaFrenchProvider(cache_path=tmp_path) as provider:
        provider.fetch("ff3", "daily")
        cache = tmp_path / "ff3_daily.parquet"
        before = cache.read_bytes()

        def interrupted(frame, path, *args, **kwargs):
            Path(path).write_bytes(b"incomplete parquet from interrupted writer")
            raise OSError("captured full-disk interruption")

        monkeypatch.setattr(pl.DataFrame, "write_parquet", interrupted)
        with pytest.raises(OSError, match="full-disk"):
            provider.fetch("ff3", "daily", refresh=True)
        assert cache.read_bytes() == before
        assert not list(tmp_path.glob("*.tmp"))


@pytest.mark.parametrize(
    "options", [{"refresh": True, "offline": True}, {"max_age": timedelta(seconds=-1)}]
)
def test_invalid_cache_policy_fails_before_network_or_cache_mutation(
    tmp_path, french_http, options
):
    with FamaFrenchProvider(cache_path=tmp_path) as provider:
        with pytest.raises(ValueError):
            provider.fetch("ff3", "daily", **options)
    assert french_http == []
    assert list(tmp_path.iterdir()) == []


def test_offline_missing_cache_never_downloads(tmp_path, french_http):
    with FamaFrenchProvider(cache_path=tmp_path) as provider:
        with pytest.raises(DataNotAvailableError, match="cache"):
            provider.fetch("ff3", "daily", offline=True)
    assert french_http == []


@pytest.fixture
def fred_http(monkeypatch):
    calls = []
    observations = [
        {
            "date": "2024-01-02",
            "value": "5.4",
            "realtime_start": "2024-02-01",
            "realtime_end": "2024-02-01",
        },
        {
            "date": "2024-01-03",
            "value": ".",
            "realtime_start": "2024-02-01",
            "realtime_end": "2024-02-01",
        },
        {
            "date": "2024-01-04",
            "value": "5.3",
            "realtime_start": "2024-02-01",
            "realtime_end": "2024-02-01",
        },
    ]

    def get(client, url, *, params, **kwargs):
        calls.append((url, dict(params)))
        body = (
            {"observations": observations}
            if url.endswith("/observations")
            else {
                "seriess": [
                    {
                        "id": params["series_id"],
                        "units": "Percent",
                        "frequency": "Daily",
                        "seasonal_adjustment": "Not Seasonally Adjusted",
                    }
                ]
            }
        )
        return httpx.Response(200, json=body, request=httpx.Request("GET", url))

    monkeypatch.setattr(httpx.Client, "get", get)
    monkeypatch.setenv("FRED_API_KEY", "fixture-key-not-a-credential")
    return calls, observations


def test_native_fred_scalar_preserves_reported_missing_dates_units_and_vintage_without_ohlcv(
    fred_http,
):
    calls, _ = fred_http
    with FREDProvider() as provider:
        result = provider.fetch_series(
            "DGS3MO", "2024-01-02", "2024-01-04", vintage_date="2024-02-01"
        )
    assert result["timestamp"].to_list() == [date(2024, 1, day) for day in (2, 3, 4)]
    assert result["value"].to_list() == [5.4, None, 5.3]
    assert result["units"].to_list() == ["Percent"] * 3
    assert result["series_id"].to_list() == ["DGS3MO"] * 3
    assert result["vintage_date"].to_list() == [date(2024, 2, 1)] * 3
    assert result["available_at"].null_count() == 3
    assert result["realtime_start"].to_list() == [date(2024, 2, 1)] * 3
    assert not {"open", "high", "low", "close", "volume"}.intersection(result.columns)
    observation_request = next(params for url, params in calls if url.endswith("/observations"))
    assert "frequency" not in observation_request
    assert observation_request["realtime_start"] == "2024-02-01"
    assert observation_request["realtime_end"] == "2024-02-01"


@pytest.mark.parametrize("value", ["infinity", "nan", "bad-value"])
def test_fred_scalar_rejects_malformed_source_values_instead_of_silently_filling(fred_http, value):
    _, observations = fred_http
    observations[0]["value"] = value
    with FREDProvider() as provider, pytest.raises(DataValidationError):
        provider.fetch_series("DGS3MO", "2024-01-02", "2024-01-04")


def test_legacy_factor_import_consumes_actual_zip_parser_without_a_stale_inner_cache(
    french_http, tmp_path, monkeypatch
):
    monkeypatch.setenv("ML4T_DATA_PATH", str(tmp_path))
    actual = compat.get_fama_french("2024-01-02", "2024-01-05")
    original = json.loads((CAPTURES / "famafrench-provider-response.json").read_text())["records"]
    parts = []
    for name in ("F-F_Research_Data_Factors_daily", "F-F_Momentum_Factor_daily"):
        record = original[name]
        parts.append(
            pd.DataFrame(
                record["data"],
                columns=record["columns"],
                index=pd.to_datetime(record["index"], utc=True),
            ).div(100)
        )
    expected = pd.concat(parts, axis=1).dropna().rename(columns=str.strip)
    assert_frame_equal(actual, expected, check_freq=False, check_dtype=False)
    assert actual.columns.to_list() == ["Mkt-RF", "SMB", "HML", "RF", "Mom"]
    assert actual.iloc[0]["RF"] == 0.0002
    assert actual.iloc[0]["Mkt-RF"] == pytest.approx(-0.007, rel=0, abs=4e-18)
    assert len(french_http) == 2
    compat.get_fama_french("2024-01-02", "2024-01-05")
    assert len(french_http) == 4
    assert not list(tmp_path.rglob("*.parquet"))


def test_unknown_factor_dataset_fails_without_any_source_request(french_http):
    with pytest.raises(ValueError, match="Unsupported"):
        compat.get_fama_french(datasets=("not-a-registered-dataset",))
    assert french_http == []


def test_monthly_factor_names_keep_month_labels_decimal_rf_and_native_column_mapping(french_http):
    actual = compat.get_fama_french(
        "2024-01-01",
        "2024-01-31",
        ("F-F_Research_Data_Factors", "F-F_Momentum_Factor"),
    )
    assert actual.index.to_list() == [pd.Timestamp("2024-01-01", tz="UTC")]
    assert actual.columns.to_list() == ["Mkt-RF", "SMB", "HML", "RF", "Mom"]
    assert actual.iloc[0].to_dict() == pytest.approx(
        {"Mkt-RF": 0.0073, "SMB": -0.0504, "HML": -0.0237, "RF": 0.0047, "Mom": 0.0546},
        rel=0,
        abs=1e-17,
    )
    assert actual.attrs["frequency"] == "monthly"
    assert actual.attrs["timestamp_semantics"] == "session_label"


def test_a_valid_factor_file_in_the_wrong_dataset_cache_is_not_silently_accepted(
    tmp_path, french_http
):
    with FamaFrenchProvider(cache_path=tmp_path) as provider:
        momentum = provider.fetch("mom", "daily")
        momentum.write_parquet(tmp_path / "ff3_daily.parquet")
        before = (tmp_path / "ff3_daily.parquet").read_bytes()
        with pytest.raises(ValueError, match="columns"):
            provider.fetch("ff3", "daily", offline=True)
        assert (tmp_path / "ff3_daily.parquet").read_bytes() == before
    assert len(french_http) == 1


def test_csv_miss_hit_and_stale_refresh_use_the_actual_file_mtime_and_preserve_callback_kwargs(
    tmp_path,
):
    path = tmp_path / "nested" / "caller.csv"
    frame = pd.DataFrame(
        {"SPY": [0.02, -1 / 102]}, index=pd.to_datetime(["2024-01-03", "2024-01-04"], utc=True)
    )
    calls = []

    def update(**kwargs):
        calls.append(kwargs)
        return frame

    first = compat.get_returns_cached(
        path, update, pd.Timestamp("2025-01-01", tz="UTC"), symbol="SPY", arbitrary=17
    )
    assert calls == [{"symbol": "SPY", "arbitrary": 17}]
    assert path.is_file()
    original = frame.copy(deep=True)
    assert_frame_equal(first, original)

    def fail(**kwargs):
        pytest.fail("A fresh CSV hit must not fetch again")

    hit = compat.get_returns_cached(path, fail, pd.Timestamp("2025-01-01"))
    assert_frame_equal(hit, original)
    os.utime(path, (946684800, 946684800))
    compat.get_returns_cached(path, update, pd.Timestamp("2025-01-01", tz="UTC"), symbol="SPY")
    assert calls[-1] == {"symbol": "SPY"}
    assert len(calls) == 2
    assert_frame_equal(frame, original)


def test_a_callback_failure_leaves_the_old_csv_unchanged(tmp_path):
    path = tmp_path / "old.csv"
    frame = pd.DataFrame({"A": [0.1]}, index=pd.to_datetime(["2024-01-02"], utc=True))
    compat.get_returns_cached(path, lambda: frame, "2024-01-01")
    before = path.read_bytes()
    os.utime(path, (946684800, 946684800))

    def fail():
        raise RuntimeError("captured acquisition failure")

    with pytest.raises(RuntimeError, match="acquisition failure"):
        compat.get_returns_cached(path, fail, "2024-01-01")
    assert path.read_bytes() == before


def test_interrupted_csv_write_warns_returns_valid_fetched_values_and_keeps_old_file(
    tmp_path, monkeypatch
):
    path = tmp_path / "cache.csv"
    frame = pd.DataFrame({"A": [0.1]}, index=pd.to_datetime(["2024-01-02"], utc=True))
    compat.get_returns_cached(path, lambda: frame, "2024-01-01")
    before = path.read_bytes()
    os.utime(path, (946684800, 946684800))

    def interrupted(data, filename, *args, **kwargs):
        Path(filename).write_text("incomplete csv")
        raise OSError("captured interrupted write")

    monkeypatch.setattr(pd.DataFrame, "to_csv", interrupted)
    with pytest.warns(UserWarning, match="could not update cache"):
        result = compat.get_returns_cached(path, lambda: frame * 2, "2024-01-01")
    assert_frame_equal(result, frame * 2)
    assert path.read_bytes() == before
    assert not list(tmp_path.glob("*.tmp"))


@pytest.mark.parametrize(
    "latest", ["2025-03-10T01:00:00-04:00", "2025-03-10T05:00:00Z", "2025-03-10T05:00:00"]
)
def test_equivalent_dst_and_naive_utc_cache_deadlines_make_the_same_freshness_decision(
    tmp_path, latest
):
    path = tmp_path / "aware.csv"
    frame = pd.DataFrame(
        {"A": [0.1]},
        index=pd.DatetimeIndex(["2025-03-07", "2025-03-10"], tz="America/New_York")[:1],
    )
    compat.get_returns_cached(path, lambda: frame, "2025-01-01")
    os.utime(path, (1741582801, 1741582801))

    def fail():
        pytest.fail("Equivalent instants must not cause another refresh")

    actual = compat.get_returns_cached(path, fail, latest)
    expected = frame.copy()
    expected.index = expected.index.tz_convert("UTC")
    assert_frame_equal(actual, expected)


@pytest.mark.parametrize(
    "body",
    [
        "date,A\nnot-a-date,.1\n",
        "date,A\n2024-01-02,inf\n",
        "date,A,A\n2024-01-02,.1,.2\n",
        "date,A\n2024-01-02,.1\n2024-01-02,.2\n",
    ],
)
def test_corrupt_cache_is_not_reported_as_a_successful_hit(tmp_path, body):
    path = tmp_path / "bad.csv"
    path.write_text(body)
    with pytest.raises((ValueError, TypeError)):
        compat.get_returns_cached(
            path, lambda: pytest.fail("Do not silently overwrite corrupt cache"), "2024-01-01"
        )


@pytest.mark.parametrize("period", ["1MO", "3MO", "6MO", "1", "5", "10"])
def test_treasury_compatibility_keeps_annual_percent_units_shape_and_reported_missingness(
    fred_http, period
):
    calls, _ = fred_http
    actual = compat.get_treasury_yield("2024-01-02", "2024-01-04", period)
    assert isinstance(actual, pd.DataFrame)
    assert actual.columns.to_list() == ["DGS" + period]
    assert actual.iloc[:, 0].to_list() == [5.4, 5.4, 5.3]
    assert actual.index.to_list() == list(
        pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])
    )
    assert actual.attrs["source_missing_mask"] == (False, True, False)
    assert actual.attrs["units"] == "annual_percent_yield"
    assert all(params["series_id"] == "DGS" + period for _, params in calls)


@pytest.fixture
def captured_yahoo(monkeypatch):
    from ml4t.data.providers import yahoo

    records = json.loads((CAPTURES / "yahoo-adjustment-provider.json").read_text())["records"]
    captured = records["default"]
    frame = pd.DataFrame(
        captured["data"],
        columns=pd.MultiIndex.from_tuples(captured["columns"]),
        index=pd.to_datetime(captured["index"]),
    )
    frame.index.name = "Date"
    calls = []

    def download(symbol, **kwargs):
        calls.append((symbol, kwargs))
        return frame.copy(deep=True)

    monkeypatch.setattr(yahoo.yf, "download", download)
    return calls, frame, records


def test_adjusted_yahoo_route_matches_captured_legacy_adj_close_without_dividend_double_count(
    captured_yahoo,
):
    calls, _, records = captured_yahoo
    actual = compat.get_symbol_returns_from_yahoo("SPY", "2024-01-02", "2024-01-06")
    old = records["explicit"]
    raw = pd.DataFrame(
        old["data"],
        columns=pd.MultiIndex.from_tuples(old["columns"]),
        index=pd.to_datetime(old["index"], utc=True),
    )
    expected = raw["Adj Close"]["SPY"].pct_change(fill_method=None).dropna().to_frame()
    assert_frame_equal(actual, expected, check_dtype=False)
    assert calls[0][1]["start"] == "2024-01-02"
    assert calls[0][1]["end"] == "2024-01-06"
    assert calls[0][1]["auto_adjust"] is True
    assert calls[0][1]["actions"] is False


@pytest.mark.parametrize(
    "start,end",
    [
        (pd.Timestamp("2024-01-02T12:15:00-05:00"), pd.Timestamp("2024-01-06T09:15:00-05:00")),
        ("2024-01-02", "2024-01-06"),
    ],
)
def test_legacy_yahoo_bounds_reach_the_same_upstream_sdk_without_day_truncation(
    captured_yahoo, start, end
):
    calls, _, _ = captured_yahoo
    compat.get_symbol_returns_from_yahoo("SPY", start, end)
    assert calls[0][1]["start"] == start
    assert calls[0][1]["end"] == end


def test_missing_yahoo_price_is_not_filled_or_dropped_before_return_derivation(captured_yahoo):
    _, frame, _ = captured_yahoo
    frame[("Close", "SPY")] = [100, np.nan, 102, 101]
    actual = compat.get_symbol_returns_from_yahoo("SPY", "2024-01-02", "2024-01-06")
    expected = pd.DataFrame({"SPY": [-1 / 102]}, index=pd.to_datetime(["2024-01-05"], utc=True))
    assert_frame_equal(actual, expected)


def test_default_returns_callable_preserves_named_series_and_creates_spy_csv(
    captured_yahoo, tmp_path, monkeypatch
):
    monkeypatch.setenv("EMPYRICAL_CACHE_DIR", str(tmp_path))
    frame = compat.get_symbol_returns_from_yahoo("SPY", "2024-01-02", "2024-01-06")
    actual = compat.default_returns_func("SPY", "2024-01-02", "2024-01-06")
    assert_series_equal(actual, frame["SPY"])
    assert (tmp_path / "spy.csv").is_file()
    assert (
        compat.cache_dir({"EMPYRICAL_CACHE_DIR": "legacy", "XDG_CACHE_HOME": "other"}) == "legacy"
    )
    assert compat.cache_dir({"XDG_CACHE_HOME": "other"}) == str(Path("other") / "empyrical")
