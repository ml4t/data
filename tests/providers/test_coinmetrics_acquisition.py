"""Public candle acquisition and disk recovery using the official Coin Metrics example."""

import copy
import json
import os
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from ml4t.data import DataManager
from ml4t.data.core.exceptions import (
    AuthenticationError,
    DataValidationError,
    NetworkError,
    RateLimitError,
)
from ml4t.data.core.models import Frequency
from ml4t.data.providers.base import BaseProvider
from ml4t.data.providers.registry import get_provider_spec
from ml4t.data.storage.config import StorageConfig
from ml4t.data.storage.flat import FlatStorage
from ml4t.data.storage.hive import HiveStorage
from ml4t.data.update_manager import BackfillManager, GapDetector
from ml4t.specs import FeedSpec

CAPTURE = json.loads((Path(__file__).parents[1] / "fixtures/coinmetrics_candles.json").read_text())
ROWS = CAPTURE["response"]["data"]
MARKET = "coinbase-btc-usd-spot"
START = datetime(2020, 6, 8, 20, 45, tzinfo=UTC)
END = START + timedelta(minutes=10)


@pytest.fixture
def provider():
    cls = get_provider_spec("coinmetrics").load_class()
    with cls(api_key="offline-unit-test-key", rate_limit=(10000, 1)) as provider:
        yield provider


def test_public_manager_can_acquire_coinmetrics_market_candles(use_transport):
    """Prevent missing typed configuration/registry routing for the supported source."""
    with DataManager(providers={"coinmetrics": {"api_key": "offline-unit-test-key"}}) as manager:
        provider = manager._get_provider("coinmetrics")
        requests = use_transport(provider, lambda _: httpx.Response(200, json={"data": ROWS}))
        result = manager.fetch(
            MARKET, "2020-06-08", "2020-06-08", frequency="5min", provider="coinmetrics"
        )
    assert result.height == 2
    assert result["symbol"].to_list() == [MARKET.upper()] * 2
    assert requests[0].url.params["end_time"].startswith("2020-06-09T00:00:00")
    assert provider.session.is_closed


@pytest.mark.parametrize(
    "frequency,native_frequency",
    [
        (Frequency.FIVE_MINUTE, "5m"),
        (Frequency.FIFTEEN_MINUTE, "15m"),
        (Frequency.THIRTY_MINUTE, "30m"),
        (Frequency.HOUR, "1h"),
        ("10min", "10m"),
    ],
)
def test_manager_frequency_values_reach_native_candle_intervals(
    use_transport, frequency, native_frequency
):
    """Prevent supported Data frequencies failing before a valid candle request."""
    with DataManager(providers={"coinmetrics": {"api_key": "offline-unit-test-key"}}) as manager:
        provider = manager._get_provider("coinmetrics")
        requests = use_transport(provider, lambda _: httpx.Response(200, json={"data": []}))

        result = manager.fetch(
            MARKET, "2020-06-08", "2020-06-08", frequency=frequency, provider="coinmetrics"
        )

    assert result.is_empty()
    assert requests[0].url.params["frequency"] == native_frequency


def test_exact_range_pages_without_losing_filters_or_keeping_end_bar(provider, use_transport):
    """Prevent rounded dates, lost pagination and accidental inclusive-end duplication."""
    calls = []

    def serve(request):
        calls.append(request)
        q = request.url.params
        assert q["markets"] == MARKET
        assert q["frequency"] == "5m"
        assert q["start_inclusive"] == "true" and q["end_inclusive"] == "false"
        assert datetime.fromisoformat(q["start_time"]) == START
        assert datetime.fromisoformat(q["end_time"]) == START + timedelta(minutes=5)
        if len(calls) == 1:
            return httpx.Response(
                200,
                json={
                    "data": [],
                    "next_page_url": str(
                        request.url.copy_add_param("next_page_token", "opaque-token")
                    ),
                },
            )
        assert q["next_page_token"] == "opaque-token"
        return httpx.Response(200, json={"data": ROWS})

    use_transport(provider, serve)
    result = provider.fetch_ohlcv_range(MARKET, START, START + timedelta(minutes=5), "5min")
    assert result.height == 1
    assert result["timestamp"].to_list() == [START]
    assert result["open"].to_list() == [float(ROWS[0]["price_open"])]
    assert result["volume"].to_list() == [float(ROWS[0]["volume"])]
    assert len(calls) == 2


@pytest.mark.parametrize("backend", [FlatStorage, HiveStorage])
@pytest.mark.parametrize("response_rows,expected_added", [(ROWS, 1), ([], 0)])
def test_disk_report_targets_source_and_preserves_real_rows_and_residuals(
    provider, use_transport, tmp_path, backend, response_rows, expected_added
):
    """Prevent fabricated no-trade bars, broad downloads, or overwritten stored observations."""
    use_transport(provider, lambda _: httpx.Response(200, json={"data": ROWS}))
    complete = provider.fetch_ohlcv_range(MARKET, START, END, "5min")
    observed = complete.filter(pl.col("timestamp") != START)
    storage = backend(StorageConfig(base_path=tmp_path))
    feed = FeedSpec(
        time_contract_version=2,
        calendar="24/7",
        data_frequency="5min",
        timestamp_semantics="event_time",
        bar_grid_anchor="session_open",
        bar_label="left",
        partial_bar_policy="include",
    )
    key = "crypto/candles/BTC"
    metadata = {**feed.to_dict(), "provider": "coinmetrics", "source_symbol": MARKET}
    storage.write(observed, key, metadata, preserve_metadata=True)
    before_metadata = storage.get_metadata(key)
    report = GapDetector().detect_gaps_in_storage(storage, key, START, START + timedelta(minutes=5))
    assert report == [{"start": START, "end": START, "size_bars": 1}]
    calls = use_transport(provider, lambda _: httpx.Response(200, json={"data": response_rows}))
    result = BackfillManager(storage, None).execute_backfill(key, report, provider)
    assert result.success, result.errors
    assert result.rows_added == expected_added
    assert result.remaining_gaps == ([] if expected_added else report)
    assert len(calls) == 1
    assert calls[0].url.params["markets"] == MARKET
    assert datetime.fromisoformat(calls[0].url.params["start_time"]) == START
    assert datetime.fromisoformat(calls[0].url.params["end_time"]) == START + timedelta(minutes=5)
    actual = storage.read(key).collect()
    assert_frame_equal(actual, complete if expected_added else observed)
    assert_frame_equal(actual.filter(pl.col("timestamp") != START), observed)
    actual_metadata = storage.get_metadata(key)["custom"]
    for field, value in before_metadata["custom"].items():
        assert actual_metadata[field] == value


@pytest.mark.parametrize(
    "symbol,frequency,start,end",
    [
        ("coinbase-*-usd-spot", "minute", START, END),
        (MARKET + ",binance-btc-usdt-spot", "minute", START, END),
        ("btc", "minute", START, END),
        (MARKET, "ticks", START, END),
        (MARKET, "minute", START.replace(tzinfo=None), END),
        (MARKET, "minute", START, START),
    ],
)
def test_unsupported_or_ambiguous_requests_fail_before_network(
    provider, use_transport, symbol, frequency, start, end
):
    calls = use_transport(provider, lambda _: pytest.fail("invalid request reached network"))
    with pytest.raises(ValueError):
        provider.fetch_ohlcv_range(symbol, start, end, frequency)
    assert not calls


@pytest.mark.parametrize(
    "payload",
    [
        [],
        {},
        {"data": {}},
        {"data": [dict(ROWS[0], market="binance-btc-usdt-spot")]},
        {"data": [dict(ROWS[0], price_close="NaN")]},
        {"data": [dict(ROWS[0], time="not-a-time")]},
        {"data": [dict(ROWS[0], price_high="1")]},
        {"data": [], "next_page_token": "missing-url"},
    ],
)
def test_malformed_source_data_is_not_stored_as_observations(provider, use_transport, payload):
    use_transport(provider, lambda _: httpx.Response(200, json=payload))
    with pytest.raises(DataValidationError):
        provider.fetch_ohlcv_range(MARKET, START, END, "5min")


@pytest.mark.parametrize(
    "next_url",
    [
        "https://evil.invalid/v4/timeseries/market-candles?next_page_token=a",
        "https://api.coinmetrics.io/v4/timeseries/asset-metrics?next_page_token=a",
        "https://api.coinmetrics.io/v4/timeseries/market-candles?markets=binance-btc-usdt-spot&next_page_token=a",
        "https://api.coinmetrics.io/v4/timeseries/market-candles?start_time=2000-01-01&next_page_token=a",
    ],
)
def test_continuation_cannot_leak_credentials_or_expand_market_range(
    provider, use_transport, next_url
):
    calls = use_transport(
        provider, lambda _: httpx.Response(200, json={"data": [], "next_page_url": next_url})
    )
    with pytest.raises(DataValidationError):
        provider.fetch_ohlcv_range(MARKET, START, END, "5min")
    assert len(calls) == 1


def test_repeated_continuation_is_bounded(provider, use_transport):
    next_url = "https://api.coinmetrics.io/v4/timeseries/market-candles?next_page_token=stuck"
    calls = use_transport(
        provider, lambda _: httpx.Response(200, json={"data": ROWS, "next_page_url": next_url})
    )
    with pytest.raises(DataValidationError, match="repeated"):
        provider.fetch_ohlcv_range(MARKET, START, END, "5min")
    assert len(calls) == 2


@pytest.mark.parametrize(
    "status,expected",
    [
        (401, AuthenticationError),
        (403, AuthenticationError),
        (429, RateLimitError),
        (503, NetworkError),
    ],
)
def test_http_failures_use_shared_contract_without_secret_leaks(
    provider, use_transport, monkeypatch, status, expected
):
    monkeypatch.setattr(BaseProvider._request.retry, "sleep", lambda _: None)
    use_transport(
        provider,
        lambda _: httpx.Response(
            status, headers={"Retry-After": "2"}, json={"error": provider.api_key}
        ),
    )
    with pytest.raises(expected) as caught:
        provider.fetch_ohlcv_range(MARKET, START, END, "5min")
    assert provider.api_key not in str(caught.value)


def test_conflicting_page_duplicates_fail_instead_of_picking_a_price(provider, use_transport):
    changed = copy.deepcopy(ROWS[0])
    changed["price_close"] = "9706"
    use_transport(provider, lambda _: httpx.Response(200, json={"data": [ROWS[0], changed]}))
    with pytest.raises(DataValidationError, match="Conflicting"):
        provider.fetch_ohlcv_range(MARKET, START, END, "5min")


@pytest.mark.parametrize("backend", [FlatStorage, HiveStorage])
def test_partial_download_keeps_unreturned_observations_in_residual_report(
    provider, use_transport, tmp_path, backend
):
    use_transport(provider, lambda _: httpx.Response(200, json={"data": ROWS}))
    complete = provider.fetch_ohlcv_range(MARKET, START, END, "5min")
    stored = complete.filter(pl.col("timestamp") > START)
    storage = backend(StorageConfig(base_path=tmp_path))
    feed = FeedSpec(
        time_contract_version=2,
        calendar="24/7",
        data_frequency="5min",
        timestamp_semantics="event_time",
        bar_grid_anchor="session_open",
        bar_label="left",
        partial_bar_policy="include",
    )
    storage.write(
        stored, "BTC", {**feed.to_dict(), "provider": "coinmetrics", "source_symbol": MARKET}
    )
    missing_start = START - timedelta(minutes=5)
    detector = GapDetector()
    report = detector.detect_gaps_in_storage(storage, "BTC", missing_start, START)
    assert report == [{"start": missing_start, "end": START, "size_bars": 2}]
    calls = use_transport(provider, lambda _: httpx.Response(200, json={"data": ROWS}))
    result = BackfillManager(storage, None).execute_backfill("BTC", report, provider)
    assert result.success and result.rows_added == 1 and result.gaps_filled == 0
    assert result.remaining_gaps == [{"start": missing_start, "end": missing_start, "size_bars": 1}]
    assert len(calls) == 1
    assert_frame_equal(storage.read("BTC").collect(), complete)


def test_transport_error_does_not_expose_encoded_credential(provider, use_transport, monkeypatch):
    monkeypatch.setattr(BaseProvider._request.retry, "sleep", lambda _: None)

    def fail(request):
        raise httpx.ConnectError(f"failed request {request.url}", request=request)

    use_transport(provider, fail)
    with pytest.raises(NetworkError) as error:
        provider.fetch_ohlcv_range(MARKET, START, END, "5min")
    assert provider.api_key not in str(error.value)


def test_invalid_continuation_url_is_typed_validation_error(provider, use_transport):
    use_transport(
        provider,
        lambda _: httpx.Response(200, json={"data": [], "next_page_url": "https://[broken"}),
    )
    with pytest.raises(DataValidationError):
        provider.fetch_ohlcv_range(MARKET, START, END, "5min")


@pytest.mark.parametrize("explicit_close", [True, False])
def test_provider_error_and_instrumented_session_do_not_segfault_at_interpreter_exit(
    tmp_path, explicit_close
):
    """Prevent destructor logging into torn-down datetime state after a real API error."""
    script = """
import httpx
from datetime import UTC, datetime, timedelta
from ml4t.data.providers import CoinMetricsProvider
from ml4t.data.core.exceptions import AuthenticationError

events = []
provider = CoinMetricsProvider(api_key="offline-unit-test-key")
provider.session.close()
provider.session = httpx.Client(transport=httpx.MockTransport(
    lambda request: httpx.Response(403, json={"error": "forbidden"})))
provider.session.event_hooks["response"] = [lambda response: events.append(response.status_code)]
start = datetime(2020, 6, 8, 20, 45, tzinfo=UTC)
try:
    provider.fetch_ohlcv_range("coinbase-btc-usd-spot", start, start+timedelta(minutes=5), "5min")
except AuthenticationError:
    print("expected account error", events)
else:
    raise AssertionError("HTTP403 did not produce the account error")
"""
    if explicit_close:
        script += "provider.close()\nassert provider.session.is_closed\n"
    result = subprocess.run(
        [sys.executable, "-X", "faulthandler", "-c", script],
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[2] / "src")},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "expected account error [403]" in result.stdout
