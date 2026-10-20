"""Real provider/storage workflows with only the external Databento SDK stubbed."""

from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import polars as pl
import pytest
from polars.testing import assert_frame_equal

pytest.importorskip("databento")

from ml4t.data.providers import SyntheticProvider
from ml4t.data.providers.databento import DataBentoProvider
from ml4t.data.sessions import SessionCompleter
from ml4t.data.storage.config import StorageConfig
from ml4t.data.storage.flat import FlatStorage
from ml4t.data.storage.hive import HiveStorage
from ml4t.data.storage.metadata_tracker import MetadataTracker
from ml4t.data.update_manager import (
    BackfillManager,
    GapDetector,
    IncrementalUpdater,
    UpdateStrategy,
)
from ml4t.specs import FeedSpec


def instant(clock):
    return datetime.fromisoformat(f"2026-09-05T{clock}:00+00:00")


class RecordedStore:
    def __init__(self, frame):
        self.frame = frame

    def to_df(self):
        return (
            self.frame.rename({"timestamp": "ts_event"})
            .drop("symbol")
            .with_columns(pl.lit("BTCU6").alias("raw_symbol"))
            .to_pandas()
        )


@pytest.fixture
def source(monkeypatch):
    with SyntheticProvider(seed=42, calendar_mode="continuous") as producer:
        frame = producer.fetch_ohlcv("BTC.c.0", "2026-09-05", "2026-09-05", "minute")
    frame = frame.filter(
        pl.col("timestamp").dt.strftime("%H:%M").is_in(["12:00", "12:01", "12:02"])
    )
    calls = []

    def get_range(**kwargs):
        calls.append(kwargs)
        return RecordedStore(frame)

    client = SimpleNamespace(timeseries=SimpleNamespace(get_range=get_range))
    monkeypatch.setattr("ml4t.data.providers.databento.Historical", lambda _: client)
    with DataBentoProvider(api_key="offline-unit-test-key") as provider:
        yield provider, frame, calls


def test_databento_timestamp_request_is_not_rounded_or_shifted_to_session_dates(source):
    provider, expected, calls = source
    provider.adjust_session_dates = True
    result = provider.fetch_ohlcv_range(
        "BTC.c.0", instant("12:01"), instant("12:02"), "minute", stype_in="continuous"
    )
    assert calls == [
        {
            "dataset": "GLBX.MDP3",
            "start": instant("12:01"),
            "end": instant("12:02"),
            "symbols": ["BTC.c.0"],
            "schema": "ohlcv-1m",
            "stype_in": "continuous",
        }
    ]
    assert_frame_equal(
        result.select(expected.columns), expected.filter(pl.col("timestamp") == instant("12:01"))
    )
    assert result["contract"].to_list() == ["BTCU6"]


@pytest.mark.parametrize("backend", [FlatStorage, HiveStorage])
def test_calendar_report_downloads_only_actual_cme_observation_and_preserves_stored_rows(
    source, backend, tmp_path
):
    provider, expected, calls = source
    provider.stype_in = "continuous"
    complete = provider._validate_ohlcv(
        provider._transform_data(RecordedStore(expected), "BTC.c.0"), provider.name, "BTC.c.0"
    )
    observed = complete.filter(pl.col("timestamp") != instant("12:01"))
    storage = backend(StorageConfig(base_path=tmp_path))
    feed = FeedSpec(
        time_contract_version=2,
        calendar="CME Globex Crypto",
        data_frequency="minute",
        timestamp_semantics="event_time",
        bar_grid_anchor="session_open",
        bar_label="left",
        partial_bar_policy="include",
    )
    key = "BTC.c.0"
    storage.write(observed, key, {**feed.to_dict(), "provider": "databento"})
    report = GapDetector().detect_gaps_in_storage(storage, key, instant("12:00"), instant("12:02"))
    assert report == [{"start": instant("12:01"), "end": instant("12:01"), "size_bars": 1}]
    result = BackfillManager(storage, MetadataTracker(tmp_path)).execute_backfill(
        key, report, provider
    )
    assert result.success, result.errors
    assert result.rows_added == 1
    assert result.remaining_gaps == []
    assert len(calls) == 1
    assert calls[0]["schema"] == "ohlcv-1m"
    assert calls[0]["stype_in"] == "continuous"
    assert calls[0]["start"] == instant("12:01")
    assert calls[0]["end"] > instant("12:01")
    assert calls[0]["end"] <= instant("12:02")
    actual = storage.read(key).collect()
    assert_frame_equal(actual, complete)
    assert_frame_equal(actual.filter(pl.col("timestamp") != instant("12:01")), observed)
    assert (
        FeedSpec.from_any(storage.get_metadata(key)["custom"]).resolve_calendar()
        == "CME Globex Crypto"
    )


@pytest.mark.parametrize(
    "start,end",
    [
        (datetime(2026, 9, 5, 12, 1), instant("12:02")),
        (instant("12:01"), datetime(2026, 9, 5, 12, 2)),
        (instant("12:01"), instant("12:01")),
        (instant("12:02"), instant("12:01")),
    ],
)
def test_invalid_timestamp_bounds_are_rejected_before_a_billable_request(source, start, end):
    provider, _, calls = source
    with pytest.raises(ValueError, match="aware|before|UTC|range"):
        provider.fetch_ohlcv_range("BTC.c.0", start, end, "minute")
    assert calls == []


@pytest.mark.parametrize("frequency", ["ticks", "weekly", "nonsense"])
def test_unsupported_targeted_schema_is_not_silently_downloaded_as_daily(source, frequency):
    provider, _, calls = source
    with pytest.raises(ValueError, match="frequency|OHLCV|supported"):
        provider.fetch_ohlcv_range("BTC.c.0", instant("12:01"), instant("12:02"), frequency)
    assert calls == []


def test_targeted_single_series_cannot_expand_a_parent_into_unrelated_contracts(source):
    provider, _, calls = source
    with pytest.raises(ValueError, match="stype|single|parent"):
        provider.fetch_ohlcv_range(
            "BTC.FUT", instant("12:01"), instant("12:02"), "minute", stype_in="parent"
        )
    assert calls == []


def stored_source(source, backend, tmp_path, *, key="BTC.c.0", alias=None):
    provider, expected, calls = source
    provider.stype_in = "continuous"
    complete = provider._validate_ohlcv(
        provider._transform_data(RecordedStore(expected), "BTC.c.0"), provider.name, "BTC.c.0"
    )
    if alias is not None:
        complete = complete.with_columns(pl.lit(alias).alias("symbol"))
    observed = complete.filter(pl.col("timestamp") != instant("12:01"))
    storage = backend(StorageConfig(base_path=tmp_path))
    declaration = FeedSpec(
        time_contract_version=2,
        calendar="CME Globex Crypto",
        data_frequency="minute",
        timestamp_semantics="event_time",
        bar_grid_anchor="session_open",
        bar_label="left",
        partial_bar_policy="include",
    )
    storage.write(observed, key, {**declaration.to_dict(), "provider": "databento"})
    report = GapDetector().detect_gaps_in_storage(storage, key, instant("12:00"), instant("12:02"))
    return provider, complete, observed, calls, storage, report


@pytest.mark.parametrize("backend", [FlatStorage, HiveStorage])
def test_storage_key_is_not_mistaken_for_the_vendor_instrument(source, backend, tmp_path):
    key = "futures/minute/BTC"
    provider, complete, observed, calls, storage, report = stored_source(
        source, backend, tmp_path, key=key, alias="BTC"
    )
    result = BackfillManager(storage, MetadataTracker(tmp_path)).execute_backfill(
        key, report, provider, source_symbol="BTC.c.0"
    )
    assert result.success, result.errors
    assert result.rows_added == 1
    assert result.remaining_gaps == []
    assert calls[0]["symbols"] == ["BTC.c.0"]
    actual = storage.read(key).collect()
    assert_frame_equal(actual.select(complete.columns), complete)
    assert_frame_equal(
        actual.filter(pl.col("timestamp") != instant("12:01")).select(observed.columns), observed
    )
    assert actual.filter(pl.col("timestamp") == instant("12:01"))["contract"].item() == "BTCU6"


@pytest.mark.parametrize("backend", [FlatStorage, HiveStorage])
def test_no_trade_response_keeps_the_actual_gap_and_does_not_rewrite_storage(
    source, backend, tmp_path, monkeypatch
):
    provider, _, observed, calls, storage, report = stored_source(source, backend, tmp_path)
    original = storage.get_metadata("BTC.c.0")

    def no_trades(**kwargs):
        calls.append(kwargs)
        return RecordedStore(observed.head(0))

    monkeypatch.setattr(provider.client.timeseries, "get_range", no_trades)
    result = BackfillManager(storage, MetadataTracker(tmp_path)).execute_backfill(
        "BTC.c.0", report, provider
    )
    assert result.success, result.errors
    assert result.rows_added == result.gaps_filled == 0
    assert result.remaining_gaps == report
    assert len(calls) == 1
    assert storage.get_metadata("BTC.c.0") == original
    assert_frame_equal(storage.read("BTC.c.0").collect(), observed)


@pytest.mark.parametrize("backend", [FlatStorage, HiveStorage])
def test_a_stale_report_does_not_redownload_an_observation_already_recovered(
    source, backend, tmp_path
):
    provider, complete, _, calls, storage, report = stored_source(source, backend, tmp_path)
    storage.write(complete, "BTC.c.0", preserve_metadata=True)
    original = storage.get_metadata("BTC.c.0")
    result = BackfillManager(storage, MetadataTracker(tmp_path)).execute_backfill(
        "BTC.c.0", report, provider
    )
    assert result.success, result.errors
    assert result.rows_added == result.gaps_filled == 0
    assert result.remaining_gaps == []
    assert calls == []
    assert storage.get_metadata("BTC.c.0") == original


def test_timezone_aware_offsets_preserve_the_exact_requested_instants(source):
    provider, _, calls = source
    start = datetime.fromisoformat("2026-09-05T08:01:00-04:00")
    provider.fetch_ohlcv_range("BTC.c.0", start, start + timedelta(minutes=1), "minute")
    assert calls[0]["start"] == instant("12:01")
    assert calls[0]["end"] == instant("12:02")


@pytest.mark.parametrize("backend", [FlatStorage, HiveStorage])
def test_real_nullable_grid_placeholder_is_replaced_only_by_a_source_observation(
    source, backend, tmp_path
):
    provider, complete, observed, calls, storage, _ = stored_source(source, backend, tmp_path)
    expanded = SessionCompleter("CME Globex Crypto").complete_sessions(observed)
    assert expanded["is_imputed"].to_list() == [False, True, False]
    storage.write(expanded, "BTC.c.0", preserve_metadata=True)
    report = GapDetector().detect_gaps_in_storage(
        storage, "BTC.c.0", instant("12:00"), instant("12:02")
    )
    result = BackfillManager(storage, MetadataTracker(tmp_path)).execute_backfill(
        "BTC.c.0", report, provider
    )
    assert result.success, result.errors
    assert result.rows_added == 0
    assert result.rows_updated == 1
    assert result.remaining_gaps == []
    actual = storage.read("BTC.c.0").collect()
    assert expanded.schema["timestamp"] == pl.Datetime("ns", "UTC")
    assert_frame_equal(
        actual.select(complete.columns),
        complete.with_columns(pl.col("timestamp").cast(pl.Datetime("ns", "UTC"))),
    )
    assert not actual["is_imputed"].any()
    assert len(calls) == 1


@pytest.mark.parametrize("backend", [FlatStorage, HiveStorage])
def test_generic_backfill_cannot_put_another_instrument_in_a_missing_bar(source, backend, tmp_path):
    _, complete, observed, _, storage, _ = stored_source(source, backend, tmp_path)
    wrong = complete.with_columns(pl.lit("MBT.C.0").alias("symbol"))
    original = storage.get_metadata("BTC.c.0")
    with pytest.raises(ValueError, match="instrument|entity|identity"):
        IncrementalUpdater().apply_strategy(storage, "BTC.c.0", wrong, UpdateStrategy.BACKFILL)
    assert_frame_equal(storage.read("BTC.c.0").collect(), observed)
    assert storage.get_metadata("BTC.c.0") == original


@pytest.mark.parametrize("symbol", ["ALL_SYMBOLS", " ALL_SYMBOLS ", "ALL_SYMBOLS,", "BTCU6,MBTU6"])
def test_targeted_single_series_cannot_expand_to_all_vendor_instruments(source, symbol):
    provider, _, calls = source
    with pytest.raises(ValueError, match="single|universe|instrument"):
        provider.fetch_ohlcv_range(symbol, instant("12:01"), instant("12:02"), "minute")
    assert calls == []


@pytest.mark.parametrize("frequency", ["1min", "1minute"])
def test_native_minute_alias_does_not_become_daily_or_lose_targeted_support(source, frequency):
    provider, _, calls = source
    frame = provider.fetch_ohlcv_range("BTC.c.0", instant("12:01"), instant("12:02"), frequency)
    assert len(frame) == 1
    assert calls[0]["schema"] == "ohlcv-1m"


@pytest.mark.parametrize("backend", [FlatStorage, HiveStorage])
def test_partial_source_response_reports_the_remaining_missing_bar(source, backend, tmp_path):
    provider, complete, _, calls, storage, _ = stored_source(source, backend, tmp_path)
    observed = complete.head(1)
    storage.write(observed, "BTC.c.0", preserve_metadata=True)
    report = GapDetector().detect_gaps_in_storage(
        storage, "BTC.c.0", instant("12:00"), instant("12:03")
    )
    result = BackfillManager(storage, MetadataTracker(tmp_path)).execute_backfill(
        "BTC.c.0", report, provider
    )
    assert result.success, result.errors
    assert result.rows_added == 2
    assert result.gaps_filled == 0
    assert result.remaining_gaps == [
        {"start": instant("12:03"), "end": instant("12:03"), "size_bars": 1}
    ]
    assert len(calls) == 1
    assert_frame_equal(storage.read("BTC.c.0").collect(), complete)


@pytest.mark.parametrize("backend", [FlatStorage, HiveStorage])
def test_leading_and_trailing_report_ranges_are_both_recovered(source, backend, tmp_path):
    provider, complete, _, calls, storage, _ = stored_source(source, backend, tmp_path)
    storage.write(complete.slice(1, 1), "BTC.c.0", preserve_metadata=True)
    report = GapDetector().detect_gaps_in_storage(
        storage, "BTC.c.0", instant("12:00"), instant("12:02")
    )
    result = BackfillManager(storage, MetadataTracker(tmp_path)).execute_backfill(
        "BTC.c.0", report, provider
    )
    assert result.success, result.errors
    assert result.rows_added == 2
    assert result.gaps_filled == 2
    assert result.remaining_gaps == []
    assert len(calls) == 2
    assert_frame_equal(storage.read("BTC.c.0").collect(), complete)


@pytest.mark.parametrize("backend", [FlatStorage, HiveStorage])
def test_gap_recovery_does_not_replace_a_present_observation_containing_a_null(
    source, backend, tmp_path
):
    provider, _, observed, _, storage, report = stored_source(source, backend, tmp_path)
    observed = observed.with_columns(pl.lit(None, dtype=pl.Float64).alias("close"))
    storage.write(observed, "BTC.c.0", preserve_metadata=True)
    result = BackfillManager(storage, MetadataTracker(tmp_path)).execute_backfill(
        "BTC.c.0", report, provider
    )
    assert result.success, result.errors
    assert result.rows_added == 1
    assert_frame_equal(
        storage.read("BTC.c.0").collect().filter(pl.col("timestamp") != instant("12:01")),
        observed,
    )


@pytest.mark.parametrize("backend", [FlatStorage, HiveStorage])
def test_intraday_candidate_scan_uses_bar_counts_instead_of_discarding_the_dataset(
    source, backend, tmp_path
):
    _, _, _, _, storage, report = stored_source(source, backend, tmp_path)
    candidates = BackfillManager(storage, MetadataTracker(tmp_path)).identify_candidates(
        max_age_days=10000, min_gap_bars=1
    )
    assert len(candidates) == 1
    assert candidates[0]["gaps"] == report
    assert candidates[0]["total_gap_bars"] == 1
    assert "total_gap_days" not in candidates[0]


@pytest.mark.parametrize("backend", [FlatStorage, HiveStorage])
def test_documented_targeted_download_uses_actual_report_and_storage(
    source, backend, tmp_path, monkeypatch
):
    key = "futures/minute/BTC"
    _, complete, _, calls, storage, _ = stored_source(
        source, backend, tmp_path, key=key, alias="BTC"
    )
    monkeypatch.setenv("DATABENTO_API_KEY", "offline-unit-test-key")
    guide = Path(__file__).resolve().parents[1] / "docs/user-guide/incremental-updates.md"
    section = guide.read_text().split("### Download Real Observations for a Selected Report", 1)[1]
    code = section.split("```python\n", 1)[1].split("```", 1)[0]
    namespace = {
        "storage": storage,
        "feed": FeedSpec.from_any(storage.get_metadata(key)["custom"]),
        "GapDetector": GapDetector,
        "datetime": datetime,
        "UTC": UTC,
    }
    exec(compile(code, str(guide), "exec"), namespace)
    assert namespace["result"].rows_added == 1
    assert namespace["result"].remaining_gaps == []
    assert len(calls) == 1
    assert calls[0]["symbols"] == ["BTC.c.0"]
    assert_frame_equal(storage.read(key).collect().select(complete.columns), complete)


def test_documented_native_timestamp_request_uses_exact_half_open_bounds(source, monkeypatch):
    _, _, calls = source
    monkeypatch.setenv("DATABENTO_API_KEY", "offline-unit-test-key")
    guide = Path(__file__).resolve().parents[1] / "docs/providers/databento.md"
    section = guide.read_text().split("## Exact Timestamp Ranges", 1)[1]
    code = section.split("```python\n", 1)[1].split("```", 1)[0]
    namespace = {"DataBentoProvider": DataBentoProvider}
    exec(compile(code, str(guide), "exec"), namespace)
    assert len(namespace["rows"]) == 1
    assert calls[0]["start"] == instant("12:01")
    assert calls[0]["end"] == instant("12:02")
    assert calls[0]["stype_in"] == "continuous"


@pytest.mark.parametrize("mode", ["unsupported_provider", "wrong_provider", "disabled"])
def test_unsupported_recovery_fails_without_vendor_calls_or_storage_mutation(
    source, tmp_path, mode
):
    provider, _, observed, calls, storage, report = stored_source(source, FlatStorage, tmp_path)
    kwargs = {}
    if mode == "wrong_provider":
        storage.write(observed, "BTC.c.0", {"provider": "coinmetrics"}, preserve_metadata=True)
    elif mode == "disabled":
        kwargs["calendar"] = None
    before = storage.get_metadata("BTC.c.0")
    with SyntheticProvider() as unsupported:
        result = BackfillManager(storage, MetadataTracker(tmp_path)).execute_backfill(
            "BTC.c.0", report, unsupported if mode == "unsupported_provider" else provider, **kwargs
        )
    assert not result.success
    assert result.errors
    assert calls == []
    assert storage.get_metadata("BTC.c.0") == before
    assert_frame_equal(storage.read("BTC.c.0").collect(), observed)


def test_history_write_failure_still_reports_the_observations_actually_published(source, tmp_path):
    provider, complete, _, _, storage, report = stored_source(source, FlatStorage, tmp_path)
    tracker = MetadataTracker(tmp_path)
    tracker._get_metadata_path("BTC.c.0").mkdir()
    result = BackfillManager(storage, tracker).execute_backfill("BTC.c.0", report, provider)
    assert not result.success
    assert any(
        error.startswith("Observations stored but update history failed:")
        for error in result.errors
    )
    assert result.rows_added == 1
    assert result.rows_after == 3
    assert result.remaining_gaps == []
    assert_frame_equal(storage.read("BTC.c.0").collect(), complete)


def test_an_empty_actual_report_completes_without_download_or_storage_changes(source, tmp_path):
    provider, complete, _, calls, storage, _ = stored_source(source, FlatStorage, tmp_path)
    storage.write(complete, "BTC.c.0", preserve_metadata=True)
    report = GapDetector().detect_gaps_in_storage(
        storage, "BTC.c.0", instant("12:00"), instant("12:02")
    )
    assert report == []
    before = storage.get_metadata("BTC.c.0")
    result = BackfillManager(storage, MetadataTracker(tmp_path)).execute_backfill(
        "BTC.c.0", report, provider
    )
    assert result.success, result.errors
    assert result.rows_added == result.gaps_filled == 0
    assert result.rows_before == result.rows_after == 3
    assert result.remaining_gaps == []
    assert calls == []
    assert storage.get_metadata("BTC.c.0") == before
    assert_frame_equal(storage.read("BTC.c.0").collect(), complete)
